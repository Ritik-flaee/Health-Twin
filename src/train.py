"""Train the fusion model and save a clinician-ready demo slice.

The split is by patient. Thresholds are chosen on validation patients and
frozen before the test patients are scored.
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    CGM_ONLY_FEATURES,
    DATA_DIR,
    FEATURE_COLUMNS,
    MODEL_DIR,
    N_DAYS,
    N_PATIENTS,
    SEED,
)
from src.features import build_feature_table
from src.synthesize import generate_cohort


def _split_ids(patient_ids: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    ids = np.array(sorted(patient_ids))
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def _best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    candidates = np.linspace(0.15, 0.85, 71)
    scores = [f1_score(y_true, proba >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def _clf() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=180,
        min_samples_leaf=40,
        l2_regularization=0.1,
        random_state=SEED,
        class_weight="balanced",
    )


def _bundle_metrics(name: str, y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = proba >= threshold
    out = {
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else None,
        "average_precision": float(average_precision_score(y_true, proba)) if y_true.sum() else None,
        "brier": float(brier_score_loss(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "prevalence": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }
    return out


def _slice_auc(frame: pd.DataFrame, proba: np.ndarray, y_col: str = "y_hyper") -> float | None:
    mask = frame["glucose_mgdl"].to_numpy() < 140
    y = frame[y_col].to_numpy()[mask]
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, proba[mask]))


def _pick_demo_moments(test: pd.DataFrame) -> dict:
    rising = test["glucose_delta_30m"].abs() <= 10
    spike = test[(test["glucose_mgdl"] < 145) & (test["y_hyper"] == 1) & rising].sort_values(
        ["hyper_proba", "patient_id", "timestamp"], ascending=[False, True, True]
    )
    if spike.empty:
        spike = test[(test["glucose_mgdl"] < 160) & (test["y_hyper"] == 1)].sort_values(
            ["hyper_proba", "patient_id", "timestamp"], ascending=[False, True, True]
        )
    calm = test[(test["glucose_mgdl"] < 130) & (test["y_hyper"] == 0) & (test["hyper_proba"] < 0.25)]
    calm = calm.sort_values(
        ["hyper_proba", "patient_id", "timestamp"], ascending=[True, True, True]
    )

    def pack(row: pd.Series, label: str) -> dict:
        return {
            "label": label,
            "patient_id": row["patient_id"],
            "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
            "glucose_mgdl": float(row["glucose_mgdl"]),
            "hyper_proba": float(row["hyper_proba"]),
            "future_peak": float(row["future_peak"]),
            "predicted_peak": float(row["predicted_peak"]),
        }

    moments = {}
    if not spike.empty:
        moments["spike"] = pack(spike.iloc[0], "Predicted spike while glucose is still in range")
    if not calm.empty:
        moments["calm"] = pack(calm.iloc[0], "Quiet hour")
    return moments


def _time_split_metrics(fused: pd.DataFrame) -> dict:
    """Train on the first eight days and score the last three.

    The two-hour label is kept from crossing the cut. The same people can
    appear on both sides, so this is a later-week check, separate from the
    unseen-patient test.
    """
    start = fused["timestamp"].min()
    train_end = start + pd.Timedelta(days=8) - pd.Timedelta(hours=2)
    test_start = start + pd.Timedelta(days=9)
    train_rows = fused[fused["timestamp"] <= train_end]
    test_rows = fused[fused["timestamp"] >= test_start]
    model = _clf()
    model.fit(train_rows[FEATURE_COLUMNS], train_rows["y_hyper"])
    proba = model.predict_proba(test_rows[FEATURE_COLUMNS])[:, 1]
    y = test_rows["y_hyper"].to_numpy()
    return {
        "description": "Days 1-8 train, days 10-12 test, labels kept off the cut",
        "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else None,
        "n_train": int(len(train_rows)),
        "n_test": int(len(test_rows)),
        "prevalence": float(np.mean(y)),
    }


def _permutation_importance(model, frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    sample = frame.sample(n=min(4000, len(frame)), random_state=SEED)
    x = sample[FEATURE_COLUMNS]
    y = sample["y_hyper"].to_numpy()
    base = roc_auc_score(y, model.predict_proba(x)[:, 1])
    rows = []
    for col in FEATURE_COLUMNS:
        shuffled = x.copy()
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
        score = roc_auc_score(y, model.predict_proba(shuffled)[:, 1])
        rows.append({"feature": col, "auc_drop": float(base - score)})
    ranked = pd.DataFrame(rows).sort_values("auc_drop", ascending=False)
    return ranked


def train(n_patients: int = N_PATIENTS, n_days: int = N_DAYS, seed: int = SEED) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_patients} synthetic patients x {n_days} days...")
    patients, timeseries = generate_cohort(n_patients, n_days, seed)
    print(f"Building features on {len(timeseries):,} wearable rows...")
    fused = build_feature_table(patients, timeseries)

    rng = np.random.default_rng(seed)
    splits = _split_ids(fused["patient_id"].unique(), rng)
    parts = {name: fused[fused["patient_id"].isin(ids)].copy() for name, ids in splits.items()}
    for name, part in parts.items():
        overlap = set(part["patient_id"]) & set().union(
            *[set(other["patient_id"]) for key, other in parts.items() if key != name]
        )
        if overlap:
            raise RuntimeError(f"Patient leakage in {name}: {overlap}")

    clf = _clf()
    clf.fit(parts["train"][FEATURE_COLUMNS], parts["train"]["y_hyper"])
    cgm = _clf()
    cgm.fit(parts["train"][CGM_ONLY_FEATURES], parts["train"]["y_hyper"])

    reg = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=160,
        min_samples_leaf=40,
        l2_regularization=0.1,
        random_state=seed,
    )
    reg.fit(parts["train"][FEATURE_COLUMNS], parts["train"]["future_peak"])

    hypo = None
    hypo_threshold = None
    if parts["train"]["y_hypo"].sum() >= 80 and parts["val"]["y_hypo"].nunique() > 1:
        hypo = _clf()
        hypo.fit(parts["train"][FEATURE_COLUMNS], parts["train"]["y_hypo"])
        hypo_val = hypo.predict_proba(parts["val"][FEATURE_COLUMNS])[:, 1]
        if parts["val"]["y_hypo"].nunique() > 1:
            hypo_threshold = _best_f1_threshold(parts["val"]["y_hypo"].to_numpy(), hypo_val)

    val_proba = clf.predict_proba(parts["val"][FEATURE_COLUMNS])[:, 1]
    threshold = _best_f1_threshold(parts["val"]["y_hyper"].to_numpy(), val_proba)
    cgm_val = cgm.predict_proba(parts["val"][CGM_ONLY_FEATURES])[:, 1]
    cgm_threshold = _best_f1_threshold(parts["val"]["y_hyper"].to_numpy(), cgm_val)

    test = parts["test"].copy()
    test["hyper_proba"] = clf.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    test["cgm_only_proba"] = cgm.predict_proba(test[CGM_ONLY_FEATURES])[:, 1]
    test["predicted_peak"] = reg.predict(test[FEATURE_COLUMNS])
    if hypo is not None:
        test["hypo_proba"] = hypo.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    else:
        test["hypo_proba"] = 0.0

    y_test = test["y_hyper"].to_numpy()
    metrics = {
        "seed": seed,
        "n_patients": int(n_patients),
        "n_days": int(n_days),
        "target": "Peak CGM glucose >= 180 mg/dL at any point in the next 2 hours",
        "cohort": "Ages 30 to 55 with type 2 diabetes or prediabetes, plus hypertension or another cardiovascular risk factor",
        "split": "Patient-level 70% train / 15% validation / 15% test",
        "n_train_patients": int(len(splits["train"])),
        "n_val_patients": int(len(splits["val"])),
        "n_test_patients": int(len(splits["test"])),
        "n_train_rows": int(len(parts["train"])),
        "n_test_rows": int(len(test)),
        "threshold": threshold,
        "cgm_only_threshold": cgm_threshold,
        "hypo_threshold": hypo_threshold,
        "fusion_test": _bundle_metrics("fusion", y_test, test["hyper_proba"].to_numpy(), threshold),
        "cgm_only_test": _bundle_metrics(
            "cgm", y_test, test["cgm_only_proba"].to_numpy(), cgm_threshold
        ),
        "lead_time_auc_fusion": _slice_auc(test, test["hyper_proba"].to_numpy()),
        "lead_time_auc_cgm_only": _slice_auc(test, test["cgm_only_proba"].to_numpy()),
        "peak_mae": float(mean_absolute_error(test["future_peak"], test["predicted_peak"])),
        "peak_residual_std": float(np.std(test["future_peak"] - test["predicted_peak"])),
        "feature_groups": {
            "ehr": [
                "age",
                "sex_male",
                "bmi",
                "hba1c",
                "fasting_glucose",
                "diabetes_duration_years",
                "on_metformin",
                "on_insulin",
                "on_sglt2",
                "family_history",
                "tcf7l2_alleles",
                "kcnj11_alleles",
                "sbp",
                "ldl",
                "egfr",
                "hypertension",
            ],
            "wearable": [
                "glucose_mgdl",
                "glucose_delta_30m",
                "glucose_mean_1h",
                "glucose_std_2h",
                "glucose_min_6h",
                "glucose_max_6h",
                "heart_rate_1h",
                "hrv_6h",
                "steps_2h",
                "steps_24h",
                "sleep_hours_prev",
                "deep_sleep_min_prev",
                "hour_sin",
                "hour_cos",
            ],
        },
    }
    in_range = test["glucose_mgdl"] < 140
    if in_range.any() and test.loc[in_range, "y_hyper"].nunique() > 1:
        metrics["lead_time_recall_fusion"] = float(
            recall_score(
                test.loc[in_range, "y_hyper"],
                test.loc[in_range, "hyper_proba"] >= threshold,
                zero_division=0,
            )
        )

    metrics["time_split"] = _time_split_metrics(fused)

    print("Scoring feature influence on a test sample...")
    importance = _permutation_importance(clf, test, rng)
    moments = _pick_demo_moments(test)

    patients.to_csv(DATA_DIR / "patients.csv", index=False)
    test_ids = set(splits["test"])
    demo_ts = timeseries[timeseries["patient_id"].isin(test_ids)].copy()
    demo_ts.to_csv(DATA_DIR / "demo_timeseries.csv.gz", index=False, compression="gzip")
    extra_cols = [
        "patient_id",
        "timestamp",
        "future_peak",
        "future_min",
        "y_hyper",
        "y_hypo",
        "hyper_proba",
        "hypo_proba",
        "cgm_only_proba",
        "predicted_peak",
    ]
    keep_cols = list(dict.fromkeys(extra_cols + FEATURE_COLUMNS))
    test[keep_cols].to_csv(DATA_DIR / "scored.csv.gz", index=False, compression="gzip")
    joblib.dump(clf, MODEL_DIR / "glytwin_hyper.joblib")
    joblib.dump(reg, MODEL_DIR / "glytwin_peak.joblib")
    if hypo is not None:
        joblib.dump(hypo, MODEL_DIR / "glytwin_hypo.joblib")
    importance.to_csv(MODEL_DIR / "importance.csv", index=False)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (DATA_DIR / "demo_moment.json").write_text(json.dumps(moments, indent=2), encoding="utf-8")

    fusion_auc = metrics["fusion_test"]["roc_auc"]
    cgm_auc = metrics["cgm_only_test"]["roc_auc"]
    print(
        f"Test ROC-AUC fusion {fusion_auc:.3f} | CGM-only {cgm_auc:.3f} | "
        f"in-range AUC fusion {metrics['lead_time_auc_fusion']:.3f} | "
        f"CGM-only {metrics['lead_time_auc_cgm_only']:.3f}"
    )
    print(f"Threshold {threshold:.2f} | peak MAE {metrics['peak_mae']:.1f} mg/dL")
    print(f"Demo moment: {moments.get('spike', moments.get('calm'))}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data and train GlyTwin")
    parser.add_argument("--patients", type=int, default=N_PATIENTS)
    parser.add_argument("--days", type=int, default=N_DAYS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train(args.patients, args.days, args.seed)


if __name__ == "__main__":
    main()
