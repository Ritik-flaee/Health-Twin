"""Clinician view of one virtual patient at one moment in time."""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import DATA_DIR, FEATURE_COLUMNS, HYPER_THRESHOLD, HYPO_THRESHOLD, MODEL_DIR
from src.explain import explain_row


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    patients = pd.read_csv(DATA_DIR / "patients.csv")
    scored = pd.read_csv(DATA_DIR / "scored.csv.gz", parse_dates=["timestamp"])
    series = pd.read_csv(DATA_DIR / "demo_timeseries.csv.gz", parse_dates=["timestamp"])
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    moments = json.loads((DATA_DIR / "demo_moment.json").read_text(encoding="utf-8"))
    importance = pd.read_csv(MODEL_DIR / "importance.csv")
    scored = scored.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    return {
        "patients": patients,
        "scored": scored,
        "series": series,
        "metrics": metrics,
        "moments": moments,
        "importance": importance,
    }


def artifacts_ready() -> bool:
    needed = [
        DATA_DIR / "patients.csv",
        DATA_DIR / "scored.csv.gz",
        DATA_DIR / "demo_timeseries.csv.gz",
        DATA_DIR / "demo_moment.json",
        MODEL_DIR / "metrics.json",
        MODEL_DIR / "importance.csv",
        MODEL_DIR / "glytwin_hyper.joblib",
        MODEL_DIR / "glytwin_peak.joblib",
    ]
    return all(path.exists() for path in needed)


def patient_table(bundle: dict) -> pd.DataFrame:
    """One row per virtual patient, ranked by the highest risk in the last 48 hours."""
    scored = bundle["scored"].sort_values(["patient_id", "timestamp"])
    latest = scored.groupby("patient_id", as_index=False).tail(1)
    end = scored.groupby("patient_id")["timestamp"].transform("max")
    recent = scored[scored["timestamp"] >= end - pd.Timedelta(hours=48)]
    peak_idx = recent.groupby("patient_id")["hyper_proba"].idxmax()
    peak = scored.loc[peak_idx, ["patient_id", "timestamp", "hyper_proba", "glucose_mgdl", "hypo_proba"]].rename(
        columns={
            "timestamp": "attention_at",
            "hyper_proba": "peak_risk_48h",
            "glucose_mgdl": "attention_glucose",
            "hypo_proba": "attention_hypo",
        }
    )
    chart_cols = ["patient_id", "name", "age", "sex", "city", "bmi", "sbp", "glycemic_status", "hypertension"]
    overlap = [column for column in chart_cols if column != "patient_id" and column in latest.columns]
    latest = latest.drop(columns=overlap)
    chart = bundle["patients"][chart_cols]
    table = latest.merge(chart, on="patient_id", how="left").merge(peak, on="patient_id", how="left")
    table["risk_band"] = pd.cut(
        table["peak_risk_48h"],
        bins=[-0.01, 0.35, 0.6, 1.01],
        labels=["Lower", "Watch", "High"],
    )
    return table.sort_values(["peak_risk_48h", "hyper_proba"], ascending=False)


def clinic_times(bundle: dict) -> list[pd.Timestamp]:
    one = bundle["scored"]["patient_id"].iloc[0]
    times = bundle["scored"].loc[bundle["scored"]["patient_id"] == one, "timestamp"]
    return list(times.sort_values())


def snapshot_ward(bundle: dict, when: pd.Timestamp) -> pd.DataFrame:
    """One row per person at a single clinic time, so the list matches that hour."""
    scored = bundle["scored"]
    snap = scored[scored["timestamp"] == when].copy()
    if snap.empty:
        snap = (
            scored.assign(_gap=(scored["timestamp"] - when).abs())
            .sort_values("_gap")
            .groupby("patient_id", as_index=False)
            .head(1)
            .drop(columns="_gap")
        )
    chart_cols = ["patient_id", "name", "age", "sex", "city", "bmi", "sbp", "glycemic_status", "hypertension"]
    overlap = [column for column in chart_cols if column != "patient_id" and column in snap.columns]
    snap = snap.drop(columns=overlap)
    table = snap.merge(bundle["patients"][chart_cols], on="patient_id", how="left")
    table["attention_at"] = when
    return table


def moment_row(bundle: dict, patient_id: str, timestamp: pd.Timestamp) -> pd.Series:
    scored = bundle["scored"]
    part = scored[scored["patient_id"] == patient_id]
    if part.empty:
        raise KeyError(patient_id)
    exact = part[part["timestamp"] == timestamp]
    if not exact.empty:
        return exact.iloc[0]
    idx = (part["timestamp"] - timestamp).abs().idxmin()
    return part.loc[idx]


def forecast_path(row: pd.Series) -> pd.DataFrame:
    """A two-hour sketch from the current reading toward the predicted peak."""
    start = pd.Timestamp(row["timestamp"])
    current = float(row["glucose_mgdl"])
    peak = float(row["predicted_peak"])
    minutes = [0, 30, 60, 90, 120]
    # Peak is the max over the window, so place it at 60 minutes and ease back.
    if peak >= current:
        values = [
            current,
            current + 0.55 * (peak - current),
            peak,
            current + 0.72 * (peak - current),
            current + 0.5 * (peak - current),
        ]
    else:
        values = [current + (peak - current) * (m / 120) for m in minutes]
    return pd.DataFrame(
        {
            "timestamp": [start + pd.Timedelta(minutes=m) for m in minutes],
            "predicted_glucose": values,
        }
    )


def history_window(bundle: dict, patient_id: str, timestamp: pd.Timestamp, hours: int = 16) -> pd.DataFrame:
    series = bundle["series"]
    part = series[series["patient_id"] == patient_id]
    start = timestamp - pd.Timedelta(hours=hours)
    end = timestamp + pd.Timedelta(hours=2)
    return part[(part["timestamp"] >= start) & (part["timestamp"] <= end)].sort_values("timestamp")


@lru_cache(maxsize=1)
def _models() -> dict:
    hypo_path = MODEL_DIR / "glytwin_hypo.joblib"
    return {
        "hyper": joblib.load(MODEL_DIR / "glytwin_hyper.joblib"),
        "peak": joblib.load(MODEL_DIR / "glytwin_peak.joblib"),
        "hypo": joblib.load(hypo_path) if hypo_path.exists() else None,
    }


def what_if(row: pd.Series, extra_steps: float, sleep_hours: float, meal_delay_min: int) -> dict:
    """Re-score one moment after a walk, a different night's sleep, or a later meal.

    The CGM trace is left as recorded. Meal delay only moves the clock features,
    which this model treats as a stand-in for when the person eats.
    """
    features = {column: row[column] for column in FEATURE_COLUMNS}
    features["steps_2h"] = float(row["steps_2h"]) + extra_steps
    features["steps_24h"] = float(row["steps_24h"]) + extra_steps
    features["sleep_hours_prev"] = float(sleep_hours)
    recorded_sleep = float(row["sleep_hours_prev"])
    if recorded_sleep > 0:
        features["deep_sleep_min_prev"] = float(row["deep_sleep_min_prev"]) * (float(sleep_hours) / recorded_sleep)
    when = pd.Timestamp(row["timestamp"]) + pd.Timedelta(minutes=int(meal_delay_min))
    hour = when.hour + when.minute / 60.0
    features["hour_sin"] = float(np.sin(2 * np.pi * hour / 24.0))
    features["hour_cos"] = float(np.cos(2 * np.pi * hour / 24.0))
    frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    models = _models()
    hyper = float(models["hyper"].predict_proba(frame)[0, 1])
    peak = float(models["peak"].predict(frame)[0])
    hypo = float(models["hypo"].predict_proba(frame)[0, 1]) if models["hypo"] is not None else 0.0
    return {"hyper_proba": hyper, "predicted_peak": peak, "hypo_proba": hypo}


def view_model(bundle: dict, patient_id: str, timestamp: pd.Timestamp) -> dict:
    row = moment_row(bundle, patient_id, timestamp)
    matches = bundle["patients"][bundle["patients"]["patient_id"] == patient_id]
    patient = matches.iloc[0]
    threshold = float(bundle["metrics"]["threshold"])
    return {
        "row": row,
        "patient": patient,
        "reasons": explain_row(row),
        "forecast": forecast_path(row),
        "history": history_window(bundle, patient_id, pd.Timestamp(row["timestamp"])),
        "alert": float(row["hyper_proba"]) >= threshold,
        "threshold": threshold,
        "hyper_line": HYPER_THRESHOLD,
        "hypo_line": HYPO_THRESHOLD,
    }
