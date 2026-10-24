"""Fuse a static chart with a past-only wearable window.

Labels look forward two hours. Features never do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    HORIZON_STEPS,
    HYPER_THRESHOLD,
    HYPO_THRESHOLD,
    LOOKBACK_STEPS,
    STEP_MINUTES,
)


def future_window_stat(series: pd.Series, horizon: int, how: str) -> pd.Series:
    """Aggregate the next `horizon` samples, excluding the current row."""
    pieces = [series.shift(-k) for k in range(1, horizon + 1)]
    window = pd.concat(pieces, axis=1)
    if how == "max":
        return window.max(axis=1)
    if how == "min":
        return window.min(axis=1)
    raise ValueError(how)


def _previous_night_sleep(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the last completed night. The current night is not included."""
    out = frame.sort_values("timestamp").copy()
    ts = out["timestamp"]
    hour = ts.dt.hour + ts.dt.minute / 60.0
    night_end = ts.dt.normalize()
    night_end = night_end.where(hour < 12, night_end + pd.Timedelta(days=1))
    asleep = out["sleep_stage"].gt(0)
    deep = out["sleep_stage"].eq(2)
    nightly = (
        pd.DataFrame({"night_end": night_end, "asleep": asleep, "deep": deep})
        .groupby("night_end", as_index=False)
        .sum(numeric_only=True)
    )
    nightly["ready_at"] = nightly["night_end"] + pd.Timedelta(hours=7)
    nightly["sleep_hours_prev"] = nightly["asleep"] * STEP_MINUTES / 60.0
    nightly["deep_sleep_min_prev"] = nightly["deep"] * STEP_MINUTES
    nightly = nightly.sort_values("ready_at")
    merged = pd.merge_asof(
        out.sort_values("timestamp"),
        nightly[["ready_at", "sleep_hours_prev", "deep_sleep_min_prev"]],
        left_on="timestamp",
        right_on="ready_at",
        direction="backward",
    )
    return merged.drop(columns=["ready_at"])


def build_feature_table(patients: pd.DataFrame, timeseries: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for patient_id, raw in timeseries.groupby("patient_id", sort=False):
        raw = raw.sort_values("timestamp").copy()
        raw["glucose_delta_30m"] = raw["glucose_mgdl"].diff(2)
        raw["glucose_mean_1h"] = raw["glucose_mgdl"].rolling(4, min_periods=4).mean()
        raw["glucose_std_2h"] = raw["glucose_mgdl"].rolling(8, min_periods=8).std()
        raw["glucose_min_6h"] = raw["glucose_mgdl"].rolling(24, min_periods=24).min()
        raw["glucose_max_6h"] = raw["glucose_mgdl"].rolling(24, min_periods=24).max()
        raw["heart_rate_1h"] = raw["heart_rate"].rolling(4, min_periods=4).mean()
        raw["hrv_6h"] = raw["hrv_rmssd"].rolling(24, min_periods=24).mean()
        raw["steps_2h"] = raw["steps"].rolling(8, min_periods=8).sum()
        raw["steps_24h"] = raw["steps"].rolling(LOOKBACK_STEPS, min_periods=LOOKBACK_STEPS).sum()
        raw["future_peak"] = future_window_stat(raw["glucose_mgdl"], HORIZON_STEPS, "max")
        raw["future_min"] = future_window_stat(raw["glucose_mgdl"], HORIZON_STEPS, "min")
        slept = _previous_night_sleep(raw)
        frames.append(slept)

    fused = pd.concat(frames, ignore_index=True)
    chart = patients.copy()
    chart["sex_male"] = (chart["sex"] == "Male").astype(int)
    keep = [
        "patient_id",
        "name",
        "sex",
        "city",
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
        "glycemic_status",
    ]
    fused = fused.merge(chart[keep], on="patient_id", how="left")
    hour = fused["timestamp"].dt.hour + fused["timestamp"].dt.minute / 60.0
    fused["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    fused["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    fused["y_hyper"] = (fused["future_peak"] >= HYPER_THRESHOLD).astype(int)
    fused["y_hypo"] = (fused["future_min"] <= HYPO_THRESHOLD).astype(int)

    needed = FEATURE_COLUMNS + ["future_peak", "future_min", "y_hyper", "y_hypo", "sleep_hours_prev"]
    fused = fused.dropna(subset=needed).reset_index(drop=True)
    return fused
