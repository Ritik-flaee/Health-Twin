"""Shared constants for synthesis, training, and the clinician dashboard."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

SEED = 2026
N_PATIENTS = 150
N_DAYS = 12
STEP_MINUTES = 15
STEPS_PER_DAY = 24 * 60 // STEP_MINUTES
HORIZON_STEPS = 8  # 2 hours
LOOKBACK_STEPS = 96  # 24 hours required before a row is usable

HYPER_THRESHOLD = 180.0  # mg/dL
HYPO_THRESHOLD = 70.0

STATIC_FEATURES = [
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
]

DYNAMIC_FEATURES = [
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
]

FEATURE_COLUMNS = STATIC_FEATURES + DYNAMIC_FEATURES

CGM_ONLY_FEATURES = [
    "glucose_mgdl",
    "glucose_delta_30m",
    "glucose_mean_1h",
    "glucose_std_2h",
]
