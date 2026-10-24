"""Synthetic EHR cohort and wearable time series for a Type 2 diabetes twin.

Every record is generated. Nothing here is a real patient. The generator is
seeded so the cohort, the model, and the dashboard story can be reproduced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import N_DAYS, N_PATIENTS, SEED, STEP_MINUTES, STEPS_PER_DAY

FEMALE_NAMES = [
    "Diya", "Ananya", "Sara", "Meera", "Zoya", "Kiara", "Aisha", "Navya", "Lakshmi", "Pooja",
]
MALE_NAMES = [
    "Aarav", "Ishaan", "Vihaan", "Kabir", "Arjun", "Reyansh", "Dev", "Aditya", "Rahul", "Imran",
]
LAST_NAMES = [
    "Sharma", "Iyer", "Reddy", "Khan", "Patel", "Nair", "Singh", "Das",
    "Gupta", "Joshi", "Menon", "Chowdhury", "Kulkarni", "Banerjee", "Shetty",
]
CITIES = [
    "Bengaluru", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Pune", "Kolkata",
    "Ahmedabad",
]


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def generate_patients(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw a coherent synthetic Type 2 diabetes registry."""
    rows = []
    for i in range(n):
        age = int(rng.integers(30, 56))
        sex = "Male" if rng.random() < 0.56 else "Female"
        bmi = _clip(rng.normal(27.2, 3.4), 19.5, 38.0)
        prediabetes = rng.random() < 0.28
        if prediabetes:
            hba1c = _clip(rng.normal(6.05, 0.22), 5.7, 6.4)
            duration = 0.0
            status = "Prediabetes"
        else:
            duration = _clip(rng.normal(5.5, 3.0), 0.4, 20)
            hba1c = _clip(rng.normal(8.0, 1.1) + max(0, duration - 8) * 0.05, 6.5, 12.0)
            status = "Type 2 diabetes"
        estimated_average = 28.7 * hba1c - 46.7
        fasting = _clip(rng.normal(estimated_average * 0.92, 12), 90, 280)
        sbp = _clip(rng.normal(144 if rng.random() < 0.7 else 126, 9) + (bmi - 25) * 0.4, 108, 175)
        ldl = _clip(rng.normal(128, 26) + (bmi - 25) * 1.1, 55, 220)
        egfr = _clip(rng.normal(96 - max(0, age - 40) * 0.4, 10), 45, 120)
        family = int(rng.random() < 0.48)
        hypertension = int(sbp >= 130)
        if hypertension == 0 and ldl < 130 and family == 0 and bmi < 27:
            sbp = float(rng.integers(132, 148))
            hypertension = 1
        tcf7l2 = int(rng.choice([0, 1, 2], p=[0.42, 0.43, 0.15]))
        kcnj11 = int(rng.choice([0, 1, 2], p=[0.55, 0.36, 0.09]))
        on_metformin = int((not prediabetes and rng.random() < 0.9) or (prediabetes and rng.random() < 0.22))
        on_insulin = int((not prediabetes) and hba1c >= 8.2 and rng.random() < 0.62)
        on_sglt2 = int(hypertension == 1 and hba1c >= 7.0 and rng.random() < 0.3)
        on_bp_med = int(hypertension == 1 and rng.random() < 0.48)
        rows.append(
            {
                "patient_id": f"GT-{i + 1:03d}",
                "name": f"{rng.choice(FEMALE_NAMES if sex == 'Female' else MALE_NAMES)} {rng.choice(LAST_NAMES)}",
                "age": age,
                "sex": sex,
                "bmi": round(bmi, 1),
                "glycemic_status": status,
                "hba1c": round(hba1c, 1),
                "fasting_glucose": round(fasting, 0),
                "diabetes_duration_years": round(duration, 1),
                "on_metformin": on_metformin,
                "on_insulin": on_insulin,
                "on_sglt2": on_sglt2,
                "family_history": family,
                "tcf7l2_alleles": tcf7l2,
                "kcnj11_alleles": kcnj11,
                "sbp": round(sbp, 0),
                "hypertension": hypertension,
                "on_antihypertensive": on_bp_med,
                "ldl": round(ldl, 0),
                "egfr": round(egfr, 0),
                "city": str(rng.choice(CITIES)),
            }
        )
    return pd.DataFrame(rows)


def _add_kernel(target: np.ndarray, center: int, amplitude: float, peak_lag: int = 4, width: float = 3.0, span: int = 18) -> None:
    start = max(0, center - 1)
    end = min(len(target), center + span)
    if start >= end or amplitude == 0:
        return
    xs = np.arange(start, end)
    peak = center + peak_lag
    target[start:end] += amplitude * np.exp(-0.5 * ((xs - peak) / width) ** 2)


def simulate_patient(patient: pd.Series, rng: np.random.Generator, n_days: int, origin: pd.Timestamp) -> pd.DataFrame:
    """Build a 15-minute wearable stream whose spikes depend on EHR and behaviour."""
    n = n_days * STEPS_PER_DAY
    idx = np.arange(n)
    minute_of_day = (idx * STEP_MINUTES) % (24 * 60)
    hour = minute_of_day / 60.0
    day = idx // STEPS_PER_DAY

    hba1c = float(patient["hba1c"])
    bmi = float(patient["bmi"])
    insulin_resistance = (hba1c / 7.0) ** 1.1
    insulin_resistance *= 1.08 if patient["on_metformin"] == 0 else 0.84
    if patient["on_insulin"]:
        insulin_resistance *= 0.68
    if patient["on_sglt2"]:
        insulin_resistance *= 0.9

    short_night = rng.random(n_days) < 0.2
    sleep_onset = np.where(short_night, 0.8, 23.1)
    wake_hour = np.where(short_night, 5.2, 6.6)
    # onset stored on 0-24 scale; values under 12 mean after midnight
    asleep = np.zeros(n, dtype=bool)
    for d in range(n_days):
        onset = float(sleep_onset[d])
        wake = float(wake_hour[d])
        hours = hour[day == d]
        if onset >= 12:
            mask = (hours >= onset) | (hours < wake)
        else:
            mask = (hours >= onset) & (hours < wake)
        asleep[day == d] = mask

    awakening = asleep & (rng.random(n) < (0.04 + max(0, bmi - 27) * 0.008))
    asleep = asleep & ~awakening

    cycle = (minute_of_day % 90) / 90.0
    sleep_stage = np.zeros(n, dtype=int)
    sleep_stage[asleep & (cycle < 0.22)] = 2  # deep
    sleep_stage[asleep & (cycle >= 0.22) & (cycle < 0.72)] = 1  # light
    sleep_stage[asleep & (cycle >= 0.72)] = 3  # REM

    stress_day = rng.random(n_days) < 0.16
    stress = stress_day[day]

    sedentary = rng.random(n_days) < 0.22
    commute = ((hour >= 7.0) & (hour < 9.0)) | ((hour >= 17.5) & (hour < 19.5))
    evening_walk = (hour >= 18.5) & (hour < 19.5) & ~sedentary[day]
    base_steps = np.where(asleep, 0, np.where(commute, 160, 28))
    base_steps = np.where(evening_walk, base_steps + 220, base_steps)
    base_steps = np.where(sedentary[day] & ~asleep, base_steps * 0.45, base_steps)
    steps = rng.poisson(np.clip(base_steps, 0, None)).astype(float)

    # Between-meal baseline sits below the HbA1c-implied average so spikes
    # have somewhere to rise from. That is the two-hour-ahead case.
    glucose = np.full(n, 112.0 + (hba1c - 7.0) * 9.0)
    glucose += (6 + (hba1c - 7) * 1.5) * np.sin((hour - 5) / 24 * 2 * np.pi)
    dawn = (hour >= 4.0) & (hour <= 8.5)
    glucose[dawn] += max(0.0, hba1c - 6.4) * 4.0

    night_hours = []
    for d in range(n_days):
        night_hours.append(float(asleep[day == d].sum() * STEP_MINUTES / 60))
    night_hours = np.asarray(night_hours)

    for d in range(n_days):
        prior_sleep = 7.0 if d == 0 else night_hours[d - 1]
        prior_steps = 6500 if d == 0 else float(steps[day == (d - 1)].sum())
        sleep_penalty = 1.34 if prior_sleep < 6.0 else (1.16 if prior_sleep < 6.6 else 1.0)
        activity_penalty = 1.28 if prior_steps < 4500 else (0.86 if prior_steps > 9000 else 1.0)
        stress_penalty = 1.26 if stress_day[d] else 1.0
        day_scale = insulin_resistance * sleep_penalty * activity_penalty * stress_penalty

        meals = [(8.0, 72.0), (13.4, 80.0), (20.4, 76.0)]
        if rng.random() < 0.28:
            meals.append((float(rng.uniform(16.0, 17.3)), 42.0))
        for base_hour, base_amp in meals:
            if rng.random() < 0.14:
                continue
            when = base_hour + float(rng.normal(0, 1.05))
            center = d * STEPS_PER_DAY + int(round(when * 60 / STEP_MINUTES))
            amp = base_amp * day_scale * float(rng.normal(1.0, 0.08))
            _add_kernel(glucose, center, amp)

        if patient["on_insulin"] and rng.random() < 0.25:
            center = d * STEPS_PER_DAY + int(round(3.1 * 60 / STEP_MINUTES))
            _add_kernel(glucose, center, -float(rng.uniform(38, 62)), peak_lag=2, width=2.4, span=10)

    activity = np.convolve(steps, np.ones(8) / 8.0, mode="same")
    glucose -= activity * 0.018
    white = rng.normal(0, 6.5, size=n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.82 * ar[i - 1] + white[i]
    glucose = np.clip(glucose + ar, 48, 380)

    heart = (
        68
        + (bmi - 25) * 0.55
        + np.where(asleep, -8.0, 0.0)
        + steps * 0.012
        + np.where(stress, 7.0, 0.0)
        + 3.5 * np.sin((hour - 15) / 24 * 2 * np.pi)
        + rng.normal(0, 2.4, size=n)
    )
    heart = np.clip(heart, 48, 165)
    hrv = (
        44
        - (bmi - 25) * 0.8
        - (hba1c - 7) * 1.4
        + np.where(asleep, 14.0, 0.0)
        - np.where(stress, 9.0, 0.0)
        + rng.normal(0, 3.2, size=n)
    )
    hrv = np.clip(hrv, 8, 95)

    timestamps = origin + pd.to_timedelta(idx * STEP_MINUTES, unit="min")
    return pd.DataFrame(
        {
            "patient_id": patient["patient_id"],
            "timestamp": timestamps,
            "glucose_mgdl": np.round(glucose, 1),
            "heart_rate": np.round(heart, 1),
            "hrv_rmssd": np.round(hrv, 1),
            "steps": steps.astype(int),
            "sleep_stage": sleep_stage.astype(int),
        }
    )


def generate_cohort(
    n_patients: int = N_PATIENTS,
    n_days: int = N_DAYS,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    patients = generate_patients(n_patients, rng)
    origin = pd.Timestamp("2026-09-01 00:00:00")
    frames = [
        simulate_patient(patients.loc[i], rng, n_days, origin)
        for i in range(len(patients))
    ]
    timeseries = pd.concat(frames, ignore_index=True)
    return patients, timeseries
