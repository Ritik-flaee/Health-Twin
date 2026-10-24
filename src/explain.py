"""Plain-language reasons a clinician can read beside the risk score."""

from __future__ import annotations

import pandas as pd


def explain_row(row: pd.Series) -> list[str]:
    """Rank observations that support or soften the two-hour spike call."""
    notes: list[tuple[int, str]] = []
    glucose = float(row["glucose_mgdl"])
    delta = float(row["glucose_delta_30m"])
    hour = pd.Timestamp(row["timestamp"]).hour + pd.Timestamp(row["timestamp"]).minute / 60.0
    sleep_h = float(row["sleep_hours_prev"])
    steps = float(row["steps_24h"])
    hba1c = float(row["hba1c"])
    hrv = float(row["hrv_6h"])

    if delta >= 12:
        notes.append((5, f"Glucose has already risen {delta:.0f} mg/dL in the last 30 minutes."))
    elif delta <= -12:
        notes.append((2, f"Glucose is falling ({delta:.0f} mg/dL over 30 minutes)."))

    if glucose >= 180:
        notes.append((5, f"The latest reading is already {glucose:.0f} mg/dL, above the 180 line."))
    elif glucose >= 140:
        notes.append((3, f"Glucose is raised at {glucose:.0f} mg/dL, still short of 180."))
    else:
        notes.append((1, f"Glucose right now is {glucose:.0f} mg/dL, in a range many clinics treat as acceptable."))

    if 7.2 <= hour <= 9.2 or 12.4 <= hour <= 14.6 or 19.4 <= hour <= 21.6:
        notes.append((4, "This is a usual meal time, so a rise may still be on the way."))

    if hba1c >= 8.5:
        notes.append((4, f"The 3-month sugar average (HbA1c) is {hba1c:.1f}%. Over 7 means sugar runs high, so meals hit harder."))
    elif hba1c >= 7.5:
        notes.append((2, f"The 3-month sugar average (HbA1c) is {hba1c:.1f}%. Doctors usually want this under 7."))

    if sleep_h < 6:
        notes.append((4, f"Last completed night was short ({sleep_h:.1f} h), which raises next-day glucose."))
    elif sleep_h < 6.6:
        notes.append((2, f"Last night's sleep was borderline ({sleep_h:.1f} h)."))

    if steps < 4000:
        notes.append((3, f"Only about {steps:,.0f} steps in the last 24 hours."))
    elif steps > 9000:
        notes.append((1, f"Activity is solid at about {steps:,.0f} steps in the last 24 hours."))

    if hrv < 28:
        notes.append((3, "The watch shows a stressed heart rhythm over the last six hours."))

    if int(row["on_insulin"]) == 1:
        notes.append((2, "Insulin is on the chart. Watch the same window for a dip as well as a spike."))
    elif int(row["on_metformin"]) == 0:
        notes.append((2, "No diabetes medicine is on record."))

    if int(row["family_history"]) == 1 and int(row["tcf7l2_alleles"]) >= 1:
        notes.append((1, "Diabetes runs in the family, and a practice risk marker is on the file."))

    if int(row.get("hypertension", 0)) == 1:
        notes.append((2, "Blood pressure is high on the chart. That is context, not a prediction of a heart attack."))

    notes.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in notes[:5]]
