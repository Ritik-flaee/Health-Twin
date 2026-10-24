"""Checks that labels look forward and features do not."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features import build_feature_table, future_window_stat
from src.synthesize import generate_cohort


class FeatureTests(unittest.TestCase):
    def test_future_window_excludes_current_point(self) -> None:
        series = pd.Series([1, 5, 9, 2, 4, 8, 3, 7, 6, 0], dtype=float)
        got = future_window_stat(series, horizon=3, how="max")
        self.assertEqual(got.iloc[0], 9)
        self.assertEqual(got.iloc[1], 9)
        self.assertTrue(np.isnan(got.iloc[-1]))

    def test_patient_rows_have_no_future_in_the_delta(self) -> None:
        patients, series = generate_cohort(n_patients=4, n_days=3, seed=7)
        table = build_feature_table(patients, series)
        self.assertGreater(len(table), 100)
        self.assertFalse(table["glucose_delta_30m"].isna().any())
        one = series[series["patient_id"] == patients.iloc[0]["patient_id"]].sort_values("timestamp")
        row = table[table["patient_id"] == patients.iloc[0]["patient_id"]].iloc[10]
        ts = row["timestamp"]
        current = one.loc[one["timestamp"] == ts, "glucose_mgdl"].iloc[0]
        prior = one.loc[one["timestamp"] == ts - pd.Timedelta(minutes=30), "glucose_mgdl"].iloc[0]
        self.assertAlmostEqual(row["glucose_delta_30m"], current - prior, places=4)
        self.assertGreaterEqual(row["future_peak"], current - 80)

    def test_cohort_is_working_age_with_cardiometabolic_risk(self) -> None:
        patients, _series = generate_cohort(n_patients=30, n_days=2, seed=11)
        self.assertTrue(patients["age"].between(30, 55).all())
        self.assertTrue(set(patients["glycemic_status"]) <= {"Type 2 diabetes", "Prediabetes"})
        covered = (patients["hypertension"] == 1) | (patients["ldl"] >= 130) | (patients["family_history"] == 1) | (patients["bmi"] >= 27)
        self.assertTrue(covered.all())

    def test_names_are_attached(self) -> None:
        patients, series = generate_cohort(n_patients=2, n_days=3, seed=3)
        table = build_feature_table(patients, series)
        self.assertIn("name", table.columns)
        self.assertTrue(table["name"].notna().all())


if __name__ == "__main__":
    unittest.main()
