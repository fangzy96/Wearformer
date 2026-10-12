"""Synthetic-data and boundary checks for the training-day criterion."""

import csv
import math
from pathlib import Path
import tempfile
import unittest

from analysis.training_days import (
    BudgetPoint,
    Interval,
    compute_intervals,
    read_curves,
    select_training_days,
)


class TrainingDaysTests(unittest.TestCase):
    def test_synthetic_csv_selection_and_unrounded_all_endpoint(self):
        total, participants = 311, 17
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "setting", "train_days", "joint_mae",
                    "total_qualifying_training_days", "training_participants",
                ])
                for day, error in [("1", 1.0), ("2", 0.9), ("5", 0.899), ("All", 0.898)]:
                    writer.writerow(["synthetic", day, error, total, participants])
            curves = read_curves(path)
        self.assertEqual(set(curves), {"synthetic"})
        intervals = compute_intervals(curves["synthetic"])
        self.assertEqual(select_training_days(intervals), 2)
        self.assertEqual(intervals[-1].start, "5")
        self.assertEqual(intervals[-1].end, "All")
        self.assertEqual(intervals[-1].added_days, total / participants - 5)
        self.assertNotEqual(intervals[-1].added_days, round(total / participants, 1) - 5)

    def test_daily_rate_uses_previous_mae_and_added_days(self):
        interval = compute_intervals([
            BudgetPoint("3", 3, 2.0), BudgetPoint("5", 5, 1.8),
        ])[0]
        self.assertAlmostEqual(interval.relative_reduction_percent, 10.0)
        self.assertAlmostEqual(interval.daily_relative_reduction_percent, 5.0)

    def test_earliest_budget_requires_all_remaining_intervals(self):
        # The first low rate cannot qualify because a later interval exceeds 0.2.
        starts = ["1", "2", "3", "5", "7"]
        ends = ["2", "3", "5", "7", "All"]
        rates = [0.3, 0.1, 0.4, 0.2, 0.1]
        intervals = [
            Interval(a, b, 1, 1, 1, rate, rate)
            for a, b, rate in zip(starts, ends, rates)
        ]
        self.assertEqual(select_training_days(intervals), 5)
        self.assertIsNone(select_training_days(intervals, min_later_intervals=3))

    def test_one_remaining_interval_cannot_select_a_budget(self):
        intervals = [Interval("40", "All", 85, 1, 0.99, 1, 1 / 85)]
        self.assertIsNone(select_training_days(intervals))

    def test_worsening_is_not_clipped_or_changed_to_a_gain(self):
        interval = compute_intervals([
            BudgetPoint("1", 1, 1.0), BudgetPoint("2", 2, 1.01),
        ])[0]
        self.assertAlmostEqual(interval.daily_relative_reduction_percent, -1.0)

    def test_invalid_or_unordered_budgets_are_rejected(self):
        bad_curves = [
            [BudgetPoint("0", 0, 1), BudgetPoint("2", 2, 0.9)],
            [BudgetPoint("1.5", 1.5, 1), BudgetPoint("2", 2, 0.9)],
            [BudgetPoint("2", 2, 1), BudgetPoint("2", 2, 0.9)],
            [BudgetPoint("3", 3, 1), BudgetPoint("2", 2, 0.9)],
            [BudgetPoint("All", 10, 1), BudgetPoint("20", 20, 0.9)],
            [BudgetPoint("40", 40, 1), BudgetPoint("All", 40, 0.9)],
            [BudgetPoint("1", 1, math.nan), BudgetPoint("2", 2, 0.9)],
        ]
        for points in bad_curves:
            with self.subTest(points=points), self.assertRaises(ValueError):
                compute_intervals(points)

    def test_csv_cannot_change_training_cohort_between_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curve.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "setting", "train_days", "joint_mae",
                    "total_qualifying_training_days", "training_participants",
                ])
                writer.writerow(["example", "1", 1, 200, 10])
                writer.writerow(["example", "All", 0.8, 210, 10])
            with self.assertRaisesRegex(ValueError, "counts differ"):
                read_curves(path)


if __name__ == "__main__":
    unittest.main()
