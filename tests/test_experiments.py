"""Chronological training-pool checks on synthetic dates and arrays."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.training_days import build_pools, chronological_indices


def synthetic_rows():
    # Missing calendar dates illustrate that first-N qualifying days are not
    # necessarily N consecutive calendar dates.
    records = [("synthetic_a", 8), ("synthetic_b", 3), ("synthetic_a", 1),
               ("synthetic_c", 1), ("synthetic_b", 1), ("synthetic_a", 4),
               ("synthetic_b", 2), ("synthetic_c", 2)]
    return [{"array_row": str(i), "participant_id": person,
             "window_start": f"2000-01-{day:02d} 00:00:00", "base_window_id": f"train:{person}:{day}"}
            for i, (person, day) in enumerate(records)]


class ChronologicalDaysTests(unittest.TestCase):
    def test_first_n_is_nested_chronological_and_fixed_cohort(self):
        rows = synthetic_rows()
        one = chronological_indices(rows, 1, 3)
        two = chronological_indices(rows, 2, 3)
        all_days = chronological_indices(rows, "All", 3)
        np.testing.assert_array_equal(one, [2, 4])
        np.testing.assert_array_equal(two, [2, 5, 4, 6])
        np.testing.assert_array_equal(all_days, [2, 5, 0, 4, 6, 1])
        self.assertTrue(set(one) <= set(two) <= set(all_days))
        self.assertEqual({rows[i]["participant_id"] for i in all_days}, {"synthetic_a", "synthetic_b"})

    def test_pool_writer_preserves_values_and_renumbers_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "source"; source.mkdir()
            rows = synthetic_rows()
            with (source / "train_base_windows.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            values = np.arange(len(rows) * 4 * 2, dtype=np.float32).reshape(len(rows), 4, 2)
            observed = np.ones_like(values, dtype=bool); observed[5, 1, 1] = False
            np.save(source / "train_values.npy", values)
            np.save(source / "train_original_observed.npy", observed)
            destinations = build_pools(source, root / "output", [1, 2, "All"], 3)
            np.testing.assert_array_equal(np.load(destinations[1] / "train_values.npy"), values[[2, 5, 4, 6]])
            np.testing.assert_array_equal(np.load(destinations[1] / "train_original_observed.npy"), observed[[2, 5, 4, 6]])
            with (destinations[1] / "train_base_windows.csv").open() as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual([row["array_row"] for row in written], ["0", "1", "2", "3"])
            self.assertEqual([row["base_window_id"] for row in written], [rows[i]["base_window_id"] for i in [2, 5, 4, 6]])
            with self.assertRaisesRegex(ValueError, "already exists"):
                build_pools(source, root / "output", [1], 3)

    def test_duplicate_dates_and_invalid_budgets_rejected(self):
        rows = synthetic_rows()
        for days in [0, -1, 4]:
            with self.subTest(days=days), self.assertRaises(ValueError):
                chronological_indices(rows, days, 3)
        rows[5]["window_start"] = rows[2]["window_start"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            chronological_indices(rows, 1, 3)

    def test_heldout_overlap_rejected(self):
        from experiments.training_days import check_heldout_dates
        rows = synthetic_rows()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "validation_base_windows.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerow(rows[0])
            with self.assertRaisesRegex(ValueError, "held-out"):
                check_heldout_dates(rows, temporary)


if __name__ == "__main__":
    unittest.main()
