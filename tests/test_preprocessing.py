import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.aggregate_fitbit import aggregate
from preprocessing.days import assign_days, build
from preprocessing.gaps import sample_starts
from preprocessing.raw import align_apple, align_fitbit


class PreprocessingTests(unittest.TestCase):
    def test_fitbit_duplicates_zero_and_hr_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hr.csv").write_text("datetime,heartrate\n2000-01-01 00:00:00,60\n2000-01-01 00:00:30,80\n2000-01-01 00:02:00,65\n2000-01-01 00:03:00,65\n")
            (root / "st.csv").write_text("datetime,steps\n2000-01-01 00:00:00,4\n2000-01-01 00:00:20,6\n2000-01-01 00:01:00,8\n2000-01-01 00:02:00,0\n")
            out = align_fitbit(root / "hr.csv", root / "st.csv")
            np.testing.assert_equal(out.heartrate.to_numpy(), [70, np.nan, 65, 65])
            np.testing.assert_equal(out.steps.to_numpy(), [6, np.nan, 0, 0])

    def test_apple_overlap_and_long_interval_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hr.csv").write_text("device,datetime,heartrate\nHK Apple Watch,2000-01-01 00:00:00,60\nHK Apple Watch,2000-01-01 00:10:00,70\nHK Apple Watch,2000-01-01 00:20:00,65\n")
            (root / "st.csv").write_text("device,start_datetime,end_datetime,steps\nHK Apple Watch,2000-01-01 00:05:00,2000-01-01 00:15:00,100\nHK Apple Watch,2000-01-01 00:00:00,2000-01-01 02:00:00,1000\n")
            out = align_apple(root / "hr.csv", root / "st.csv")
            np.testing.assert_equal(out.steps.to_numpy(), [50, 50, 0])

    def test_partial_bin_not_rescaled(self):
        values = np.full((1, 1440, 2), np.nan, np.float32)
        values[0, :2] = [[60, 10], [80, 30]]
        out, mask = aggregate(values, np.isfinite(values))
        np.testing.assert_equal(out[0, 0], [70, 40])
        self.assertEqual(mask.sum(), 2)

    def test_split_disjoint_and_order_independent(self):
        values = np.ones((30, 144, 2))
        split = assign_days(values, "synthetic-person", 1)
        self.assertEqual([len(split[k]) for k in ("train", "validation", "internal_test")], [15, 5, 10])
        self.assertEqual(len(set().union(*[set(v) for v in split.values()])), 30)
        self.assertEqual(split, assign_days(values, "synthetic-person", 1))

    def test_pool_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({"datetime": pd.date_range("2000-01-01", periods=25*144, freq="10min"),
                          "heartrate": 60., "steps": 0.}).to_csv(root / "record.csv", index=False)
            (root / "records.csv").write_text("participant_id,path\nsynthetic-person,record.csv\n")
            build(root / "records.csv", root / "pool", 10, 1)
            ids = []
            for split, n in (("train", 10), ("validation", 5), ("internal_test", 10)):
                values = np.load(root / "pool" / f"{split}_values.npy")
                self.assertEqual(values.shape, (n, 144, 2))
                with (root / "pool" / f"{split}_base_windows.csv").open() as handle:
                    ids.extend(r["window_start"] for r in csv.DictReader(handle))
            self.assertEqual(len(set(ids)), 25)

    def test_gap_context_and_repeatability(self):
        values = np.ones((2, 144, 2))
        values[1] = np.nan
        kwargs = dict(draws=4, flank=6, flank_observed=3, seed=7)
        out = sample_starts(values, np.isfinite(values), [1, 72], **kwargs)
        again = sample_starts(values, np.isfinite(values), [1, 72], **kwargs)
        for key, starts in out.items():
            length = int(key)
            self.assertTrue(((starts[0] >= 6) & (starts[0] + length + 6 <= 144)).all())
            self.assertTrue((starts[1] == -1).all())
            np.testing.assert_equal(starts, again[key])


if __name__ == "__main__":
    unittest.main()
