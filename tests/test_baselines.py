"""Comparator tests use generated inputs only."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from baselines.evaluation import load_starts, preserve_rng, score_gaps, select_best_epoch
from baselines.models import StandardizedPredictor
from baselines.train import expected_gap_positions, main
from baselines.train_mean import TrainingMean, fit_training_mean, impute, observed_feature_stats_f64


def synthetic_pool(directory, split, n=4):
    values = np.arange(n * 144 * 2, dtype=np.float32).reshape(n, 144, 2) / 100 + 1
    observed = np.ones(values.shape, dtype=bool)
    observed[:, 0, 1] = False
    np.save(directory / f"{split}_values.npy", values)
    np.save(directory / f"{split}_original_observed.npy", observed)
    with (directory / f"{split}_base_windows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["array_row", "participant_id"])
        writer.writeheader()
        writer.writerows({"array_row": i, "participant_id": f"synthetic_{i // 2}"} for i in range(n))
    return values, observed


class MeanTests(unittest.TestCase):
    def test_training_mean_ignores_unobserved_and_nonfinite_cells(self):
        values = np.array([[[2, 3], [4, 5], [1000, np.nan], [6, 7]]], dtype=np.float32)
        observed = np.ones(values.shape, dtype=bool)
        observed[0, 2, 0] = False
        means, _, counts = observed_feature_stats_f64(values, observed)
        np.testing.assert_array_equal(means, [4, 5])
        np.testing.assert_array_equal(counts, [3, 3])

    def test_mean_predictions_independent_of_batch_order_and_visible_values(self):
        train = np.arange(48, dtype=np.float32).reshape(4, 6, 2)
        record = fit_training_mean(train, np.ones_like(train, dtype=bool))
        predictor = TrainingMean(record["mean"])
        values = np.arange(60, dtype=np.float32).reshape(5, 6, 2)
        visible = np.ones_like(values, dtype=bool)
        together = predictor.predict(values, visible)
        split = np.concatenate([predictor.predict(values[:1], visible[:1]), predictor.predict(values[1:], visible[1:])])
        shuffled = predictor.predict(values[::-1] * 500, visible[::-1])[::-1]
        np.testing.assert_array_equal(together, split)
        np.testing.assert_array_equal(together, shuffled)
        filled = impute(values, ~visible, record["mean"])
        np.testing.assert_array_equal(filled, values)

    def test_cli_mean_fit_and_reload_use_all_seven_ten_minute_gaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool, manifest = root / "pool", root / "manifest"
            pool.mkdir(); manifest.mkdir()
            synthetic_pool(pool, "train")
            for split in ("validation", "internal_test"):
                synthetic_pool(pool, split)
                np.savez(manifest / f"{split}_v2_starts.npz", **{
                    str(length): np.full((4, 1), 20, dtype=np.int32)
                    for length in expected_gap_positions(10)
                })
            common = ["--model", "Mean", "--train-dir", str(pool), "--protocol-dir", str(pool),
                      "--manifest", str(manifest), "--time-step-minutes", "10", "--k", "1"]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(common + ["--output-dir", str(root / "run")]), 0)
                self.assertEqual(main(common + ["--load-run", str(root / "run"), "--output-dir", str(root / "loaded")]), 0)
            first = json.loads((root / "run" / "internal_test.json").read_text())
            second = json.loads((root / "loaded" / "internal_test.json").read_text())
            self.assertEqual(first, second)
            self.assertEqual(len(first["by_gap_positions"]), 7)
            self.assertEqual(first["output_policy"], "raw")


class EvaluationTests(unittest.TestCase):
    def test_validation_restores_every_rng_even_on_error(self):
        random.seed(91); np.random.seed(92); torch.manual_seed(93)
        expected = (random.random(), np.random.rand(), torch.rand(2))
        random.seed(91); np.random.seed(92); torch.manual_seed(93)
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with preserve_rng(999):
                random.random(); np.random.rand(); torch.rand(100)
                raise RuntimeError("synthetic failure")
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        self.assertTrue(torch.equal(torch.rand(2), expected[2]))

    def test_macro_aggregation_does_not_overweight_people_with_more_days(self):
        values = np.array([[[2, 4]], [[2, 4]], [[8, 10]]], dtype=np.float32)
        visible = np.ones_like(values, dtype=bool)
        starts = {1: np.zeros((3, 1), dtype=np.int32)}
        result = score_gaps(lambda x, mask: np.zeros_like(x), values, visible,
                            ["synthetic_a", "synthetic_a", "synthetic_b"], starts, [1, 1], k=1, batch_size=2)
        self.assertEqual(result["overall"], {"hr_mae": 5, "steps_mae": 7, "joint_mae": 6})

    def test_raw_negative_steps_predictions_are_not_clipped(self):
        class Model:
            def predict(self, data):
                return {"imputation": np.full(data["X"].shape, -5.0, dtype=np.float32)}
        values = np.zeros((1, 2, 2), dtype=np.float32)
        predictor = StandardizedPredictor(Model(), [0, 0], [1, 1])
        result = score_gaps(predictor.predict, values, np.ones_like(values, dtype=bool),
                            ["synthetic"], {1: np.zeros((1, 1), dtype=np.int32)}, [1, 1], k=1)
        self.assertEqual(result["overall"]["steps_mae"], 5)

    def test_selection_is_unrounded_and_ties_choose_earliest(self):
        history = [{"epoch": 1, "validation_joint_mae": 0.40000009},
                   {"epoch": 2, "validation_joint_mae": 0.40000001},
                   {"epoch": 3, "validation_joint_mae": 0.40000001}]
        self.assertEqual(select_best_epoch(history), 2)

    def test_one_minute_manifest_requires_the_eighth_duration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main, extra = root / "main", root / "extra"
            main.mkdir(); extra.mkdir()
            expected = expected_gap_positions(1)
            np.savez(main / "validation_v2_starts.npz", **{
                str(length): np.zeros((2, 1), dtype=np.int32)
                for length in expected if length != 10
            })
            with self.assertRaisesRegex(ValueError, "all eight"):
                load_starts([main], "validation", 2, 1440, expected)
            np.savez(extra / "validation_v2_starts.npz", **{"10": np.zeros((2, 1), dtype=np.int32)})
            self.assertEqual(sorted(load_starts([main, extra], "validation", 2, 1440, expected)), expected)


if __name__ == "__main__":
    unittest.main()
