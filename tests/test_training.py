"""Training/evaluation checks on generated daily signals only."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from wearformer.data import (Pool, TrainingDataset, build_profiles, observed_feature_stats,
                             assert_disjoint_dates)
from wearformer.train import equal_feature_masked_loss, main as train_main
from wearformer.evaluate import main as evaluate_main


def synthetic_values(count, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(144)
    values = np.zeros((count, 144, 2), np.float32)
    values[..., 0] = 65 + 8 * np.sin(2 * np.pi * (t - 50) / 144) + rng.normal(size=(count, 144))
    values[..., 1] = np.maximum(0, 8 * np.sin(2 * np.pi * (t - 50) / 144))
    return values


def config():
    return {"time_step_minutes": 10,
            "model": {"seq_len": 144, "d_model": 16, "n_heads": 2, "d_k": 8,
                      "d_v": 8, "d_ffn": 16, "n_layers": 1, "local_conv": 9},
            "training": {"epochs": 2, "batch_size": 2, "learning_rate": 0.001,
                "weight_decay": 0, "grad_clip": 1, "seed": 123,
                "samples_per_participant": 2, "scatter_rate": 0.15,
                "anchors": [1, 3, 6, 12, 24, 48, 72, 84], "flank": 6,
                "flank_observed": 3, "min_observed_frac": 0.5, "num_workers": 0},
            "evaluation": {"gap_lengths": [1, 3], "events_per_day": 1, "batch_size": 2}}


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_loss_weights_channels_equally(self):
        prediction = torch.tensor([[[2., 6.], [4., 99.]]])
        truth = torch.zeros_like(prediction)
        mask = torch.tensor([[[True, True], [True, False]]])
        # Channel means are 3 and 6, despite having different numbers of targets.
        self.assertEqual(float(equal_feature_masked_loss(prediction, truth, mask)), 4.5)

    def test_masking_is_repeatable_and_never_targets_natural_missing_values(self):
        values = synthetic_values(4, 7)
        values[:, 20:25, :] = np.nan
        observed = np.isfinite(values)
        rows = [{"participant_id": f"synthetic-{i // 2}", "window_start": f"2024-01-0{i + 1}T00:00:00"}
                for i in range(4)]
        pool = Pool(values, observed, rows)
        mean, std, _ = observed_feature_stats(values, observed)
        profiles, _ = build_profiles(pool)
        dataset = TrainingDataset(pool, mean, std, profiles, config())
        dataset.set_epoch(1)
        first, second = dataset[0], dataset[0]
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a, b))
        x, _, visible, targets, _ = first
        self.assertFalse(targets[20:25].any())
        self.assertFalse((visible & targets).any())
        self.assertTrue(torch.all(x[~visible] == 0))
        dataset.set_epoch(2)
        self.assertFalse(torch.equal(first[3], dataset[0][3]))

    def test_overlapping_dates_are_rejected(self):
        values = synthetic_values(1, 1)
        pool = Pool(values, np.isfinite(values), [{"participant_id": "synthetic", "window_start": "2024-01-01T00:00:00"}])
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_disjoint_dates(pool, pool)

    def test_train_select_reload_and_test_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for split, offset in (("train", 1), ("validation", 10), ("internal_test", 20)):
                values = synthetic_values(4, offset)
                np.save(path / f"{split}_values.npy", values)
                np.save(path / f"{split}_original_observed.npy", np.isfinite(values))
                with (path / f"{split}_base_windows.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("array_row", "participant_id", "window_start"))
                    writer.writeheader()
                    for i in range(4):
                        writer.writerow({"array_row": i, "participant_id": f"synthetic-{i // 2}",
                                         "window_start": f"2024-01-{offset + i:02d}T00:00:00"})
                if split != "train":
                    np.savez(path / f"{split}_v2_starts.npz", **{"1": np.full((4, 1), 30, np.int32),
                                                               "3": np.full((4, 1), 60, np.int32)})
            cfg = path / "config.json"
            cfg.write_text(json.dumps(config()))
            train_main(["--config", str(cfg), "--protocol-dir", str(path),
                        "--output-dir", str(path / "run"), "--device", "cpu"])
            checkpoint = path / "run/best.pt"
            self.assertTrue(checkpoint.exists())
            history = json.loads((path / "run/training_history.json").read_text())
            self.assertEqual(len(history), 2)
            selected = torch.load(checkpoint, weights_only=True)
            self.assertEqual(selected["validation_joint_mae"], min(row["validation_joint_mae"] for row in history))
            evaluate_main(["--checkpoint", str(checkpoint), "--protocol-dir", str(path),
                           "--output-dir", str(path / "evaluation"), "--device", "cpu"])
            metrics = json.loads((path / "evaluation/metrics.json").read_text())
            self.assertEqual(set(metrics["by_length"]), {"1", "3"})
            self.assertTrue(metrics["steps_clipped"])
            self.assertTrue(np.isfinite(metrics["joint"]))


if __name__ == "__main__":
    unittest.main()
