"""A synthetic end-to-end head run through the native daily-pool interfaces."""

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from confidence.pipeline import run_evaluation, run_train
from downstream.impute import run_fill
from wearformer.checkpoint import save_checkpoint
from wearformer.model import TimeGapSAITS


class NativeConfidencePipelineTests(unittest.TestCase):
    def test_train_select_score_and_natural_fill(self):
        torch.manual_seed(4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            time = np.arange(144, dtype=np.float32)
            values = np.stack([np.stack([65 + index + np.sin(time / 8),
                                        (time % 7) * (index + 1)], axis=-1)
                               for index in range(2)])
            observed = np.ones_like(values, dtype=bool)
            metadata = [{"participant_id": f"synthetic-{index}",
                         "window_start": "2050-01-01T00:00:00"} for index in range(2)]
            for split_index, split in enumerate(("train", "validation", "internal_test", "natural"), start=1):
                for row in metadata:
                    row["window_start"] = f"2050-01-{split_index:02}T00:00:00"
                current = values.copy()
                mask = observed.copy()
                if split == "natural":
                    mask[:, 50:54] = False
                    current[~mask] = np.nan
                np.save(data / f"{split}_values.npy", current)
                np.save(data / f"{split}_original_observed.npy", mask)
                with open(data / f"{split}_base_windows.csv", "w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(metadata[0]))
                    writer.writeheader()
                    writer.writerows(metadata)
                np.savez(data / f"{split}_v2_starts.npz", **{"2": np.array([[20], [30]])})
            config = {"time_step_minutes": 10,
                      "evaluation": {"gap_lengths": [2], "events_per_day": 1},
                      "model": {"seq_len": 144, "d_model": 8, "n_layers": 1,
                                "n_heads": 2, "d_k": 4, "d_v": 4, "d_ffn": 8, "local_conv": 3},
                      "training": {"seed": 4, "epochs": 1, "samples_per_participant": 1,
                                   "scatter_rate": 0.15, "anchors": [1, 2, 3], "flank": 2,
                                   "flank_observed": 1, "min_observed_frac": 0.5}}
            prep = {"mean": [65.0, 3.0], "std": [4.0, 2.0],
                    "descriptor_mean": [0.0] * 8, "descriptor_std": [1.0] * 8,
                    "reference_std": [4.0, 2.0]}
            checkpoint = root / "model.pt"
            save_checkpoint(checkpoint, TimeGapSAITS(**config["model"]), config, prep, 1, 0.0)
            shared = {"checkpoint": str(checkpoint), "data_dir": str(data),
                      "device": "cpu", "workers": 0, "batch_size": 2}
            head_dir = root / "heads"
            training = SimpleNamespace(**shared, out_dir=str(head_dir), epochs=2, seed=4)
            result = run_train(training)
            self.assertTrue(result["backbone_unchanged"])
            selection = SimpleNamespace(**shared, head_dir=str(head_dir), manifest=None,
                                        starts_per_day=1, folds=None, fold_salt="synthetic")
            self.assertEqual(run_evaluation(selection, True)["events"], 2)
            scoring = SimpleNamespace(**shared, head_dir=str(head_dir), manifest=None,
                                      starts_per_day=1, split="internal_test", out_dir=str(root / "scores"))
            result = run_evaluation(scoring, False)
            with open(result["output"], newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(0 <= float(row["confidence"]) <= 1 for row in rows))
            fill = SimpleNamespace(**shared, split="natural", out_dir=str(root / "filled"))
            completion = run_fill(fill)
            self.assertEqual(completion["filled_cells"], [8, 8])
            output = np.load(root / "filled" / "completed.npz")
            original_mask = np.load(data / "natural_original_observed.npy")
            np.testing.assert_array_equal(output["values"][original_mask], values[original_mask])


if __name__ == "__main__":
    unittest.main()
