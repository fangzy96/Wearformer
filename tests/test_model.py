"""Synthetic model and checkpoint checks; no participant data or model weights."""

import tempfile
import unittest
from pathlib import Path

import torch

from wearformer.checkpoint import load_model, save_checkpoint
from wearformer.model import LocalConvBranch, TimeGapSAITS


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_local_branch_starts_as_identity(self):
        branch = LocalConvBranch(8, 9)
        x = torch.randn(2, 24, 8)
        self.assertTrue(torch.equal(x, branch(x)))

    def test_checkpoint_preserves_predictions_and_visible_values(self):
        config = {"model": dict(seq_len=144, d_model=16, n_heads=2, d_k=8,
                    d_v=8, d_ffn=16, n_layers=1, local_conv=9)}
        model = TimeGapSAITS(**config["model"]).eval()
        x = torch.randn(2, 144, 2)
        visible = torch.rand(x.shape) > 0.25
        x[~visible] = 0
        side = torch.randn(2, 144, 16)
        before = model(x, visible, side)
        self.assertTrue(torch.equal(before["imputed"][visible], x[visible]))
        self.assertEqual(before["encoded_1"].shape, (2, 144, 16))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, config, {}, 1, 0.5)
            restored, checkpoint = load_model(path)
            after = restored(x, visible, side)
            self.assertEqual(checkpoint["epoch"], 1)
            for key in before:
                self.assertTrue(torch.equal(before[key], after[key]), key)


if __name__ == "__main__":
    unittest.main()
