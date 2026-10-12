"""Synthetic checks of the error head, weighting, and confidence transformation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from confidence.head import (ScaleHead, backbone_state_sha, balanced_nll,
                             confidence_rank, laplace_loss, participant_folds,
                             predicted_joint_error)
from confidence.pipeline import evaluate_heads, load_heads, select_head, train_head


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(2, 4)
        self.readout = nn.Linear(4, 2)

    def forward(self, values, observed, side):
        encoded = self.features(values)
        return {"encoded_1": encoded, "x_tilde_3": self.readout(encoded)}


class ConfidenceMathTests(unittest.TestCase):
    def test_loss_weights_channels_equally_not_by_target_count(self):
        b = torch.ones((1, 3, 2))
        z = torch.tensor([[[1.0, 10.0], [2.0, 99.0], [99.0, 99.0]]])
        target = torch.tensor([[[True, True], [True, False], [False, False]]])
        self.assertEqual(float(laplace_loss(b, torch.zeros_like(z), z, target)), 5.75)
        with self.assertRaises(ValueError):
            laplace_loss(b, z, z, torch.zeros_like(target))

    def test_selection_weights_participants_then_durations(self):
        values = np.array([[1, 1], [1, 1], [9, 9], [100, 100]], float)
        result = balanced_nll(values, ["a", "a", "b", "a"], [1, 1, 1, 10])
        self.assertEqual(result, 52.5)

    def test_confidence_is_validation_rank_with_right_ties(self):
        ranks = confidence_rank([0, 1, 2, 3, 4], [1, 2, 2, 3])
        np.testing.assert_array_equal(ranks, [1, 0.75, 0.25, 0, 0])
        np.testing.assert_allclose(predicted_joint_error([[1, 2]], [2, 3], [4, 6]), [0.75])
        with self.assertRaises(ValueError):
            confidence_rank([1], [])

    def test_fold_order_and_selection_tie_are_deterministic(self):
        self.assertEqual(participant_folds(["a", "b", "c", "d"]),
                         participant_folds(["d", "b", "a", "c"]))
        events = [{"participant_id": "a", "gap_slots": 2},
                  {"participant_id": "b", "gap_slots": 2}]
        best, _, reference = select_head([2, 1], np.ones((2, 2, 2)), np.ones((2, 2, 2)),
                                         events, {"a": "fit", "b": "cal"}, [1, 1])
        self.assertEqual(best, 1)
        np.testing.assert_array_equal(reference, [1, 1])

    def test_native_scale_retains_original_scaler_precision(self):
        model = TinyBackbone().eval()
        head = ScaleHead(4)
        for parameter in head.parameters():
            parameter.data.zero_()
        values = torch.ones(1, 3, 2)
        visible = torch.ones_like(values, dtype=torch.bool)
        visible[:, 1] = False
        dataset = TensorDataset(values * visible, values, visible, ~visible,
                                torch.tensor([0]), torch.zeros(1, 3, 1))
        dataset.events = [{"start": 1, "end": 2}]
        std = np.array([1.00000006, 2.00000012], np.float64)
        _, scales = evaluate_heads(model, [(1, head)], dataset, [0, 0], std)
        b = float(torch.nn.functional.softplus(torch.tensor(0.0)) + 0.001)
        np.testing.assert_array_equal(scales[0, 0], b * std)
        self.assertFalse(np.array_equal(scales[0, 0], b * std.astype(np.float32).astype(np.float64)))

    def test_head_training_updates_head_only_and_rejects_wrong_backbone(self):
        torch.manual_seed(17)
        model = TinyBackbone()
        before = backbone_state_sha(model)
        flags = [parameter.requires_grad for parameter in model.parameters()]
        values = torch.randn(4, 8, 2)
        visible = torch.ones_like(values, dtype=torch.bool)
        visible[:, 2:5] = False
        target = ~visible
        x = values * visible
        data = TensorDataset(x, values, visible, target, torch.zeros(4, 8, 1))
        with tempfile.TemporaryDirectory() as directory:
            summary = train_head(model, data, directory, 4, epochs=2, batch_size=2, seed=7)
            self.assertEqual(len(summary["heads"]), 2)
            self.assertEqual(before, backbone_state_sha(model))
            self.assertEqual(flags, [parameter.requires_grad for parameter in model.parameters()])
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            heads = load_heads(directory, model, "cpu")
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(heads[0][1].parameters(), heads[1][1].parameters())))
            with self.assertRaises(ValueError):
                load_heads(directory, TinyBackbone(), "cpu")
            with self.assertRaises(FileExistsError):
                train_head(model, data, directory, 4, epochs=1)


if __name__ == "__main__":
    unittest.main()
