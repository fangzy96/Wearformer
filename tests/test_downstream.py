"""Synthetic checks of observation-preserving natural-gap completion."""

import unittest

import numpy as np
import torch
from torch import nn

from downstream.impute import complete_pool, fill_missing_values
from wearformer.data import Pool


class ConstantBackbone(nn.Module):
    def forward(self, values, observed, side):
        self.last_side = side.detach().cpu().numpy()
        return {"x_tilde_3": torch.ones_like(values)}


class CompletionTests(unittest.TestCase):
    def test_observations_zero_steps_and_empty_hr_days_are_preserved(self):
        values = np.array([[[70, 0], [np.nan, np.nan], [80, 4]],
                           [[np.nan, 0], [np.nan, np.nan], [np.nan, 3]]], dtype=np.float64)
        observed = np.isfinite(values)
        predictions = np.full_like(values, -2)
        completed, filled = fill_missing_values(values, observed, predictions)
        np.testing.assert_array_equal(completed[observed], values[observed])
        np.testing.assert_array_equal(completed[0, 1], [-2, 0])
        self.assertTrue(np.isnan(completed[1, :, 0]).all())
        self.assertFalse(filled[1].any())
        self.assertEqual(int(filled.sum()), 2)

    def test_nonfinite_predictions_cannot_be_written(self):
        values = np.array([[[70, 0], [np.nan, np.nan]]], dtype=np.float32)
        with self.assertRaises(ValueError):
            fill_missing_values(values, np.isfinite(values), np.full_like(values, np.nan))

    def test_complete_pool_standardizes_and_uses_mean_profile_fallback(self):
        values = np.array([[[70, 0], [np.nan, np.nan], [80, 4]]], dtype=np.float32)
        pool = Pool(values, np.isfinite(values), [{"participant_id": "synthetic-person", "window_start": "2050-01-01T00:00:00"}])
        model = ConstantBackbone()
        completed, filled = complete_pool(model, pool, {}, [60, 2], [10, 3], 1)
        np.testing.assert_array_equal(completed[0, 1], [70, 5])
        np.testing.assert_array_equal(model.last_side[:, :, 8:], 0)
        self.assertEqual(int(filled.sum()), 2)
        np.testing.assert_array_equal(completed[pool.observed], values[pool.observed])


if __name__ == "__main__":
    unittest.main()
