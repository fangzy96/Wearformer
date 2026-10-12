"""Fill naturally missing daily values while preserving every observed value.

This is an imputation bridge. It does not implement disease detection, decide
clinical case eligibility, or reproduce a published detector's endpoint rules.
"""

import csv
import json
from pathlib import Path

import numpy as np


def fill_missing_values(values, observed, predictions, eligible_days=None):
    """Return completed values and a mask of newly filled cells.

    Shapes are [days, time, 2] (HR, Steps). By default only days with at least
    one observed HR are filled. Observed Steps=0 are retained; only imputed
    Steps are constrained to be nonnegative. Unfilled missing cells are NaN.
    """
    values = np.asarray(values)
    observed = np.asarray(observed, dtype=bool)
    predictions = np.asarray(predictions)
    if values.ndim != 3 or values.shape[-1] != 2 or observed.shape != values.shape or predictions.shape != values.shape:
        raise ValueError("values, observed and predictions must have shape [days, time, 2]")
    if not np.isfinite(values[observed]).all():
        raise ValueError("observed values must be finite")
    eligible = observed[:, :, 0].any(1) if eligible_days is None else np.asarray(eligible_days, dtype=bool)
    if eligible.shape != (len(values),):
        raise ValueError("eligible_days must contain one flag per day")
    imputed = ~observed & eligible[:, None, None]
    if not np.isfinite(predictions[imputed]).all():
        raise ValueError("predictions to be inserted must be finite")
    completed = values.astype(np.result_type(values.dtype, np.float32), copy=True)
    completed[~observed] = np.nan
    completed[imputed] = predictions[imputed]
    steps = completed[:, :, 1]
    steps[imputed[:, :, 1]] = np.maximum(steps[imputed[:, :, 1]], 0)
    return completed, imputed


def complete_pool(model, pool, profiles, mean, std, time_step_minutes, *, device="cpu", batch_size=32):
    """Run daily inference on natural observation masks, then preserve observations."""
    import torch
    from wearformer.features import time_gap_side_information, window_start_minute_of_week

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    if mean.shape != (2,) or std.shape != (2,) or np.any(std <= 0) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("two finite channel means and positive standard deviations are required")
    original = np.asarray(pool.observed, dtype=bool)
    eligible = original[:, :, 0].any(1)
    predictions = np.full(pool.values.shape, np.nan, np.float32)
    rows = np.flatnonzero(eligible)
    fallback = np.zeros((pool.values.shape[1], 8), np.float32)
    model.to(device).eval()
    with torch.no_grad():
        for offset in range(0, len(rows), batch_size):
            indices = rows[offset:offset + batch_size]
            raw = np.asarray(pool.values[indices], dtype=np.float32)
            visible = original[indices]
            if not np.isfinite(raw[visible]).all():
                raise ValueError("observed values must be finite")
            x = np.where(visible, (raw - mean) / std, 0).astype(np.float32)
            side = []
            for index, observation in zip(indices, visible):
                row = pool.metadata[index]
                profile = profiles.get(str(row["participant_id"]), fallback)
                side.append(time_gap_side_information(window_start_minute_of_week(row), observation,
                                                       profile, time_step_minutes))
            output = model(torch.from_numpy(x).to(device),
                           torch.from_numpy(visible.astype(np.float32)).to(device),
                           torch.from_numpy(np.stack(side)).to(device))
            prediction = output["x_tilde_3"].float().cpu().numpy()
            predictions[indices] = prediction * std + mean
    return fill_missing_values(pool.values, original, predictions, eligible)


def run_fill(args):
    from wearformer.checkpoint import load_model
    from wearformer.data import build_profiles, load_pool, validate_pool_resolution

    destination = Path(args.out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    targets = [destination / name for name in ("completed.npz", "days.csv", "imputation.json")]
    if any(path.exists() for path in targets):
        raise FileExistsError("imputation output exists; choose a new output directory")
    model, checkpoint = load_model(args.checkpoint, device=args.device)
    prep = checkpoint["preprocessing"]
    history = load_pool(args.data_dir, "train")
    validate_pool_resolution(history, checkpoint["config"])
    profiles, _ = build_profiles(history, descriptor_reference=(np.asarray(prep["descriptor_mean"]),
                                                               np.asarray(prep["descriptor_std"])))
    pool = load_pool(args.data_dir, args.split)
    validate_pool_resolution(pool, checkpoint["config"])
    completed, imputed = complete_pool(model, pool, profiles, prep["mean"], prep["std"],
                                       int(checkpoint["config"]["time_step_minutes"]),
                                       device=args.device, batch_size=args.batch_size)
    np.savez_compressed(targets[0], values=completed, observed=np.asarray(pool.observed, bool), imputed=imputed)
    with open(targets[1], "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pool.metadata[0]))
        writer.writeheader()
        writer.writerows(pool.metadata)
    summary = {
        "schema": "wearformer-completed-daily-values/1", "split": args.split,
        "channels": ["HR", "Steps"], "time_step_minutes": int(checkpoint["config"]["time_step_minutes"]),
        "days": len(completed), "filled_cells": imputed.sum(axis=(0, 1)).tolist(),
        "rule": "fill missing cells on days with observed HR; preserve observed cells; clip imputed Steps at zero",
        "downstream_detector_run": False,
    }
    targets[2].write_text(json.dumps(summary, indent=2) + "\n")
    return summary
