"""Fit one observed-training mean per channel and use it for every missing cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def validate_arrays(values, observed):
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("values must have shape [days, time points, 2]")
    if observed.shape != values.shape:
        raise ValueError("values and observed must have identical shapes")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("values must be numeric")


def observed_feature_stats_f64(values, observed):
    """Training mean/SD/count, with the model runner's float64 accumulation.

    Only cells marked observed and containing finite values contribute. Chunk
    size and arithmetic match the observed-training statistics in the runner.
    """
    validate_arrays(values, observed)
    sums = np.zeros(2, dtype=np.float64)
    sums_sq = np.zeros(2, dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    for start in range(0, values.shape[0], 512):
        chunk = np.asarray(values[start:start + 512], dtype=np.float64)
        mask = np.asarray(observed[start:start + 512], dtype=bool).copy()
        mask &= np.isfinite(chunk)
        for feature in range(2):
            selected = chunk[..., feature][mask[..., feature]]
            sums[feature] += selected.sum(dtype=np.float64)
            sums_sq[feature] += np.square(selected, dtype=np.float64).sum(dtype=np.float64)
            counts[feature] += selected.size
    if np.any(counts < 2):
        raise ValueError("Each channel needs at least two observed training values")
    means = sums / counts
    variances = np.maximum(sums_sq / counts - np.square(means), 1e-12)
    return means, np.sqrt(variances), counts


def fit_training_mean(values, observed):
    means, _, counts = observed_feature_stats_f64(values, observed)
    return {
        "method": "training_mean", "features": ["heart_rate", "steps"],
        "mean": means.tolist(), "mean_hex": [float(x).hex() for x in means],
        "n_observed": counts.tolist(), "output_policy": "raw",
    }


def read_means(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    means = np.asarray([float.fromhex(x) for x in record["mean_hex"]], dtype=np.float64)
    if means.shape != (2,) or not np.isfinite(means).all():
        raise ValueError("The saved training means must contain two finite numbers")
    if record["mean"] != means.tolist():
        raise ValueError("Saved mean and hexadecimal representation disagree")
    return means


def impute(values, hide, means, observed=None):
    """Fill hide=True with fixed means; preserve all other available values."""
    values = np.asarray(values)
    hide = np.asarray(hide, dtype=bool)
    validate_arrays(values, hide)
    means = np.asarray(means, dtype=np.float64)
    if means.shape != (2,) or not np.isfinite(means).all():
        raise ValueError("means must contain two finite values")
    result = np.asarray(values, dtype=np.float64).copy()
    if observed is not None:
        validate_arrays(values, np.asarray(observed))
        result[~np.asarray(observed, dtype=bool)] = np.nan
    return np.where(hide, means, result)


class TrainingMean:
    """Native-unit predictor whose outputs are independent of evaluation data."""

    def __init__(self, means):
        self.means = np.asarray(means, dtype=np.float64)
        if self.means.shape != (2,) or not np.isfinite(self.means).all():
            raise ValueError("means must contain two finite values")

    def predict(self, values, visible):
        validate_arrays(np.asarray(values), np.asarray(visible))
        return np.broadcast_to(self.means, values.shape).copy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Fit observed training-cell means")
    fit.add_argument("--train-values", type=Path, required=True)
    fit.add_argument("--train-observed", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fill = commands.add_parser("impute", help="Fill a user-supplied missing mask")
    fill.add_argument("--means", type=Path, required=True)
    fill.add_argument("--values", type=Path, required=True)
    fill.add_argument("--hide", type=Path, required=True)
    fill.add_argument("--observed", type=Path)
    fill.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Output exists; choose a new output path")
    try:
        if args.command == "fit":
            values = np.load(args.train_values, mmap_mode="r", allow_pickle=False)
            observed = np.load(args.train_observed, mmap_mode="r", allow_pickle=False)
            record = fit_training_mean(values, observed)
            args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        else:
            values = np.load(args.values, allow_pickle=False)
            hide = np.load(args.hide, allow_pickle=False)
            observed = np.load(args.observed, allow_pickle=False) if args.observed else None
            result = impute(values, hide, read_means(args.means), observed)
            with args.out.open("wb") as handle:
                np.save(handle, result, allow_pickle=False)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
