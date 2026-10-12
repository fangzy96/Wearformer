"""Frozen-gap scoring and independent random streams for comparator evaluation."""

from __future__ import annotations

from contextlib import contextmanager
import csv
from pathlib import Path
import random
import statistics

import numpy as np

from .train_mean import validate_arrays


@contextmanager
def preserve_rng(seed=None):
    """Restore Python, NumPy, PyTorch CPU/CUDA RNG states, even on exceptions."""
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def load_pool(directory, split):
    directory = Path(directory)
    values = np.load(directory / f"{split}_values.npy", mmap_mode="r", allow_pickle=False)
    observed = np.load(directory / f"{split}_original_observed.npy", mmap_mode="r", allow_pickle=False)
    validate_arrays(values, observed)
    with (directory / f"{split}_base_windows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(values):
        raise ValueError("Pool metadata row count differs from array row count")
    if any(int(row["array_row"]) != index for index, row in enumerate(rows)):
        raise ValueError("Pool metadata array_row must match its zero-based row order")
    participants = [row["participant_id"] for row in rows]
    if any(not name for name in participants):
        raise ValueError("Every row needs a participant_id")
    return values, observed, participants


def load_starts(manifests, split, n_rows, n_steps, expected_lengths):
    """Merge disjoint manifest files; keys are gap lengths in sampling positions."""
    starts = {}
    for directory in manifests:
        with np.load(Path(directory) / f"{split}_v2_starts.npz", allow_pickle=False) as data:
            for key in data.files:
                length = int(key)
                if key != str(length) or length <= 0 or length > n_steps:
                    raise ValueError("Manifest has an invalid gap-length key")
                if length in starts:
                    raise ValueError(f"Gap length {length} appears in more than one manifest")
                array = data[key]
                if array.ndim != 2 or array.shape[0] != n_rows or not np.issubdtype(array.dtype, np.integer):
                    raise ValueError("Manifest starts must be integer arrays with shape [days, starts]")
                if np.any(array < -1) or np.any((array >= 0) & (array + length > n_steps)):
                    raise ValueError("A manifest gap extends beyond its daily input")
                starts[length] = array.copy()
    if set(starts) != set(expected_lengths):
        raise ValueError(
            f"Expected gap lengths {sorted(expected_lengths)}, received {sorted(starts)}. "
            "One-minute selection requires all eight durations, including the 10-minute gap."
        )
    return starts


def score_gaps(predict, values, observed, participants, starts, scales, *, k=4, batch_size=64):
    """Event MAE -> participant mean -> duration mean, with raw predictions.

    predict(values, visible) must return native-unit predictions of the same
    shape. Target cells were observed before a simultaneous two-channel gap.
    No output clipping is applied to any comparator.
    """
    validate_arrays(values, observed)
    scales = np.asarray(scales, dtype=np.float64)
    if scales.shape != (2,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Two positive finite Joint scoring scales are required")
    if k < 1 or batch_size < 1:
        raise ValueError("k and batch_size must be positive")
    if len(participants) != len(values) or not starts:
        raise ValueError("Participants and nonempty gap manifests are required")
    unique, person_index = np.unique(participants, return_inverse=True)
    time = np.arange(values.shape[1])
    by_gap = {}
    for length in sorted(starts):
        table = starts[length]
        if table.ndim != 2 or table.shape[0] != len(values) or table.shape[1] < k:
            raise ValueError("Manifest row count or number of sampled starts is inconsistent")
        jobs = [(row, int(start)) for row in range(len(values)) for start in table[row, :k] if start >= 0]
        totals = np.zeros((len(unique), 2), dtype=np.float64)
        counts = np.zeros((len(unique), 2), dtype=np.int64)
        for offset in range(0, len(jobs), batch_size):
            batch = jobs[offset:offset + batch_size]
            indices = np.asarray([item[0] for item in batch])
            gap_start = np.asarray([item[1] for item in batch])
            truth = np.asarray(values[indices], dtype=np.float32)
            available = np.asarray(observed[indices], dtype=bool) & np.isfinite(truth)
            gap = (time[None, :] >= gap_start[:, None]) & (time[None, :] < gap_start[:, None] + length)
            target = available & gap[..., None]
            visible = available & ~gap[..., None]
            prediction = np.asarray(predict(truth, visible))
            if prediction.shape != truth.shape:
                raise ValueError("Predictor output shape does not match the daily inputs")
            if not np.isfinite(prediction[target]).all():
                raise ValueError("Predictor returned non-finite values at evaluated cells")
            # np.where excludes unavailable NaNs without multiplying them by zero.
            errors = np.where(target, np.abs(prediction - truth), 0.0)
            cell_counts = target.sum(axis=1)
            event_sums = errors.sum(axis=1)
            for channel in range(2):
                has_target = cell_counts[:, channel] > 0
                event_mae = event_sums[has_target, channel] / cell_counts[has_target, channel]
                np.add.at(totals[:, channel], person_index[indices[has_target]], event_mae)
                np.add.at(counts[:, channel], person_index[indices[has_target]], 1)
        means = []
        for channel in range(2):
            present = counts[:, channel] > 0
            if not present.any():
                raise ValueError(f"No observed targets for gap {length}, channel {channel}")
            means.append(float(np.mean(totals[present, channel] / counts[present, channel])))
        by_gap[str(length)] = {
            "hr_mae": means[0], "steps_mae": means[1],
            "joint_mae": float(np.mean(np.asarray(means) / scales)),
        }
    hr = statistics.fmean(row["hr_mae"] for row in by_gap.values())
    steps = statistics.fmean(row["steps_mae"] for row in by_gap.values())
    overall = {"hr_mae": hr, "steps_mae": steps,
               "joint_mae": 0.5 * (hr / scales[0] + steps / scales[1])}
    return {"overall": overall, "by_gap_positions": by_gap, "output_policy": "raw"}


def select_best_epoch(history):
    """Exact validation Joint argmin; the earliest epoch resolves a tie."""
    if not history or any(not np.isfinite(row["validation_joint_mae"]) for row in history):
        raise ValueError("Selection requires nonempty, finite validation history")
    return min(history, key=lambda row: (row["validation_joint_mae"], row["epoch"]))["epoch"]
