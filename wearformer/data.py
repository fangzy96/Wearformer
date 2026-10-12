"""Daily-array protocol, train-only statistics, and original masking sampler."""

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import (person_descriptor_table, time_gap_side_information,
                       window_start_minute_of_week)


@dataclass
class Pool:
    values: np.ndarray
    observed: np.ndarray
    metadata: list


def load_pool(directory, split):
    """Load native arrays without pickle or eager duplication of their contents."""
    directory = Path(directory)
    values = np.load(directory / f"{split}_values.npy", mmap_mode="r", allow_pickle=False)
    observed = np.load(directory / f"{split}_original_observed.npy", mmap_mode="r", allow_pickle=False)
    with (directory / f"{split}_base_windows.csv").open(newline="") as handle:
        metadata = list(csv.DictReader(handle))
    if values.ndim != 3 or values.shape[2] != 2 or observed.shape != values.shape:
        raise ValueError(f"{split}: expected matching [days,time,2] values and observation masks")
    if len(metadata) != len(values) or not metadata:
        raise ValueError(f"{split}: metadata row count differs from arrays or pool is empty")
    for index, row in enumerate(metadata):
        if not str(row.get("participant_id", "")).strip():
            raise ValueError("Every metadata row must contain participant_id")
        window_start_minute_of_week(row)
        if row.get("array_row", "") != "" and int(row["array_row"]) != index:
            raise ValueError(f"{split}: array_row disagrees with metadata order at row {index}")
    return Pool(values, observed, metadata)


def validate_pool_resolution(pool, config):
    step = int(config["time_step_minutes"])
    if step not in (1, 10) or int(config["model"]["seq_len"]) != 1440 // step:
        raise ValueError("Model sequence length must equal 1440 / time_step_minutes (1 or 10)")
    if pool.values.shape[1] != config["model"]["seq_len"]:
        raise ValueError("Daily-array length does not match the model sampling resolution")


def assert_disjoint_dates(first, second):
    """Reject dates from the same participant that occur in both data partitions."""
    def keys(pool):
        return {(str(row["participant_id"]), datetime.fromisoformat(row["window_start"]).date())
                for row in pool.metadata}
    if keys(first) & keys(second):
        raise ValueError("Data pools overlap in participant and calendar date")


def observed_feature_stats(values, observed):
    """Source scaler: float64 accumulation and population variance, training only."""
    sums = np.zeros(2, dtype=np.float64)
    sums_sq = np.zeros(2, dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    for start in range(0, len(values), 512):
        chunk = np.asarray(values[start:start + 512], dtype=np.float64)
        mask = np.asarray(observed[start:start + 512], dtype=bool).copy()
        mask &= np.isfinite(chunk)
        for feature in range(2):
            selected = chunk[..., feature][mask[..., feature]]
            sums[feature] += selected.sum(dtype=np.float64)
            sums_sq[feature] += np.square(selected, dtype=np.float64).sum(dtype=np.float64)
            counts[feature] += selected.size
    if np.any(counts < 2):
        raise ValueError("Each channel requires at least two observed training values")
    means = sums / counts
    variances = np.maximum(sums_sq / counts - np.square(means), 1e-12)
    return means, np.sqrt(variances), counts


class _ObservedValues:
    """Apply masks lazily to one participant's rows while computing summaries."""
    def __init__(self, pool):
        self.pool, self.shape = pool, pool.values.shape

    def __getitem__(self, key):
        return np.where(self.pool.observed[key], self.pool.values[key], np.nan)


def build_profiles(train_pool, descriptor_reference=None):
    """Build personal summaries; transfer may supply source descriptor statistics."""
    profiles = person_descriptor_table(_ObservedValues(train_pool), train_pool.metadata,
                                       reference=descriptor_reference)
    return profiles, person_descriptor_table.last_stats


def _sample(pool, row_index, start, end, means, stds, profile, step, scatter_rate=0, rng=None):
    truth_raw = np.asarray(pool.values[row_index], dtype=np.float32).copy()
    original = np.asarray(pool.observed[row_index], dtype=bool).copy() & np.isfinite(truth_raw)
    target = np.zeros_like(original)
    target[start:end, :] = original[start:end, :]
    visible = original.copy()
    visible[start:end, :] = False
    if scatter_rate > 0:
        scatter = (rng.random(original.shape) < scatter_rate) & visible
        visible &= ~scatter
        target |= scatter
    normalized = (truth_raw - means) / stds
    x = np.where(visible, normalized, 0.0).astype(np.float32)
    z = np.nan_to_num(normalized, nan=0.0).astype(np.float32)
    side = time_gap_side_information(window_start_minute_of_week(pool.metadata[row_index]),
                                     visible, profile, step)
    return x, z, visible, target, side, np.nan_to_num(truth_raw, nan=0.0).astype(np.float32)


class TrainingDataset(Dataset):
    """Exact primary participant-uniform, stratified block-plus-scatter sampler.

    The random generator for each item is keyed by seed, epoch and sample index.
    Strata, eligible-row choice, log-uniform length draw, retry/fallback behavior,
    and scatter draws follow the experimental implementation.
    """
    def __init__(self, pool, mean, std, profiles, config):
        self.pool = pool
        self.values, self.observed, self.metadata = pool.values, pool.observed, pool.metadata
        self.means, self.stds = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        training = config["training"]
        self.seed, self.epoch = int(training["seed"]), 0
        self.samples_per_participant = int(training["samples_per_participant"])
        self.scatter_rate = float(training["scatter_rate"])
        self.time_step_minutes = int(config["time_step_minutes"])
        zero = np.zeros((self.values.shape[1], 8), np.float32)
        self.row_profile = [profiles.get(str(row["participant_id"]), zero) for row in self.metadata]
        rows_by_pid = defaultdict(list)
        flank = int(training["flank"])
        for i, row in enumerate(self.metadata):
            hr = np.asarray(self.values[i, :, 0])
            mask = np.asarray(self.observed[i, :, 0], dtype=bool)
            if int((mask & np.isfinite(hr)).sum()) >= 2 * flank + 1:
                rows_by_pid[str(row["participant_id"])].append(i)
        self.rows_by_pid = {pid: tuple(rows) for pid, rows in rows_by_pid.items()}
        self.participants = tuple(sorted(self.rows_by_pid))
        if not self.participants:
            raise ValueError("No participant has enough observed HR for training blocks")
        self.enable_v2_strata(training["anchors"], training["min_observed_frac"],
                              training["flank_observed"], flank)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.participants) * self.samples_per_participant

    def enable_v2_strata(self, anchors, min_observed_frac, flank_observed, flank):
        anchors = [int(a) for a in anchors]
        if len(anchors) < 2 or any(b <= a for a, b in zip(anchors, anchors[1:])):
            raise ValueError("Training anchors must be strictly increasing")
        self.v2_strata = tuple(zip(anchors[:-1], anchors[1:]))
        self.v2_min_observed_frac = float(min_observed_frac)
        self.v2_flank_observed, self.v2_flank = int(flank_observed), int(flank)
        rows = sorted({r for rows in self.rows_by_pid.values() for r in rows})
        seq_len = self.values.shape[1]
        if max(anchors) + 2 * self.v2_flank > seq_len:
            raise ValueError("Largest training gap plus flanks exceeds a daily input")
        table = np.zeros((len(self.values), seq_len + 1), dtype=np.int32)
        for row_index in rows:
            hr = np.asarray(self.values[row_index, :, 0])
            ok = np.asarray(self.observed[row_index, :, 0], dtype=bool) & np.isfinite(hr)
            table[row_index, 1:] = np.cumsum(ok)
        self._v2_hr_cumsum = table
        deepest = np.full(len(self.values), -1, dtype=np.int8)
        for row_index in rows:
            for s_index, (lo_anchor, _) in enumerate(self.v2_strata):
                if self._v2_feasible_starts(table[row_index], lo_anchor).size:
                    deepest[row_index] = s_index
                else:
                    break
        self._v2_row_deepest = deepest
        self._v2_rows = []
        for s_index in range(len(self.v2_strata)):
            per_pid = {}
            for pid, pid_rows in self.rows_by_pid.items():
                keep = tuple(r for r in pid_rows if deepest[r] >= s_index)
                if keep:
                    per_pid[pid] = keep
            self._v2_rows.append(per_pid)

    def _v2_feasible_starts(self, cs, length):
        high = cs.shape[0] - 1 - self.v2_flank - length
        if high < self.v2_flank:
            return np.empty(0, dtype=np.int64)
        starts = np.arange(self.v2_flank, high + 1)
        inside = cs[starts + length] - cs[starts]
        left = cs[starts] - cs[starts - self.v2_flank]
        right = cs[starts + length + self.v2_flank] - cs[starts + length]
        need = max(1, int(self.v2_min_observed_frac * length))
        return starts[(inside >= need) & (left >= self.v2_flank_observed) & (right >= self.v2_flank_observed)]

    def _v2_pick(self, pid, rng):
        s_index = int(rng.integers(len(self.v2_strata)))
        while s_index > 0 and pid not in self._v2_rows[s_index]:
            s_index -= 1
        eligible = self._v2_rows[s_index].get(pid) or self.rows_by_pid[pid]
        row_index = int(rng.choice(eligible))
        start, length = self._v2_draw_in_stratum(row_index, s_index, rng)
        return row_index, start, length

    def _v2_draw_in_stratum(self, row_index, s_index, rng):
        deepest = int(self._v2_row_deepest[row_index])
        if deepest >= 0:
            s_index = min(s_index, deepest)
        lo, hi = self.v2_strata[s_index]
        cs = self._v2_hr_cumsum[row_index]
        for _ in range(12):
            length = int(round(float(np.exp(rng.uniform(float(np.log(lo)), float(np.log(hi)))))))
            length = max(1, min(length, hi))
            feasible = self._v2_feasible_starts(cs, length)
            if feasible.size:
                return int(rng.choice(feasible)), length
        feasible = self._v2_feasible_starts(cs, int(lo))
        if feasible.size:
            return int(rng.choice(feasible)), int(lo)
        raise RuntimeError(f"No feasible block for training row {row_index} in stratum [{lo},{hi})")

    def __getitem__(self, index):
        pid = self.participants[index % len(self.participants)]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        row, start, gap = self._v2_pick(pid, rng)
        x, z, visible, target, side, _ = _sample(self.pool, row, start, start + gap,
            self.means, self.stds, self.row_profile[row], self.time_step_minutes, self.scatter_rate, rng)
        if not target[:, 0].any():
            raise RuntimeError("Training block has no observed HR target")
        return tuple(torch.from_numpy(a) for a in (x, z, visible, target, side))


class FixedEventDataset(Dataset):
    """One masked daily input per fixed NPZ event; no scatter masking at evaluation."""
    def __init__(self, pool, manifest_path, mean, std, profiles, time_step_minutes, k=4):
        self.pool = pool
        self.means, self.stds = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self.time_step_minutes = int(time_step_minutes)
        self.events = []
        zero = np.zeros((pool.values.shape[1], 8), np.float32)
        self.row_profile = [profiles.get(str(row["participant_id"]), zero) for row in pool.metadata]
        with np.load(manifest_path, allow_pickle=False) as manifest:
            for key in sorted(manifest.files, key=int):
                gap = int(key)
                starts = manifest[key]
                if starts.ndim != 2 or len(starts) != len(pool.values) or gap < 1:
                    raise ValueError(f"Invalid manifest entry {key}")
                if k < 1 or starts.shape[1] < k:
                    raise ValueError("Manifest does not contain the requested number of events")
                for row_index in range(len(starts)):
                    for start in starts[row_index, :k]:
                        start = int(start)
                        if start < 0:
                            continue
                        if start + gap > pool.values.shape[1]:
                            raise ValueError("Manifest gap lies outside a daily input")
                        meta = pool.metadata[row_index]
                        self.events.append({"participant_id": str(meta["participant_id"]),
                            "day_id": meta["window_start"], "base_index": row_index,
                            "start": start, "end": start + gap, "gap_slots": gap,
                            "event_index": len(self.events)})
        self.records = self.events
        if not self.events:
            raise ValueError("No eligible events in evaluation manifest")

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        event = self.events[index]
        row = event["base_index"]
        x, _, visible, target, side, raw = _sample(self.pool, row, event["start"], event["end"],
            self.means, self.stds, self.row_profile[row], self.time_step_minutes)
        if not target[:, 0].any():
            raise ValueError("Evaluation event has no observed HR target")
        return (torch.from_numpy(x), torch.from_numpy(raw), torch.from_numpy(visible),
                torch.from_numpy(target), index, torch.from_numpy(side))
