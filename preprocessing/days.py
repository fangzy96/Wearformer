"""Build within-participant training, validation and test arrays from aligned records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .raw import read_records, resolve_path

SPLITS = ("train", "validation", "internal_test")
FIELDS = ("participant_id", "base_window_id", "base_window_rank", "array_row",
          "window_start_index", "window_start", "window_minutes")


def daily_arrays(path: Path, resolution: int):
    frame = pd.read_csv(path, usecols=["datetime", "heartrate", "steps"])
    stamps = pd.to_datetime(frame.pop("datetime"), errors="raise")
    if stamps.dt.tz is not None or stamps.duplicated().any() or not stamps.is_monotonic_increasing:
        raise ValueError(f"Require sorted unique, timezone-naive timestamps in {path}")
    if frame.empty or not stamps.equals(stamps.dt.floor(f"{resolution}min")):
        raise ValueError(f"Require timestamps aligned to {resolution}-minute bins in {path}")
    # Match the released within-person pool builders: Fitbit drops partial
    # boundary days; Apple first pads boundary days and applies HR coverage.
    if resolution == 1:
        first = stamps.iloc[0].ceil("D")
        end = (stamps.iloc[-1] + pd.Timedelta(minutes=1)).floor("D")
    else:
        first = stamps.iloc[0].normalize()
        end = stamps.iloc[-1].normalize() + pd.Timedelta(days=1)
    if end <= first:
        return np.empty((0, 1440 // resolution, 2), np.float32), first
    grid = pd.date_range(first, end, freq=f"{resolution}min", inclusive="left")
    frame.index = pd.DatetimeIndex(stamps)
    values = frame.reindex(grid).to_numpy(dtype=np.float32)
    if np.isinf(values).any():
        raise ValueError(f"Infinite measurements in {path}")
    return values.reshape(-1, 1440 // resolution, 2), first


def assign_days(values, participant, seed, *, min_coverage=0.8, n_validation=5,
                n_test=10, min_training=10):
    eligible = np.flatnonzero(np.isfinite(values[:, :, 0]).mean(axis=1) >= min_coverage)
    if len(eligible) < n_validation + n_test + min_training:
        return None
    def key(day):
        return int.from_bytes(hashlib.blake2b(
            f"{seed}:{participant}:{int(day)}".encode(), digest_size=8).digest(), "big")
    ordered = sorted(eligible.tolist(), key=key)
    return {"internal_test": ordered[:n_test],
            "validation": ordered[n_test:n_test + n_validation],
            "train": ordered[n_test + n_validation:]}


def build(records: Path, output: Path, resolution: int, seed: int):
    rows = read_records(records, ("participant_id", "path"))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    plans, counts = [], {s: 0 for s in SPLITS}
    for row in rows:
        path = resolve_path(records, row["path"])
        values, first = daily_arrays(path, resolution)
        parts = assign_days(values, row["participant_id"], seed)
        if parts:
            source_first = pd.to_datetime(pd.read_csv(path, usecols=["datetime"], nrows=1)["datetime"].iloc[0])
            offset = int((first - source_first).total_seconds() // 60) if resolution == 1 else 0
            plans.append((row["participant_id"], path, first, offset, parts))
            for split in SPLITS:
                counts[split] += len(parts[split])
    if not plans:
        raise ValueError("No participant has at least 25 days with 80% HR coverage")
    arrays, masks, metadata = {}, {}, {s: [] for s in SPLITS}
    for split in SPLITS:
        shape = (counts[split], 1440 // resolution, 2)
        arrays[split] = np.lib.format.open_memmap(output / f"{split}_values.npy",
                                                  mode="w+", dtype=np.float32, shape=shape)
        masks[split] = np.lib.format.open_memmap(output / f"{split}_original_observed.npy",
                                                 mode="w+", dtype=np.float32, shape=shape)
    for participant, path, first, offset, parts in plans:
        values, _ = daily_arrays(path, resolution)
        for split, days in parts.items():
            for rank, day in enumerate(days, 1):
                index = len(metadata[split])
                arrays[split][index] = values[day]
                masks[split][index] = np.isfinite(values[day])
                stamp = first + pd.Timedelta(days=day)
                metadata[split].append(dict(participant_id=participant,
                    base_window_id=f"{split}:{participant}:{day:04d}", base_window_rank=rank,
                    array_row=index, window_start_index=offset + day * (1440 // resolution),
                    window_start=stamp.isoformat(sep=" "), window_minutes=1440))
    for split in SPLITS:
        arrays[split].flush()
        masks[split].flush()
        with (output / f"{split}_base_windows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(metadata[split])
    (output / "preparation.json").write_text(json.dumps({
        "time_step_minutes": resolution, "split_seed": seed,
        "validation_days_per_participant": 5, "test_days_per_participant": 10,
        "minimum_training_days": 10, "minimum_hr_coverage": 0.8}, indent=2) + "\n")
    print(f"Prepared {len(plans)} participants in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True,
                        help="CSV: participant_id,path; aligned-record paths relative to CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, choices=(1, 10), required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    build(args.records, args.output_dir, args.resolution, args.seed)


if __name__ == "__main__":
    main()
