"""Build nested first-N qualifying training-day pools in chronological order."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np


def chronological_indices(rows, days, minimum_days=40):
    """Fixed cohort: participants with >=minimum_days qualifying training rows.

    rows come exclusively from the training split. Held-out and nonqualifying
    dates are never counted as training days. All returns that fixed cohort's
    complete training histories in chronological order.
    """
    if minimum_days < 1 or (days != "All" and (not isinstance(days, int) or days < 1 or days > minimum_days)):
        raise ValueError("days must be All or a positive integer no larger than minimum_days")
    by_person = {}
    for index, row in enumerate(rows):
        if int(row["array_row"]) != index:
            raise ValueError("array_row must match metadata row order")
        participant = row["participant_id"]
        if not participant:
            raise ValueError("Every training day needs a participant_id")
        start = datetime.fromisoformat(row["window_start"])
        if (start.hour, start.minute, start.second, start.microsecond) != (0, 0, 0, 0):
            raise ValueError("Daily inputs must start at midnight")
        by_person.setdefault(participant, []).append((start, index))
    selected = []
    for participant in sorted(by_person):
        records = sorted(by_person[participant])
        if len({day for day, _ in records}) != len(records):
            raise ValueError("A participant has duplicate training dates")
        if len(records) >= minimum_days:
            selected.extend(index for _, index in (records if days == "All" else records[:days]))
    if not selected:
        raise ValueError("No participants meet the fixed-cohort training-day requirement")
    return np.asarray(selected, dtype=np.int64)


def check_heldout_dates(train_rows, protocol_dir):
    """Reject overlap if held-out metadata is provided alongside training data."""
    seen = {(row["participant_id"], datetime.fromisoformat(row["window_start"]).date()) for row in train_rows}
    for split in ("validation", "internal_test"):
        path = Path(protocol_dir) / f"{split}_base_windows.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if (row["participant_id"], datetime.fromisoformat(row["window_start"]).date()) in seen:
                        raise ValueError("A training date is also present in held-out metadata")


def build_pools(train_dir, output_dir, budgets, minimum_days=40, protocol_dir=None):
    train_dir, output_dir = Path(train_dir), Path(output_dir)
    with (train_dir / "train_base_windows.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    values = np.load(train_dir / "train_values.npy", mmap_mode="r", allow_pickle=False)
    observed = np.load(train_dir / "train_original_observed.npy", mmap_mode="r", allow_pickle=False)
    if values.shape != observed.shape or values.ndim != 3 or values.shape[-1] != 2 or len(values) != len(rows):
        raise ValueError("Training arrays and metadata have inconsistent shapes")
    check_heldout_dates(rows, protocol_dir or train_dir)
    if len(set(budgets)) != len(budgets) or not budgets:
        raise ValueError("Provide distinct, nonempty training budgets")
    selections = [(budget, chronological_indices(rows, budget, minimum_days)) for budget in budgets]
    destinations = [output_dir / ("days_all" if budget == "All" else f"days_{budget:02d}") for budget, _ in selections]
    if any(path.exists() for path in destinations):
        raise ValueError("A requested output pool already exists; choose a new output directory")
    for (_, indices), destination in zip(selections, destinations):
        destination.mkdir(parents=True)
        for name, source in [("train_values.npy", values), ("train_original_observed.npy", observed)]:
            result = np.lib.format.open_memmap(destination / name, mode="w+", dtype=source.dtype,
                                               shape=(len(indices),) + source.shape[1:])
            for start in range(0, len(indices), 256):
                result[start:start + 256] = source[indices[start:start + 256]]
            result.flush()
            del result
        with (destination / "train_base_windows.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, original in enumerate(indices):
                writer.writerow({**rows[original], "array_row": index})
    return destinations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, help="Optional held-out metadata directory for overlap checks")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--days", default="1,2,3,5,7,10,14,20,30,40,All")
    parser.add_argument("--minimum-days", type=int, default=40, help="Fixed-cohort inclusion threshold")
    args = parser.parse_args(argv)
    try:
        budgets = ["All" if item.strip().lower() == "all" else int(item.strip()) for item in args.days.split(",")]
        paths = build_pools(args.train_dir, args.output_dir, budgets, args.minimum_days, args.protocol_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
