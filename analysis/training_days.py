#!/usr/bin/env python3
"""Calculate a training-day criterion from user-supplied aggregate Joint MAE.

This analysis requires only the Python standard library. It does not fit models.
See analysis/README.md for the data schema and interpretation of the criterion.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class BudgetPoint:
    label: str
    days: float
    joint_mae: float


@dataclass(frozen=True)
class Interval:
    start: str
    end: str
    added_days: float
    start_joint_mae: float
    end_joint_mae: float
    relative_reduction_percent: float
    daily_relative_reduction_percent: float


def _positive_integer(value: str, name: str) -> int:
    if not value.isdecimal() or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer; received {value!r}")
    return int(value)


def read_curves(path: Path) -> dict[str, list[BudgetPoint]]:
    """Read ordered curves, deriving All's day count from unrounded totals.

    Rows must occur in increasing budget order within each setting, with All last.
    Integer training-cohort counts must be constant within each setting.
    """
    required = {
        "setting", "train_days", "joint_mae",
        "total_qualifying_training_days", "training_participants",
    }
    curves: dict[str, list[BudgetPoint]] = {}
    counts: dict[str, tuple[int, int]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(key) is None for key in required):
                raise ValueError(f"CSV row {row_number} has an incorrect number of fields")
            setting = row["setting"].strip()
            if not setting:
                raise ValueError(f"CSV row {row_number} has an empty setting")
            total = _positive_integer(
                row["total_qualifying_training_days"].strip(),
                "total_qualifying_training_days",
            )
            participants = _positive_integer(
                row["training_participants"].strip(), "training_participants"
            )
            cohort_counts = (total, participants)
            if setting in counts and counts[setting] != cohort_counts:
                raise ValueError(f"Training-cohort counts differ within {setting!r}")
            counts[setting] = cohort_counts
            label = row["train_days"].strip()
            days = total / participants if label == "All" else float(
                _positive_integer(label, "train_days")
            )
            try:
                mae = float(row["joint_mae"])
            except ValueError as exc:
                raise ValueError(f"Invalid joint_mae in CSV row {row_number}") from exc
            curves.setdefault(setting, []).append(BudgetPoint(label, days, mae))
    if not curves:
        raise ValueError("The CSV contains no data rows")
    for setting, points in curves.items():
        if points[-1].label != "All":
            raise ValueError(f"{setting!r} must end with an All row")
        compute_intervals(points)
    return curves


def compute_intervals(points: Sequence[BudgetPoint]) -> list[Interval]:
    """Calculate signed percentage MAE reduction per added training day.

    For adjacent points a and b: 100 * (E_a - E_b) / E_a / (b - a).
    No smoothing, monotonic fitting or clipping is applied.
    """
    if len(points) < 2:
        raise ValueError("At least two training budgets are required")
    for index, point in enumerate(points):
        if not math.isfinite(point.days) or point.days <= 0:
            raise ValueError("Training-day counts must be positive and finite")
        if not math.isfinite(point.joint_mae) or point.joint_mae <= 0:
            raise ValueError("Joint MAE values must be positive and finite")
        if point.label == "All":
            if index != len(points) - 1:
                raise ValueError("All must be the final training budget")
        elif _positive_integer(point.label, "train_days") != point.days:
            raise ValueError("Numeric budget labels must equal their day counts")
        if index and point.days <= points[index - 1].days:
            raise ValueError("Training-day counts must be strictly increasing")
    intervals = []
    for first, second in zip(points, points[1:]):
        added_days = second.days - first.days
        reduction = 100 * (first.joint_mae - second.joint_mae) / first.joint_mae
        intervals.append(Interval(
            start=first.label, end=second.label, added_days=added_days,
            start_joint_mae=first.joint_mae, end_joint_mae=second.joint_mae,
            relative_reduction_percent=reduction,
            daily_relative_reduction_percent=reduction / added_days,
        ))
    return intervals


def select_training_days(
    intervals: Sequence[Interval],
    threshold: float = 0.20,
    min_later_intervals: int = 2,
) -> int | None:
    """Select the earliest budget whose remaining interval rates meet the rule.

    The outgoing interval counts toward min_later_intervals. An isolated final
    interval therefore cannot determine the selected budget with the default 2.
    """
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if isinstance(min_later_intervals, bool) or not isinstance(min_later_intervals, int):
        raise ValueError("min_later_intervals must be a positive integer")
    if min_later_intervals < 1:
        raise ValueError("min_later_intervals must be a positive integer")
    for index in range(len(intervals) - min_later_intervals + 1):
        if all(
            item.daily_relative_reduction_percent <= threshold
            for item in intervals[index:]
        ):
            return _positive_integer(intervals[index].start, "interval start")
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Aggregate training-day CSV")
    parser.add_argument(
        "--threshold", type=float, default=0.20,
        help="Maximum daily relative MAE reduction in percent (default: 0.20)",
    )
    parser.add_argument(
        "--min-later-intervals", type=int, default=2,
        help="Required intervals from a candidate budget through All (default: 2)",
    )
    parser.add_argument(
        "--intervals-out", type=Path,
        help="Optional CSV output with all adjacent interval calculations",
    )
    args = parser.parse_args(argv)
    try:
        curves = read_curves(args.csv)
        rows = []
        summaries = {}
        for setting, points in curves.items():
            intervals = compute_intervals(points)
            selected = select_training_days(
                intervals, args.threshold, args.min_later_intervals
            )
            summaries[setting] = {
                "selected_training_days": selected,
                "all_mean_training_days": points[-1].days,
            }
            rows.extend({"setting": setting, **asdict(item)} for item in intervals)
        if args.intervals_out:
            with args.intervals_out.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "threshold_percent_per_added_day": args.threshold,
        "minimum_remaining_intervals": args.min_later_intervals,
        "settings": summaries,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
