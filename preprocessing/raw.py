"""Align a user-supplied roster of raw wearable exports to regular time grids."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from ._raw import BIN_NS, _parse_hr, _parse_steps, _prorate_steps, read_signal


def align_fitbit(hr_file: Path, steps_file: Path) -> pd.DataFrame:
    """Median HR and maximum recorded minute-total Steps; gate Steps on HR."""
    hr, _ = read_signal(hr_file, "heartrate")
    steps, _ = read_signal(steps_file, "steps")
    if (steps < 0).any() or ((steps % 1) != 0).any():
        raise ValueError("Fitbit Steps must be nonnegative integer minute totals")
    hr = hr.resample("1min").median()
    steps = steps.groupby(steps.index.floor("min")).max().reindex(hr.index)
    steps = steps.fillna(0).where(hr.notna())
    return pd.DataFrame({"datetime": hr.index, "heartrate": hr.to_numpy(),
                         "steps": steps.to_numpy()})


def align_apple(hr_file: Path, steps_file: Path) -> pd.DataFrame:
    """Median HR and overlap-weighted interval Steps on ten-minute bins."""
    times, values, _ = _parse_hr(hr_file)
    if len(times) == 0:
        raise ValueError("No usable Apple HR samples")
    hr = pd.Series(values, index=pd.DatetimeIndex(times)).resample("10min").median()
    starts, ends, counts, _ = _parse_steps(steps_file)
    bins, mass, _, _ = _prorate_steps(starts, ends, counts)
    positions = bins - int(hr.index[0].value // BIN_NS)
    in_span = (positions >= 0) & (positions < len(hr))
    steps = np.zeros(len(hr), dtype=np.float64)
    steps[positions[in_span]] = mass[in_span]
    steps[hr.isna().to_numpy()] = np.nan
    return pd.DataFrame({"datetime": hr.index, "heartrate": hr.to_numpy(), "steps": steps})


def read_records(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(columns).issubset(reader.fieldnames or []):
            raise ValueError(f"{path} requires columns {columns}")
        rows = list(reader)
    ids = [r["participant_id"] for r in rows]
    if not ids or len(set(ids)) != len(ids) or any(not p for p in ids):
        raise ValueError("Supply exactly one record per nonempty participant ID")
    return sorted(rows, key=lambda r: r["participant_id"])


def resolve_path(roster: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else roster.resolve().parent / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=("fitbit", "apple"))
    parser.add_argument("--records", required=True, type=Path,
                        help="CSV: participant_id,hr_file,steps_file; paths relative to CSV")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    rows = read_records(args.records, ("participant_id", "hr_file", "steps_file"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = args.output_dir / "records.csv"
    if output_manifest.exists() or any(args.output_dir.glob("record_*.csv")):
        raise FileExistsError("Use a new output directory to avoid overwriting records")
    align = align_fitbit if args.device == "fitbit" else align_apple
    outputs = []
    for index, row in enumerate(rows):
        frame = align(resolve_path(args.records, row["hr_file"]),
                      resolve_path(args.records, row["steps_file"]))
        name = f"record_{index:06d}.csv"
        frame.to_csv(args.output_dir / name, index=False, na_rep="",
                     date_format="%Y-%m-%d %H:%M:%S",
                     float_format="%.6g" if args.device == "fitbit" else "%.17g")
        outputs.append({"participant_id": row["participant_id"], "path": name})
    with output_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("participant_id", "path"))
        writer.writeheader()
        writer.writerows(outputs)
    print(f"Wrote {len(outputs)} aligned records to {args.output_dir}")


if __name__ == "__main__":
    main()
