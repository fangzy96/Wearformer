"""Raw-export parsers and Apple interval allocation used by the preprocessing CLI."""
from __future__ import annotations
from pathlib import Path
from typing import Any

APPLE_LABELS = ("HK Apple Watch", "HK Apple", "HK iPhone")
BIN_NS = 10 * 60 * 10**9
MAX_INTERVAL_NS = 60 * 60 * 10**9
MAX_BINS_PER_INTERVAL = 7


def read_signal(path: Path, value_column: str) -> tuple["Any", int]:
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(
        path,
        usecols=["datetime", value_column],
        skip_blank_lines=True,
        dtype={"datetime": "string", value_column: "float64"},
    )
    raw_rows = len(frame)
    if raw_rows == 0:
        raise ValueError(f"No data rows in {path}")

    timestamps = pd.to_datetime(
        frame.pop("datetime"),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    values = frame.pop(value_column)
    bad_timestamp = int(timestamps.isna().sum())
    bad_value = int(values.isna().sum())
    nonfinite_value = int((~np.isfinite(values.to_numpy())).sum())
    if bad_timestamp or bad_value or nonfinite_value:
        raise ValueError(
            f"Invalid {value_column} input in {path}: "
            f"timestamps={bad_timestamp}, missing_values={bad_value}, "
            f"nonfinite_values={nonfinite_value}"
        )
    series = pd.Series(values.to_numpy(copy=False), index=timestamps, name=value_column)
    return series, raw_rows

def _read_apple_csv(path: Path, expected_columns: list[str]) -> tuple["Any", int]:
    """Robustly read a raw Apple export.

    Handles the leading blank line, occasional duplicated header rows embedded
    mid-file, and junk rows: every field is read as a string and coerced.
    Returns the string frame and the raw row count.
    """
    import pandas as pd

    frame = pd.read_csv(
        path,
        skip_blank_lines=True,
        dtype="string",
        header=0,
        on_bad_lines="skip",
        engine="python",
    )
    missing = [column for column in expected_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    return frame.loc[:, expected_columns], len(frame)

def _parse_timestamps(raw: "Any") -> tuple["Any", int]:
    """Parse naive timestamps in either 24-hour or 12-hour (AM/PM) notation.

    Returns the parsed datetimes (NaT where unparseable) and how many rows
    required the 12-hour fallback (some raw Apple exports use AM/PM).
    """
    import pandas as pd

    parsed = pd.to_datetime(raw, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    retry = parsed.isna() & raw.notna()
    ampm_rows = 0
    if retry.any():
        fallback = pd.to_datetime(
            raw[retry], format="%Y-%m-%d %I:%M:%S %p", errors="coerce"
        )
        ampm_rows = int(fallback.notna().sum())
        parsed = parsed.copy()
        parsed[retry] = fallback
    return parsed, ampm_rows

def _parse_hr(path: Path) -> tuple["Any", "Any", dict[str, int]]:
    import numpy as np
    import pandas as pd

    frame, raw_rows = _read_apple_csv(path, ["device", "datetime", "heartrate"])
    device_ok = frame["device"].isin(APPLE_LABELS).fillna(False).to_numpy()
    timestamps, ampm_rows = _parse_timestamps(frame["datetime"])
    values = pd.to_numeric(frame["heartrate"], errors="coerce")
    parse_ok = timestamps.notna().to_numpy() & np.isfinite(
        values.to_numpy(dtype="float64", na_value=np.nan)
    )
    # A duplicated embedded header row fails both the device filter and the
    # coercions; count it once, as an invalid row.
    dropped_invalid = int((~parse_ok).sum())
    dropped_device = int((parse_ok & ~device_ok).sum())
    keep = parse_ok & device_ok
    counters = {
        "raw_rows": raw_rows,
        "dropped_invalid": dropped_invalid,
        "dropped_wrong_device": dropped_device,
        "ampm_rows": ampm_rows,
    }
    return (
        timestamps.to_numpy()[keep],
        values.to_numpy(dtype="float64")[keep],
        counters,
    )

def _parse_steps(path: Path) -> tuple["Any", "Any", "Any", dict[str, Any]]:
    import numpy as np
    import pandas as pd

    frame, raw_rows = _read_apple_csv(
        path, ["device", "start_datetime", "end_datetime", "steps"]
    )
    device_ok = frame["device"].isin(APPLE_LABELS).fillna(False).to_numpy()
    starts, start_ampm = _parse_timestamps(frame["start_datetime"])
    ends, end_ampm = _parse_timestamps(frame["end_datetime"])
    # Some exports record point events with a blank (or whitespace-only)
    # end_datetime. Treat them as instantaneous events at start: the whole
    # step count is assigned to the bin containing start.
    blank_end = (
        frame["end_datetime"].fillna("").str.strip().eq("").to_numpy()
        & starts.notna().to_numpy()
    )
    ends = ends.copy()
    ends[blank_end] = starts[blank_end]
    values = pd.to_numeric(frame["steps"], errors="coerce")
    numeric = values.to_numpy(dtype="float64", na_value=np.nan)
    parse_ok = (
        starts.notna().to_numpy()
        & ends.notna().to_numpy()
        & np.isfinite(numeric)
    )
    dropped_invalid = int((~parse_ok).sum())
    dropped_device = int((parse_ok & ~device_ok).sum())
    keep = parse_ok & device_ok
    negative = keep & (numeric < 0)
    dropped_negative = int(negative.sum())
    keep &= ~negative
    counters = {
        "raw_rows": raw_rows,
        "dropped_invalid": dropped_invalid,
        "dropped_wrong_device": dropped_device,
        "dropped_negative_steps": dropped_negative,
        "explicit_zero_rows": int((numeric[keep] == 0).sum()),
        "ampm_rows": max(start_ampm, end_ampm),
        "blank_end_instantaneous": int((blank_end & keep).sum()),
    }
    return (
        starts.to_numpy()[keep].astype("datetime64[ns]").astype(np.int64),
        ends.to_numpy()[keep].astype("datetime64[ns]").astype(np.int64),
        numeric[keep],
        counters,
    )

def _interval_overlap_seconds(starts: "Any", ends: "Any") -> tuple[float, float]:
    """Total interval seconds and double-counted (overlapping) seconds."""
    import numpy as np

    if len(starts) == 0:
        return 0.0, 0.0
    order = np.argsort(starts, kind="mergesort")
    s = starts[order]
    e = ends[order]
    total = float((e - s).sum()) / 1e9
    union = 0
    current_start = int(s[0])
    current_end = int(e[0])
    for i in range(1, len(s)):
        si = int(s[i])
        ei = int(e[i])
        if si > current_end:
            union += current_end - current_start
            current_start, current_end = si, ei
        else:
            current_end = max(current_end, ei)
    union += current_end - current_start
    return total, max(0.0, total - union / 1e9)

def _prorate_steps(
    starts: "Any", ends: "Any", steps: "Any"
) -> tuple["Any", "Any", "Any", dict[str, Any]]:
    """Pro-rate interval step counts onto the 10-minute bin grid.

    Returns (bin_index_int64, step_mass, contributing_interval_count) for bins
    with at least one contributing interval, plus counters. Bin index k covers
    [k * BIN_NS, (k + 1) * BIN_NS).
    """
    import numpy as np

    duration = ends - starts
    negative_duration = duration < 0
    over_long = ~negative_duration & (duration > MAX_INTERVAL_NS)
    keep = ~negative_duration & ~over_long
    counters: dict[str, Any] = {
        "dropped_negative_duration": int(negative_duration.sum()),
        "dropped_over_60min": int(over_long.sum()),
        "dropped_over_60min_step_sum": float(steps[over_long].sum()),
    }
    starts = starts[keep]
    ends = ends[keep]
    steps = steps[keep]
    duration = duration[keep]
    total_seconds, overlap_seconds = _interval_overlap_seconds(starts, ends)
    counters["interval_total_seconds"] = total_seconds
    counters["interval_overlap_seconds"] = overlap_seconds
    counters["retained_step_sum"] = float(steps.sum())
    if len(starts) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, np.empty(0, dtype=np.float64), empty, counters

    first_bin = starts // BIN_NS
    grid_min = int(first_bin.min())
    grid_max = int(((np.maximum(ends, starts + 1) - 1) // BIN_NS).max())
    size = grid_max - grid_min + 1
    mass = np.zeros(size, dtype=np.float64)
    contributors = np.zeros(size, dtype=np.int64)

    instantaneous = duration == 0
    if instantaneous.any():
        offsets = (first_bin[instantaneous] - grid_min).astype(np.int64)
        mass += np.bincount(offsets, weights=steps[instantaneous], minlength=size)
        contributors += np.bincount(offsets, minlength=size)

    positive = ~instantaneous
    if positive.any():
        s = starts[positive]
        e = ends[positive]
        v = steps[positive]
        d = duration[positive].astype(np.float64)
        b0 = s // BIN_NS
        for k in range(MAX_BINS_PER_INTERVAL):
            bin_start = (b0 + k) * BIN_NS
            overlap = np.minimum(e, bin_start + BIN_NS) - np.maximum(s, bin_start)
            has = overlap > 0
            if not has.any():
                break
            offsets = (b0[has] + k - grid_min).astype(np.int64)
            weights = v[has] * (overlap[has].astype(np.float64) / d[has])
            mass += np.bincount(offsets, weights=weights, minlength=size)
            contributors += np.bincount(offsets, minlength=size)

    occupied = contributors > 0
    bin_indices = grid_min + np.flatnonzero(occupied).astype(np.int64)
    return bin_indices, mass[occupied], contributors[occupied], counters
