"""Time and personal-summary features from the primary training implementation."""
import math
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
HR_INDEX = 0
STEPS_INDEX = 1


def window_start_minute_of_week(row: Mapping[str, str]) -> int:
    """Read local wall-clock time from a daily input metadata row."""
    raw = str(row.get('window_start', '')).strip()
    if not raw:
        raise ValueError('Time-aware SAITS requires window_start in every base-window metadata row')
    try:
        start = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f'Cannot parse window_start={raw!r}') from exc
    return start.weekday() * 1440 + start.hour * 60 + start.minute

def cyclic_time_features(start_minute_of_week: int, seq_len: int, time_step_minutes: int=1) -> np.ndarray:
    """Daily and weekly sine/cosine features on the local sample grid."""
    minute_of_week = (int(start_minute_of_week) + np.arange(seq_len, dtype=np.int64) * int(time_step_minutes)) % (7 * 1440)
    minute_of_day = minute_of_week % 1440
    daily_phase = 2.0 * np.pi * minute_of_day.astype(np.float32) / 1440.0
    weekly_phase = 2.0 * np.pi * minute_of_week.astype(np.float32) / (7.0 * 1440.0)
    return np.stack((np.sin(daily_phase), np.cos(daily_phase), np.sin(weekly_phase), np.cos(weekly_phase)), axis=-1).astype(np.float32, copy=False)

def gap_distance_features(observed: np.ndarray) -> np.ndarray:
    """Log-scaled backward/forward distances to visible observations."""
    mask = np.asarray(observed, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f'Expected observed mask [T,F], got {mask.shape}')
    (seq_len, n_features) = mask.shape
    output = np.empty((seq_len, n_features * 2), dtype=np.float32)
    scale = math.log1p(seq_len)
    indices = np.arange(seq_len, dtype=np.int32)[:, None]
    previous_index = np.maximum.accumulate(np.where(mask, indices, -1), axis=0)
    previous = np.where(mask, 0, np.where(previous_index >= 0, indices - previous_index, seq_len))
    following_index = np.minimum.accumulate(np.where(mask, indices, seq_len)[::-1], axis=0)[::-1]
    following = np.where(mask, 0, np.where(following_index < seq_len, following_index - indices, seq_len))
    output[:, 0::2] = np.log1p(previous) / scale
    output[:, 1::2] = np.log1p(following) / scale
    return output

def person_descriptor_table(values: np.ndarray, metadata: Sequence[Mapping[str, str]], reference: Optional[Tuple[np.ndarray, np.ndarray]]=None) -> Dict[str, np.ndarray]:
    """Eight train-only personal features, standardized across training participants."""
    by_person: Dict[str, List[int]] = {}
    for (row, meta) in enumerate(metadata):
        by_person.setdefault(str(meta['participant_id']), []).append(row)
    T = int(values.shape[1])
    step = max(1, int(1440 // T)) if T else 1
    min_still = max(1, 60 // step)
    wake_win = max(1, 900 // step)
    raw: Dict[str, np.ndarray] = {}
    for (pid, rows) in by_person.items():
        v = np.asarray(values[rows], dtype=np.float32)
        fin = np.isfinite(v)
        hr = v[:, :, HR_INDEX]
        st = v[:, :, STEPS_INDEX]
        sleep = np.zeros(hr.shape, bool)
        wakes = []
        for d in range(v.shape[0]):
            vis = fin[d, :, HR_INDEX]
            if vis.sum() < 120:
                continue
            med = np.median(hr[d, vis])
            still = (np.nan_to_num(st[d]) == 0) & fin[d, :, STEPS_INDEX]
            e = np.flatnonzero(np.diff(np.concatenate(([0], still.astype(np.int8), [0]))))
            for (j, k2) in zip(e[::2], e[1::2]):
                if k2 - j >= min_still and np.median(np.nan_to_num(hr[d, j:k2])) < med:
                    sleep[d, j:k2] = True
            w = np.flatnonzero(sleep[d, :wake_win])
            if w.size:
                wakes.append(float(w.max()))
        hm = fin[:, :, HR_INDEX]
        sm = fin[:, :, STEPS_INDEX]
        f = [np.nanmean(hr[hm]) if hm.any() else np.nan, np.nanstd(hr[hm]) if hm.any() else np.nan, np.nanmean(hr[sleep & hm]) if (sleep & hm).any() else np.nan, np.nanmean(hr[~sleep & hm]) if (~sleep & hm).any() else np.nan, np.nanmean(st[sm]) if sm.any() else np.nan, float(np.mean(st[sm] > 0)) if sm.any() else np.nan, float(sleep[hm].mean()) if hm.any() else np.nan, float(np.median(wakes)) if wakes else np.nan]
        raw[pid] = np.asarray(f, np.float32)
    M = np.stack(list(raw.values()))
    if reference is not None:
        (mu, sd) = (np.asarray(reference[0], np.float32), np.asarray(reference[1], np.float32))
    else:
        mu = np.nanmean(M, axis=0)
        sd = np.nanstd(M, axis=0) + 1e-06
    table: Dict[str, np.ndarray] = {}
    table_stats = (mu, sd)
    for (pid, f) in raw.items():
        z = np.nan_to_num((f - mu) / sd, nan=0.0).astype(np.float32)
        table[pid] = np.tile(z[None, :], (T, 1))
    person_descriptor_table.last_stats = table_stats
    return table

def person_descriptor_stats(values: np.ndarray, metadata: Sequence[Mapping[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
    """Return the population mean and standard deviation of personal features."""
    person_descriptor_table(values, metadata)
    return person_descriptor_table.last_stats


def time_gap_side_information(start_minute_of_week, input_observed, profile, time_step_minutes=1):
    """Concatenate four time, four distance and eight personal features."""
    time = cyclic_time_features(start_minute_of_week, input_observed.shape[0], time_step_minutes)
    gap = gap_distance_features(input_observed)
    return np.concatenate((time, gap, profile), axis=-1).astype(np.float32, copy=False)
