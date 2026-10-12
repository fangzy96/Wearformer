"""Generate and freeze shared validation/test gap starts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def sample_starts(values, observed, lengths, *, draws=4, flank=60,
                  flank_observed=30, min_fraction=0.5, seed=20260906):
    if values.shape != observed.shape or values.ndim != 3 or values.shape[2] != 2:
        raise ValueError("Expected matching [days, positions, 2] values and masks")
    if draws < 1 or flank < 1 or not 1 <= flank_observed <= flank:
        raise ValueError("Invalid number of draws or context requirement")
    if not 0 < min_fraction <= 1:
        raise ValueError("min_fraction must lie in (0,1]")
    ok = np.asarray(observed[:, :, 0], dtype=bool) & np.isfinite(values[:, :, 0])
    n, size = ok.shape
    cs = np.concatenate([np.zeros((n, 1), np.int64), np.cumsum(ok, 1)], 1)
    rng = np.random.default_rng(seed)
    store = {}
    for length in lengths:
        if length < 1 or length + 2 * flank > size:
            raise ValueError(f"Gap {length} does not fit in a day with context")
        starts = np.arange(flank, size - flank - length + 1)
        inside = cs[:, starts + length] - cs[:, starts]
        left = cs[:, starts] - cs[:, starts - flank]
        right = cs[:, starts + length + flank] - cs[:, starts + length]
        good = ((inside >= max(1, int(min_fraction * length))) &
                (left >= flank_observed) & (right >= flank_observed))
        chosen = np.full((n, draws), -1, np.int32)
        for i in range(n):
            candidates = starts[good[i]]
            if candidates.size:
                chosen[i] = rng.choice(candidates, size=draws, replace=candidates.size < draws)
        store[str(length)] = chosen
    return store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resolution", type=int, choices=(1, 10), required=True)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--draws", type=int, default=4)
    args = parser.parse_args()
    out = args.output_dir or args.protocol_dir
    out.mkdir(parents=True, exist_ok=True)
    lengths = [1, 10, 30, 60, 120, 240, 480, 720] if args.resolution == 1 else [1, 3, 6, 12, 24, 48, 72]
    flank = 60 // args.resolution
    for split in ("validation", "internal_test"):
        dest = out / f"{split}_v2_starts.npz"
        if dest.exists():
            raise FileExistsError(f"Refusing to replace fixed events: {dest}")
        values = np.load(args.protocol_dir / f"{split}_values.npy", mmap_mode="r")
        observed = np.load(args.protocol_dir / f"{split}_original_observed.npy", mmap_mode="r")
        starts = sample_starts(values, observed, lengths, draws=args.draws, flank=flank,
                               flank_observed=flank // 2, seed=args.seed)
        np.savez_compressed(dest, **starts)
    (out / "gap_generation.json").write_text(json.dumps({
        "seed": args.seed, "draws_per_day": args.draws, "gap_lengths_positions": lengths,
        "time_step_minutes": args.resolution, "flank": flank,
        "flank_observed": flank // 2, "min_observed_fraction": 0.5}, indent=2) + "\n")
    print(f"Wrote fixed gap starts to {out}")


if __name__ == "__main__":
    main()
