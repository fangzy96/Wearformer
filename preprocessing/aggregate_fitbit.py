"""Aggregate prepared one-minute Fitbit pools while preserving participant/date splits."""
from __future__ import annotations

import argparse
import shutil
import warnings
from pathlib import Path

import numpy as np


def aggregate(values, observed):
    observed = np.asarray(observed, bool)
    if values.shape != observed.shape or values.shape[1:] != (1440, 2):
        raise ValueError("Expected matching [days,1440,2] arrays")
    if not np.array_equal(observed[..., 0], observed[..., 1]):
        raise ValueError("Fitbit HR and Steps observation masks must coincide")
    v = values.reshape(len(values), 144, 10, 2)
    o = observed.reshape(len(values), 144, 10, 2)
    seen = o[..., 0].sum(axis=2) >= 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hr = np.nanmedian(np.where(o[..., 0], v[..., 0], np.nan), axis=2)
    steps = np.nansum(np.where(o[..., 1], v[..., 1], 0.0), axis=2)
    out = np.full((len(values), 144, 2), np.nan, np.float32)
    out[..., 0] = np.where(seen, hr, np.nan)
    out[..., 1] = np.where(seen, steps, np.nan)
    return out, np.repeat(seen[..., None], 2, axis=2).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use an empty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "internal_test"):
        values = np.load(args.data_dir / f"{split}_values.npy", mmap_mode="r")
        mask = np.load(args.data_dir / f"{split}_original_observed.npy", mmap_mode="r")
        shape = (len(values), 144, 2)
        result = np.lib.format.open_memmap(args.output_dir / f"{split}_values.npy",
                                            mode="w+", dtype=np.float32, shape=shape)
        observed = np.lib.format.open_memmap(args.output_dir / f"{split}_original_observed.npy",
                                              mode="w+", dtype=np.float32, shape=shape)
        for start in range(0, len(values), 512):
            result[start:start+512], observed[start:start+512] = aggregate(
                values[start:start+512], mask[start:start+512])
        result.flush()
        observed.flush()
        shutil.copyfile(args.data_dir / f"{split}_base_windows.csv",
                        args.output_dir / f"{split}_base_windows.csv")
    print(f"Wrote ten-minute pools to {args.output_dir}; generate new ten-minute gap manifests next")


if __name__ == "__main__":
    main()
