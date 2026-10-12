"""Create a small synthetic ten-minute dataset and one-epoch smoke configuration."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from preprocessing.gaps import sample_starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use an empty output directory")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/apple_10min.json").read_text())
    config["model"].update(d_model=16, n_heads=2, d_k=8, d_v=8, d_ffn=16)
    config["training"].update(epochs=1, batch_size=2, samples_per_participant=2)
    config["evaluation"].update(events_per_day=1, batch_size=4)
    for split, days in (("train", 4), ("validation", 2), ("internal_test", 2)):
        values = np.empty((days, 144, 2), dtype=np.float32)
        metadata = []
        for i in range(days):
            phase = np.arange(144) / 144 * 2 * np.pi
            values[i, :, 0] = 65 - 8 * np.cos(phase) + rng.normal(0, 2, 144) + 2 * (i % 2)
            values[i, :, 1] = rng.poisson(10, 144)
            values[i, :36, 1] = 0
            month = {"train": 1, "validation": 2, "internal_test": 3}[split]
            metadata.append({"participant_id": f"synthetic-{i % 2}", "array_row": i,
                "base_window_id": f"{split}-{i}", "base_window_rank": i+1,
                "window_start": f"2000-{month:02d}-{i+1:02d} 00:00:00", "window_minutes": 1440})
        observed = np.ones_like(values, dtype=np.float32)
        np.save(root / f"{split}_values.npy", values)
        np.save(root / f"{split}_original_observed.npy", observed)
        with (root / f"{split}_base_windows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
            writer.writeheader()
            writer.writerows(metadata)
        if split != "train":
            starts = sample_starts(values, observed, config["evaluation"]["gap_lengths"],
                draws=1, flank=6, flank_observed=3, seed=11)
            np.savez_compressed(root / f"{split}_v2_starts.npz", **starts)
    (root / "smoke_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Synthetic inputs written to {root}; these are not study data")


if __name__ == "__main__":
    main()
