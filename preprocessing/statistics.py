"""Save fixed reference scales fitted only to a full training pool."""
import argparse
import json
from pathlib import Path

from wearformer.data import load_pool, observed_feature_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = load_pool(args.train_dir, "train")
    mean, std, count = observed_feature_stats(pool.values, pool.observed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump({"mean": mean.tolist(), "std": std.tolist(), "count": count.tolist(),
                   "fit_split": "train", "ddof": 0}, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote training-only reference scales to {args.output}")


if __name__ == "__main__":
    main()
