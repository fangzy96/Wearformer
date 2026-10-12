"""Run with python -m confidence {train,select,score} --help."""

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estimate Wearformer imputation error and validation-relative confidence")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "select", "score"):
        sub = commands.add_parser(command)
        sub.add_argument("--checkpoint", required=True, help="trained Wearformer checkpoint")
        sub.add_argument("--data-dir", required=True, help="native daily pools and fixed-gap manifests")
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--workers", type=int, default=0)
        sub.add_argument("--batch-size", type=int, default=64 if command == "train" else 32)
        if command == "train":
            sub.add_argument("--out-dir", required=True)
            sub.add_argument("--epochs", type=int, default=20)
            sub.add_argument("--seed", type=int)
        else:
            sub.add_argument("--head-dir", required=True)
            sub.add_argument("--manifest", help="defaults to DATA_DIR/<split>_v2_starts.npz")
            sub.add_argument("--starts-per-day", type=int, default=4)
            if command == "select":
                sub.add_argument("--folds", help="CSV with participant_id,fold (fit or cal)")
                sub.add_argument("--fold-salt", default="wearformer-confidence-v1")
            else:
                sub.add_argument("--split", choices=("validation", "internal_test"), default="internal_test")
                sub.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    from .pipeline import run_evaluation, run_train
    result = run_train(args) if args.command == "train" else run_evaluation(args, args.command == "select")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
