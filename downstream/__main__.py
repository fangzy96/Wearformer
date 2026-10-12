"""Command-line interface for naturally missing daily values."""

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fill native daily pools before a separate downstream analysis")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="internal_test",
                        help="native pool prefix, e.g. train, validation, internal_test, or natural")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    from .impute import run_fill
    print(json.dumps(run_fill(args), indent=2))


if __name__ == "__main__":
    main()
