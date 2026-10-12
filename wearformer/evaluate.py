"""Evaluate fixed gaps: event means, participant means, then equal gap weights."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import load_model, resolve_device
from .data import (FixedEventDataset, build_profiles, load_pool,
                   validate_pool_resolution, assert_disjoint_dates)


@torch.no_grad()
def evaluate_model(model, dataset, reference_std, device="cpu", batch_size=32, clamp_steps=True):
    """Score raw-unit predictions using fixed scales; validation remains unclipped."""
    model.eval()
    device = torch.device(device)
    reference_std = np.asarray(reference_std, np.float64)
    if reference_std.shape != (2,) or not np.isfinite(reference_std).all() or (reference_std <= 0).any():
        raise ValueError("Two finite, positive reference standard deviations are required")
    acc = defaultdict(lambda: {"hr": defaultdict(list), "steps": defaultdict(list)})
    counts = defaultdict(int)
    for x, truth, visible, target, indices, side in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        out = model(x.to(device), visible.to(device), side.to(device))["x_tilde_3"]
        prediction = out.cpu().numpy() * dataset.stds + dataset.means
        if clamp_steps:
            prediction[..., 1] = np.maximum(prediction[..., 1], 0)
        error = np.abs(prediction - truth.numpy())
        mask = target.numpy()
        # Preserve the source's float32 sum followed by integer-count division.
        totals, supports = (error * mask).sum(axis=1), mask.sum(axis=1)
        for j, index in enumerate(indices.tolist()):
            event = dataset.events[index]
            gap, pid = event["gap_slots"], event["participant_id"]
            counts[gap] += 1
            for c, name in enumerate(("hr", "steps")):
                if supports[j, c]:
                    acc[gap][name][pid].append(float(totals[j, c] / supports[j, c]))
    by_gap = {}
    for gap in sorted(acc):
        people = {name: {pid: statistics.fmean(values) for pid, values in groups.items()}
                  for name, groups in acc[gap].items()}
        if not all(people.values()):
            raise ValueError(f"Gap {gap} has no scored observations in one channel")
        hr, steps = (statistics.fmean(people[name].values()) for name in ("hr", "steps"))
        by_gap[str(gap)] = {"hr": hr, "steps": steps,
            "joint": float(0.5 * (hr / reference_std[0] + steps / reference_std[1])),
            "events": counts[gap], "per_participant": people}
    hr, steps = (statistics.fmean(row[name] for row in by_gap.values()) for name in ("hr", "steps"))
    return {"hr": hr, "steps": steps,
            "joint": float(0.5 * (hr / reference_std[0] + steps / reference_std[1])),
            "by_length": by_gap, "steps_clipped": bool(clamp_steps)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol-dir", "--data-dir", dest="protocol_dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, help="Target training days for personal features; defaults to protocol-dir")
    parser.add_argument("--manifest", "--manifest-dir", dest="manifest", type=Path)
    parser.add_argument("--split", default="internal_test", choices=("validation", "internal_test", "external_test"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--k", type=int, help="Number of fixed starts per day; defaults to checkpoint config")
    parser.add_argument("--reference-scales", type=Path, help="JSON containing std: [HR,Steps]; use target scales for transfer")
    parser.add_argument("--unclipped", action="store_true", help="Use only for validation/model selection")
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    pre = checkpoint["preprocessing"]
    train_pool = load_pool(args.train_dir or args.protocol_dir, "train")
    evaluation_pool = load_pool(args.protocol_dir, args.split)
    validate_pool_resolution(train_pool, checkpoint["config"])
    validate_pool_resolution(evaluation_pool, checkpoint["config"])
    assert_disjoint_dates(train_pool, evaluation_pool)
    if args.split != "validation" and (args.protocol_dir / "validation_base_windows.csv").exists():
        validation_pool = load_pool(args.protocol_dir, "validation")
        assert_disjoint_dates(validation_pool, evaluation_pool)
    profiles, _ = build_profiles(train_pool, (pre["descriptor_mean"], pre["descriptor_std"]))
    reference = pre["reference_std"]
    if args.reference_scales:
        reference = json.loads(args.reference_scales.read_text())["std"]
    evaluation = checkpoint["config"]["evaluation"]
    dataset = FixedEventDataset(evaluation_pool,
        (args.manifest or args.protocol_dir) / f"{args.split}_v2_starts.npz",
        pre["mean"], pre["std"], profiles, checkpoint["config"]["time_step_minutes"],
        args.k if args.k is not None else evaluation["events_per_day"])
    actual_gaps = {event["gap_slots"] for event in dataset.events}
    if actual_gaps != set(evaluation["gap_lengths"]):
        raise ValueError(f"Evaluation gaps {sorted(actual_gaps)} differ from configured gaps")
    result = evaluate_model(model, dataset, reference, device, args.batch_size, not args.unclipped)
    result.update({"split": args.split, "time_step_minutes": checkpoint["config"]["time_step_minutes"],
                   "reference_std": list(reference)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("hr", "steps", "joint")}, indent=2))


if __name__ == "__main__":
    main()
