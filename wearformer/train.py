"""Train the primary Wearformer model and select its checkpoint on validation gaps."""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import resolve_device, save_checkpoint
from .data import (FixedEventDataset, TrainingDataset, build_profiles, load_pool,
                   observed_feature_stats, validate_pool_resolution, assert_disjoint_dates)
from .evaluate import evaluate_model
from .model import TimeGapSAITS


def equal_feature_masked_loss(prediction, truth, mask):
    """Mean over targets per channel, equally over channels, then daily inputs."""
    error = torch.abs(prediction - truth)
    weights = mask.to(dtype=error.dtype)
    counts = weights.sum(dim=1)
    valid = counts > 0
    feature_means = (error * weights).sum(dim=1) / counts.clamp_min(1)
    if torch.any(valid.sum(dim=1) == 0):
        raise ValueError("A training input has no target values")
    return ((feature_means * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)).mean()


def capture_rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng_state(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total, seen = 0.0, 0
    for x, truth, visible, target, side in loader:
        x, truth, visible, target, side = (a.to(device) for a in (x, truth, visible, target, side))
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, visible, side)["x_tilde_3"]
        loss = equal_feature_masked_loss(prediction, truth, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.detach()) * len(x)
        seen += len(x)
    return total / seen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-dir", "--data-dir", dest="protocol_dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--manifest", "--manifest-dir", dest="manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--time-step-minutes", type=int, choices=(1, 10))
    parser.add_argument("--reference-scales", type=Path, help="JSON std: [HR,Steps] from the full training pool for training-day analyses")
    parser.add_argument("--epochs", type=int, help="Override for smoke checks; primary runs use the config value")
    parser.add_argument("--save-every-epoch", action="store_true")
    args = parser.parse_args(argv)
    config = copy.deepcopy(json.loads(args.config.read_text()))
    training = config["training"]
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.time_step_minutes is not None:
        config["time_step_minutes"] = args.time_step_minutes
    if int(training["epochs"]) < 1:
        raise ValueError("epochs must be positive")
    device = resolve_device(args.device)
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    train_pool = load_pool(args.train_dir or args.protocol_dir, "train")
    validation_pool = load_pool(args.protocol_dir, "validation")
    validate_pool_resolution(train_pool, config)
    validate_pool_resolution(validation_pool, config)
    assert_disjoint_dates(train_pool, validation_pool)
    mean, std, _ = observed_feature_stats(train_pool.values, train_pool.observed)
    profiles, (descriptor_mean, descriptor_std) = build_profiles(train_pool)
    reference = std
    if args.reference_scales:
        reference = np.asarray(json.loads(args.reference_scales.read_text())["std"], np.float64)
    preprocessing = {"mean": mean.tolist(), "std": std.tolist(),
        "descriptor_mean": descriptor_mean.tolist(), "descriptor_std": descriptor_std.tolist(),
        "reference_std": reference.tolist()}
    dataset = TrainingDataset(train_pool, mean, std, profiles, config)
    evaluation = config["evaluation"]
    validation = FixedEventDataset(validation_pool,
        (args.manifest or args.protocol_dir) / "validation_v2_starts.npz",
        mean, std, profiles, config["time_step_minutes"], evaluation["events_per_day"])
    actual_gaps = {event["gap_slots"] for event in validation.events}
    if actual_gaps != set(evaluation["gap_lengths"]):
        raise ValueError(f"Validation gaps {sorted(actual_gaps)} differ from configured gaps")
    model = TimeGapSAITS(**config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"],
                                 weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["epochs"], eta_min=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "best.pt").exists():
        raise FileExistsError("Output already contains best.pt; choose a new run directory")
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    best, history = float("inf"), []
    for epoch_index in range(int(training["epochs"])):
        # Sampling/shuffling use the original zero-based epoch; saved epochs are one-based.
        epoch = epoch_index + 1
        dataset.set_epoch(epoch_index)
        generator = torch.Generator().manual_seed(seed + epoch_index)
        loader = DataLoader(dataset, batch_size=training["batch_size"], shuffle=True,
            generator=generator, num_workers=training.get("num_workers", 0),
            pin_memory=device.type == "cuda", drop_last=False)
        loss = train_one_epoch(model, loader, optimizer, device, training["grad_clip"])
        state = capture_rng_state()
        try:
            metrics = evaluate_model(model, validation, reference, device,
                                     evaluation["batch_size"], clamp_steps=False)
        finally:
            restore_rng_state(state)
        score = metrics["joint"]
        history.append({"epoch": epoch, "training_mae": loss, "validation_joint_mae": score,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        scheduler.step()
        # Strict inequality resolves tied validation scores in favour of the earlier epoch.
        if score < best:
            best = score
            save_checkpoint(args.output_dir / "best.pt", model, config, preprocessing, epoch, score)
        if args.save_every_epoch:
            save_checkpoint(args.output_dir / f"epoch_{epoch:02d}.pt", model, config, preprocessing, epoch, score)
        (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"epoch {epoch}: training MAE={loss:.6f}; validation Joint MAE={score:.6f}", flush=True)


if __name__ == "__main__":
    main()
