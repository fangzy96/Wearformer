"""Train, select and score an error head using the public Wearformer data API."""

import copy
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .head import (
    ScaleHead, assert_one_block, backbone_state_sha, balanced_nll,
    confidence_rank, frozen_forward, laplace_loss, participant_folds,
    predicted_joint_error,
)


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def train_head(backbone, dataset, output_dir, d_model, *, epochs=20,
               batch_size=64, seed=20260907, device="cpu", workers=0,
               learning_rate=1e-3, width=128, eps=1e-3,
               backbone_checkpoint_sha=None):
    """Fit only a scale head, saving every epoch for later validation selection.

    Dataset items are (input, standardized truth, visibility, target mask, side
    information), as returned by TrainingDataset with scatter_rate=0.
    """
    if epochs < 1 or batch_size < 1 or len(dataset) == 0:
        raise ValueError("positive epochs/batch size and a nonempty dataset are required")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if list(destination.glob("head_ep*.pt")) or (destination / "heads.json").exists():
        raise FileExistsError("head output already exists; choose a new output directory")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    backbone = backbone.to(device).eval()
    if any(parameter.grad is not None for parameter in backbone.parameters()):
        raise ValueError("backbone must have no existing gradients")
    before = backbone_state_sha(backbone)
    grad_flags = [parameter.requires_grad for parameter in backbone.parameters()]
    head = ScaleHead(d_model, width, eps).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate, weight_decay=0)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    records = []
    for epoch_index in range(epochs):
        epoch = epoch_index + 1
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch_index)
        generator = torch.Generator().manual_seed(seed + epoch_index)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            generator=generator, num_workers=workers)
        head.train()
        loss_sum, samples = 0.0, 0
        for model_input, truth, visible, target, side in loader:
            model_input, truth, visible, target, side = (
                tensor.to(device) for tensor in (model_input, truth, visible, target, side)
            )
            assert_one_block(visible.bool(), target.bool())
            encoded, prediction = frozen_forward(backbone, model_input, visible, side)
            optimizer.zero_grad(set_to_none=True)
            loss = laplace_loss(head(encoded, visible), prediction, truth, target)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite scale-head training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(model_input)
            samples += len(model_input)
        schedule.step()
        if any(parameter.grad is not None for parameter in backbone.parameters()):
            raise RuntimeError("scale-head training created a backbone gradient")
        if backbone_state_sha(backbone) != before:
            raise RuntimeError("scale-head training changed the backbone")
        if grad_flags != [parameter.requires_grad for parameter in backbone.parameters()]:
            raise RuntimeError("backbone requires_grad flags changed")
        filename = f"head_ep{epoch:03d}.pt"
        checkpoint = {
            "schema": "wearformer-error-head/1", "epoch": epoch,
            "d_model": int(d_model), "width": int(width), "eps": float(eps),
            "backbone_state_sha256": before,
            "backbone_checkpoint_sha256": backbone_checkpoint_sha,
            "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        }
        torch.save(checkpoint, destination / filename)
        records.append({"epoch": epoch, "file": filename,
                        "sha256": file_sha(destination / filename),
                        "training_nll": loss_sum / samples})
    summary = {
        "schema": "wearformer-error-head-training/1", "heads": records,
        "backbone_state_sha256": before,
        "backbone_checkpoint_sha256": backbone_checkpoint_sha,
        "backbone_unchanged": True, "requires_grad_unchanged": True,
        "training": {"epochs": epochs, "batch_size": batch_size, "seed": seed,
                     "learning_rate": learning_rate, "weight_decay": 0,
                     "gradient_clip": 1.0, "scatter_rate": 0.0},
    }
    write_json(destination / "heads.json", summary)
    return summary


def load_heads(head_dir, backbone, device, selected_epoch=None):
    """Load head checkpoints only if bound to this exact backbone state."""
    directory = Path(head_dir)
    manifest = json.loads((directory / "heads.json").read_text())
    fingerprint = backbone_state_sha(backbone)
    if manifest["backbone_state_sha256"] != fingerprint:
        raise ValueError("head was trained on a different backbone")
    loaded = []
    for row in manifest["heads"]:
        if selected_epoch is not None and row["epoch"] != selected_epoch:
            continue
        path = directory / row["file"]
        if file_sha(path) != row["sha256"]:
            raise ValueError(f"head checkpoint changed: {path.name}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["backbone_state_sha256"] != fingerprint:
            raise ValueError("head checkpoint backbone mismatch")
        head = ScaleHead(state["d_model"], state["width"], state["eps"])
        head.load_state_dict(state["state_dict"], strict=True)
        loaded.append((int(row["epoch"]), head.to(device).eval()))
    if not loaded:
        raise ValueError("no matching scale-head checkpoints")
    return loaded


def evaluate_heads(backbone, heads, dataset, mean, std, *, device="cpu", batch_size=32, workers=0):
    """Compute event/channel NLL and full-gap mean scales in native units.

    The NLL residual is formed in float32, then scale and residual are promoted
    to float64 for logarithms and reductions, matching the original evaluator.
    All heads share the same frozen backbone forward for each batch.
    """
    events = dataset.events
    if not events:
        raise ValueError("no evaluation gap events")
    backbone.eval()
    native_std = np.asarray(std, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    nll = np.full((len(heads), len(events), 2), np.nan, dtype=np.float64)
    scales = np.full_like(nll, np.nan)
    visited = np.zeros(len(events), dtype=bool)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    with torch.no_grad():
        for model_input, raw_truth, visible, target, indices, side in loader:
            idx = indices.numpy().astype(int)
            if visited[idx].any():
                raise ValueError("duplicate evaluation event index")
            visited[idx] = True
            standardized = ((raw_truth.numpy().astype(np.float32) - mean) / std).astype(np.float32)
            standardized = np.nan_to_num(standardized, nan=0.0)
            truth = torch.from_numpy(standardized).to(device)
            model_input, visible, target, side = (
                tensor.to(device) for tensor in (model_input, visible, target, side)
            )
            encoded, prediction = frozen_forward(backbone, model_input, visible, side)
            residual = (truth - prediction).abs().double()
            mask = target.bool()
            counts = mask.sum(1)
            for head_index, (_, head) in enumerate(heads):
                b = head(encoded, visible).double()
                cell_nll = torch.log(b) + residual / b
                totals = torch.where(mask, cell_nll, 0.0).sum(1)
                means = totals / counts.clamp_min(1)
                means = torch.where(counts > 0, means, float("nan"))
                nll[head_index, idx] = means.cpu().numpy()
                for batch_index, event_index in enumerate(idx):
                    event = events[event_index]
                    start, end = int(event["start"]), int(event["end"])
                    if not 0 <= start < end <= b.shape[1]:
                        raise ValueError("invalid event boundaries")
                    native = b[batch_index, start:end].cpu().numpy() * native_std
                    scales[head_index, event_index] = native.mean(0)
    if not visited.all() or not np.isfinite(scales).all():
        raise RuntimeError("incomplete or nonfinite scale-head evaluation")
    return nll, scales


def load_fold_file(path, participants):
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    assignments = {}
    for row in rows:
        pid, fold = row["participant_id"], row["fold"]
        if pid in assignments or fold not in {"fit", "cal"}:
            raise ValueError("fold CSV requires unique participant_id and fit/cal labels")
        assignments[pid] = fold
    missing = set(map(str, participants)) - assignments.keys()
    if missing:
        raise ValueError(f"fold CSV lacks {len(missing)} validation participants")
    if {assignments[str(pid)] for pid in participants} != {"fit", "cal"}:
        raise ValueError("both fit and cal validation participants are required")
    return assignments


def select_head(epochs, nll, native_scales, events, folds, reference_std):
    ids = np.array([str(event["participant_id"]) for event in events])
    gaps = np.array([int(event["gap_slots"]) for event in events])
    scores = {"fit": {}, "cal": {}, "all": {}}
    for fold in scores:
        use = np.ones(len(ids), bool) if fold == "all" else np.array([folds[pid] == fold for pid in ids])
        for index, epoch in enumerate(epochs):
            scores[fold][int(epoch)] = balanced_nll(nll[index], ids, gaps, use)
    best = min(scores["fit"], key=lambda epoch: (scores["fit"][epoch], epoch))
    best_index = list(epochs).index(best)
    predicted_error = predicted_joint_error(native_scales[best_index], [1.0, 1.0], reference_std)
    return best, scores, predicted_error


def write_event_scores(path, events, predicted_error, confidence, time_step_minutes):
    """Write per-gap outputs for the user's data; no participant data ship here."""
    fields = ["participant_id", "day_id", "event_index", "start", "end", "gap_minutes",
              "predicted_joint_error", "confidence"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, event in enumerate(events):
            row = {key: event[key] for key in fields[:5]}
            row.update(gap_minutes=int(event["gap_slots"]) * time_step_minutes,
                       predicted_joint_error=float(predicted_error[i]), confidence=float(confidence[i]))
            writer.writerow(row)


def core_context(checkpoint_path, data_dir, device):
    from wearformer.checkpoint import load_model
    from wearformer.data import build_profiles, load_pool, validate_pool_resolution

    backbone, checkpoint = load_model(checkpoint_path, device=device)
    backbone.eval()
    preprocessing = checkpoint["preprocessing"]
    training_pool = load_pool(data_dir, "train")
    validate_pool_resolution(training_pool, checkpoint["config"])
    descriptor_reference = (np.asarray(preprocessing["descriptor_mean"]),
                            np.asarray(preprocessing["descriptor_std"]))
    profiles, _ = build_profiles(training_pool, descriptor_reference=descriptor_reference)
    return backbone, checkpoint, training_pool, profiles


def run_train(args):
    from wearformer.data import TrainingDataset

    backbone, checkpoint, pool, profiles = core_context(args.checkpoint, args.data_dir, args.device)
    config = copy.deepcopy(checkpoint["config"])
    config["training"]["scatter_rate"] = 0.0
    seed = args.seed if args.seed is not None else int(config["training"].get("seed", 20260907))
    config["training"]["seed"] = seed
    prep = checkpoint["preprocessing"]
    dataset = TrainingDataset(pool, np.asarray(prep["mean"]), np.asarray(prep["std"]), profiles, config)
    return train_head(backbone, dataset, args.out_dir, config["model"]["d_model"],
                      epochs=args.epochs, batch_size=args.batch_size, seed=seed,
                      device=args.device, workers=args.workers,
                      backbone_checkpoint_sha=file_sha(args.checkpoint))


def run_evaluation(args, selecting):
    from wearformer.data import FixedEventDataset, load_pool, validate_pool_resolution, assert_disjoint_dates

    backbone, checkpoint, training_pool, profiles = core_context(args.checkpoint, args.data_dir, args.device)
    directory = Path(args.head_dir)
    prep = checkpoint["preprocessing"]
    minutes = int(checkpoint["config"]["time_step_minutes"])
    split = "validation" if selecting else args.split
    mean, std = np.asarray(prep["mean"], np.float32), np.asarray(prep["std"], np.float64)
    reference_std = np.asarray(prep["reference_std"], np.float64)
    pool = load_pool(args.data_dir, split)
    validate_pool_resolution(pool, checkpoint["config"])
    assert_disjoint_dates(training_pool, pool)
    if split == "internal_test":
        assert_disjoint_dates(load_pool(args.data_dir, "validation"), pool)
    manifest = args.manifest or str(Path(args.data_dir) / f"{split}_v2_starts.npz")
    dataset = FixedEventDataset(pool, manifest, mean, std, profiles, minutes, k=args.starts_per_day)
    expected_gaps = set(checkpoint["config"]["evaluation"]["gap_lengths"])
    if {event["gap_slots"] for event in dataset.events} != expected_gaps:
        raise ValueError("confidence evaluation durations differ from the backbone evaluation configuration")
    if selecting:
        selected_epoch = None
    else:
        selection = json.loads((directory / "selection.json").read_text())
        if selection["backbone_checkpoint_sha256"] != file_sha(args.checkpoint):
            raise ValueError("selection used a different backbone checkpoint")
        selected_epoch = int(selection["selected_epoch"])
    heads = load_heads(directory, backbone, args.device, selected_epoch)
    nll, scales = evaluate_heads(backbone, heads, dataset, mean, std,
                                device=args.device, batch_size=args.batch_size, workers=args.workers)
    if selecting:
        ids = [str(event["participant_id"]) for event in dataset.events]
        folds = load_fold_file(args.folds, ids) if args.folds else participant_folds(ids, args.fold_salt)
        epoch, scores, predicted = select_head([x[0] for x in heads], nll, scales,
                                               dataset.events, folds, reference_std)
        reference = np.sort(predicted)
        np.save(directory / "validation_reference.npy", reference, allow_pickle=False)
        write_json(directory / "validation_folds.json", folds)
        selection = {
            "schema": "wearformer-error-head-selection/1", "selected_epoch": epoch,
            "nll_by_epoch": scores,
            "rule": "minimum fit-fold participant-balanced NLL; ties use earlier epoch",
            "fold_rule": "explicit CSV" if args.folds else "SHA256 ordering; first half fit",
            "fold_salt": None if args.folds else args.fold_salt,
            "backbone_checkpoint_sha256": file_sha(args.checkpoint),
            "reference_sha256": file_sha(directory / "validation_reference.npy"),
            "reference_gap_count": len(reference), "reference_std": reference_std.tolist(),
            "validation_manifest_sha256": file_sha(manifest),
        }
        write_json(directory / "selection.json", selection)
        output = directory / "validation_confidence.csv"
    else:
        reference_path = directory / "validation_reference.npy"
        if file_sha(reference_path) != selection["reference_sha256"]:
            raise ValueError("validation confidence reference changed")
        reference = np.load(reference_path, allow_pickle=False)
        if not np.array_equal(reference_std, np.asarray(selection["reference_std"])):
            raise ValueError("Joint MAE reference standard deviations changed")
        predicted = predicted_joint_error(scales[0], [1.0, 1.0], reference_std)
        output = Path(args.out_dir) / f"{split}_confidence.csv"
    ranks = confidence_rank(predicted, reference)
    write_event_scores(output, dataset.events, predicted, ranks, minutes)
    return {"selected_epoch": selection["selected_epoch"], "events": len(dataset.events), "output": str(output)}
