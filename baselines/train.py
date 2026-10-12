"""Train and select a comparator on all validation durations, then evaluate it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluation import load_pool, load_starts, preserve_rng, score_gaps, select_best_epoch
from .models import MODEL_NAMES, NEURAL_MODELS, StandardizedPredictor, construct_model, observed_input_stats
from .train_mean import TrainingMean, fit_training_mean, observed_feature_stats_f64


def expected_gap_positions(time_step_minutes):
    return [1, 10, 30, 60, 120, 240, 480, 720] if time_step_minutes == 1 else [1, 3, 6, 12, 24, 48, 72]


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args):
    """Run one new fit, or evaluate a previously saved run without retraining."""
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new directory")
    pool_values, pool_observed, _ = load_pool(args.train_dir, "train")
    n_steps = pool_values.shape[1]
    if n_steps != 1440 // args.time_step_minutes:
        raise ValueError("Daily input length does not match --time-step-minutes")
    splits = [split.strip() for split in args.split.split(",")]
    if not splits or any(split not in ("validation", "internal_test") for split in splits):
        raise ValueError("Only validation and internal_test evaluation splits are supported")
    saved = json.loads((args.load_run / "run.json").read_text()) if args.load_run else None
    target_epochs = saved.get("training_epochs_target", saved.get("epochs_trained", args.epochs)) if saved else args.epochs
    if saved:
        if args.model != saved["model"] or args.time_step_minutes != saved["time_step_minutes"]:
            raise ValueError("Requested model or time resolution differs from the saved run")
        means = np.asarray(saved["input_mean"], dtype=np.float64)
        stds = np.asarray(saved["input_std"], dtype=np.float64)
        seed, overrides = saved["initialization_seed"], saved["model_overrides"]
        evaluation_seed = saved["evaluation_seed"]
        if args.k != saved["k"]:
            raise ValueError("k differs from the saved evaluation protocol")
    else:
        reference_means, reference_stds, _ = observed_feature_stats_f64(pool_values, pool_observed)
        means, stds = ((reference_means, reference_stds) if args.model == "Mean"
                      else observed_input_stats(pool_values, pool_observed))
        seed = args.seed
        evaluation_seed = args.evaluation_seed
        overrides = json.loads(args.model_config.read_text()) if args.model_config else {}
    scales = [args.hr_norm, args.steps_norm]
    if (scales[0] is None) != (scales[1] is None):
        raise ValueError("Supply both --hr-norm and --steps-norm, or neither")
    if scales[0] is None:
        if saved:
            scales = saved["joint_scales"]
        else:
            scales = reference_stds.tolist()
    if not np.isfinite(scales).all() or np.any(np.asarray(scales) <= 0):
        raise ValueError("Joint scoring scales must be positive and finite")
    lengths = expected_gap_positions(args.time_step_minutes)
    manifests = [args.manifest] + list(args.extra_manifest or [])
    cache = {}

    def evaluate(predictor, split):
        if split not in cache:
            values, observed, participants = load_pool(args.protocol_dir, split)
            if values.shape[1] != n_steps:
                raise ValueError("Training and evaluation inputs have different lengths")
            starts = load_starts(manifests, split, len(values), n_steps, lengths)
            cache[split] = (values, observed, participants, starts)
        with preserve_rng(evaluation_seed):
            return score_gaps(predictor.predict, *cache[split], scales,
                              k=args.k, batch_size=args.eval_batch)

    # Validate the complete validation manifest before constructing or fitting a
    # neural model; missing one-minute L=10 must never silently select on 7 gaps.
    if not saved and args.model in NEURAL_MODELS:
        values, observed, participants = load_pool(args.protocol_dir, "validation")
        starts = load_starts(manifests, "validation", len(values), n_steps, lengths)
        cache["validation"] = (values, observed, participants, starts)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    if args.model == "Mean":
        means_record = saved.get("training_mean") if saved else fit_training_mean(pool_values, pool_observed)
        predictor = TrainingMean(means_record["mean"])
        selected_epoch = None
    else:
        model = construct_model(args.model, n_steps, args.batch, args.device, seed, overrides, epochs=target_epochs)
        predictor = StandardizedPredictor(model, means, stds)
        if saved:
            if args.model == "MICE":
                import joblib
                model = joblib.load(args.load_run / "mice.joblib")
                predictor.model = model
                joblib.dump(model, output / "mice.joblib")
            elif args.model in NEURAL_MODELS:
                model.load(str(args.load_run / "best.pypots"))
                model.save(str(output / "best.pypots"))
            selected_epoch = saved["selected_epoch"]
        elif args.model in NEURAL_MODELS or args.model == "MICE":
            visible = np.asarray(pool_observed, dtype=bool) & np.isfinite(pool_values)
            training = np.where(visible, (pool_values - predictor.mean) / predictor.std, np.nan).astype(np.float32)
            if args.model == "MICE":
                import joblib
                model.fit({"X": training})
                joblib.dump(model, output / "mice.joblib")
                selected_epoch = None
            else:
                for epoch in range(1, args.epochs + 1):
                    model.batch_size = args.batch
                    model.epochs = 1
                    model.fit({"X": training})
                    model.batch_size = args.eval_batch
                    validation = evaluate(predictor, "validation")
                    history.append({"epoch": epoch, "validation_joint_mae": validation["overall"]["joint_mae"],
                                    "by_gap_positions": validation["by_gap_positions"]})
                    selected_epoch = select_best_epoch(history)
                    if selected_epoch == epoch:
                        model.save(str(output / "best.pypots"), overwrite=True)
                    _write_json(output / "history.json", history)
                    print(json.dumps({"epoch": epoch, "validation_joint_mae": history[-1]["validation_joint_mae"]}), flush=True)
                model.load(str(output / "best.pypots"))
        else:
            selected_epoch = None
        if hasattr(model, "batch_size"):
            model.batch_size = args.eval_batch
    record = {
        "model": args.model, "time_step_minutes": args.time_step_minutes,
        "input_mean": means.tolist(), "input_std": stds.tolist(), "joint_scales": list(scales),
        "initialization_seed": seed, "evaluation_seed": evaluation_seed,
        "model_overrides": overrides, "selected_epoch": selected_epoch,
        "selection": "minimum raw validation Joint MAE across all durations; ties select earliest epoch",
        "gap_lengths_positions": lengths, "gap_lengths_minutes": [length * args.time_step_minutes for length in lengths],
        "training_batch_size": args.batch, "evaluation_batch_size": args.eval_batch,
        "k": args.k, "output_policy": "raw", "fit_performed": not bool(saved),
        "epochs_trained": (saved["epochs_trained"] if saved else len(history)),
        "training_epochs_target": target_epochs,
    }
    if args.model == "Mean":
        record["training_mean"] = means_record
    _write_json(output / "run.json", record)
    for split in splits:
        result = evaluate(predictor, split)
        result["selected_epoch"] = selected_epoch
        result["gap_units"] = "sampling positions"
        if split == "validation" and history:
            selected = next(row for row in history if row["epoch"] == selected_epoch)
            if not np.isclose(result["overall"]["joint_mae"], selected["validation_joint_mae"], rtol=0, atol=1e-8):
                raise ValueError("Reloaded checkpoint does not reproduce its selected validation score")
        _write_json(output / f"{split}.json", result)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True,
                        help="Mean is the fixed training mean; TimeMixerPP denotes TimeMixer++")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extra-manifest", type=Path, action="append",
                        help="Additional disjoint gap manifest, such as the one-minute L=10 supplement")
    parser.add_argument("--time-step-minutes", type=int, choices=(1, 10), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=64)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--evaluation-seed", type=int, default=20260921)
    parser.add_argument("--hr-norm", type=float)
    parser.add_argument("--steps-norm", type=float)
    parser.add_argument("--model-config", type=Path, help="JSON object overriding model constructor hyperparameters")
    parser.add_argument("--load-run", type=Path, help="Evaluate a saved run without fitting")
    parser.add_argument("--split", default="validation,internal_test")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch, args.eval_batch, args.k) < 1:
        parser.error("Epochs, batch sizes and k must be positive")
    try:
        run(args)
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
