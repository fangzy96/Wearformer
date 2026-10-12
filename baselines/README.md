# Comparison methods

These entry points fit and evaluate comparison methods on user-prepared daily
arrays and frozen gap manifests. They include no data, fitted models or results.
PyPOTS methods require `pypots==1.5`; MICE also uses scikit-learn. The standalone
training-mean fit/impute command requires only NumPy.

## Train, select and evaluate

```bash
python -m baselines.train --model SAITS \
  --train-dir prepared --protocol-dir prepared --manifest gaps \
  --time-step-minutes 10 --output-dir runs/saits --epochs 60 --device cpu
```

The prepared directory contains `{split}_values.npy`,
`{split}_original_observed.npy`, and `{split}_base_windows.csv` for `train`,
`validation`, and `internal_test`. Arrays have shape `[days, time points, 2]`,
with HR first and Steps second. Metadata contains `array_row` and
`participant_id`. Manifest files are `{split}_v2_starts.npz`; each numeric key
is a gap length in sampling positions and its integer array has shape
`[days, starts]`. A start of -1 denotes an unavailable event.

Selection uses every evaluated duration: eight at one-minute resolution
(1, 10, 30, 60, 120, 240, 480 and 720 minutes) and seven at ten-minute
resolution (10, 30, 60, 120, 240, 480 and 720 minutes). A missing duration is an
error. If the one-minute 10-minute-gap events are in a separate manifest,
add `--extra-manifest path/to/additional_gaps`.

Each neural model is fitted one epoch at a time on the same model and optimizer.
After each epoch, all validation durations are scored with an independent,
restored random stream. The lowest unrounded Overall Joint MAE selects the
checkpoint; a tie selects the earliest epoch. Test values are read after
selection. Raw predictions are scored without clipping negative Steps.

Inputs use observed-training means and standard deviations accumulated in
float64, cast to float32 for model input. By default, Joint MAE uses the model
runner's training standard deviations. Supply both `--hr-norm` and
`--steps-norm` when a fixed reference dataset defines the scoring scales, such
as when training on a subset of days. These scoring scales are independent of
the model's input scaler.

`--model` accepts SAITS, ImputeFormer, Informer, ETSformer, Crossformer, PatchTST,
FEDformer, Autoformer, BRITS, SegRNN, TCN, SCINet, FiLM, TimeMixer, TimeMixerPP,
DLinear, FreTS, MICE, LOCF, Lerp and Mean. `TimeMixerPP` is TimeMixer++ and `Lerp`
is linear interpolation. **Mean is always the fixed training mean**; this entry
point does not invoke PyPOTS' evaluation-window mean.

Constructor hyperparameters retain the comparison wrapper's defaults.
`--model-config` accepts a JSON object of explicit constructor overrides.
`--batch` controls training batches, `--eval-batch` controls evaluation batches,
and `--k` defaults to four starts per day and duration. Reduce evaluation batches
if they exceed device memory. Stochastic inference may depend on batch size;
keep evaluation batches fixed when checking reproducibility.

To evaluate a saved selected checkpoint without fitting:

```bash
python -m baselines.train --model SAITS \
  --train-dir prepared --protocol-dir prepared --manifest gaps \
  --time-step-minutes 10 --load-run runs/saits --output-dir runs/saits_read
```

Keep `--k` and the evaluation batch size consistent with the original run.
The saved model initialization seed is reused before reconstruction, including
FEDformer's random Fourier-mode indices. A new output directory is required.
This CLI supports fresh training and evaluation of a selected checkpoint;
resuming an interrupted training job is not implemented.

## Fixed training mean

```bash
python -m baselines.train_mean fit \
  --train-values prepared/train_values.npy \
  --train-observed prepared/train_original_observed.npy --out means.json
python -m baselines.train_mean impute --means means.json \
  --values inputs.npy --hide missing_mask.npy --out imputed.npy
```

The fit uses finite, observed training cells only. Every hidden cell receives
the same fixed mean for its channel, independently of the visible evaluation
values and batch composition. The impute command preserves other values;
`--observed` can identify cells that should remain unavailable when they are
not selected by `--hide`.

These are reorganized public entry points. They preserve the documented model
defaults and evaluation rules, but do not recreate historical datasets, stored
gap manifests, checkpoints, or all platform-dependent numerical behavior.
