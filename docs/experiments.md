# Running additional experiments

Run commands from the repository root after [preparing inputs](preprocessing.md).
Generated predictions and scores remain local; none are distributed here.

## Comparison methods

The adapters cover SAITS, ImputeFormer, Informer, ETSformer, Crossformer,
PatchTST, FEDformer, Autoformer, BRITS, SegRNN, TCN, SCINet, FiLM, TimeMixer,
TimeMixer++, DLinear, FreTS, MICE, LOCF, linear interpolation and training-mean
imputation. Use `TimeMixerPP` for TimeMixer++ and `Lerp` for interpolation.

```bash
python -m baselines.train --model SAITS \
  --train-dir data/processed/apple_watch \
  --protocol-dir data/processed/apple_watch \
  --manifest data/processed/apple_watch --time-step-minutes 10 \
  --output-dir runs/apple_saits --device cuda
```

Replace `SAITS` with another model name. Optional constructor overrides are
provided through `--model-config settings.json`. The wrapper retains each
implementation's training loss. Neural validation runs after each epoch, saves
and restores Python/NumPy/PyTorch random-number states, and selects the earliest
minimum Joint MAE across all eight or seven durations. Raw comparator predictions
are evaluated without Steps clipping. Mean imputation uses one mean per channel
from the training observations, independent of evaluation batches.

All methods must use the same daily splits, gap manifests and Joint reference
scales. The default scales come from the supplied full training pool; use
`--hr-norm` and `--steps-norm` to provide fixed full-pool scales when training on
a subset. These flags affect scoring, not model input normalization.

## Transfer without retraining

Use a source checkpoint and the target dataset's training dates for personal
summaries. Source input-normalization and personal-feature reference statistics
stay fixed. Joint scoring uses the **target** full training pool's scales.

```bash
python -m preprocessing.statistics --train-dir data/processed/small_fitbit \
  --output outputs/small_fitbit_reference.json

python -m wearformer.evaluate \
  --checkpoint runs/large_fitbit/best.pt \
  --protocol-dir data/processed/small_fitbit \
  --reference-scales outputs/small_fitbit_reference.json \
  --output-dir outputs/transfer_to_small --device cuda
```

For Apple Watch transfer, use a ten-minute Fitbit checkpoint and Apple Watch
data/reference scales. Source and target must have the same sampling interval.
There is no optimizer or model update in the evaluation command.

## How many training days?

First create the fixed scoring reference from the original full training pool:

```bash
python -m preprocessing.statistics --train-dir data/processed/apple_watch \
  --output outputs/apple_reference.json

python -m experiments.training_days \
  --train-dir data/processed/apple_watch \
  --protocol-dir data/processed/apple_watch \
  --output-dir data/day_subsets/apple_watch
```

The subset builder keeps a fixed cohort with at least 40 qualifying training
dates. It selects each person's first `N` training dates in chronological order,
for `1,2,3,5,7,10,14,20,30,40,All`. Held-out dates stay in the original protocol
directory. `All` uses all available training dates for that same fixed cohort.

Train a separate model for each generated subset, using `--train-dir` for that
subset and `--protocol-dir` for the unchanged validation/test pools. Always pass
`--reference-scales outputs/apple_reference.json` to both training and evaluation.
Input normalization and personal summaries are refitted on each training subset;
reference scales remain fixed. Evaluation must receive the same subset via
`--train-dir`, so it does not recompute summaries from additional days.

Use [analysis/training_days.py](../analysis/README.md) to summarize the relative
MAE reduction per added day from locally computed errors. A threshold and the
number of later intervals required for a sustained plateau are explicit analysis
choices; they are not hardcoded into model training.

## Confidence and downstream completion

See [confidence training, selection and scoring](../confidence/README.md) and
[the downstream completion bridge](../downstream/README.md). The bridge fills
missing values for an external analysis. It does not reproduce a disease
detector or its study-specific case selection.
