# Wearformer

Code for **Wearformer enables accurate wearable data imputation across devices and gap lengths**.

Wearformer jointly imputes heart rate and step counts using local and global
temporal patterns, time of day and personal history. It supports one-minute
Fitbit records and ten-minute Fitbit or Apple Watch records.

## What is included

- Raw-record preprocessing, daily train/validation/test pools and fixed test gaps.
- Wearformer model, training, validation checkpoint selection and evaluation.
- Adapters for the 21 comparison methods, including training-mean imputation.
- Transfer evaluation and chronological training-day experiments.
- A frozen-backbone error head for ranking imputation confidence.
- A bridge for filling missing values before downstream analysis.

The repository contains **code and documentation only**. Study data, participant
rosters, date/gap assignments, trained weights and aggregate results are not
included. The downstream bridge does not include the published COVID-19
detection pipelines; see its [scope and links](downstream/README.md).

## Installation

Use Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

For comparison methods, also install:

```bash
python -m pip install -r requirements-baselines.txt
```

The reference stack uses PyTorch 2.8 and PyPOTS 1.5. Install a PyTorch build
appropriate for your CUDA environment when running on a GPU. CPU execution is
supported for checking the workflow.

## Quick check with synthetic data

This small example trains for one epoch with a reduced model. It does not use
study data or reproduce the paper's results.

```bash
python -m examples.make_synthetic --output-dir outputs/smoke/data
python -m wearformer.train \
  --config outputs/smoke/data/smoke_config.json \
  --protocol-dir outputs/smoke/data \
  --output-dir outputs/smoke/model --device cpu
python -m wearformer.evaluate \
  --checkpoint outputs/smoke/model/best.pt \
  --protocol-dir outputs/smoke/data --k 1 \
  --output-dir outputs/smoke/evaluation --device cpu
python -m unittest discover -s tests -v
```

Choose a new output directory when repeating training. Generated data and
checkpoints stay under ignored directories.

## Train and evaluate

First follow the [data preparation instructions](docs/preprocessing.md). Then:

```bash
python -m wearformer.train \
  --config configs/fitbit_large_1min.json \
  --protocol-dir data/processed/large_fitbit \
  --output-dir runs/large_fitbit --device cuda

python -m wearformer.evaluate \
  --checkpoint runs/large_fitbit/best.pt \
  --protocol-dir data/processed/large_fitbit \
  --output-dir outputs/large_fitbit --device cuda
```

Input normalization and personal summaries are fitted on training dates only.
Checkpoint selection uses validation Joint MAE across all evaluated durations,
with ties resolved by the earlier epoch. Test targets do not select the model.

| Setting | Configuration |
| --- | --- |
| Large Fitbit, one minute | `configs/fitbit_large_1min.json` |
| Small Fitbit, one minute | `configs/fitbit_small_1min.json` |
| Apple Watch, ten minutes | `configs/apple_10min.json` |
| Large Fitbit, ten minutes | `configs/fitbit_large_10min.json` |

See the [reproduction protocol](docs/reproduction.md) for the model, masking,
evaluation and scoring rules, and [experiment commands](docs/experiments.md)
for comparisons, transfer and training-day subsets.

## Repository layout

```text
wearformer/     Model, training, checkpoint loading and evaluation
preprocessing/  Raw records, daily arrays, temporal aggregation and fixed gaps
configs/        Four benchmark configurations
baselines/      Comparison-method adapters and fixed training-mean imputation
experiments/    Chronological training-day subsets
confidence/     Error-head training, selection and confidence scoring
downstream/     Missing-value completion before external downstream pipelines
analysis/       Relative MAE reduction per added training day
examples/       Synthetic installation check
tests/          Synthetic numerical and workflow checks
docs/           Data and reproduction instructions
```

[Data sources](docs/data.md) · [Confidence](confidence/README.md) ·
[Training-day analysis](analysis/README.md)

A license for the original Wearformer code has not yet been selected.
