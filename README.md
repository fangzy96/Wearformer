# Wearformer

Code for **Wearformer enables accurate wearable data imputation across devices and gap lengths**.

Wearformer is a Transformer-based model for jointly imputing heart rate and step counts using local and global temporal patterns, time of day and personal history.

## Status

The final preprocessing, model, training and evaluation implementation is being prepared for release. Currently available: the experiment protocol and a standalone training-day analysis utility. Executable model-training instructions will accompany the implementation.

## Available code

`analysis/training_days.py` calculates the relative MAE reduction per added training day from a user-supplied CSV. It uses Python's standard library and requires no GPU.

```bash
python analysis/training_days.py --help
python -m unittest discover -s tests -v
```

See [analysis/README.md](analysis/README.md) for the input schema and commands.

## Documentation

- [Data sources and daily input format](docs/data.md)
- [Model configuration and reproduction protocol](docs/reproduction.md)

## Layout

```text
analysis/       Analysis utilities
docs/           Data and reproduction instructions
tests/          Checks using synthetic inputs
```

Keep downloaded data under `data/` and generated artifacts under `outputs/`, `runs/` or `checkpoints/`; these directories are ignored by Git.
