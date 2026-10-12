# Training-day subsets

Build nested chronological training pools from prepared qualifying training
days:

```bash
python -m experiments.training_days --train-dir prepared \
  --protocol-dir prepared --output-dir day_pools \
  --days 1,2,3,5,7,10,14,20,30,40,All --minimum-days 40
```

The input uses `train_values.npy`, `train_original_observed.npy`, and
`train_base_windows.csv`. Arrays have shape `[days, time points, 2]`; metadata
must contain `array_row`, `participant_id`, and `window_start`, an ISO timestamp
at midnight. All other metadata fields are preserved.

Participants with at least `--minimum-days` qualifying training dates form one
fixed cohort. Each numeric budget uses that person's first N training dates
in time order. All retains that same cohort's complete training histories.
Missing calendar dates and held-out dates do not count as training days.
The selected dates therefore need not be consecutive calendar dates.

Outputs are `days_01`, `days_02`, ... and `days_all`, each containing only a
training pool. Values and observation masks are copied without transformation;
metadata row indices are renumbered. Existing output pools are not overwritten.
When held-out metadata is available under `--protocol-dir`, overlap between
training dates and held-out dates is rejected.

Use each generated directory as `--train-dir` while retaining the original
`--protocol-dir` and gap manifests for validation and test. Keep evaluation
participants, sampled inputs per participant, and reference Joint scoring
scales fixed when comparing training budgets. The aggregate-curve calculation
is documented separately in [analysis/README.md](../analysis/README.md).
