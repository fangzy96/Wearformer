# Applying imputation before a downstream analysis

This directory provides a runnable bridge from a trained Wearformer checkpoint
to completed daily HR and Steps arrays. It **does not include or run the
published COVID-19 detectors**, their clinical case selection, or their scoring
pipelines. It does not reproduce the paper's downstream results by itself.

From the repository root:

```bash
python -m downstream \
  --checkpoint runs/model/best.pt \
  --data-dir data/daily \
  --split natural \
  --out-dir runs/completed
```

The model checkpoint and daily pools use the format described in
[the data guide](../docs/data.md). A `natural` pool follows the same array and
metadata format as other splits: `natural_values.npy`,
`natural_original_observed.npy` and `natural_base_windows.csv`. It can contain
days outside the training eligibility criteria. The `train` pool provides each
participant's personal history; use the same history used to fit the checkpoint.
Participants absent from that history use the mean standardized personal
features (a zero feature vector). These files are supplied by the user.

Every originally observed value is retained, including zero Steps. Missing cells
are filled only on days with at least one observed HR value; other missing cells
remain NaN. Imputed Steps are constrained to be nonnegative. Predictions remain
continuous values. No rounding, clinical labels, symptom dates, cohort lists or
algorithm-specific eligibility rules are applied here.

Outputs are:

- `completed.npz`: `values`, original `observed` flags, and newly `imputed` flags,
  each with shape `[days, time, 2]` and channels `[HR, Steps]`.
- `days.csv`: metadata rows in the same order as the arrays.
- `imputation.json`: completion settings and counts for the supplied data.

No data, results or pretrained weights are distributed in this repository.
Generated outputs contain the user's participant information and should remain
outside the source repository.

## Published detectors

The manuscript used the authors' published algorithms with their study-specific
case definitions and parameters:

- Mishra et al., *Pre-symptomatic detection of COVID-19 from smartwatch data*,
  Nature Biomedical Engineering (2020):
  [WearableDetection](https://github.com/mwgrassgreen/WearableDetection) and
  [AnomalyDetect](https://github.com/gireeshkbogu/AnomalyDetect).
- Alavi et al., *Real-time alerting system for COVID-19 and other stress events
  using wearable data*, Nature Medicine (2022):
  [wearable-infection](https://github.com/StanfordBioinformatics/wearable-infection),
  including NightSignal.

Obtain and run those implementations separately under their own terms. A full
reproduction also requires matching their input preparation, clinical reference
dates and detection-window/lead-time definitions. This bridge deliberately
does not infer those rules from the imputed arrays or claim that the same rules
apply to a new study.
