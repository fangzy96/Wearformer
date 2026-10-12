# Imputation confidence

The confidence head estimates the error of a **fixed trained Wearformer**. A
two-layer network reads the model's final encoded features and the two visibility
flags, predicts positive Laplace error scales, and is trained using the Laplace
negative log-likelihood on known values hidden in one continuous gap. Scattered
masking is disabled for head training. Defaults are 128 hidden units, 20 epochs,
Adam with learning rate 0.001 and zero weight decay, cosine learning-rate decay,
and gradient clipping at 1. The positive scale is `softplus(output) + 0.001`.

The backbone stays in evaluation mode. Its forward passes use `torch.no_grad()`;
its `requires_grad` flags are left unchanged, preserving the attention execution
path. Only head parameters enter the optimizer. The training routine verifies
that the backbone has no gradients and its parameters and buffers have not
changed after each epoch.

Run the following commands from the repository root after training Wearformer:

```bash
python -m confidence train \
  --checkpoint runs/model/best.pt --data-dir data/daily \
  --out-dir runs/confidence --device cpu

python -m confidence select \
  --checkpoint runs/model/best.pt --data-dir data/daily \
  --head-dir runs/confidence --device cpu

python -m confidence score \
  --checkpoint runs/model/best.pt --data-dir data/daily \
  --head-dir runs/confidence --split internal_test \
  --out-dir runs/confidence/test --device cpu
```

Use `--device cuda` on a compatible GPU. The native daily arrays, training history
and fixed-gap manifests use [the data guide's format](../docs/data.md). By
default, selection reads `validation_v2_starts.npz` and scoring reads
`internal_test_v2_starts.npz`, with four starts per day and duration; `--manifest`
and `--starts-per-day` allow explicit paths and counts. A head checkpoint is
bound to the exact backbone state on which it was trained.

## Selecting the head

Every head epoch is evaluated on the same validation gaps. Selection uses only
the `fit` half of validation participants. NLL is averaged over target cells
within an event and channel, then events within each participant, then
participants, and finally equally across channels and gap durations. The
minimum selects the epoch; ties choose the earlier epoch. The remaining `cal`
participants do not select the epoch.

By default, participants are ordered by a SHA256 hash of a configurable salt and
their participant label. The first half form the `fit` group. The public default
salt is `wearformer-confidence-v1`; it generates a new assignment and is **not
a copy of the study's private assignment**. Use `--fold-salt` or a supplied
`--folds folds.csv` file to control this. A fold file has the columns
`participant_id,fold`, unique participant labels, and `fit` or `cal` values.
Exact historical selection requires the original fold assignments, input data,
manifests and backbone checkpoint; these are not distributed here.

## From predicted error to confidence

For each gap, the head's scale is averaged over **all positions in the hidden
interval**, converted to native units with the backbone's channel standard
deviations, divided by the evaluation reference standard deviations, and
averaged across HR and Steps. This gives the predicted Joint error.

The reference is all validation gaps, pooling durations and both participant
groups. Confidence is:

```text
1 - number of validation gaps with predicted error <= this gap's predicted error
    / total number of validation gaps
```

A score of 0.8 means lower predicted error than approximately 80% of validation
gaps. It is a relative ranking, not an 80% probability that the values are
correct. Test targets are not used to build the validation reference. This
implementation produces the manuscript's confidence score; it does not add
conformal prediction intervals or claim probability calibration.

`select` writes the chosen epoch, fold assignments, sorted validation reference
and validation event scores. `score` uses that fixed reference to write a
per-gap CSV containing participant/date keys, predicted Joint error and
confidence. These are user-generated outputs, not data included in the code
release. No cohort results, participant metadata or trained weights ship here.
