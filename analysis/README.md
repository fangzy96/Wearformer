# Training-day analysis

This standard-library script calculates daily relative MAE reduction and applies
a diminishing-gain criterion to a user-supplied aggregate curve. It requires
Python 3.10 or later and does not train or run Wearformer. No experiment results
or participant data are included.

From the repository root, supply your own CSV:

```bash
python analysis/training_days.py /path/to/training_days.csv
python analysis/training_days.py /path/to/training_days.csv --intervals-out intervals.csv
python -m unittest discover -s tests -p 'test_training_days.py' -v
```

## Input format

| Column | Meaning |
| --- | --- |
| `setting` | Name identifying each dataset or experimental condition |
| `train_days` | Positive integer training budget, or `All` |
| `joint_mae` | Aggregate Joint MAE measured at this budget |
| `total_qualifying_training_days` | Total qualifying training days in the fixed training cohort |
| `training_participants` | Number of participants in that training cohort |

For example, the following **synthetic values** illustrate the schema and can be
saved to a temporary CSV to try the command:

```csv
setting,train_days,joint_mae,total_qualifying_training_days,training_participants
synthetic,1,1.0,311,17
synthetic,2,0.9,311,17
synthetic,5,0.899,311,17
synthetic,All,0.898,311,17
```

Within each setting, budgets must increase strictly and end with All; cohort
counts must stay fixed. MAEs must be positive and finite. The All mean must
exceed the last numeric budget. Keep the error definition, evaluation data and
aggregation procedure consistent across budgets.

## Calculation

For successive budgets a and b with errors E_a and E_b, the daily relative MAE
reduction is:

```text
100 * (E_a - E_b) / E_a / (b - a)
```

Its unit is **percent per added training day**. The relative reduction uses the
previous budget's MAE as its denominator.

For the final interval, b is the mean number of qualifying training days in the
fixed cohort: `total_qualifying_training_days / training_participants`. The
script computes this directly from the integer counts, without rounding the
mean before calculating the interval. All remains an endpoint in the analysis.

The default selected budget is the earliest evaluated day count for which its
outgoing interval and **every subsequent interval** have a daily relative
reduction of at most 0.20%. At least two intervals, including the outgoing
interval, must remain. If none qualifies, the script reports `null`. A low rate
in one interval cannot qualify a budget if a later rate exceeds the threshold.

Rates remain signed: negative values mean MAE increased. The script does not
smooth errors or fit a monotonic curve. Interpret negative rates as deterioration
when applying this descriptive criterion to your data.

The threshold and minimum number of remaining intervals can be changed:

```bash
python analysis/training_days.py /path/to/training_days.csv --threshold 0.10 --min-later-intervals 2
```

The optional interval CSV preserves the start and end budgets, added days,
endpoint errors, total relative reduction and daily relative reduction. Console
output is JSON. Parent directories for output files must already exist.
