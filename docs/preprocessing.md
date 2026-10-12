# Preparing model inputs

Start with records downloaded from the [original data repositories](data.md).
Keep data and generated files outside Git. The commands below take a local
participant roster, so cohort eligibility can be applied before preprocessing.
They do not infer clinical status from filenames or download participant lists.

## Raw records

Create a local CSV with columns `participant_id,hr_file,steps_file`. File paths
are absolute or relative to that CSV. Use one row per participant. For Fitbit,
the raw HR file has `datetime,heartrate` columns and the Steps file has
`datetime,steps`. For Apple, use the source export's `device,datetime,heartrate`
and `device,start_datetime,end_datetime,steps` columns.

```bash
python -m preprocessing.raw --device fitbit \
  --records data/fitbit_roster.csv --output-dir data/aligned/fitbit

python -m preprocessing.raw --device apple \
  --records data/apple_roster.csv --output-dir data/aligned/apple
```

Fitbit HR uses the median within each minute. Repeated Steps minute totals
use their maximum. Steps without HR are discarded; an HR-observed minute with
no Steps entry is assigned zero. Apple HR uses the median within each ten-minute
bin. Interval Steps are distributed in proportion to the overlap with each bin;
the exported preprocessing procedure excludes intervals longer than 60 minutes.
Empty Steps bins with observed HR are assigned zero. Missing HR makes both
channels missing. Partially observed bins are not rescaled.

The output is an aligned CSV per participant, plus `records.csv` containing
`participant_id,path`. Alternatively, supply your own aligned records with
`datetime,heartrate,steps` columns and a matching `records.csv`.

## Daily arrays and splits

```bash
python -m preprocessing.days --records data/aligned/fitbit/records.csv \
  --resolution 1 --seed 20260907 --output-dir data/processed/large_fitbit

python -m preprocessing.days --records data/aligned/apple/records.csv \
  --resolution 10 --seed 20260907 --output-dir data/processed/apple_watch
```

Days qualify with at least 80% HR availability. Participants need 25 qualifying
days. A seeded hash orders each participant's qualifying days: ten go to test,
five to validation and all remaining dates to training. Fitbit inputs discard
incomplete boundary days; the Apple pool builder pads boundary days to midnight
and applies the coverage threshold, matching the supplied preprocessing code.

For ten-minute Fitbit, aggregate the **already split** one-minute arrays to retain
the same participants and dates:

```bash
python -m preprocessing.aggregate_fitbit --data-dir data/processed/large_fitbit \
  --output-dir data/processed/fitbit_10min
```

Each ten-minute bin uses the median observed HR and sum of observed Steps. One
observed minute is sufficient. Run the one-minute commands separately for Small
Fitbit using its roster; do not create its split from the Large Fitbit pool.

## Fixed artificial gaps

```bash
python -m preprocessing.gaps --protocol-dir data/processed/large_fitbit \
  --resolution 1 --seed 20260906

python -m preprocessing.gaps --protocol-dir data/processed/apple_watch \
  --resolution 10 --seed 20260906

python -m preprocessing.gaps --protocol-dir data/processed/fitbit_10min \
  --resolution 10 --seed 20260917
```

Generate gaps once and share them across methods and training-day conditions.
The builder samples four feasible starts per date and duration. It checks
observed HR within the gap and on both sides. It refuses to replace existing
manifests. A new split or seed defines a new reproducible evaluation; use the
original external manifests when exact historical event assignments are needed.

## Prepared-file schema

For `split` in `train`, `validation`, `internal_test`:

| File | Content |
| --- | --- |
| `{split}_values.npy` | float32 `[days, positions, 2]`, channels HR and Steps; NaN for missing |
| `{split}_original_observed.npy` | matching 0/1 array, including observed zero Steps |
| `{split}_base_windows.csv` | one row per array row; `participant_id`, `array_row`, `window_start` and window identifiers |
| `{split}_v2_starts.npz` | validation/test only; key is duration in **positions**, value is int32 `[days, draws]`; `-1` means infeasible |

`window_start` is a timezone-naive ISO timestamp, used for daily/weekly encodings
and chronological ordering. One position is one minute or ten minutes according
to the configuration. Array row ordering and metadata row ordering must agree.
Do not mix a manifest from one pool with another pool.
`window_minutes` is informational and remains 1,440 for a full day at either
resolution; use the array shape and configuration for the number of positions.

This repository supplies code, not individual records, cohort rosters, date/gap
manifests, fitted normalization values or trained weights. The commands above
generate these locally. Exact numerical reproduction of a historical run also
requires its original inputs, assignments and software environment.
