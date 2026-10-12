# Data preparation

Wearformer uses heart-rate measurements and step counts from the public releases of two wearable studies.

| Release | Wearformer cohorts | Source study | Download |
| --- | --- | --- | --- |
| COVID-19 Phase 2 | Large Fitbit and Apple Watch | Alavi et al., *Real-time alerting system for COVID-19 and other stress events using wearable data*, Nature Medicine (2022) | [Original data archive](https://storage.googleapis.com/gbsc-gcp-project-ipop_public/COVID-19-Phase2/COVID-19-Phase2-Wearables.zip) |
| COVID-19 | Small Fitbit | Mishra et al., *Pre-symptomatic detection of COVID-19 from smartwatch data*, Nature Biomedical Engineering (2020) | [Original data archive](https://storage.googleapis.com/gbsc-gcp-project-ipop_public/COVID-19/COVID-19-Wearables.zip) |

Consult the source studies for data-use conditions and the original collection procedures:

- [Alavi et al.](https://doi.org/10.1038/s41591-021-01593-2)
- [Mishra et al.](https://doi.org/10.1038/s41551-020-00640-6)

## Local data layout

Download the archives from their original repositories and keep them outside version control. The following layout separates source records from derived inputs:

```text
data/
  raw/
    phase1/
    phase2/
  processed/
    large_fitbit/
    small_fitbit/
    apple_watch/
    fitbit_10min/
```

`data/` is ignored by Git. Raw wearable records, individual date assignments and fitted model files are not included in this repository. Use locally generated outputs as inputs to the analysis scripts.

## Expected daily inputs

Each daily input contains two channels in the order **HR, Steps**:

| Resolution | Signal shape | Units |
| --- | --- | --- |
| One minute | `(1440, 2)` | HR: beats/min; Steps: counts/min |
| Ten minutes | `(144, 2)` | HR: beats/min; Steps: counts/ten-minute interval |

Observation masks distinguish naturally missing values from measured zeros. After normalization, hidden inputs are zero-filled and accompanied by visibility masks; zero alone does not indicate missingness.

Fit input normalization and personal summaries using training dates only. Preserve each participant's validation and test dates across methods and training-day conditions. See the [reproduction protocol](reproduction.md) for eligibility, temporal aggregation, normalization and evaluation rules.

See [preprocessing commands and file schemas](preprocessing.md) for the raw-file
roster, daily pool builder, ten-minute aggregation and frozen evaluation gaps.
