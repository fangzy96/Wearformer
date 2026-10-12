# Reproduction protocol

This document describes the protocol used in the Wearformer manuscript. The complete training and evaluation code is being prepared for release; executable instructions will accompany that code.

## Data and evaluation settings

Large Fitbit and Apple Watch use data from Alavi et al.; Small Fitbit uses data from Mishra et al. The ten-minute Fitbit setting aggregates the Large Fitbit records and retains the same participants and dates.

| Setting | Sampling interval | Positions per day | Evaluated gap lengths (minutes) |
| --- | ---: | ---: | --- |
| Large Fitbit | 1 min | 1,440 | 1, 10, 30, 60, 120, 240, 480, 720 |
| Small Fitbit | 1 min | 1,440 | 1, 10, 30, 60, 120, 240, 480, 720 |
| Apple Watch | 10 min | 144 | 10, 30, 60, 120, 240, 480, 720 |
| Ten-minute Fitbit | 10 min | 144 | 10, 30, 60, 120, 240, 480, 720 |

Each input contains one calendar day of heart rate (HR) and step counts (Steps), with a separate observation mask for each channel. HR is measured in beats per minute; Steps is measured in counts per sampling interval.

- Qualifying days have at least 80% data availability, using HR observations as the indicator. Incomplete first and last days are excluded.
- Participants have at least 25 qualifying days. Their qualifying dates are randomly ordered using a fixed seed: ten dates are assigned to test, five to validation, and all remaining dates to training. A date belongs to only one split.
- Under the Fitbit recording convention, missing Steps entries at HR-observed minutes are assigned zero and treated as observed. Steps recorded without HR are discarded.
- At ten-minute resolution, any valid HR observation makes a bin observed. HR is the median of available samples. Fitbit Steps sums the counts at HR-observed minutes; Apple Watch interval counts are allocated in proportion to their overlap with each bin. Partially observed bins are not rescaled.

## Model and training configuration

| Parameter | Value |
| --- | --- |
| Encoder layers | 2 |
| Representation / feed-forward width | 256 / 256 |
| Attention heads / key and value width | 4 / 64 |
| Attention mask | Positions may attend to themselves; no causal mask |
| Additional inputs | Daily and weekly time encodings, four observation-distance features, eight personal summary features |
| Local convolution | Depthwise temporal convolution with circular padding, followed by a pointwise projection |
| Convolution width | 91 positions at one-minute resolution; 9 positions at ten-minute resolution |
| Convolution initialization | Pointwise projection weights and biases initialized to zero |
| Dropout | 0 |
| Optimizer | Adam |
| Learning rate | 0.001, cosine decay to zero over 60 epochs |
| Weight decay / gradient norm clipping | 0 / 1 |
| Batch size | 64 |
| Epochs | 60 |
| Training inputs per participant per epoch | 52 sampled daily inputs, with repeats allowed |
| Training seed | 20260907 |
| Training objective | MAE on artificially hidden observed values |

Training hides one contiguous interval in both channels and independently masks 15% of the remaining observed values. Each input samples one of seven length ranges with equal probability, then samples a length within the range on a log scale. The range boundaries are `1, 30, 60, 120, 240, 480, 720, 840` positions at one-minute resolution and `1, 3, 6, 12, 24, 48, 72, 84` positions at ten-minute resolution. Sampling falls back to feasible shorter lengths when observation availability requires it.

A gap of `L` positions requires at least `max(1, floor(0.5 * L))` observed HR positions. Each side requires 60 minutes of context containing at least 30 observed HR minutes, or six positions containing at least three observations at ten-minute resolution. Naturally missing values never become reconstruction targets. Loss is averaged over targets within each channel, equally across channels with targets, then across daily inputs.

## Normalization and personal features

Compute each channel's mean and population standard deviation (`ddof=0`) from observed values on **training dates only**, separately for each setting. Use float64 accumulation for these statistics. Validation and test dates do not contribute. Artificially hidden and naturally missing inputs are set to zero after standardization, with their input masks set to zero.

Personal summaries also use training dates only. Their feature means and standard deviations are computed across training participants; undefined standardized features are set to zero. Recompute distance features from the visible input masks after artificial masking. Distances are measured in sampling positions, transformed as `log(1 + d) / log(1 + T)`, and set to `d = T` when no visible observation exists on a side.

The eight personal features are HR mean and standard deviation, HR means during sleep-like and other observed periods, mean Steps including zeros, the fraction of observed positions with nonzero Steps, the fraction of HR-observed positions classified as sleep-like, and a habitual waking-time proxy. The manuscript Methods defines these summaries and their eligibility rules.

## Checkpoint selection and evaluation

- Hold the validation and test gap manifests fixed across methods. Each day contributes four sampled gap starts per evaluated duration; ten test days therefore give 40 events per participant and duration.
- Evaluate each gap in a separate daily input. Hide both channels and score only values that were originally observed.
- Select the checkpoint with the lowest validation Joint MAE, equally weighting the setting's evaluated durations: eight at one-minute resolution and seven at ten-minute resolution. Resolve ties by the earlier epoch. Test results must not determine checkpoint selection.
- Preserve the training random-number state across validation. In particular, save and restore the relevant Python, NumPy, PyTorch and CUDA states around baseline validation so validation does not reset later training orders or masks.
- Use unconstrained predictions for Wearformer training and validation selection. At inference, convert predictions to original units and clip negative Steps predictions to zero.
- Mean imputation uses one fixed mean per channel, fitted from observed training values. It must not estimate its means from evaluation batches.

For each duration, average channel MAE within each gap, then across a participant's gaps, then equally across participants. Overall MAE equally weights the evaluated durations. Joint MAE is dimensionless:

```text
Joint MAE = 0.5 * (HR MAE / HR reference SD + Steps MAE / Steps reference SD)
```

The reference scales are population standard deviations computed from each setting's qualifying training days. Share these scales across methods within a setting and hold them fixed in the training-day analyses. Retain their full precision when calculating errors.

## Transfer and training-day analyses

Transfer uses the Large Fitbit model for Small Fitbit and the ten-minute Fitbit model for Apple Watch, without retraining. Retain the source cohort's input normalization and personal-feature reference statistics. Compute target participants' personal summaries from their training dates only. Evaluate local and transferred models on the same target test gaps and target reference scales.

The training-day analysis uses a fixed training cohort of participants who have at least 40 qualifying training dates after holding out validation and test dates. Select the first `N` qualifying training dates in chronological order for `N = 1, 2, 3, 5, 7, 10, 14, 20, 30, 40`, followed by all available qualifying training dates. Skip held-out and non-qualifying dates; `N` counts training days, not elapsed calendar days. Keep validation and test dates unchanged, and refit input normalization and personal summaries for each training subset. Evaluated participants outside the fixed training cohort receive mean personal features, represented by zeros after standardization.

## Recorded software environment

The experiments used Python 3.10 and PyTorch 2.8 on a single NVIDIA A100 or V100 GPU, with CUDA 12.8 or 12.6. Recorded NumPy versions were 2.2.6 for the A100 stack and 1.26.4 for the V100 stack. Neural comparator implementations used PyPOTS 1.5; MICE used scikit-learn 1.7.2. These are recorded experiment versions, not yet a complete installation specification. Exact environments and executable entry points will be included with the code release.
