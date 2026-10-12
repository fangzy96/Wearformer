"""Adapters for the comparison methods and their training-time defaults."""

from __future__ import annotations

import inspect
import random

import numpy as np


NEURAL_MODELS = (
    "SAITS", "ImputeFormer", "Informer", "ETSformer", "Crossformer", "PatchTST",
    "FEDformer", "Autoformer", "BRITS", "SegRNN", "TCN", "SCINet", "FiLM",
    "TimeMixer", "TimeMixerPP", "DLinear", "FreTS",
)
MODEL_NAMES = NEURAL_MODELS + ("MICE", "LOCF", "Lerp", "Mean")


def observed_input_stats(values, observed):
    """Float64 training scaler using the comparator wrapper's 2048-row chunks."""
    from .train_mean import validate_arrays

    validate_arrays(values, observed)
    sums = np.zeros(2, dtype=np.float64)
    squared = np.zeros(2, dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    for start in range(0, len(values), 2048):
        block = np.asarray(values[start:start + 2048], dtype=np.float64)
        mask = np.asarray(observed[start:start + 2048], dtype=bool) & np.isfinite(block)
        for channel in range(2):
            selected = block[..., channel][mask[..., channel]]
            sums[channel] += selected.sum()
            squared[channel] += (selected * selected).sum()
            counts[channel] += selected.size
    if np.any(counts < 2):
        raise ValueError("Each channel needs at least two observed training values")
    means = sums / counts
    stds = np.sqrt(np.maximum(squared / counts - means * means, 0.0))
    if np.any(stds <= 0):
        raise ValueError("Each comparator input channel needs nonzero training variance")
    return means, stds


class MICE:
    """Regress across time-of-day x channel features, fitted on training days."""

    def __init__(self):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        self.imputer = IterativeImputer(
            max_iter=5, n_nearest_features=64, sample_posterior=False,
            random_state=0, keep_empty_features=True,
        )

    def fit(self, data):
        values = data["X"]
        self.imputer.fit(values.reshape(len(values), -1))

    def predict(self, data):
        values = data["X"]
        return {"imputation": self.imputer.transform(values.reshape(len(values), -1)).reshape(values.shape)}


def construct_model(name, n_steps, batch_size=32, device="cpu", seed=20260926, overrides=None, epochs=60):
    """Construct a fresh model with seeded Fourier modes as well as weights.

    FEDformer samples Fourier-mode indices at construction. They are not all
    checkpoint tensors, so reconstruction must use the saved initialization seed.
    """
    if name == "MICE":
        return MICE()
    if name not in NEURAL_MODELS + ("LOCF", "Lerp"):
        raise ValueError("Use TrainingMean for Mean; unknown PyPOTS model name")
    try:
        import torch
        import pypots.imputation as imputation
    except ImportError as exc:
        raise ImportError("Install the optional comparator dependencies: pip install pypots==1.5") from exc
    cls = getattr(imputation, name)
    signature = inspect.signature(cls.__init__).parameters
    defaults = {
        "n_steps": n_steps, "n_features": 2, "epochs": epochs, "batch_size": batch_size,
        "device": device, "rnn_hidden_size": 128, "n_layers": 2, "d_model": 256,
        "n_heads": 4, "d_ffn": 256, "d_k": 64, "d_v": 64, "dropout": 0.1,
        "top_k": 3, "n_kernels": 5, "n_diffusion_steps": 50, "n_channels": 64,
        "d_time_embedding": 32, "d_feature_embedding": 3, "d_diffusion_embedding": 32,
        "target_strategy": "random", "n_temporal_heads": 4, "d_input_embed": 32,
        "d_learnable_embed": 32, "d_proj": 32, "d_ffn_ie": 64, "n_temporal_layers": 2,
    }
    fallback = {
        "n_pred_steps": 0, "n_pred_features": 2, "term": "short", "d_hidden": 128,
        "attn_dropout": 0.1, "patch_size": 24, "patch_len": 24, "stride": 12,
        "seg_len": 24, "moving_avg_window_size": 25, "factor": 3, "win_size": 2,
        "n_levels": 3, "n_stacks": 1, "n_groups": 1, "n_decoder_layers": 1,
        "n_encoder_layers": 2, "d_bottleneck": 64, "latent_size": 32,
        "n_features_hidden": 64, "window_size": [4, 4], "n_series": 2,
        "apply_nonstationary_norm": True, "individual": False, "downsampling_window": 2,
        "downsampling_layers": 3, "channel_independence": True, "n_nodes": 2,
        "version": "fourier", "modes": 32, "ORT_weight": 1.0, "MIT_weight": 1.0,
        "lambda_mse": 1.0, "consistency_weight": 0.1, "imputation_weight": 1.0,
        "num_workers": 0, "proj_dim": 64, "expansion": 2, "kernel_size": 25,
        "ffn_ratio": 1, "n_blocks": [1], "large_size": [31], "small_size": [5],
        "dims": [64], "ortho_weight": 1.0, "multiscale": [1, 2, 4],
        "n_seg_steps": 48, "d_dynamic": 128, "n_hidden_layers": 2,
        "conv_kernel": [12, 16], "patch_stride": 12, "downsampling_ratio": 2,
        "num_blocks": [1], "d_projector_hidden": [64, 64], "n_projector_hidden_layers": 2,
        "inner_size": 5, "bucket_size": 36, "n_hashes": 4, "causal": False,
        "d_feature_encode": 32, "d_temporal_decoder_hidden": 64, "d_embedding": 64,
    }
    kwargs = {key: value for key, value in defaults.items() if key in signature}
    for key, parameter in signature.items():
        if key == "self" or parameter.kind in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL):
            continue
        if parameter.default is parameter.empty and key not in kwargs:
            if key not in fallback:
                raise ValueError(f"No recorded default for required {name} parameter {key!r}")
            kwargs[key] = fallback[key]
    if name == "FEDformer":
        kwargs["n_heads"] = 8
    overrides = overrides or {}
    if set(overrides) - set(signature):
        raise ValueError(f"Unsupported {name} parameters: {sorted(set(overrides) - set(signature))}")
    if any(key in overrides for key in ("n_steps", "n_features", "epochs", "batch_size", "device")):
        raise ValueError("Set data dimensions, epochs, batch size and device using the main CLI")
    kwargs.update(overrides)
    # Imports occur before seeding, so optional-library initialization cannot alter
    # the construction stream between a training run and checkpoint loading.
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    model = cls(**kwargs)
    if name == "SAITS" and isinstance(getattr(model, "model", None), torch.nn.Module):
        def omit_unused_attention(module, inputs, outputs):
            if not module.training and isinstance(outputs, dict):
                return {key: value for key, value in outputs.items()
                        if "attn" not in key and key != "combining_weights"}
            return outputs
        model.model.register_forward_hook(omit_unused_attention)
    return model


class StandardizedPredictor:
    """Convert native inputs to the fixed training scaler and return raw outputs."""

    def __init__(self, model, mean, std):
        self.model = model
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != (2,) or self.std.shape != (2,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("The training scaler must contain two finite means and positive standard deviations")

    def predict(self, values, visible):
        standardized = np.where(visible, (values - self.mean) / self.std, np.nan).astype(np.float32)
        output = self.model.predict({"X": standardized})
        prediction = np.asarray(output["imputation"])
        if prediction.ndim == 4:
            prediction = prediction.mean(axis=1)
        return prediction * self.std + self.mean
