"""Portable checkpoints containing model weights and fitted preprocessing."""

from pathlib import Path
import torch
from .model import TimeGapSAITS


def resolve_device(value="auto"):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_model(path, device="cpu"):
    """Return ``(model, checkpoint)``; load tensors with restricted deserialization."""
    device = resolve_device(str(device))
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Expected a Wearformer portable checkpoint (format_version=1)")
    model = TimeGapSAITS(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def save_checkpoint(path, model, config, preprocessing, epoch, validation_joint_mae):
    """Write atomically; no participant identifiers or daily data are embedded."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format_version": 1, "model_state_dict": model.state_dict(),
                "config": config, "preprocessing": preprocessing, "epoch": int(epoch),
                "validation_joint_mae": float(validation_joint_mae)}, temporary)
    temporary.replace(path)
