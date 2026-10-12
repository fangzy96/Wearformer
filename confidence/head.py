"""Laplace error head and validation-relative confidence scores.

The head predicts an error scale; its final confidence score is an empirical
rank against validation gaps, not a probability that an imputation is correct.
"""

import hashlib

import numpy as np
import torch
from torch import nn


class ScaleHead(nn.Module):
    """One positive error scale per time point and channel, in standardized units."""

    def __init__(self, d_model, width=128, eps=1e-3, n_out=2):
        super().__init__()
        self.d_in = int(d_model) + 2
        self.l1 = nn.Linear(self.d_in, int(width))
        self.l2 = nn.Linear(int(width), int(n_out))
        self.eps = float(eps)

    def forward(self, enc, vis):
        h = torch.relu(self.l1(torch.cat((enc, vis.to(enc.dtype)), dim=-1)))
        return nn.functional.softplus(self.l2(h)) + self.eps


def frozen_forward(net, model_input, input_observed, side_information):
    """Forward without backbone gradients; leave its requires_grad flags intact.

    Callers set the backbone to eval mode and optimize only the head. Changing
    requires_grad on the backbone can change the attention execution path.
    """
    with torch.no_grad():
        out = net(model_input, input_observed, side_information)
    return out['encoded_1'], out['x_tilde_3']


def laplace_nll(b, zhat, z):
    return torch.log(b) + (z - zhat).abs() / b


def laplace_loss(b, zhat, z, mask):
    """Equal channel means within each daily sample, then an equal sample mean."""
    nll = laplace_nll(b, zhat, z)
    w = mask.to(dtype=nll.dtype)
    counts = w.sum(dim=1)
    valid = counts > 0
    fm = (nll * w).sum(dim=1) / counts.clamp_min(1)
    per = (fm * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    if torch.any(valid.sum(dim=1) == 0):
        raise ValueError('a training sample has no target support')
    return per.mean()


def assert_one_block(input_observed, target):
    """Require one span hidden in both channels, with at least one target value."""
    t_any = target.any(dim=-1)
    if not bool(t_any.any(dim=1).all()):
        raise RuntimeError('a sample without target cells')
    T = t_any.shape[1]
    ar = torch.arange(T, device=t_any.device)
    first = torch.where(t_any, ar, T).min(dim=1).values
    last = torch.where(t_any, ar, -1).max(dim=1).values
    span = (ar[None, :] >= first[:, None]) & (ar[None, :] <= last[:, None])
    if bool((target & ~span[:, :, None]).any()):
        raise RuntimeError('target cells outside one contiguous block')
    if bool((input_observed & span[:, :, None]).any()):
        raise RuntimeError('a block cell is visible to the model')
    return int((last - first + 1).sum().item())


def backbone_state_sha(net):
    """Hash parameters and buffers to verify that head training leaves them fixed."""
    h = hashlib.sha256()
    for name, value in net.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def participant_folds(participant_ids, salt="wearformer-confidence-v1"):
    """Deterministically allocate half of participants to head selection.

    An explicit fold file can instead be supplied to the CLI. Participant labels
    and the salt affect this assignment; persist it with each analysis.
    """
    ids = sorted(set(str(pid) for pid in participant_ids))
    if len(ids) < 2:
        raise ValueError("at least two validation participants are required")
    ordered = sorted(ids, key=lambda p: (hashlib.sha256(f"{salt}:{p}".encode()).hexdigest(), p))
    return {pid: ("fit" if i < len(ids) // 2 else "cal") for i, pid in enumerate(ordered)}


def balanced_nll(event_channel_nll, participant_ids, gap_slots, selected=None):
    """Mean events per participant, then participants, then channels/durations.

    Input is [event, channel]. NaN means an event has no observed target in that
    channel. Every duration/channel must have at least one eligible participant.
    This differs from pooling all target cells or averaging all events directly.
    """
    values = np.asarray(event_channel_nll, dtype=np.float64)
    ids = np.asarray(participant_ids, dtype=str)
    gaps = np.asarray(gap_slots)
    if values.ndim != 2 or values.shape != (len(ids), 2) or gaps.shape != ids.shape:
        raise ValueError("expected [events, 2] NLL and one participant/duration per event")
    use = np.ones(len(ids), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
    if use.shape != ids.shape or not use.any():
        raise ValueError("selection must include validation events")
    means = []
    for gap in np.unique(gaps):
        at_gap = use & (gaps == gap)
        for channel in range(2):
            valid = at_gap & np.isfinite(values[:, channel])
            participants = np.unique(ids[valid])
            if not len(participants):
                raise ValueError(f"no finite validation NLL for gap {gap}, channel {channel}")
            means.append(np.mean([values[valid & (ids == pid), channel].mean() for pid in participants]))
    return float(np.mean(means))


def predicted_joint_error(gap_mean_scale, train_std, reference_std):
    """Convert gap-mean scales from model units to the Joint MAE reference scale.

    Average scales over the full hidden interval, including naturally missing
    positions; unlike MAE this does not require ground-truth target values.
    """
    b = np.asarray(gap_mean_scale, dtype=np.float64)
    train = np.asarray(train_std, dtype=np.float64)
    reference = np.asarray(reference_std, dtype=np.float64)
    if b.ndim < 1 or b.shape[-1] != 2 or train.shape != (2,) or reference.shape != (2,):
        raise ValueError("expected two channels in scales and standard deviations")
    if not np.isfinite(b).all() or np.any(b <= 0):
        raise ValueError("predicted scales must be finite and positive")
    if not np.isfinite(train).all() or not np.isfinite(reference).all() or np.any(train <= 0) or np.any(reference <= 0):
        raise ValueError("standard deviations must be finite and positive")
    return (b * train / reference).mean(axis=-1)


def confidence_rank(predicted_error, validation_error):
    """One minus the validation empirical CDF, including ties at the query value."""
    query = np.asarray(predicted_error, dtype=np.float64)
    reference = np.asarray(validation_error, dtype=np.float64)
    if reference.ndim != 1 or reference.size == 0 or not np.isfinite(reference).all():
        raise ValueError("a nonempty finite validation reference is required")
    if not np.isfinite(query).all():
        raise ValueError("predicted errors must be finite")
    return 1.0 - np.searchsorted(np.sort(reference), query, side="right") / reference.size
