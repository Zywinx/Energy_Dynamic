"""Dynamic instance-loss weighting utilities for Energy-aware CLAM-SB.

Loss-scale convention
---------------------
Original CLAM-SB uses:
    total_loss = bag_weight * bag_loss + (1 - bag_weight) * instance_loss

ECLAM-SB defines lambda_t as the instance-loss weight relative to bag loss:
    lambda_t = schedule_t * inst_lambda_max
    inst_lambda_max = (1 - bag_weight) / bag_weight  when not explicitly given

To strictly preserve the original CLAM loss scale, the train loop must use:
    total_loss = bag_weight * (bag_loss + lambda_t * instance_loss)

When dynamic_inst_weight='constant' and bag_weight=0.7:
    lambda_t = 0.3 / 0.7 = 0.428571...
    total_loss = 0.7 * (bag_loss + 0.428571 * instance_loss)
               = 0.7 * bag_loss + 0.3 * instance_loss

Do NOT use total_loss = bag_loss + lambda_t * instance_loss for the constant
sanity check, because that changes the global gradient scale.
"""

from __future__ import annotations

import math
from typing import Optional


def _validate_bag_weight(bag_weight: float) -> float:
    bag_weight = float(bag_weight)
    if not 0.0 < bag_weight <= 1.0:
        raise ValueError(f"bag_weight must be in (0, 1], got {bag_weight}")
    return bag_weight


def default_lambda_from_bag_weight(bag_weight: float) -> float:
    """Return lambda_max = (1 - bag_weight) / bag_weight."""
    bag_weight = _validate_bag_weight(bag_weight)
    return (1.0 - bag_weight) / bag_weight


def get_instance_lambda(
    epoch: int,
    max_epochs: int,
    mode: str,
    bag_weight: float,
    inst_lambda_max: Optional[float] = None,
    warmup_epochs: int = 10,
    gamma: float = 10.0,
    attn_sep: Optional[float] = None,
    sep_min: float = 0.05,
    sep_max: float = 0.30,
) -> float:
    """Return lambda_t, the instance-loss weight relative to bag loss.

    This function only returns lambda_t. The caller must compute:
        total_loss = bag_weight * (bag_loss + lambda_t * instance_loss)

    Modes:
      constant:
        lambda_t = inst_lambda_max, defaulting to (1 - bag_weight) / bag_weight.
      linear_warmup:
        lambda_t = inst_lambda_max * min(epoch / warmup_epochs, 1).
      sigmoid_warmup:
        lambda_t = inst_lambda_max / (1 + exp(-gamma * (progress - center))).
      attention_separation:
        if attn_sep is None, fallback to linear_warmup; otherwise map attention
        separation from [sep_min, sep_max] to [0, 1].
    """
    bag_weight = _validate_bag_weight(bag_weight)
    if inst_lambda_max is None:
        inst_lambda_max = default_lambda_from_bag_weight(bag_weight)
    inst_lambda_max = max(float(inst_lambda_max), 0.0)
    epoch = max(int(epoch), 0)
    max_epochs = max(int(max_epochs), 1)
    warmup_epochs = max(int(warmup_epochs), 1)

    if mode == "constant":
        lambda_t = inst_lambda_max
    elif mode == "linear_warmup":
        schedule_t = min(float(epoch) / float(warmup_epochs), 1.0)
        lambda_t = schedule_t * inst_lambda_max
    elif mode == "sigmoid_warmup":
        progress = float(epoch) / float(max_epochs)
        center = float(warmup_epochs) / float(max_epochs)
        schedule_t = 1.0 / (1.0 + math.exp(-float(gamma) * (progress - center)))
        lambda_t = schedule_t * inst_lambda_max
    elif mode == "attention_separation":
        if attn_sep is None:
            schedule_t = min(float(epoch) / float(warmup_epochs), 1.0)
        else:
            denom = max(float(sep_max) - float(sep_min), 1e-12)
            schedule_t = (float(attn_sep) - float(sep_min)) / denom
            schedule_t = min(max(schedule_t, 0.0), 1.0)
        lambda_t = schedule_t * inst_lambda_max
    else:
        raise ValueError(
            f"Unsupported dynamic instance weight mode: {mode}. "
            "Expected constant, linear_warmup, sigmoid_warmup, or attention_separation."
        )
    return float(max(lambda_t, 0.0))


def get_instance_weight(
    epoch: int,
    max_epochs: int,
    mode: str,
    base_bag_weight: float,
    lambda_max: Optional[float] = None,
    warmup_epochs: int = 10,
    gamma: float = 10.0,
    attn_sep: Optional[float] = None,
    sep_min: float = 0.05,
    sep_max: float = 0.30,
) -> float:
    """Backward-compatible alias for earlier ECLAM patches.

    Earlier local patches called this function get_instance_weight and used
    argument names base_bag_weight/lambda_max. New docs call it
    get_instance_lambda with bag_weight/inst_lambda_max.
    """
    return get_instance_lambda(
        epoch=epoch,
        max_epochs=max_epochs,
        mode=mode,
        bag_weight=base_bag_weight,
        inst_lambda_max=lambda_max,
        warmup_epochs=warmup_epochs,
        gamma=gamma,
        attn_sep=attn_sep,
        sep_min=sep_min,
        sep_max=sep_max,
    )
