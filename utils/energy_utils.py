"""Energy and uncertainty utilities for the ECLAM experimental branch.

Research interpretation notes
-----------------------------
Attention scores in CLAM describe the relative contribution of patches within a
single WSI bag. They are not class labels. In particular, low-attention patches
must not be directly interpreted as an ``unknown`` class.

The free-energy score implemented here is computed from slide-level logits and
is used as a slide-level compatibility / uncertainty signal for the known
benign/malignant classes. This implementation does not create a patch-level
unknown classifier and does not perform open-set classification. If reliable
unknown-slide labels or ROI annotations become available later, this branch can
be extended to a C+1 detector.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

ArrayLike = Union[torch.Tensor, np.ndarray]


def compute_free_energy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute slide-level free energy: E(x) = -T * logsumexp(logits / T)."""
    if not torch.is_tensor(logits):
        raise TypeError("logits must be a torch.Tensor")
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if logits.dim() != 2:
        raise ValueError(f"logits must have shape [B, C], got {tuple(logits.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return -float(temperature) * torch.logsumexp(logits / float(temperature), dim=1)


def compute_softmax_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute predictive entropy H(p) = -sum_c p_c log(p_c)."""
    if not torch.is_tensor(probs):
        raise TypeError("probs must be a torch.Tensor")
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
    probs = probs.clamp(min=eps)
    return -(probs * probs.log()).sum(dim=1)


def compute_confidence(probs: torch.Tensor) -> torch.Tensor:
    """Return max softmax probability for each slide."""
    if not torch.is_tensor(probs):
        raise TypeError("probs must be a torch.Tensor")
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
    return torch.max(probs, dim=1).values


def compute_margin(probs: torch.Tensor) -> torch.Tensor:
    """Return top1_prob - top2_prob for binary or multiclass outputs."""
    if not torch.is_tensor(probs):
        raise TypeError("probs must be a torch.Tensor")
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
    if probs.size(1) < 2:
        raise ValueError("margin requires at least two classes")
    top2 = torch.topk(probs, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def _zscore(x: ArrayLike, eps: float = 1e-12) -> ArrayLike:
    if torch.is_tensor(x):
        x_float = x.float()
        if x_float.numel() <= 1:
            return x_float
        return (x_float - x_float.mean()) / (x_float.std(unbiased=False) + eps)
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.size <= 1:
        return x_arr
    return (x_arr - x_arr.mean()) / (x_arr.std() + eps)


def compute_active_score(
    energy: ArrayLike,
    entropy: Optional[ArrayLike] = None,
    margin: Optional[ArrayLike] = None,
    mode: str = "energy_entropy",
) -> ArrayLike:
    """Compute a ranking-only active-learning score.

    Higher score means earlier review priority. This score is not used by training.
    """
    if mode == "energy":
        return energy
    if mode == "entropy":
        if entropy is None:
            raise ValueError("entropy is required for mode='entropy'")
        return entropy
    if mode == "energy_entropy":
        if entropy is None:
            raise ValueError("entropy is required for mode='energy_entropy'")
        return _zscore(energy) + _zscore(entropy)
    if mode == "energy_low_margin":
        if margin is None:
            raise ValueError("margin is required for mode='energy_low_margin'")
        return _zscore(energy) - _zscore(margin)
    raise ValueError("Unsupported active score mode")
