"""Flat-vector helpers and seeding."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_flat(model: nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()


def set_flat(model: nn.Module, flat: torch.Tensor) -> None:
    torch.nn.utils.vector_to_parameters(flat, model.parameters())


def layer_slices(model: nn.Module) -> list[slice]:
    """Slices of the flat parameter vector, one per parameter *group*.

    Weight and bias of the same module are merged into one group, so a
    ``Linear``/``Conv`` layer counts as one layer L in Eq. (feat).
    """
    slices, start, prev_mod = [], 0, None
    for name, p in model.named_parameters():
        mod = name.rsplit(".", 1)[0]
        n = p.numel()
        if mod == prev_mod:
            last = slices[-1]
            slices[-1] = slice(last.start, last.stop + n)
        else:
            slices.append(slice(start, start + n))
        start += n
        prev_mod = mod
    return slices


def lower_median(x: torch.Tensor) -> torch.Tensor:
    """The ceil(N/2)-th smallest value (Lemma 1 uses this definition)."""
    n = x.numel()
    k = (n + 1) // 2  # ceil(n/2), 1-indexed
    return torch.kthvalue(x.flatten(), k).values


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Cosine similarity of each row of ``a`` with vector ``b``."""
    if a.dim() == 1:
        a = a.unsqueeze(0)
    return (a @ b) / (a.norm(dim=1) * b.norm() + eps)
