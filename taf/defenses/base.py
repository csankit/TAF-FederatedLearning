"""Defence interface.

``aggregate(U, ctx)`` receives the (N, d) matrix of client updates for one
round (rows ordered by client id) and returns ``(delta, info)``:
``delta`` is the aggregated update applied as w <- w + eta * delta, and
``info["weights"]`` (length N, sums to 1 or is all-zero) is the effective
per-client weight used to compute the malicious trust share.
"""
from __future__ import annotations

import torch


class Defense:
    name = "base"

    def aggregate(self, U: torch.Tensor, ctx: dict):
        raise NotImplementedError


def one_hot_weights(n, idx):
    w = torch.zeros(n)
    w[idx] = 1.0 / len(idx)
    return w
