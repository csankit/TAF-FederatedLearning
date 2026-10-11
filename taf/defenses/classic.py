"""FedAvg, Krum, coordinate-wise median, trimmed mean, FLTrust.

Note: FedAvg is the *unweighted* mean of client updates, so that TAF with
equal trust reduces to it exactly (Section 4.5). Data-size weighting would
add a difference unrelated to robustness.
"""
from __future__ import annotations

import torch

from ..utils import cosine
from .base import Defense, one_hot_weights


class FedAvg(Defense):
    name = "FedAvg"

    def aggregate(self, U, ctx):
        n = U.shape[0]
        return U.mean(0), {"weights": torch.full((n,), 1.0 / n)}


class Krum(Defense):
    """Blanchard et al. (2017), single-Krum with f = rho*N."""
    name = "Krum"

    def aggregate(self, U, ctx):
        n = U.shape[0]
        f = int(ctx["rho"] * n)
        k = max(1, n - f - 2)
        D = torch.cdist(U, U) ** 2
        scores = torch.sort(D, dim=1).values[:, 1:k + 1].sum(1)
        i = int(torch.argmin(scores))
        return U[i].clone(), {"weights": one_hot_weights(n, [i])}


class Median(Defense):
    name = "Median"

    def aggregate(self, U, ctx):
        return U.median(0).values, {"weights": None}


class TrimmedMean(Defense):
    """Trim the true rho fraction from each end (favours the baseline)."""
    name = "TrimmedMean"

    def aggregate(self, U, ctx):
        n = U.shape[0]
        k = min(int(ctx["rho"] * n), (n - 1) // 2)
        if k == 0:
            return U.mean(0), {"weights": None}
        S = torch.sort(U, dim=0).values
        return S[k:n - k].mean(0), {"weights": None}


class FLTrust(Defense):
    """Cao et al. (NDSS 2021). Root data: 100 examples held out from TRAIN."""
    name = "FLTrust"

    def aggregate(self, U, ctx):
        g0 = ctx["server_update"]()            # update on the root set
        ts = torch.clamp(cosine(U, g0), min=0.0)
        Un = U * (g0.norm() / (U.norm(dim=1) + 1e-12))[:, None]
        Z = ts.sum()
        if Z <= 0:
            return torch.zeros_like(g0), {"weights": torch.zeros(U.shape[0])}
        p = ts / Z
        return (p[:, None] * Un).sum(0), {"weights": p.cpu()}
