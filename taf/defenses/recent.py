"""FLAME (Nguyen et al., USENIX Security 2022) and DeepSight (Rieger et al.,
NDSS 2022), reimplemented from the papers' descriptions.

Both are REIMPLEMENTATIONS. Known deviations from the originals are listed in
docs/BASELINES.md and must be validated (e.g. reproduce each paper's
qualitative behaviour on its own backdoor setting) before results are
reported.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*default value of `copy`.*", category=FutureWarning)
import numpy as np
import torch
from sklearn.cluster import HDBSCAN

from ..utils import cosine, lower_median
from .base import Defense


def _cos_dist(X: torch.Tensor) -> np.ndarray:
    # float64: distances between full model weights are ~1e-6, below float32
    # resolution of the cosine; then force exact symmetry (sklearn checks it).
    X = X.double()
    Xn = X / (X.norm(dim=1, keepdim=True) + 1e-300)
    D = (1.0 - Xn @ Xn.T).clamp(min=0).cpu().numpy()
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def _hdbscan(D: np.ndarray, min_cluster_size: int) -> np.ndarray:
    return HDBSCAN(min_cluster_size=max(2, min_cluster_size), min_samples=1,
                   metric="precomputed", allow_single_cluster=True).fit_predict(D)


class FLAME(Defense):
    """Cosine HDBSCAN filtering -> median-norm clipping -> Gaussian noise."""
    name = "FLAME"

    def __init__(self, noise_lambda=0.001, **_):
        self.lam = noise_lambda

    def aggregate(self, U, ctx):
        n = U.shape[0]
        # Algorithm 1, line 6 of the paper: cosine distances between the local
        # MODELS W_i = G + u_i, not between the updates u_i.
        W = ctx["global_flat"].unsqueeze(0) + U
        labels = _hdbscan(_cos_dist(W), n // 2 + 1)
        if (labels >= 0).any():
            major = np.bincount(labels[labels >= 0]).argmax()
            keep = np.flatnonzero(labels == major)
        else:
            keep = np.arange(n)
        norms = U.norm(dim=1)
        S_t = lower_median(norms)
        Uk = U[keep] * torch.clamp(S_t / (norms[keep] + 1e-12), max=1.0)[:, None]
        delta = Uk.mean(0)
        delta = delta + torch.randn_like(delta) * (self.lam * S_t)
        w = torch.zeros(n)
        w[keep] = 1.0 / len(keep)
        return delta, {"weights": w, "kept": keep.tolist()}


class DeepSight(Defense):
    """Classification of updates via NEUP threshold exceedings, ensemble
    clustering on DDifs / NEUPs / output-bias cosine, cluster-level rejection,
    then median-norm clipping."""
    name = "DeepSight"

    def __init__(self, n_probe=512, n_seeds=3, reject_frac=1 / 3, **_):
        self.n_probe, self.n_seeds, self.tau = n_probe, n_seeds, reject_frac

    def aggregate(self, U, ctx):
        n = U.shape[0]
        out_sl = ctx["slices"][-1]              # output layer (weight + bias)
        n_cls = ctx["num_classes"]
        w_out = U[:, out_sl]
        n_w = w_out.shape[1] - n_cls
        dW = w_out[:, :n_w].reshape(n, n_cls, -1)
        dB = w_out[:, n_w:]

        # NEUPs and threshold exceedings
        energy = dW.abs().sum(2) + dB.abs()                     # (n, C)
        neup = energy ** 2 / (energy ** 2).sum(1, keepdim=True).clamp(min=1e-12)
        xi = torch.clamp(neup.max(1, keepdim=True).values * max(0.01, 1 / n_cls), min=0)
        te = (neup >= xi).sum(1).float()
        suspicious = (te <= te.median() / 2).cpu().numpy()

        # DDifs on random probe inputs
        g = ctx["global_flat"]
        clusterings = []
        for s in range(self.n_seeds):
            gen = torch.Generator(device="cpu").manual_seed(1000 + s)
            x = torch.rand((self.n_probe, *ctx["input_shape"]), generator=gen).to(U.device)
            p_g = ctx["softmax"](g, x).clamp(min=1e-12)
            dd = torch.stack([(ctx["softmax"](g + U[i], x) / p_g).mean(0) for i in range(n)])
            clusterings.append(_hdbscan(_cos_dist(dd), 2))
        clusterings.append(_hdbscan(_cos_dist(neup), 2))
        clusterings.append(_hdbscan(_cos_dist(dB), 2))

        # ensemble: distance = fraction of clusterings separating i and j
        D = np.zeros((n, n))
        for lab in clusterings:
            lab = lab.copy()
            noise = lab < 0
            lab[noise] = lab.max() + 1 + np.arange(noise.sum())   # noise = singletons
            D += (lab[:, None] != lab[None, :])
        D /= len(clusterings)
        final = _hdbscan(D, 2)
        final = final.copy()
        noise = final < 0
        final[noise] = final.max() + 1 + np.arange(noise.sum())

        keep = []
        for c in np.unique(final):
            members = np.flatnonzero(final == c)
            if suspicious[members].mean() < self.tau:
                keep.extend(members.tolist())
        keep = np.array(sorted(keep), dtype=int)
        diag = {"ds_suspicious": torch.as_tensor(suspicious, dtype=torch.float32),
                "ds_te": te.float().cpu(), "ds_n_clusters": int(len(np.unique(final))),
                "ds_cluster_sizes": sorted(np.bincount(final).tolist(), reverse=True)[:5]}
        if len(keep) == 0:
            return torch.zeros_like(U[0]), {"weights": torch.zeros(n), "kept": [], **diag}
        norms = U.norm(dim=1)
        S_t = lower_median(norms)
        Uk = U[keep] * torch.clamp(S_t / (norms[keep] + 1e-12), max=1.0)[:, None]
        w = torch.zeros(n)
        w[keep] = 1.0 / len(keep)
        return Uk.mean(0), {"weights": w, "kept": keep.tolist(), **diag}


class FoolsGold(Defense):
    """FoolsGold (Fung, Yoon, Beschastnikh, RAID 2020), Algorithm 1.

    Keeps the running sum of each client's updates, computes pairwise cosine
    similarity of these histories, pardons honest clients that resemble a
    more similar one, and maps 1 - max similarity through a logit with
    kappa = 1. Departure from the paper: the weighted sum is normalised by
    the sum of weights (a weighted mean), so that its scale matches the
    other aggregators; the paper scales by the learning rate instead. No
    feature importance weighting (the paper's default for its main results
    is also the full update). ``clip=True`` adds TAF's lower median norm
    bound to every update before aggregation (revision 2 variant).
    """
    name = "FoolsGold"

    def __init__(self, n_clients=None, kappa=1.0, clip=False, **_):
        self.kappa, self.n, self.clip = kappa, n_clients, clip
        self.Hist = None

    def aggregate(self, U, ctx):
        n = U.shape[0]
        ids = torch.as_tensor(ctx.get("ids", list(range(n))), dtype=torch.long, device=U.device)
        if self.Hist is None:
            self.Hist = torch.zeros(self.n or n, U.shape[1], device=U.device, dtype=U.dtype)
        self.Hist[ids] += U
        Hn = self.Hist[ids].double()
        Hn = Hn / (Hn.norm(dim=1, keepdim=True) + 1e-12)
        cs = (Hn @ Hn.T).cpu().numpy() - np.eye(n)
        v = cs.max(1)
        for i in range(n):
            for j in range(n):
                if i != j and v[i] < v[j] and v[j] > 0:
                    cs[i, j] *= v[i] / v[j]
        wv = 1.0 - cs.max(1)
        wv = np.clip(wv, 0.0, 1.0)
        if wv.max() > 0:
            wv = wv / wv.max()
        wv[wv >= 1.0] = 0.99
        with np.errstate(divide="ignore"):
            wv = self.kappa * (np.log(wv / (1.0 - wv)) + 0.5)
        wv[~np.isfinite(wv)] = 0.0
        wv = np.clip(wv, 0.0, 1.0)
        w = torch.as_tensor(wv, dtype=U.dtype, device=U.device)
        Z = w.sum()
        if Z <= 0:
            return torch.zeros_like(U[0]), {"weights": torch.zeros(n)}
        p = w / Z
        info = {"weights": p.cpu()}
        if self.clip:
            # Revision 2 variant: TAF's norm bound (lower median of the round's
            # update norms) applied to every update before aggregation. The
            # similarity history above still uses the raw updates, so the
            # weights are exactly those of FoolsGold.
            nrm = U.norm(dim=1)
            tau = lower_median(nrm)
            U = U * torch.clamp(tau / (nrm + 1e-12), max=1.0)[:, None]
            info["tau"] = float(tau)
        return (p[:, None] * U).sum(0), info
