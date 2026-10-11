"""Trust-Aware Aggregation Framework.

Two versions, selectable by keyword:

v1 (original submission, kept for the paper's comparison):
    detector="iforest", collusion=False, combine="linear", clean_consensus=False

v2 (revised; default):
    detector="robust", collusion=True, combine="gated", clean_consensus=True

Per round:
  1. norm bound (lower-median norm)                       -> u~_i
  2. anomaly penalty A_i in [0,1]  (robust z of layer features, or iForest)
  3. collusion penalty C_i in [0,1] (robust z of mean top-k peer cosine)
  4. consensus = coordinate median over clients with gate (1-A)(1-C) >= 1/2
     (all clients if fewer than half pass)                -> u_bar
  5. S_i = max(0, cos(u~_i, u_bar));  H_i asymmetric EMA of S_i
  6. v2: T_i = (a S_i + b H_i)/(a+b) * (1-A_i) * (1-C_i)
     v1: T_i = clip(a S_i + b H_i - g A_i, 0, 1)
  7. aggregate sum_i T_i u~_i / sum_j T_j

Pilot evidence that motivated v2: under non-IID data honest clients agree
weakly with the median while a coordinated backdoor block agrees strongly, so
v1 gave attackers MORE trust than honest clients (CIFAR-10, BD, rho=0.3).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.ensemble import IsolationForest

from ..utils import cosine, lower_median
from .base import Defense


def _robust_z(x: np.ndarray, axis=0, eps=1e-9):
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
    return (x - med) / (1.4826 * mad + eps)


def _hinge(z, z0, width):
    """0 below z0, rising linearly to 1 at z0 + width."""
    return np.clip((z - z0) / width, 0.0, 1.0)


class TAF(Defense):
    name = "TAF"

    def __init__(self, n_clients, alpha=0.4, beta=0.4, gamma=0.2,
                 lam_up=0.1, lam_down=0.5, h0=0.5, norm_bound=True,
                 detector="robust", collusion=True, combine="gated",
                 clean_consensus=True, anom_z0=3.0, anom_w=3.0,
                 coll_z0=2.0, coll_w=2.0, coll_k=0.1, if_trees=100, seed=0,
                 mode="v2", rep_up=0.1, rep_down=0.5, cluster_gate=True, use_rep=True, **_):
        self.a, self.b, self.g = alpha, beta, gamma
        self.lam_up, self.lam_down, self.h0 = lam_up, lam_down, h0
        self.norm_bound, self.detector, self.collusion = norm_bound, detector, collusion
        self.combine, self.clean_consensus = combine, clean_consensus
        self.anom_z0, self.anom_w = anom_z0, anom_w
        self.coll_z0, self.coll_w, self.coll_k = coll_z0, coll_w, coll_k
        self.if_trees, self.seed = if_trees, seed
        self.H = torch.full((n_clients,), float(h0))
        # v3 (hybrid): acceptance reputation, starts neutral
        self.mode, self.rep_up, self.rep_down = mode, rep_up, rep_down
        self.R = torch.full((n_clients,), float(h0))
        self.cluster_gate, self.use_rep = cluster_gate, use_rep   # v3 ablation switches

    # ---- components ------------------------------------------------------
    def bound(self, U):
        norms = U.norm(dim=1)
        tau = lower_median(norms)
        if not self.norm_bound:
            return U, norms, tau
        scale = torch.clamp(tau / (norms + 1e-12), max=1.0)
        return U * scale[:, None], norms, tau

    def _features(self, U, ref, slices):
        feats = []
        for sl in slices:
            feats.append(torch.log(U[:, sl].norm(dim=1) + 1e-12))
            feats.append(cosine(U[:, sl], ref[sl]))
        return np.nan_to_num(_robust_z(torch.stack(feats, 1).cpu().numpy()))

    def anomaly(self, U, ref, slices, rnd):
        n = U.shape[0]
        if self.detector == "none":
            return np.zeros(n)
        F = self._features(U, ref, slices)            # raw U: inflated norms stay visible
        if self.detector == "robust":
            # largest per-feature robust z; median/MAD are set by honest clients
            # whenever rho < 1/2, so a large group cannot mask itself
            return _hinge(np.abs(F).max(1), self.anom_z0, self.anom_w)
        iso = IsolationForest(n_estimators=self.if_trees, max_samples=min(256, n),
                              random_state=self.seed * 100003 + rnd)
        a = -iso.fit(F).score_samples(F)
        return 1.0 / (1.0 + np.exp(-_robust_z(a)))   # v1 normalisation

    def collusion_penalty(self, Ub):
        n = Ub.shape[0]
        if not self.collusion or n < 4:
            return np.zeros(n), np.zeros(n)
        Xn = Ub / (Ub.norm(dim=1, keepdim=True) + 1e-12)
        P = (Xn @ Xn.T).cpu().numpy()
        np.fill_diagonal(P, -np.inf)
        k = max(1, math.ceil(self.coll_k * n))
        topk = np.sort(P, axis=1)[:, -k:].mean(1)     # concentration among closest peers
        z = _robust_z(topk)
        return _hinge(z, self.coll_z0, self.coll_w), topk

    # ---- one round -------------------------------------------------------
    def aggregate(self, U, ctx):
        if self.mode == "v3":
            return self._aggregate_v3(U, ctx)
        n = U.shape[0]
        assert n == self.H.numel(), "partial participation is implemented for TAF v3 only"
        Ub, raw_norms, tau = self.bound(U)
        C, peer = self.collusion_penalty(Ub)
        # provisional reference for layer-cosine features: plain median
        ref0 = Ub.median(0).values
        A = self.anomaly(U, ref0, ctx["slices"], ctx["round"])
        gate = (1 - A) * (1 - C)
        if self.clean_consensus:
            keep = np.flatnonzero(gate >= 0.5)
            if len(keep) < (n + 1) // 2:
                keep = np.arange(n)
            ubar = Ub[torch.as_tensor(keep, device=Ub.device)].median(0).values
        else:
            ubar = ref0
        S = torch.clamp(cosine(Ub, ubar), min=0.0).cpu()
        lam = torch.where(S < self.H, torch.tensor(self.lam_down), torch.tensor(self.lam_up))
        self.H = self.H + lam * (S - self.H)
        A_t = torch.as_tensor(A, dtype=torch.float32)
        C_t = torch.as_tensor(C, dtype=torch.float32)
        if self.combine == "gated":
            T = (self.a * S + self.b * self.H) / (self.a + self.b) * (1 - A_t) * (1 - C_t)
        elif self.combine == "product":
            T = S * self.H * (1 - A_t)
        else:  # v1 linear
            T = self.a * S + self.b * self.H - self.g * A_t
        T = torch.clamp(T, 0.0, 1.0)
        Z = T.sum()
        info = {"T": T, "S": S, "H": self.H.clone(), "A": A_t, "C": C_t,
                "peer": torch.as_tensor(peer, dtype=torch.float32), "tau": float(tau)}
        if Z <= 0:
            return torch.zeros_like(U[0]), {**info, "weights": torch.zeros(n)}
        p = (T / Z).to(U.device)
        return (p[:, None] * Ub).sum(0), {**info, "weights": p.cpu()}

    # ---- v3: hybrid ------------------------------------------------------
    def _aggregate_v3(self, U, ctx):
        """Hybrid (v3), motivated by pilot 2:
          * v2's similarity-based trust was gamed by an adaptive attacker that
            kept its similarity steady at the honest median (reputation then
            amplified it), while FLAME's majority-cluster test resisted;
          * FLAME let a group of near-identical updates (ALIE) take over its
            majority cluster on MNIST, which v2's collusion penalty caught.
        Gate g_i = m_i (1 - A_i)(1 - C_i), with m_i = membership of the
        weight-space majority cluster. Reputation R_i is an asymmetric EMA of
        the gate itself (acceptance history), not of similarity, so a steady
        attacker cannot out-earn honest clients. Weight T_i = g_i * R_i.
        """
        from .recent import _cos_dist, _hdbscan
        n = U.shape[0]
        Ub, raw_norms, tau = self.bound(U)
        if self.cluster_gate:
            W = ctx["global_flat"].unsqueeze(0) + U
            labels = _hdbscan(_cos_dist(W), n // 2 + 1)
        else:
            labels = np.full(n, -1)
        if (labels >= 0).any():
            major = np.bincount(labels[labels >= 0]).argmax()
            m = (labels == major).astype(float)
        else:
            m = np.ones(n)
        C, peer = self.collusion_penalty(Ub)
        ref = Ub[torch.as_tensor(np.flatnonzero(m > 0) if m.any() else np.arange(n),
                                 device=Ub.device)].median(0).values
        A = self.anomaly(U, ref, ctx["slices"], ctx["round"])
        g = torch.as_tensor(m * (1 - A) * (1 - C), dtype=torch.float32)
        ids = torch.as_tensor(ctx.get("ids", list(range(n))), dtype=torch.long)
        R = self.R[ids]                     # reputations of this round's participants
        lam = torch.where(g < R, torch.tensor(self.rep_down), torch.tensor(self.rep_up))
        R = R + lam * (g - R)
        self.R[ids] = R
        T = torch.clamp(g * R if self.use_rep else g, 0.0, 1.0)
        S = torch.clamp(cosine(Ub, ref), min=0.0).cpu()
        info = {"T": T, "S": S, "H": R.clone(), "A": torch.as_tensor(A, dtype=torch.float32),
                "C": torch.as_tensor(C, dtype=torch.float32), "M": torch.as_tensor(m, dtype=torch.float32),
                "peer": torch.as_tensor(peer, dtype=torch.float32), "tau": float(tau)}
        Z = T.sum()
        if Z <= 0:
            return torch.zeros_like(U[0]), {**info, "weights": torch.zeros(n)}
        p = (T / Z).to(U.device)
        return (p[:, None] * Ub).sum(0), {**info, "weights": p.cpu()}
