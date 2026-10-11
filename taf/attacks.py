"""Attacks of Section 5.2: LF, BD, GS, CO (ALIE) and the trust-adaptive AD.

Each round the runner computes, for malicious clients, a clean update and/or
a poisoned update (trained on ``poison_data``), then calls ``craft``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.stats import norm as _norm

from .data import stamp_trigger
from .utils import lower_median

LF_PAIRS = {"mnist": (1, 7), "cifar10": (3, 5)}  # source -> target (cat -> dog)
BD_TARGET = 0


@dataclass
class RoundCtx:
    honest: torch.Tensor           # (|H|, d) honest updates of this round
    mal_clean: torch.Tensor | None  # (|M|, d) what malicious clients would send honestly
    mal_poison: torch.Tensor | None  # (|M|, d) updates trained on poisoned data
    slices: list                   # layer slices of the flat vector
    n_total: int


class Attack:
    name = "none"
    targeted = False
    needs_clean = False
    needs_poison = False

    def __init__(self, dataset: str, **kw):
        self.dataset = dataset.replace("fake_", "")
        self.kw = kw

    def poison_data(self, x, y):
        return x, y

    def craft(self, ctx: RoundCtx) -> torch.Tensor:
        raise NotImplementedError

    # --- ASR evaluation (targeted attacks only) -----------------------
    def asr_inputs(self, test):
        return None


class LabelFlip(Attack):
    name, targeted, needs_poison = "LF", True, True

    def poison_data(self, x, y):
        src, tgt = LF_PAIRS[self.dataset]
        y = y.clone()
        y[y == src] = tgt
        return x, y

    def craft(self, ctx):
        return ctx.mal_poison

    def asr_inputs(self, test):
        src, tgt = LF_PAIRS[self.dataset]
        m = test.y == src
        return test.x[m], tgt


class Backdoor(Attack):
    name, targeted, needs_poison = "BD", True, True

    def __init__(self, dataset, boost=10.0, poison_frac=0.5, **kw):
        super().__init__(dataset, **kw)
        self.boost, self.frac = boost, poison_frac

    def poison_data(self, x, y):
        k = int(len(y) * self.frac)
        idx = torch.randperm(len(y), device="cpu")[:k].to(y.device)
        x, y = x.clone(), y.clone()
        x[idx] = stamp_trigger(x[idx], self.dataset)
        y[idx] = BD_TARGET
        return x, y

    def craft(self, ctx):
        return self.boost * ctx.mal_poison

    def asr_inputs(self, test):
        m = test.y != BD_TARGET
        return stamp_trigger(test.x[m], self.dataset), BD_TARGET


class SignFlip(Attack):
    name, needs_clean = "GS", True

    def __init__(self, dataset, scale=5.0, **kw):
        super().__init__(dataset, **kw)
        self.scale = scale

    def craft(self, ctx):
        return -self.scale * ctx.mal_clean


class ALIE(Attack):
    """'A little is enough' (Baruch et al., 2019): identical colluding update."""
    name, needs_clean = "CO", True

    def __init__(self, dataset, knowledge="full", **kw):
        super().__init__(dataset, **kw)
        self.knowledge = knowledge

    def craft(self, ctx):
        ref = ctx.honest if self.knowledge == "full" else ctx.mal_clean
        n, m = ctx.n_total, ctx.mal_clean.shape[0]
        s = n // 2 + 1 - m
        z = float(_norm.ppf((n - s) / n)) if 0 < (n - s) / n < 1 else 1.0
        sd = ref.std(0) if ref.shape[0] > 1 else torch.zeros_like(ref[0])
        u = ref.mean(0) - z * sd
        return u.unsqueeze(0).repeat(m, 1)


class TrustAdaptive(Attack):
    """Attack AD, adaptive to TAF v2 (white-box: knows TAF and its hyperparameters).

    Per layer l, malicious client i submits
        u_il = n_l * ( s_l e_l + sqrt(1 - s_l^2) * v_il ),
        v_il = a * b_perp_l + sqrt(1 - a^2) * r_il,
    where e_l is the estimated consensus direction, b_perp_l the backdoor
    direction orthogonal to it, r_il independent noise orthogonal to both, and
    n_l the median honest layer norm. Then
      - every layer norm equals the honest median     -> no norm-bound / anomaly signal
      - cos(u_il, consensus_l) = s_l                    -> similarity under attacker control
      - cos(u_i, u_j) = s^2 + (1 - s^2) a^2             -> set equal to the honest
        clients' typical top-k peer cosine q, so the collusion penalty sees nothing.
    a = sqrt((q - s^2)/(1 - s^2)) is forced by q; the payload actually delivered
    per client is sqrt(1 - s^2) * a, logged as ``diag["payload"]``.
    s = "match" uses the honest median per-layer cosine with the consensus.
    """
    name, targeted, needs_clean, needs_poison = "AD", True, True, True

    def __init__(self, dataset, s=0.7, knowledge="full", diversify=True, coll_k=0.1,
                 poison_frac=0.5, **kw):
        super().__init__(dataset, **kw)
        self.s, self.knowledge, self.diversify, self.coll_k = s, knowledge, diversify, coll_k
        self._bd = Backdoor(dataset, boost=1.0, poison_frac=poison_frac)
        self.diag = {}

    def poison_data(self, x, y):
        return self._bd.poison_data(x, y)

    def asr_inputs(self, test):
        return self._bd.asr_inputs(test)

    def _peer_q(self, ref):
        """Median over clients of the mean cosine to their top-k closest peers."""
        n = ref.shape[0]
        if n < 3:
            return 0.0
        Xn = ref / (ref.norm(dim=1, keepdim=True) + 1e-12)
        P = Xn @ Xn.T
        P.fill_diagonal_(-float("inf"))
        k = max(1, min(n - 1, int(-(-self.coll_k * n // 1))))
        return float(P.topk(k, dim=1).values.mean(1).median())

    def craft(self, ctx):
        ref = ctx.honest if self.knowledge == "full" else ctx.mal_clean
        m = ctx.mal_poison.shape[0]
        u_hat = ref.median(0).values
        b = ctx.mal_poison.mean(0)
        q = self._peer_q(ref) if self.diversify else 1.0
        out = torch.zeros(m, u_hat.numel(), device=u_hat.device, dtype=u_hat.dtype)
        payload = []
        for sl in ctx.slices:
            e = u_hat[sl] / (u_hat[sl].norm() + 1e-12)
            if self.s == "match":
                s = float((ref[:, sl] @ e / (ref[:, sl].norm(dim=1) + 1e-12)).median().clamp(0, 0.999))
            else:
                s = float(self.s)
            bp = b[sl] - (b[sl] @ e) * e
            bp = bp / (bp.norm() + 1e-12)
            a2 = min(1.0, max(0.0, (q - s * s) / (1 - s * s))) if s < 1 else 1.0
            a = a2 ** 0.5
            r = torch.randn(m, bp.numel(), device=bp.device, dtype=bp.dtype)
            r = r - (r @ e)[:, None] * e - (r @ bp)[:, None] * bp
            r = r / (r.norm(dim=1, keepdim=True) + 1e-12)
            v = a * bp + (1 - a2) ** 0.5 * r
            n_l = lower_median(ref[:, sl].norm(dim=1))
            out[:, sl] = n_l * (s * e + (1 - s * s) ** 0.5 * v)
            payload.append((1 - s * s) ** 0.5 * a)
        self.diag = {"payload": float(sum(payload) / len(payload)), "peer_q": q}
        return out


class Mimic(Attack):
    """Attack MI, adaptive to clustering defences (FLAME, TAF v3).

    Full knowledge: each malicious client copies a DIFFERENT honest client's
    update h_j (so attackers inherit the honest density structure and fall
    inside the majority cluster) and tilts it toward the backdoor:
        u_i = ||h_j|| ( sqrt(1 - eps^2) h_j/||h_j|| + eps * b_perp_j ),
    with b_perp_j the unit backdoor direction orthogonal to h_j. Norms equal an
    honest client's; cos(u_i, h_j) = sqrt(1 - eps^2). eps is the payload dial
    (logged as diag["payload"]). Copies are re-drawn every round.
    """
    name, targeted, needs_poison = "MI", True, True

    def __init__(self, dataset, eps=0.3, poison_frac=0.5, **kw):
        super().__init__(dataset, **kw)
        self.eps = eps
        self._bd = Backdoor(dataset, boost=1.0, poison_frac=poison_frac)
        self.diag = {}

    def poison_data(self, x, y):
        return self._bd.poison_data(x, y)

    def asr_inputs(self, test):
        return self._bd.asr_inputs(test)

    def craft(self, ctx):
        H = ctx.honest
        m = ctx.mal_poison.shape[0]
        b = ctx.mal_poison.mean(0)
        idx = torch.randperm(H.shape[0], device="cpu")[:m].to(H.device)
        if len(idx) < m:                       # more attackers than honest: reuse
            idx = idx.repeat((m + len(idx) - 1) // len(idx))[:m]
        out = []
        for j in idx:
            h = H[j]
            nh = h.norm() + 1e-12
            e = h / nh
            bp = b - (b @ e) * e
            bp = bp / (bp.norm() + 1e-12)
            out.append(nh * ((1 - self.eps ** 2) ** 0.5 * e + self.eps * bp))
        self.diag = {"payload": float(self.eps)}
        return torch.stack(out)


class NoisyCollusion(Attack):
    """Attack CN: coordinated but nonidentical collusion (revision 2).

    All attackers share the ALIE target u = mu - z sigma (attack CO), but each
    sends its own copy with client specific noise, layer by layer:
        u_il = ||u_l|| ( sqrt(c) u_l/||u_l|| + sqrt(1 - c) r_il ),
    with r_il a random unit vector orthogonal to u_l. Every layer norm equals
    that of the CO update (same per client budget), and the expected pairwise
    cosine between attackers is c. c = "match" sets c to the honest clients'
    median top k peer cosine, so pairwise similarity is no longer unusual.
    The shared component carries sqrt(c) of the CO payload per client.
    """
    name, needs_clean = "CN", True

    def __init__(self, dataset, c=0.9, coll_k=0.1, knowledge="full", **kw):
        super().__init__(dataset, **kw)
        self.c, self.coll_k, self.knowledge = c, coll_k, knowledge
        self._alie = ALIE(dataset, knowledge=knowledge)
        self.diag = {}

    def craft(self, ctx):
        base = self._alie.craft(ctx)[0]
        ref = ctx.honest if self.knowledge == "full" else ctx.mal_clean
        m = ctx.mal_clean.shape[0]
        if self.c == "match":
            c = TrustAdaptive._peer_q(self, ref)
        else:
            c = float(self.c)
        c = min(1.0, max(0.0, c))
        out = torch.zeros(m, base.numel(), device=base.device, dtype=base.dtype)
        for sl in ctx.slices:
            ul = base[sl]
            nl = ul.norm() + 1e-12
            e = ul / nl
            r = torch.randn(m, ul.numel(), device=ul.device, dtype=ul.dtype)
            r = r - (r @ e)[:, None] * e
            r = r / (r.norm(dim=1, keepdim=True) + 1e-12)
            out[:, sl] = nl * (c ** 0.5 * e + (1 - c) ** 0.5 * r)
        Xn = out / (out.norm(dim=1, keepdim=True) + 1e-12)
        P = Xn @ Xn.T
        iu = torch.triu_indices(m, m, 1)
        self.diag = {"c": c, "pair_cos": float(P[iu[0], iu[1]].mean()) if m > 1 else 1.0}
        return out


class Intermittent(Attack):
    """Attack IM: attackers alternate between honest and malicious rounds.

    Wraps a base attack (BD or AD). In round r the attackers are malicious when
    r % period in on_rounds, and otherwise send their honest update. The runner
    sets ``self.current_round`` before each call to craft. With the defaults
    (period 4, on round 3) the attackers poison one round in four, including
    the last training round.
    """
    name, needs_clean = "IM", True

    def __init__(self, dataset, base="BD", period=4, on_rounds=(3,), base_kw=None, **kw):
        super().__init__(dataset, **kw)
        self.base = ATTACKS[base](dataset, **(base_kw or {}))
        self.targeted = self.base.targeted
        self.needs_poison = self.base.needs_poison
        self.period, self.on_rounds = int(period), tuple(on_rounds)
        self.current_round = 0
        self.diag = {}

    def poison_data(self, x, y):
        return self.base.poison_data(x, y)

    def asr_inputs(self, test):
        return self.base.asr_inputs(test)

    def is_on(self, r=None):
        r = self.current_round if r is None else r
        return (r % self.period) in self.on_rounds

    def craft(self, ctx):
        on = self.is_on()
        self.diag = {"on": int(on)}
        if not on:
            return ctx.mal_clean
        out = self.base.craft(ctx)
        self.diag.update(getattr(self.base, "diag", {}) or {})
        return out


ATTACKS = {"none": Attack, "LF": LabelFlip, "BD": Backdoor, "GS": SignFlip,
           "CO": ALIE, "AD": TrustAdaptive, "MI": Mimic,
           "CN": NoisyCollusion, "IM": Intermittent}


def build_attack(name, dataset, **kw):
    return ATTACKS[name](dataset, **kw)
