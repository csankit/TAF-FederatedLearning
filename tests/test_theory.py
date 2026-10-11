"""Numerical checks that the implementation satisfies Section 4.6."""
import torch

from taf.attacks import RoundCtx, TrustAdaptive
from taf.defenses.taf import TAF
from taf.utils import cosine, lower_median

V1 = dict(detector="iforest", collusion=False, combine="linear", clean_consensus=False)


def _ctx(slices, r=0):
    return {"slices": slices, "round": r}


def test_lemma1_tau_in_honest_range():
    g = torch.Generator().manual_seed(0)
    for n in (5, 6, 11, 50, 51):
        n_mal = (n - 1) // 2                       # largest rho < 1/2
        honest = torch.rand(n - n_mal, generator=g) + 0.5
        for mal in (torch.full((n_mal,), 1e6), torch.full((n_mal,), 1e-6)):
            tau = lower_median(torch.cat([honest, mal]))
            assert honest.min() <= tau <= honest.max()


def test_lemma1_bounded_norms():
    U = torch.randn(20, 100)
    U[:9] *= 1e4                                   # 9 < 10 huge malicious updates
    d = TAF(20)
    Ub, norms, tau = d.bound(U)
    assert (Ub.norm(dim=1) <= tau * (1 + 1e-5)).all()
    assert tau <= norms[9:].max()


def test_prop1_influence_bound():
    torch.manual_seed(1)
    n, d_ = 20, 50
    U = torch.randn(n, d_)
    mal = list(range(6))
    U[mal] = -10 * U[mal]
    taf = TAF(n)
    slices = [slice(0, 25), slice(25, 50)]
    for r in range(5):
        delta, info = taf.aggregate(U, _ctx(slices, r))
        p = info["weights"]
        Ub, _, tau = taf.bound(U)
        mal_contrib = (p[mal, None] * Ub[mal]).sum(0).norm()
        assert mal_contrib <= tau * p[mal].sum() + 1e-5


def test_prop3_geometric_decay():
    """A client with S=0 from round t0 on loses trust geometrically (both versions)."""
    for kw in (V1, {}):
        torch.manual_seed(2)
        n, d_ = 11, 40
        base = torch.randn(d_)
        U = base + 0.05 * torch.randn(n, d_)
        taf = TAF(n, lam_up=0.1, lam_down=0.5, **kw)
        c = taf.b if taf.combine == "linear" else taf.b / (taf.a + taf.b)
        slices = [slice(0, 20), slice(20, 40)]
        for r in range(5):
            taf.aggregate(U, _ctx(slices, r))
        H_prev = float(taf.H[0])
        U[0] = -base
        for k in range(1, 8):
            _, info = taf.aggregate(U, _ctx(slices, 5 + k))
            assert float(info["S"][0]) == 0.0
            assert float(info["T"][0]) <= c * (1 - taf.lam_down) ** k * H_prev + 1e-6


def test_prop2_floor_holds_when_assumption_holds():
    torch.manual_seed(3)
    n, d_ = 15, 30
    U = torch.randn(d_) + 0.1 * torch.randn(n, d_)  # tightly aligned honest clients
    taf = TAF(n, **V1)
    slices = [slice(0, 15), slice(15, 30)]
    for r in range(10):
        _, info = taf.aggregate(U, _ctx(slices, r))
        s_min = float(info["S"].min())
        m = min(taf.h0, s_min)
        floor = (taf.a + taf.b) * m - taf.g
        if m > taf.g / (taf.a + taf.b):
            assert (info["T"] >= floor - 1e-6).all()


def test_adaptive_attack_construction():
    torch.manual_seed(4)
    slices = [slice(0, 300), slice(300, 500)]
    honest = torch.randn(12, 500) + 0.5 * torch.randn(500)
    mal_poison = torch.randn(4, 500)
    atk = TrustAdaptive("mnist", s=0.7, knowledge="full")
    U = atk.craft(RoundCtx(honest, honest[:4], mal_poison, slices, 16))
    u_hat = honest.median(0).values
    for sl in slices:
        for u in U:
            assert abs(float(cosine(u[sl], u_hat[sl])) - 0.7) < 1e-4
            assert abs(float(u[sl].norm() - lower_median(honest[:, sl].norm(dim=1)))) < 1e-4
    # pairwise cosine between attackers approximates the honest peer level
    Un = U / U.norm(dim=1, keepdim=True)
    pair = (Un @ Un.T)[~torch.eye(4, dtype=bool)].mean()
    target = max(atk.diag["peer_q"], 0.49)        # floor s^2 when q < s^2
    assert abs(float(pair) - target) < 0.1


def test_collusion_block_is_suppressed():
    """Pilot failure mode: diverse honest clients + a coordinated block.
    v1 gives the block MORE weight than its headcount; v2 must not."""
    torch.manual_seed(5)
    n, d_, n_mal = 30, 200, 9                       # rho = 0.3
    common = torch.randn(d_)
    honest = 0.6 * common + torch.randn(n - n_mal, d_)   # weakly aligned (non-IID)
    block = 0.6 * common + 1.2 * torch.randn(d_)         # shared backdoor direction
    mal = block + 0.05 * torch.randn(n_mal, d_)
    U = torch.cat([mal, honest])
    slices = [slice(0, 100), slice(100, 200)]
    shares = {}
    for name, kw in (("v1", V1), ("v2", {})):
        taf = TAF(n, **kw)
        for r in range(10):
            _, info = taf.aggregate(U, _ctx(slices, r))
        shares[name] = float(info["weights"][:n_mal].sum())
    assert shares["v1"] > n_mal / n                 # reproduces the failure
    assert shares["v2"] < 0.05                      # block is suppressed


def test_flame_distances_symmetric_at_real_scale():
    """Weight-space cosine distances are ~1e-6; float32 GPU rounding once made
    the matrix asymmetric and HDBSCAN refused it (pilot 2, CIFAR-10)."""
    import numpy as np
    from taf.defenses.recent import _cos_dist, _hdbscan
    torch.manual_seed(0)
    G = torch.randn(200_000) * 0.05
    U = torch.randn(40, 200_000) * 1e-4
    U[:12] += torch.randn(200_000) * 3e-4
    D = _cos_dist(G + U)
    assert np.array_equal(D, D.T)
    lab = _hdbscan(D, 21)
    major = np.bincount(lab[lab >= 0]).argmax()
    assert not (lab[:12] == major).any()            # coordinated group excluded


def _v3_share(U, n_mal, rounds=6):
    n, d_ = U.shape
    taf = TAF(n, mode="v3")
    G = torch.randn(d_) * 0.05
    ctx = {"slices": [slice(0, d_ // 2), slice(d_ // 2, d_)], "global_flat": G}
    for r in range(rounds):
        _, info = taf.aggregate(U, {**ctx, "round": r})
    return float(info["weights"][:n_mal].sum())


def test_v3_suppresses_identical_block():
    """ALIE-style: identical malicious updates (FLAME failed this on MNIST)."""
    torch.manual_seed(6)
    n, d_, n_mal = 40, 400, 12
    common = torch.randn(d_)
    honest = 0.5 * common + torch.randn(n - n_mal, d_)
    mal = (honest.mean(0) - 0.3 * honest.std(0)).repeat(n_mal, 1)
    assert _v3_share(torch.cat([mal, honest]), n_mal) < 0.05


def test_v3_suppresses_disguised_clique():
    """Adaptive-attack geometry that beat v2: attackers equidistant from each
    other at the honest peer level, matched norms, sharing a payload."""
    torch.manual_seed(7)
    n, d_, n_mal = 40, 400, 12
    common = torch.randn(d_)
    honest = 0.5 * common + torch.randn(n - n_mal, d_)
    hn = honest.norm(dim=1).median()
    payload = torch.randn(d_)
    mal = 0.5 * common + 0.6 * payload + 0.8 * torch.randn(n_mal, d_)
    mal = mal / mal.norm(dim=1, keepdim=True) * hn
    assert _v3_share(torch.cat([mal, honest]), n_mal) < 0.30 * 0.5


def test_mimic_construction():
    from taf.attacks import Mimic
    torch.manual_seed(8)
    honest = torch.randn(10, 300)
    atk = Mimic("mnist", eps=0.3)
    U = atk.craft(RoundCtx(honest, None, torch.randn(4, 300), [slice(0, 300)], 14))
    best = (U / U.norm(dim=1, keepdim=True)) @ (honest / honest.norm(dim=1, keepdim=True)).T
    assert torch.allclose(best.max(1).values, torch.full((4,), (1 - 0.09) ** 0.5), atol=1e-4)
    assert len(set(best.argmax(1).tolist())) == 4          # distinct honest sources
