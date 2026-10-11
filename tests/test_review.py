"""Checks for code added in the October review round."""
import torch

from taf.defenses import build_defense


def test_foolsgold_zeroes_identical_sybils():
    torch.manual_seed(0)
    n, d, m = 20, 300, 6
    honest = torch.randn(n - m, d)
    sybil = torch.randn(d).repeat(m, 1) + 1e-4 * torch.randn(m, d)
    U = torch.cat([sybil, honest])
    fg = build_defense("FoolsGold", n)
    for r in range(3):
        _, info = fg.aggregate(U, {"round": r})
    w = info["weights"]
    assert float(w[:m].sum()) < 1e-6
    assert float(w[m:].sum()) > 0.99


def test_taf_reputation_updates_only_participants():
    torch.manual_seed(1)
    n, d = 20, 200
    taf = build_defense("TAF", n, mode="v3")
    G = torch.randn(d) * 0.05
    ids = list(range(0, n, 2))
    U = torch.randn(len(ids), d)
    before = taf.R.clone()
    taf.aggregate(U, {"slices": [slice(0, d)], "round": 0, "global_flat": G, "ids": ids})
    others = [i for i in range(n) if i not in ids]
    assert torch.equal(taf.R[others], before[others])
