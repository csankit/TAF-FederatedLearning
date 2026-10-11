"""Checks for code added in the second feedback round (9 Oct 2026)."""
import torch

from taf.attacks import RoundCtx, build_attack
from taf.defenses import build_defense


def _ctx(n_h=20, m=6, d=400, L=4):
    torch.manual_seed(0)
    H = torch.randn(n_h, d) * 0.1 + 0.05
    clean = torch.randn(m, d) * 0.1 + 0.05
    poison = torch.randn(m, d) * 0.1
    step = d // L
    slices = [slice(i * step, (i + 1) * step) for i in range(L)]
    return RoundCtx(H, clean, poison, slices, n_h + m)


def test_noisy_collusion_cosine_and_norms():
    ctx = _ctx()
    co = build_attack("CO", "mnist").craft(ctx)
    for c in (0.9, 0.5):
        cn = build_attack("CN", "mnist", c=c)
        U = cn.craft(ctx)
        assert U.shape == co.shape
        for sl in ctx.slices:   # same layer norms as the CO update
            assert torch.allclose(U[:, sl].norm(dim=1), co[:, sl].norm(dim=1).expand(U.shape[0]), rtol=1e-4)
        assert abs(cn.diag["pair_cos"] - c) < 0.1
    cm = build_attack("CN", "mnist", c="match")
    cm.craft(ctx)
    assert 0.0 <= cm.diag["c"] <= 1.0


def test_intermittent_schedule():
    ctx = _ctx()
    im = build_attack("IM", "mnist", base="BD", period=4, on_rounds=(3,))
    assert im.needs_clean and im.needs_poison and im.targeted
    im.current_round = 2
    assert torch.equal(im.craft(ctx), ctx.mal_clean)
    im.current_round = 3
    assert torch.allclose(im.craft(ctx), 10.0 * ctx.mal_poison)
    assert [im.is_on(r) for r in range(8)] == [False, False, False, True] * 2


def test_foolsgold_clip_same_weights_bounded_norm():
    torch.manual_seed(2)
    n, d = 20, 300
    U = torch.randn(n, d)
    U[:3] *= 50.0                    # scaled attackers
    a = build_defense("FoolsGold", n)
    b = build_defense("FoolsGold", n, clip=True)
    da, ia = a.aggregate(U, {"round": 0})
    db, ib = b.aggregate(U, {"round": 0})
    assert torch.allclose(ia["weights"], ib["weights"])
    assert float(db.norm()) <= ib["tau"] + 1e-5
    assert float(db.norm()) < float(da.norm())
