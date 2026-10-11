"""Every defence x attack runs end-to-end on tiny synthetic data."""
import itertools
import math

import pytest

from taf.defenses import DEFENSES
from taf.runner import run

ATTACKS = ["none", "LF", "BD", "GS", "CO", "AD"]


@pytest.mark.parametrize("defense,attack", list(itertools.product(DEFENSES, ATTACKS)))
def test_runs(defense, attack):
    log = run(dict(dataset="fake_mnist", n_clients=10, rounds=2, eval_every=1, rho=0.3,
                   attack=attack, defense=defense, val_size=100, root_size=50,
                   device="cpu", seed=0), verbose=False)
    assert math.isfinite(log["final"]["acc"])
    if attack in ("LF", "BD", "AD"):
        assert "asr" in log["final"]
    if attack != "none" and defense not in ("Median", "TrimmedMean"):
        assert 0.0 <= log["rounds"][-1]["mal_share"] <= 1.0 + 1e-6


def test_taf_matches_fedavg_when_iid():
    """With near-IID data and no attack, TAF should track FedAvg (Section 4.5)."""
    accs = {}
    for d in ("FedAvg", "TAF"):
        log = run(dict(dataset="fake_mnist", n_clients=10, rounds=10, eval_every=10,
                       defense=d, device="cpu", val_size=100, root_size=50,
                       lr=0.01, kappa=100.0), verbose=False)
        accs[d] = log["final"]["acc"]
    assert accs["FedAvg"] > 0.5
    assert abs(accs["TAF"] - accs["FedAvg"]) < 0.05
