from .classic import FedAvg, FLTrust, Krum, Median, TrimmedMean
from .recent import FLAME, DeepSight, FoolsGold
from .taf import TAF

DEFENSES = {d.name: d for d in (FedAvg, Krum, Median, TrimmedMean, FLTrust, FLAME, DeepSight, FoolsGold, TAF)}


def build_defense(name, n_clients, seed=0, **kw):
    cls = DEFENSES[name]
    if cls is TAF:
        return cls(n_clients=n_clients, seed=seed, **kw)
    if cls is FoolsGold:
        return cls(n_clients=n_clients, **kw)
    return cls(**kw) if kw else cls()
