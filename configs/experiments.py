"""Experiment groups, one per section of the revised manuscript.

Run order matters: ``tune`` selects TAF's hyperparameters (MNIST, validation
split, LF+BD only). Copy the selected values into TAF_DEFAULT below before
running any other group, so no reported result involves tuning.
"""
from itertools import product

DEFENSES = ["FedAvg", "Krum", "Median", "TrimmedMean", "FLTrust", "FLAME", "DeepSight", "TAF"]
SEEDS5, SEEDS3 = [0, 1, 2, 3, 4], [0, 1, 2]
SCALE = {"mnist": dict(n_clients=50, rounds=50), "cifar10": dict(n_clients=100, rounds=100)}

# >>> Fill from the `tune` group before running anything else. <<<
TAF_DEFAULT = dict(alpha=0.4, beta=0.4, gamma=0.2, lam_up=0.1, lam_down=0.5, h0=0.5)
AD_DEFAULT = dict(s=0.7, knowledge="full")
# The original (submitted) TAF, kept so the paper can report how it fails.
TAF_V1 = dict(detector="iforest", collusion=False, combine="linear", clean_consensus=False)
# Hybrid: majority-cluster gate + collusion + anomaly, acceptance reputation.
TAF_V3 = dict(mode="v3")


def _cfg(dataset, defense, attack, seed, rho=0.3, kappa=0.5, attack_kw=None,
         defense_kw=None, **extra):
    dkw = dict(TAF_DEFAULT) if defense == "TAF" else {}
    if defense == "FLAME":
        dkw["impl"] = 2      # weight-space clustering (paper-faithful); changes run ids
    dkw.update(defense_kw or {})
    akw = dict(AD_DEFAULT) if attack == "AD" else {}
    akw.update(attack_kw or {})
    return dict(dataset=dataset, defense=defense, attack=attack, seed=seed,
                rho=rho if attack != "none" else 0.0, kappa=kappa,
                attack_kw=akw, defense_kw=dkw, **SCALE[dataset], **extra)


def pilot():
    """Run FIRST (8 runs). Measures per-run time on the actual GPU and checks
    whether honest similarities at kappa=0.5 clear the Proposition 2 threshold
    on real data (synthetic data suggested they may not)."""
    return [_cfg(ds, d, a, 0, eval_every=10) for ds, d, a in product(
        ["mnist", "cifar10"], ["FedAvg", "TAF"], ["none", "BD"])]


def pilot2():
    """Method check after the redesign (24 runs, validation split, seed 0):
    FedAvg vs TAF v1 vs TAF v2 vs FLAME under no attack, BD and CO."""
    out = []
    for ds, a in product(["mnist", "cifar10"], ["none", "BD", "CO"]):
        out.append(_cfg(ds, "FedAvg", a, 0, eval_split="val", eval_every=10))
        out.append(_cfg(ds, "FLAME", a, 0, eval_split="val", eval_every=10))
        out.append(_cfg(ds, "TAF", a, 0, eval_split="val", eval_every=10))
        out.append(dict(_cfg(ds, "TAF", a, 0, defense_kw=TAF_V1, eval_split="val",
                             eval_every=10), tag="TAF-v1"))
    # adaptive attack, built against TAF v2: fixed s=0.7, and s matched to honest clients
    for ds in ["mnist", "cifar10"]:
        for d in ["FedAvg", "FLAME", "TAF"]:
            out.append(_cfg(ds, d, "AD", 0, eval_split="val", eval_every=10))
        for d in ["FLAME", "TAF"]:
            out.append(dict(_cfg(ds, d, "AD", 0, attack_kw=dict(s="match"), eval_split="val",
                                 eval_every=10), tag=f"{d}@match"))
    return out


def pilot3():
    """Hybrid check (validation split, seed 0), 26 runs. Reuses pilot2 for
    FedAvg/FLAME under none/BD/CO/AD (read both groups in the summary).
      - TAF v3 under none, BD, CO, AD@match, MI eps 0.2 / 0.5
      - FedAvg and FLAME under MI (the attack aimed at clustering defences)
      - FedAvg under AD@match (missing from pilot2)
      - FedAvg no-attack learning-rate check, lr 0.03 and 0.1 (baseline accuracy)
    """
    out = []
    v = dict(eval_split="val", eval_every=10)
    for ds in ["mnist", "cifar10"]:
        for a, akw in [("none", None), ("BD", None), ("CO", None), ("AD", dict(s="match")),
                       ("MI", dict(eps=0.2)), ("MI", dict(eps=0.5))]:
            tag = "TAF-v3" + ("@match" if a == "AD" else f"@{akw['eps']}" if a == "MI" else "")
            out.append(dict(_cfg(ds, "TAF", a, 0, attack_kw=akw, defense_kw=TAF_V3, **v), tag=tag))
        for d in ["FedAvg", "FLAME"]:
            for eps in (0.2, 0.5):
                out.append(dict(_cfg(ds, d, "MI", 0, attack_kw=dict(eps=eps), **v), tag=f"{d}@{eps}"))
        out.append(dict(_cfg(ds, "FedAvg", "AD", 0, attack_kw=dict(s="match"), **v), tag="FedAvg@match"))
        for lr in (0.03, 0.1):
            out.append(dict(_cfg(ds, "FedAvg", "none", 0, lr=lr, **v), tag=f"FedAvg-lr{lr}"))
    return out


def pilot4():
    """Learning-rate confirmation before freezing v3 (validation split, seed 0,
    lr 0.1 for every method), 32 runs: FedAvg, FLAME, DeepSight, TAF v3 under
    none, BD, AD@match, MI eps 0.5, on both datasets. Also the first real-data
    check of the DeepSight reimplementation."""
    out = []
    v = dict(eval_split="val", eval_every=10, lr=0.1)
    for ds in ["mnist", "cifar10"]:
        for a, akw, sfx in [("none", None, ""), ("BD", None, ""),
                            ("AD", dict(s="match"), "@match"), ("MI", dict(eps=0.5), "@0.5")]:
            for d in ["FedAvg", "FLAME", "DeepSight"]:
                out.append(dict(_cfg(ds, d, a, 0, attack_kw=akw, **v), tag=d + sfx))
            out.append(dict(_cfg(ds, "TAF", a, 0, attack_kw=akw, defense_kw=TAF_V3, **v),
                            tag="TAF-v3" + sfx))
    return out


def tune():
    out = []
    grid = [(a / 10, b / 10) for a in range(1, 9) for b in range(1, 9) if a + b <= 9]
    for (a, b), lam, atk, seed in product(grid, [(0.1, 0.5), (0.05, 0.7)],
                                          ["LF", "BD"], [0]):
        kw = dict(alpha=a, beta=b, gamma=round(1 - a - b, 2), lam_up=lam[0], lam_down=lam[1])
        out.append(_cfg("mnist", "TAF", atk, seed, defense_kw=kw, eval_split="val"))
    return out


# ======================================================================
# Main experiments (v3 frozen 6 Oct 2026 after pilot 4). Every run: lr 0.1,
# test split, seeds 0-2. Secondary studies use CIFAR-10 and four defences to
# fit the compute budget (see docs/BUDGET.md).
# ======================================================================
LR = 0.1
CORE = ["FedAvg", "Median", "FLAME", "TAF"]           # secondary-study defences
MAIN_DEFS = ["FedAvg", "Krum", "Median", "TrimmedMean", "FLTrust", "FLAME", "TAF"]
MAIN_ATTACKS = [("none", None, ""), ("LF", None, ""), ("BD", None, ""), ("GS", None, ""),
                ("CO", None, ""), ("AD", dict(s="match"), "@match")]


def _run(ds, d, a, seed, akw=None, sfx="", dkw=None, tag=None, **kw):
    """One main-experiment run; TAF always means v3 unless dkw overrides."""
    if d == "TAF":
        dk = dict(TAF_V3, **(dkw or {}))
        name = "TAF-v3"
    else:
        dk, name = dkw, d
    c = _cfg(ds, d, a, seed, attack_kw=akw, defense_kw=dk, lr=LR, **kw)
    return dict(c, tag=tag or name + sfx)


def deepsight_diag():
    """DeepSight check (validation split, seed 0), 4 runs: why did it accept
    the boosted backdoor in pilot 4? Logs suspicious flags, TE counts and
    cluster sizes per round."""
    return [_run(ds, "DeepSight", a, 0, eval_split="val", eval_every=10)
            for ds in ["mnist", "cifar10"] for a in ["none", "BD"]]


def deepsight_iid():
    """Is the reimplementation right? Near-IID data (kappa=100), where
    DeepSight's NEUP labelling assumption should hold. 4 runs, val split."""
    return [_run(ds, "DeepSight", "BD", 0, kappa=k, eval_split="val", eval_every=10,
                 tag=f"DeepSight@k{k}") for ds in ["mnist", "cifar10"] for k in (1.0, 100.0)]


def main():
    """Table: 2 datasets x 7 defences x 6 attacks x 3 seeds = 252 runs."""
    return [_run(ds, d, a, s, akw, sfx) for s in SEEDS3 for ds in ["mnist", "cifar10"]
            for d in MAIN_DEFS for a, akw, sfx in MAIN_ATTACKS]


def main_deepsight():
    """DeepSight rows for the main table (36 runs). Run only once
    deepsight_diag shows the implementation behaves as published."""
    return [_run(ds, "DeepSight", a, s, akw, sfx) for s in SEEDS3
            for ds in ["mnist", "cifar10"] for a, akw, sfx in MAIN_ATTACKS]


def strength():
    """Attack-strength sweep (R2), CIFAR-10: mimic eps and adaptive s.
    7 settings x 4 defences x 3 seeds = 84 runs."""
    settings = [("MI", dict(eps=e), f"@{e}") for e in (0.1, 0.2, 0.3, 0.5)]
    settings += [("AD", dict(s=v), f"@s{v}") for v in (0.5, 0.7, 0.9)]
    return [_run("cifar10", d, a, s, akw, sfx) for s in SEEDS3 for d in CORE
            for a, akw, sfx in settings]


def rho_sweep():
    """Fraction of attackers, CIFAR-10 (rho 0.3 comes from main). 72 runs."""
    return [_run("cifar10", d, a, s, rho=r, tag=("TAF-v3" if d == "TAF" else d) + f"@rho{r}")
            for s in SEEDS3 for d in CORE for a in ["BD", "CO"] for r in (0.1, 0.2, 0.4)]


def noniid():
    """Data heterogeneity, CIFAR-10 (kappa 0.5 comes from main). 72 runs."""
    return [_run("cifar10", d, a, s, kappa=k, tag=("TAF-v3" if d == "TAF" else d) + f"@k{k}")
            for s in SEEDS3 for d in CORE for a in ["none", "BD", "CO"] for k in (0.1, 1.0)]


ABLATIONS = {
    "no_cluster_gate": dict(cluster_gate=False),
    "no_collusion": dict(collusion=False),
    "no_anomaly": dict(detector="none"),
    "no_reputation": dict(use_rep=False),
    "no_norm_bound": dict(norm_bound=False),
}


def ablation():
    """v3 components, CIFAR-10 (full v3 comes from main), plus v2 and v1 for
    the redesign story. 7 variants x 3 attacks x 3 seeds = 63 runs."""
    out = []
    for s in SEEDS3:
        for a, akw, sfx in [("BD", None, ""), ("CO", None, ""), ("AD", dict(s="match"), "@match")]:
            for name, kw in ABLATIONS.items():
                out.append(_run("cifar10", "TAF", a, s, akw, dkw=kw, tag=f"abl_{name}{sfx}"))
            out.append(dict(_cfg("cifar10", "TAF", a, s, attack_kw=akw, lr=LR), tag=f"TAF-v2{sfx}"))
            out.append(dict(_cfg("cifar10", "TAF", a, s, attack_kw=akw, defense_kw=TAF_V1, lr=LR),
                            tag=f"TAF-v1{sfx}"))
    return out


def sensitivity():
    """Reputation rates, CIFAR-10 (0.1/0.5 is the default, from main). 24 runs."""
    return [_run("cifar10", "TAF", a, s, akw, dkw=dict(rep_up=u, rep_down=d),
                 tag=f"rep{u}-{d}{sfx}")
            for s in SEEDS3 for a, akw, sfx in [("BD", None, ""), ("AD", dict(s="match"), "@match")]
            for u, d in [(0.05, 0.5), (0.2, 0.5), (0.1, 0.3), (0.1, 0.8)]]


GROUPS = dict(pilot=pilot, pilot2=pilot2, pilot3=pilot3, pilot4=pilot4, tune=tune,
              deepsight_diag=deepsight_diag, deepsight_iid=deepsight_iid, main=main, main_deepsight=main_deepsight,
              strength=strength, rho_sweep=rho_sweep, noniid=noniid, ablation=ablation,
              sensitivity=sensitivity)


# ======================================================================
# Experiments added for the internal review of 8 Oct 2026 (lr 0.1, test
# split, CIFAR-10 unless stated). See docs/REVIEW_EXPERIMENTS.md.
# ======================================================================
AD_M = dict(s="match")


def review_ablation():
    """Both penalties removed together (CIFAR-10 BD, CO, AD) and the single
    and double removals under collusion on MNIST, where FLAME failed. 18 runs."""
    both = dict(collusion=False, detector="none")
    out = [_run("cifar10", "TAF", a, s, akw, dkw=both, tag="abl_no_anom_no_coll" + sfx)
           for s in SEEDS3 for a, akw, sfx in [("BD", None, ""), ("CO", None, ""), ("AD", AD_M, "@match")]]
    for name, kw in [("abl_no_anom_no_coll", both), ("abl_no_collusion", dict(collusion=False)),
                     ("abl_no_anomaly", dict(detector="none"))]:
        out += [_run("mnist", "TAF", "CO", s, dkw=kw, tag=name) for s in SEEDS3]
    return out


def review_mimic():
    """Mimicry at eps 0.15, between the 0.1 and 0.2 of the strength sweep. 12 runs."""
    return [_run("cifar10", d, "MI", s, dict(eps=0.15), "@0.15") for s in SEEDS3 for d in CORE]


THRESH = {"zA2": dict(anom_z0=2.0), "zA4": dict(anom_z0=4.0), "zC1": dict(coll_z0=1.0),
          "zC3": dict(coll_z0=3.0), "k05": dict(coll_k=0.05), "k20": dict(coll_k=0.2)}


def review_thresholds():
    """Anomaly threshold, collusion threshold and peer set size, one at a time
    around the defaults (3, 2, 0.1), under CO and AD. 36 runs."""
    return [_run("cifar10", "TAF", a, s, akw, dkw=kw, tag=f"thr_{name}{sfx}")
            for s in SEEDS3 for name, kw in THRESH.items()
            for a, akw, sfx in [("CO", None, ""), ("AD", AD_M, "@match")]]


def review_partial():
    """Half of the clients sampled each round. 27 runs."""
    return [_run("cifar10", d, a, s, akw, sfx, participation=0.5,
                 tag=("TAF-v3" if d == "TAF" else d) + sfx + "@q0.5")
            for s in SEEDS3 for d in ["FedAvg", "FLAME", "TAF"]
            for a, akw, sfx in [("BD", None, ""), ("CO", None, ""), ("AD", AD_M, "@match")]]


def review_clients():
    """No attack, per client acceptance logging, kappa 0.5 and 0.1. 12 runs."""
    return [_run("cifar10", d, "none", s, kappa=k, tag=("TAF-v3" if d == "TAF" else d) + f"@k{k}")
            for s in SEEDS3 for d in ["FLAME", "TAF"] for k in (0.5, 0.1)]


def review_seeds():
    """Seeds 3 and 4 for the key CIFAR-10 comparisons. 24 runs."""
    return [_run("cifar10", d, a, s, akw, sfx) for s in (3, 4)
            for d in ["FLAME", "DeepSight", "FLTrust", "TAF"]
            for a, akw, sfx in [("BD", None, ""), ("CO", None, ""), ("AD", AD_M, "@match")]]


def review_foolsgold():
    """FoolsGold under the main attacks on CIFAR-10 and collusion on MNIST. 21 runs."""
    out = [_run("cifar10", "FoolsGold", a, s, akw, sfx) for s in SEEDS3 for a, akw, sfx in MAIN_ATTACKS]
    out += [_run("mnist", "FoolsGold", "CO", s) for s in SEEDS3]
    return out


GROUPS.update(review_ablation=review_ablation, review_mimic=review_mimic,
              review_thresholds=review_thresholds, review_partial=review_partial,
              review_clients=review_clients, review_seeds=review_seeds,
              review_foolsgold=review_foolsgold)


# ======================================================================
# Revision 2 (second round of feedback, 9 Oct 2026). CIFAR-10 unless stated,
# test split, rho = 0.3, kappa = 0.5. See docs/REVIEW2_EXPERIMENTS.md.
# ======================================================================
FG_CLIP = dict(clip=True)
EPS5 = (0.1, 0.15, 0.2, 0.3, 0.5)
R2_KEY = [("BD", None, ""), ("CO", None, ""), ("AD", AD_M, "@match")]


def _fg(ds, a, s, akw=None, sfx="", clip=False, **kw):
    return _run(ds, "FoolsGold", a, s, akw, dkw=FG_CLIP if clip else None,
                tag=("FoolsGold+clip" if clip else "FoolsGold") + sfx, **kw)


def r2_fg_clip():
    """FoolsGold with TAF's norm bound: no attack and the five main attacks on
    CIFAR-10, collusion on MNIST, seeds 0 to 2. 21 runs. (Tier 1)"""
    out = [_fg("cifar10", a, s, akw, sfx, clip=True) for s in SEEDS3 for a, akw, sfx in MAIN_ATTACKS]
    out += [_fg("mnist", "CO", s, clip=True) for s in SEEDS3]
    return out


def r2_fg_mimic():
    """Mimicry against FoolsGold and FoolsGold+clip, eps 0.1 to 0.5. 30 runs. (Tier 1)"""
    return [_fg("cifar10", "MI", s, dict(eps=e), f"@{e}", clip=c)
            for s in SEEDS3 for e in EPS5 for c in (False, True)]


def r2_intermittent():
    """Attackers poison one round in four (BD or matched AD) and behave
    honestly otherwise. TAF with and without reputation, FLAME, FedAvg.
    Per client reputation, gate and weight logged for TAF. 24 runs. (Tier 1)"""
    out = []
    for s in SEEDS3:
        for base, bkw, sfx in [("BD", None, "BD"), ("AD", dict(AD_DEFAULT, s="match"), "AD@match")]:
            akw = dict(base=base, base_kw=bkw, period=4, on_rounds=(3,))
            for d, dkw, name in [("FedAvg", None, "FedAvg"), ("FLAME", None, "FLAME"),
                                 ("TAF", None, "TAF-v3"), ("TAF", dict(use_rep=False), "abl_no_reputation")]:
                out.append(_run("cifar10", d, "IM", s, akw, dkw=dkw, tag=f"{name}@IM-{sfx}",
                                log_rep=(d == "TAF")))
    return out


def r2_noisy_co():
    """Coordinated, nonidentical collusion: ALIE target plus client noise with
    pairwise cosine c = 0.9 or matched to honest peers. 36 runs. (Tier 2)"""
    out = []
    for s in SEEDS3:
        for c in (0.9, "match"):
            akw, sfx = dict(c=c), f"@c{c}"
            for d in ["FedAvg", "Median", "FLAME", "TAF"]:
                out.append(_run("cifar10", d, "CN", s, akw, sfx))
            out.append(_fg("cifar10", "CN", s, akw, sfx))
            out.append(_fg("cifar10", "CN", s, akw, sfx, clip=True))
    return out


def r2_seeds():
    """Independent evaluation seeds 5 to 7 for TAF, FLAME, FoolsGold and
    FoolsGold+clip under BD, CO and matched AD, plus seeds 3 and 4 for the two
    FoolsGold variants. With the existing runs this gives seeds 1 to 7 for all
    four rules (seed 0 was used in development). 48 runs. (Tier 2)"""
    out = []
    for s in (5, 6, 7):
        for a, akw, sfx in R2_KEY:
            out += [_run("cifar10", d, a, s, akw, sfx) for d in ["FLAME", "TAF"]]
            out += [_fg("cifar10", a, s, akw, sfx), _fg("cifar10", a, s, akw, sfx, clip=True)]
    for s in (3, 4):
        for a, akw, sfx in R2_KEY:
            out += [_fg("cifar10", a, s, akw, sfx), _fg("cifar10", a, s, akw, sfx, clip=True)]
    return out


def r2_partial():
    """Participation 25% and 75% (50% is review_partial). 54 runs. (Tier 3)"""
    return [_run("cifar10", d, a, s, akw, sfx, participation=q, log_rep=(d == "TAF"),
                 tag=("TAF-v3" if d == "TAF" else d) + sfx + f"@q{q}")
            for s in SEEDS3 for q in (0.25, 0.75) for d in ["FedAvg", "FLAME", "TAF"]
            for a, akw, sfx in R2_KEY]


GROUPS.update(r2_fg_clip=r2_fg_clip, r2_fg_mimic=r2_fg_mimic, r2_intermittent=r2_intermittent,
              r2_noisy_co=r2_noisy_co, r2_seeds=r2_seeds, r2_partial=r2_partial)
