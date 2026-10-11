"""Run one federated experiment and write a JSON log.

A seed fixes: data partition, malicious client set, model init, held-out
splits. Runs with the same seed are therefore paired across defences
(Section 5.5).
"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from .attacks import RoundCtx, build_attack
from .client import evaluate, local_update, predict
from .fastclient import train_many
from .data import dirichlet_partition, load_dataset
from .defenses import build_defense
from .models import build_model
from .utils import get_flat, layer_slices, set_flat, set_seed

DEFAULTS = dict(
    dataset="mnist", n_clients=50, kappa=0.5, rho=0.3, attack="none",
    attack_kw={}, defense="FedAvg", defense_kw={}, rounds=50, local_epochs=1,
    lr=0.01, bs=64, server_lr=1.0, seed=0, eval_every=5, root_size=100,
    val_size=5000, eval_split="test", participation=1.0, data_root="./data", device="auto", chunk=32, trainer="vmap",
)


def _sync(dev):
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def run(cfg: dict, out_path: str | None = None, verbose=True) -> dict:
    cfg = {**DEFAULTS, **cfg}
    dev = cfg["device"]
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    # ---------------- data ----------------
    train, test = load_dataset(cfg["dataset"], cfg["data_root"], dev)
    perm = rng.permutation(len(train))
    vs = min(cfg["val_size"], len(train) // 10)
    rs = cfg["root_size"]
    val, root = train.subset(perm[:vs]), train.subset(perm[vs:vs + rs])
    pool = train.subset(perm[vs + rs:])
    evald = val if cfg["eval_split"] == "val" else test
    labels = pool.y.cpu().numpy()
    parts = dirichlet_partition(labels, cfg["n_clients"], cfg["kappa"], rng,
                                min_size=min(10, len(pool) // (2 * cfg["n_clients"])))
    N = cfg["n_clients"]
    n_mal = int(round(cfg["rho"] * N)) if cfg["attack"] != "none" else 0
    mal = set(rng.permutation(N)[:n_mal].tolist())
    honest_ids = [i for i in range(N) if i not in mal]
    mal_ids = sorted(mal)

    # label-distribution divergence of each client from the global one
    glob = np.bincount(labels, minlength=10) / len(labels)
    jsd = np.array([jensenshannon(np.bincount(labels[p], minlength=10) / len(p), glob)
                    for p in parts])

    # ---------------- model / attack / defence ----------------
    model = build_model(cfg["dataset"]).to(dev)
    gflat = get_flat(model)
    slices = layer_slices(model)
    attack = build_attack(cfg["attack"], cfg["dataset"], **cfg["attack_kw"])
    defense = build_defense(cfg["defense"], N, seed=cfg["seed"], **cfg["defense_kw"])
    asr_x, asr_t = (attack.asr_inputs(evald) or (None, None)) if attack.targeted else (None, None)

    client_data = [pool.subset(p) for p in parts]
    poisoned = {i: attack.poison_data(client_data[i].x, client_data[i].y) for i in mal_ids}
    probe_model = deepcopy(model)

    def softmax(flat, x):
        set_flat(probe_model, flat)
        probe_model.eval()
        with torch.no_grad():
            return F.softmax(probe_model(x), 1)

    gen = torch.Generator().manual_seed(cfg["seed"])
    lu = dict(epochs=cfg["local_epochs"], lr=cfg["lr"], bs=cfg["bs"], gen=gen)
    log = {"config": cfg, "malicious": mal_ids, "rounds": [], "evals": []}
    trust_sum = np.zeros(N)
    t0 = time.time()

    q = float(cfg.get("participation", 1.0))
    part_rng = np.random.default_rng(cfg["seed"] + 7919)
    n_part = np.zeros(N); n_acc = np.zeros(N); w_sum = np.zeros(N)
    all_honest, all_mal = honest_ids, mal_ids

    for r in range(cfg["rounds"]):
        if q < 1.0:   # partial participation: a fresh random subset each round
            k = max(4, int(round(q * N)))
            sel = sorted(part_rng.choice(N, size=k, replace=False).tolist())
        else:
            sel = list(range(N))
        pos = {c: j for j, c in enumerate(sel)}            # client id -> row in U
        honest_ids = [c for c in all_honest if c in pos]
        mal_ids = [c for c in all_mal if c in pos]
        # one batched training call: honest, then malicious clean / poisoned
        jobs = [(client_data[i].x, client_data[i].y) for i in honest_ids]
        if mal_ids and attack.needs_clean:
            jobs += [(client_data[i].x, client_data[i].y) for i in mal_ids]
        if mal_ids and attack.needs_poison:
            jobs += [poisoned[i] for i in mal_ids]
        out = train_many(model, gflat, jobs, epochs=cfg["local_epochs"], lr=cfg["lr"],
                         bs=cfg["bs"], gen=gen, chunk=cfg["chunk"],
                         trainer=cfg["trainer"])
        nh, nm = len(honest_ids), len(mal_ids)
        upd = {i: out[k] for k, i in enumerate(honest_ids)}
        mal_clean = mal_poison = None
        if mal_ids:
            off = nh
            if attack.needs_clean:
                mal_clean, off = out[off:off + nm], off + nm
            if attack.needs_poison:
                mal_poison = out[off:off + nm]
            H = out[:nh]
            if hasattr(attack, "current_round"):
                attack.current_round = r
            crafted = attack.craft(RoundCtx(H, mal_clean, mal_poison, slices, len(sel)))
            for k, i in enumerate(mal_ids):
                upd[i] = crafted[k]
        U = torch.stack([upd[i] for i in sel])
        hid = [pos[c] for c in honest_ids]               # row indices
        mid = [pos[c] for c in mal_ids]

        ctx = {"ids": sel, "rho": cfg["rho"], "slices": slices, "round": r, "global_flat": gflat,
               "num_classes": 10, "input_shape": tuple(train.x.shape[1:]),
               "softmax": softmax,
               "server_update": lambda: local_update(model, gflat, root.x, root.y, **lu)}
        _sync(dev)
        ts = time.perf_counter()
        delta, info = defense.aggregate(U, ctx)
        _sync(dev)
        agg_time = time.perf_counter() - ts
        gflat = gflat + cfg["server_lr"] * delta

        rec = {"round": r, "agg_time_s": agg_time, "n_part": len(sel), "n_mal_part": len(mal_ids)}
        n_part[sel] += 1
        w = info.get("weights")
        if w is not None:
            w = torch.as_tensor(w).float()
            n_acc[sel] += (w > 0).numpy(); w_sum[sel] += w.numpy()
            rec["mal_share"] = float(w[mid].sum()) if mid else 0.0
        if "T" in info:  # TAF diagnostics
            S = info["S"].numpy()
            trust_sum[sel] += info["T"].numpy()
            rec["S_honest_q"] = np.quantile(S[hid], [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
            rec["T_honest_mean"] = float(info["T"][hid].mean())
            if mal_ids:
                rec["T_mal_mean"] = float(info["T"][mid].mean())
                rec["A_mal_mean"] = float(info["A"][mid].mean())
            rec["A_honest_mean"] = float(info["A"][hid].mean())
            rec["C_honest_mean"] = float(info["C"][hid].mean())
            if "M" in info:
                rec["M_honest_mean"] = float(info["M"][hid].mean())
                if mal_ids:
                    rec["M_mal_mean"] = float(info["M"][mid].mean())
            if mal_ids:
                rec["C_mal_mean"] = float(info["C"][mid].mean())
            rec["tau"] = info["tau"]
            if "H" in info and getattr(defense, "mode", None) == "v3":   # reputation
                Rr = torch.as_tensor(info["H"]).float()
                rec["R_honest_mean"] = float(Rr[hid].mean())
                if mal_ids:
                    rec["R_mal_mean"] = float(Rr[mid].mean())
                if cfg.get("log_rep"):
                    full = np.full(N, np.nan)
                    full[sel] = Rr.numpy()
                    rec["R"] = [None if not np.isfinite(v) else round(float(v), 4) for v in full]
                    rec["M_all"] = [None] * N
                    if "M" in info:
                        mm = np.full(N, np.nan); mm[sel] = torch.as_tensor(info["M"]).float().numpy()
                        rec["M_all"] = [None if not np.isfinite(v) else int(v > 0.5) for v in mm]
                    rec["w_all"] = [0.0] * N
                    if w is not None:
                        ww = np.zeros(N); ww[sel] = w.numpy()
                        rec["w_all"] = [round(float(v), 5) for v in ww]
        if "ds_suspicious" in info:  # DeepSight diagnostics
            sus = info["ds_suspicious"]
            rec["ds_sus_honest"] = float(sus[hid].mean())
            if mal_ids:
                rec["ds_sus_mal"] = float(sus[mid].mean())
                rec["ds_te_mal"] = float(info["ds_te"][mid].mean())
            rec["ds_te_honest"] = float(info["ds_te"][hid].mean())
            rec["ds_n_clusters"] = info["ds_n_clusters"]
            rec["ds_cluster_sizes"] = info["ds_cluster_sizes"]
        if mal_ids and getattr(attack, "diag", None):
            rec.update({f"atk_{k}": v for k, v in attack.diag.items()})
        log["rounds"].append(rec)

        if (r + 1) % cfg["eval_every"] == 0 or r == cfg["rounds"] - 1:
            ev = {"round": r + 1, "acc": evaluate(model, gflat, evald)}
            if asr_x is not None and len(asr_x):
                ev["asr"] = float((predict(model, gflat, asr_x) == asr_t).float().mean())
            log["evals"].append(ev)
            if verbose:
                print(f"[{cfg['defense']}/{cfg['attack']}] r{r+1} acc={ev['acc']:.4f}"
                      + (f" asr={ev['asr']:.4f}" if "asr" in ev else ""), flush=True)
            if not np.isfinite(gflat.cpu().numpy()).all():
                log["diverged"] = r + 1
                break

    log["final"] = log["evals"][-1]
    log["client_stats"] = {"jsd": jsd.tolist(), "participated": n_part.tolist(),
                           "accepted": n_acc.tolist(), "weight_sum": w_sum.tolist(),
                           "malicious": all_mal}
    log["wall_time_s"] = time.time() - t0
    honest_ids, mal_ids = all_honest, all_mal
    if trust_sum.any() and len(honest_ids) > 2:
        rho_s, p = spearmanr(trust_sum[honest_ids], jsd[honest_ids])
        log["benign_penalty_spearman"] = {"rho": float(rho_s), "p": float(p)}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(log))
    return log
