"""Compact summary of any group: final accuracy / ASR per run, and TAF
trust diagnostics (honest vs malicious trust, anomaly, collusion, share).

  python scripts/summarize.py --results $RES --group pilot2
"""
import argparse
import json
from pathlib import Path

import numpy as np


def label(c):
    lab = c.get("tag") or c["defense"]
    if c["defense"] == "FLAME" and c["defense_kw"].get("impl") != 2:
        lab = lab.replace("FLAME", "FLAME(old)")   # pre-fix update-space clustering
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--group", nargs="+", default=["pilot2"])
    a = ap.parse_args()
    logs, fails = [], []
    for g in a.group:
        d = Path(a.results) / g
        logs += [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
        fails += sorted(d.glob("*.FAILED.txt"), key=lambda p: p.stat().st_mtime)
    print(f"{len(logs)} finished, {len(fails)} failed in {a.group}")
    for f in fails[-1:]:
        print("LATEST FAILURE", f.name, "\n", f.read_text()[-800:])

    rows = sorted(logs, key=lambda L: (L["config"]["dataset"], L["config"]["attack"], label(L["config"])))
    print(f"\n{'dataset':8} {'attack':6} {'defense':16} {'acc':>6} {'asr':>6} {'mal_share':>9} {'payload':>7} {'min':>5}")
    for L in rows:
        c, f = L["config"], L["final"]
        shares = [r["mal_share"] for r in L["rounds"] if "mal_share" in r]
        sh = np.mean(shares[len(shares) // 2:]) if shares else float("nan")
        pl = [r["atk_payload"] for r in L["rounds"] if "atk_payload" in r]
        pl = np.mean(pl) if pl else float("nan")
        print(f"{c['dataset']:8} {c['attack']:6} {label(c):16} {f['acc']:6.3f} "
              f"{f.get('asr', float('nan')):6.3f} {sh:9.3f} {pl:7.2f} {L['wall_time_s']/60:5.1f}")

    ds = [L for L in rows if L["config"]["defense"] == "DeepSight"
          and any("ds_sus_honest" in r for r in L["rounds"])]
    if ds:
        print("\nDeepSight diagnostics (mean over second half; sus = flagged suspicious, TE = threshold exceedings):")
        print(f"{'dataset':8} {'attack':6} {'sus_hon':>7} {'sus_mal':>7} {'TE_hon':>6} {'TE_mal':>6} "
              f"{'n_clu':>5}  largest clusters (last round)")
        for L in ds:
            c = L["config"]
            R = [r for r in L["rounds"][len(L["rounds"]) // 2:] if "ds_sus_honest" in r]
            m = lambda k: np.mean([r[k] for r in R if k in r]) if any(k in r for r in R) else float("nan")
            print(f"{c['dataset']:8} {c['attack']:6} {m('ds_sus_honest'):7.2f} {m('ds_sus_mal'):7.2f} "
                  f"{m('ds_te_honest'):6.2f} {m('ds_te_mal'):6.2f} {m('ds_n_clusters'):5.1f}  "
                  f"{R[-1]['ds_cluster_sizes']}")

    print("\nTAF diagnostics (mean over second half of training):")
    print(f"{'dataset':8} {'attack':6} {'version':16} {'T_hon':>6} {'T_mal':>6} {'A_hon':>6} "
          f"{'A_mal':>6} {'C_hon':>6} {'C_mal':>6} {'M_hon':>6} {'M_mal':>6}")
    for L in rows:
        c = L["config"]
        if c["defense"] != "TAF":
            continue
        R = L["rounds"][len(L["rounds"]) // 2:]
        m = lambda k: np.mean([r[k] for r in R if k in r]) if any(k in r for r in R) else float("nan")
        print(f"{c['dataset']:8} {c['attack']:6} {label(c):16} {m('T_honest_mean'):6.3f} "
              f"{m('T_mal_mean'):6.3f} {m('A_honest_mean'):6.2f} {m('A_mal_mean'):6.2f} "
              f"{m('C_honest_mean'):6.2f} {m('C_mal_mean'):6.2f} "
              f"{m('M_honest_mean'):6.2f} {m('M_mal_mean'):6.2f}")


if __name__ == "__main__":
    main()
