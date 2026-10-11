"""Summarise the pilot: per-run cost, projected budget, and the Prop. 2 check.

  python scripts/summarize_pilot.py --results /content/drive/MyDrive/taf_results
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.experiments import GROUPS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    logs = [json.loads(p.read_text()) for p in sorted((Path(a.results) / "pilot").glob("*.json"))]
    if not logs:
        sys.exit("no pilot results found")
    per_ds = {}
    print(f"{'dataset':8} {'defense':7} {'attack':5} {'min':>6} {'acc':>6} {'asr':>6}  honest S (5/50/95%)  frac>thr")
    for L in logs:
        c, f = L["config"], L["final"]
        mins = L["wall_time_s"] / 60
        per_ds.setdefault(c["dataset"], []).append(mins)
        line = f"{c['dataset']:8} {c['defense']:7} {c['attack']:5} {mins:6.1f} {f['acc']:6.3f} " \
               f"{f.get('asr', float('nan')):6.3f}"
        if c["defense"] == "TAF":
            kw = {"alpha": 0.4, "beta": 0.4, "gamma": 0.2, **c["defense_kw"]}  # TAF defaults
            thr = kw["gamma"] / (kw["alpha"] + kw["beta"])
            q = np.array([r["S_honest_q"] for r in L["rounds"][len(L["rounds"]) // 2:]])
            med = q.mean(0)
            # fraction of rounds whose honest median clears the threshold
            frac = float((q[:, 2] > thr).mean())
            line += f"  {med[0]:.2f}/{med[2]:.2f}/{med[4]:.2f}       {frac:.2f} (thr={thr:.2f})"
        print(line)

    print("\nProjected sequential GPU time per group (from pilot wall time):")
    rate = {ds: float(np.mean(v)) for ds, v in per_ds.items()}
    total = 0.0
    for g, fn in GROUPS.items():
        if g == "pilot":
            continue
        h = sum(rate.get(c["dataset"], 0) for c in fn()) / 60
        total += h
        print(f"  {g:12s} {h:7.1f} h")
    print(f"  {'TOTAL':12s} {total:7.1f} h  (DeepSight/AD runs are slower than the pilot average;"
          " parallel workers reduce wall time, not GPU-hours)")


if __name__ == "__main__":
    main()
