"""Aggregate a results folder over seeds: mean and std of final accuracy,
ASR and attackers' weight share (mean over second half of training).

  python scripts/tables.py --results results --out tables/
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

GROUPS = ["main", "main_deepsight", "strength", "rho_sweep", "noniid", "ablation",
          "sensitivity", "deepsight_iid", "review_ablation", "review_mimic", "review_clients",
          "review_partial", "review_foolsgold", "review_thresholds", "review_seeds",
          "r2_fg_clip"]


def load(results):
    rows = []
    for g in GROUPS:
        for p in sorted((Path(results) / g).glob("*.json")):
            L = json.loads(p.read_text())
            c, f = L["config"], L["final"]
            sh = [r["mal_share"] for r in L["rounds"] if "mal_share" in r]
            rows.append(dict(group=g, dataset=c["dataset"], attack=c["attack"], defense=c["defense"],
                             tag=c.get("tag", c["defense"]), seed=c["seed"], rho=c["rho"], kappa=c["kappa"],
                             lr=c["lr"], split=c.get("eval_split", "test"), acc=f["acc"],
                             asr=f.get("asr", np.nan),
                             share=np.mean(sh[len(sh) // 2:]) if sh else np.nan,
                             minutes=L["wall_time_s"] / 60,
                             agg_ms=1000 * np.mean([r["agg_time_s"] for r in L["rounds"]])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    df = load(a.results)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_runs.csv", index=False)
    keys = ["group", "dataset", "attack", "tag", "rho", "kappa"]
    df["kappa"] = df["kappa"].astype(float)
    agg = df.groupby(keys, dropna=False).agg(
        n=("seed", "count"), acc_mean=("acc", "mean"), acc_std=("acc", "std"),
        asr_mean=("asr", "mean"), asr_std=("asr", "std"),
        share_mean=("share", "mean"), share_std=("share", "std"),
        agg_ms=("agg_ms", "mean")).reset_index()
    agg.to_csv(out / "aggregated.csv", index=False)
    print(df.groupby("group").agg(runs=("seed", "count"), lr=("lr", "unique"), split=("split", "unique")))


if __name__ == "__main__":
    main()
