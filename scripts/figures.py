"""Paper figures from tables/aggregated.csv (output of scripts/tables.py).
Every value plotted is a mean over 3 seeds; error bars are 1 std."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DEFS = ["TAF-v3", "FLAME", "Median", "FedAvg"]
COL = {"TAF-v3": "#2a78d6", "FLAME": "#eb6834", "Median": "#1baf7a", "FedAvg": "#eda100"}
LABEL = {"TAF-v3": "TAF"}
MRK = {"TAF-v3": "o", "FLAME": "s", "Median": "^", "FedAvg": "D"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})


def base(tag):
    return tag.split("@")[0]


def series(ax, df, x, y, ylab, xlab, xticks=None, label_last=True):
    for d in DEFS:
        s = df[df["def"] == d].sort_values(x)
        if s.empty:
            continue
        ax.errorbar(s[x], s[y + "_mean"], yerr=s[y + "_std"], color=COL[d], marker=MRK[d],
                    ms=6, lw=2, capsize=3, elinewidth=1, label=LABEL.get(d, d),
                    mec="white", mew=1, zorder=3)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_ylim(-0.03, 1.03)
    if xticks is not None:
        ax.set_xticks(xticks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="tables")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    A = pd.read_csv(Path(a.tables) / "aggregated.csv")
    A["def"] = A["tag"].map(base)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    main_c = A[(A.group == "main") & (A.dataset == "cifar10")]

    # Fig: mimic attack strength (CIFAR-10)
    s = A[(A.group.isin(["strength", "review_mimic"])) & (A.attack == "MI")].copy()
    s["eps"] = s["tag"].str.split("@").str[1].astype(float)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    series(ax, s, "eps", "asr", "Backdoor success rate", "Mimic tilt ε", [0.1, 0.15, 0.2, 0.3, 0.5])
    ax.set_xticklabels(["0.1", "0.15", "0.2", "0.3", "0.5"])
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "fig_strength_mimic.pdf"); fig.savefig(out / "fig_strength_mimic.png", dpi=200)

    # Fig: attacker fraction rho (BD ASR, CO accuracy), CIFAR-10; rho=0.3 from main
    r = A[A.group == "rho_sweep"].copy()
    r = pd.concat([r, main_c[main_c["def"].isin(DEFS) & main_c.attack.isin(["BD", "CO"])]])
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6))
    series(axs[0], r[r.attack == "BD"], "rho", "asr", "Backdoor success rate", "Fraction of attackers ρ", [0.1, 0.2, 0.3, 0.4])
    axs[0].set_title("(a) Backdoor", fontsize=9, color=INK, loc="left")
    series(axs[1], r[r.attack == "CO"], "rho", "acc", "Clean accuracy", "Fraction of attackers ρ", [0.1, 0.2, 0.3, 0.4])
    axs[1].set_ylim(0, 0.7); axs[1].set_title("(b) Collusion (ALIE)", fontsize=9, color=INK, loc="left")
    axs[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "fig_rho.pdf"); fig.savefig(out / "fig_rho.png", dpi=200)

    # Fig: heterogeneity kappa; kappa=0.5 from main
    k = A[A.group == "noniid"].copy()
    k = pd.concat([k, main_c[main_c["def"].isin(DEFS) & main_c.attack.isin(["none", "BD", "CO"])]])
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for ax, (at, y, lab, t) in zip(axs, [("none", "acc", "Clean accuracy", "(a) No attack"),
                                        ("BD", "asr", "Backdoor success rate", "(b) Backdoor"),
                                        ("CO", "acc", "Clean accuracy", "(c) Collusion (ALIE)")]):
        series(ax, k[k.attack == at], "kappa", y, lab, "Dirichlet κ (log scale)")
        ax.set_xscale("log"); ax.set_xticks([0.1, 0.5, 1.0]); ax.set_xticklabels(["0.1", "0.5", "1.0"])
        if y == "acc":
            ax.set_ylim(0, 0.7)
        ax.set_title(t, fontsize=9, color=INK, loc="left")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out / "fig_kappa.pdf"); fig.savefig(out / "fig_kappa.png", dpi=200)
    print("figures written to", out)


if __name__ == "__main__":
    main()


def trust_share_figure(results, out):
    """Attackers' aggregation weight per round, CIFAR-10, CO and AD (matched).
    Mean over 3 seeds, 5-round moving average, band = 1 std over seeds."""
    import json, glob
    import numpy as np
    col = {"TAF-v3": "#2a78d6", "FLAME": "#eb6834", "DeepSight": "#4a3aa7"}
    mk = {"TAF-v3": "o", "FLAME": "s", "DeepSight": "v"}
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    for ax, (atk, title) in zip(axs, [("CO", "(a) Collusion (ALIE)"), ("AD", "(b) Adaptive attack, s matched")]):
        for d in col:
            g = "main_deepsight" if d == "DeepSight" else "main"
            S = []
            for p in glob.glob(f"{results}/{g}/*.json"):
                L = json.loads(Path(p).read_text()); c = L["config"]
                if c["dataset"] == "cifar10" and c["attack"] == atk and c.get("tag", "").replace("@match", "") == d:
                    S.append([r["mal_share"] for r in L["rounds"]])
            S = np.array(S)
            k = np.ones(5) / 5
            sm = np.array([np.convolve(s, k, mode="valid") for s in S])
            x = np.arange(sm.shape[1]) + 3
            mu, sd = sm.mean(0), sm.std(0)
            ax.fill_between(x, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=col[d], alpha=0.15, lw=0)
            ax.plot(x, mu, color=col[d], lw=2, label=LABEL.get(d, d), marker=mk[d], markevery=12, ms=6, mec="white", mew=1)
        ax.axhline(0.3, color=MUTED, lw=1, ls="--")
        if atk == "CO":
            ax.text(98, 0.28, "attackers' headcount (30%)", color=MUTED, fontsize=7.5, ha="right", va="top")
        ax.set_title(title, fontsize=9, color=INK, loc="left"); ax.set_xlabel("Round")
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_ylim(-0.02, 0.8)
    axs[0].set_ylabel("Attackers' aggregation weight")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(Path(out) / "fig_trust_share.pdf"); fig.savefig(Path(out) / "fig_trust_share.png", dpi=200)
