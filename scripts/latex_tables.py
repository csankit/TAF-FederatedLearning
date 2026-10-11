"""Write LaTeX table bodies for the paper from tables/aggregated.csv and
tables/significance.csv. Values are percentages (one decimal) of the mean
over 3 seeds, followed by the standard deviation."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

T = Path(sys.argv[1] if len(sys.argv) > 1 else "tables")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "latex")
OUT.mkdir(parents=True, exist_ok=True)
A = pd.read_csv(T / "aggregated.csv")
A["base"] = A.tag.str.replace("@match", "", regex=False)
SIG = pd.read_csv(T / "significance.csv")
ORDER = ["FedAvg", "Krum", "Median", "TrimmedMean", "FLTrust", "FLAME", "DeepSight", "TAF-v3"]
NAME = {"TrimmedMean": "Trimmed mean", "TAF-v3": "TAF (ours)", "FoolsGold+clip": "FoolsGold + norm bound"}


def cell(r, m, best=False, sd=True):
    if r is None or pd.isna(r[m + "_mean"]):
        return "--" if False else "n/a"
    v = f"{100 * r[m + '_mean']:.1f}"
    if best:
        v = r"\textbf{" + v + "}"
    if sd and not pd.isna(r[m + "_std"]):
        v += r"\,$\pm$\," + f"{100 * r[m + '_std']:.1f}"
    return v


def fmt_signed(v):
    v = round(v, 1)
    if v == 0:
        return "0.0"
    return ("$+$" if v > 0 else "$-$") + f"{abs(v):.1f}"


def get(crit):
    s = A
    for k, v in crit.items():
        s = s[s[k] == v]
    return None if s.empty else s.iloc[0]


def main_rows(ds, atk):
    g = lambda d: get({"group": "main_deepsight" if d == "DeepSight" else "main",
                       "dataset": ds, "attack": atk, "base": d})
    return {d: g(d) for d in ORDER}


def write(name, s):
    (OUT / name).write_text(s)


# Table: clean accuracy, both datasets
lines = []
for ds, lab in [("mnist", "MNIST"), ("cifar10", "CIFAR-10")]:
    lines.append(r"\multicolumn{7}{l}{\textit{" + lab + r"}}\\")
    cols = ["none", "LF", "BD", "GS", "CO", "AD"]
    R = {a: main_rows(ds, a) for a in cols}
    best = {a: max(round(100 * R[a][d]["acc_mean"], 1) for d in ORDER) for a in cols}
    for d in ORDER:
        lines.append(NAME.get(d, d) + " & " + " & ".join(
            cell(R[a][d], "acc", round(100 * R[a][d]["acc_mean"], 1) == best[a]) for a in cols) + r" \\")
    if ds == "mnist":
        lines.append(r"\midrule")
write("tab_acc.tex", "\n".join(lines))

# Table: ASR of targeted attacks, both datasets side by side
lines = []
R = {(ds, a): main_rows(ds, a) for ds in ("mnist", "cifar10") for a in ("LF", "BD", "AD")}
best = {k: min(round(100 * R[k][d]["asr_mean"], 1) for d in ORDER) for k in R}
for d in ORDER:
    lines.append(NAME.get(d, d) + " & " + " & ".join(
        cell(R[(ds, a)][d], "asr", round(100 * R[(ds, a)][d]["asr_mean"], 1) == best[(ds, a)]) for ds in ("mnist", "cifar10") for a in ("LF", "BD", "AD")) + r" \\")
write("tab_asr.tex", "\n".join(lines))

# Table: attackers' weight share (means only)
lines = []
for d in ["FedAvg", "Krum", "FLTrust", "FLAME", "DeepSight", "TAF-v3"]:
    vals = []
    for ds in ("mnist", "cifar10"):
        for a in ("LF", "BD", "GS", "CO", "AD"):
            r = main_rows(ds, a)[d]
            vals.append(f"{100 * r['share_mean']:.1f}")
    lines.append(NAME.get(d, d) + " & " + " & ".join(vals) + r" \\")
write("tab_share.tex", "\n".join(lines))

# Table: TAF minus FLAME and minus best other baseline, paired test with Holm
lines = []
for ds, lab in [("mnist", "MNIST"), ("cifar10", "CIFAR-10")]:
    for a, met in [("none", "acc"), ("LF", "asr"), ("BD", "asr"), ("GS", "acc"), ("CO", "acc"), ("AD", "asr")]:
        s = SIG[(SIG.dataset == ds) & (SIG.attack == a) & (SIG.metric == met)]
        f = s[s.baseline == "FLAME"].iloc[0]
        metlab = "accuracy" if met == "acc" else "ASR"
        atk = {"none": "No attack", "LF": "LF", "BD": "BD", "GS": "GS", "CO": "CO", "AD": "AD"}[a]
        p = f["p_holm"]
        ptxt = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        lines.append(f"{lab} & {atk} & {metlab} & {fmt_signed(100 * f['taf_minus_base'])} & {ptxt}" + r" \\")
    if ds == "mnist":
        lines.append(r"\midrule")
write("tab_sig.tex", "\n".join(lines))

# Table: ablation (CIFAR-10)
V = [("Full TAF", {"group": "main", "dataset": "cifar10", "base": "TAF-v3"}),
     ("Without cluster gate", {"group": "ablation", "base": "abl_no_cluster_gate"}),
     ("Without reputation", {"group": "ablation", "base": "abl_no_reputation"}),
     ("Without anomaly penalty", {"group": "ablation", "base": "abl_no_anomaly"}),
     ("Without collusion penalty", {"group": "ablation", "base": "abl_no_collusion"}),
     ("Without norm bound", {"group": "ablation", "base": "abl_no_norm_bound"}),
     ("Without both penalties", {"group": "review_ablation", "dataset": "cifar10", "base": "abl_no_anom_no_coll"}),
     ("Previous revision (v2)", {"group": "ablation", "base": "TAF-v2"}),
     ("Original submission (v1)", {"group": "ablation", "base": "TAF-v1"})]
lines = []
for lab, c in V:
    rb, rc, ra = (get({**c, "attack": x}) for x in ("BD", "CO", "AD"))
    lines.append(lab + " & " + " & ".join([cell(rb, "acc"), cell(rb, "asr"), cell(rc, "acc"),
                                            f"{100 * rc['share_mean']:.1f}", cell(ra, "acc"), cell(ra, "asr"),
                                            f"{100 * ra['share_mean']:.1f}"]) + r" \\")
    if lab == "Full TAF" or lab == "Without both penalties":
        lines.append(r"\midrule")
write("tab_ablation.tex", "\n".join(lines))

# Table: sensitivity to reputation rates
S = [("0.10 / 0.50 (default)", {"group": "main", "dataset": "cifar10", "base": "TAF-v3"})]
S += [(f"{u:.2f} / {d:.2f}", {"group": "sensitivity", "base": f"rep{u}-{d}"}) for u, d in
      [(0.05, 0.5), (0.2, 0.5), (0.1, 0.3), (0.1, 0.8)]]
lines = []
for lab, c in S:
    rb, ra = get({**c, "attack": "BD"}), get({**c, "attack": "AD"})
    lines.append(lab + " & " + " & ".join([cell(rb, "acc"), cell(rb, "asr"), cell(ra, "acc"), cell(ra, "asr")]) + r" \\")
write("tab_sens.tex", "\n".join(lines))

# Table: fixed s adaptive attack (CIFAR-10)
lines = []
for d in ["FedAvg", "Median", "FLAME", "TAF-v3"]:
    vals = [cell(get({"group": "strength", "tag": f"{d}@s{s}"}), "asr") for s in (0.5, 0.7, 0.9)]
    vals.append(cell(get({"group": "main", "dataset": "cifar10", "tag": f"{d}@match"}), "asr"))
    lines.append(NAME.get(d, d) + " & " + " & ".join(vals) + r" \\")
write("tab_adfixed.tex", "\n".join(lines))

# Table: overhead (ms per round)
lines = []
for d in ORDER:
    v = []
    for ds in ("mnist", "cifar10"):
        g = "main_deepsight" if d == "DeepSight" else "main"
        s = A[(A.group == g) & (A.dataset == ds) & (A.base == d)]
        v.append(f"{s.agg_ms.mean():.0f}")
    lines.append(NAME.get(d, d) + " & " + " & ".join(v) + r" \\")
write("tab_overhead.tex", "\n".join(lines))

# Table: DeepSight check
lines = []
for ds, lab in [("mnist", "MNIST"), ("cifar10", "CIFAR-10")]:
    v = []
    for k in ("1.0", "100.0"):
        r = get({"group": "deepsight_iid", "dataset": ds, "tag": f"DeepSight@k{k}"})
        v += [f"{100 * r['asr_mean']:.1f}", f"{100 * r['share_mean']:.1f}"]
    lines.append(lab + " & " + " & ".join(v) + r" \\")
write("tab_deepsight.tex", "\n".join(lines))
print("written", sorted(p.name for p in OUT.iterdir()))


# ---------------- tables added in the review round ----------------
import json
from scipy import stats as _st
RUNS = pd.read_csv(T / "all_runs.csv")
RUNS["base"] = RUNS.tag.str.replace("@match", "", regex=False)
RES = Path(sys.argv[3]) if len(sys.argv) > 3 else None

# FoolsGold against TAF on CIFAR-10 (all attacks) and MNIST collusion
lines = []
ATK6 = [("none", "No attack"), ("LF", "LF"), ("BD", "BD"), ("GS", "GS"), ("CO", "CO"), ("AD", "AD (matched)")]
for d, g in [("TAF-v3", "main"), ("FoolsGold", "review_foolsgold"), ("FoolsGold+clip", "r2_fg_clip")]:
    cells = []
    for a, _ in ATK6:
        r = get({"group": g, "dataset": "cifar10", "base": d, "attack": a})
        txt = cell(r, "acc")
        if a in ("LF", "BD", "AD"):
            txt += " / " + cell(r, "asr")
        cells.append(txt)
    lines.append(NAME.get(d, d) + " & " + " & ".join(cells) + r" \\")
write("tab_foolsgold.tex", "\n".join(lines))

# Attackers' weight share (second half mean, %) and paired tests, FoolsGold variants vs TAF
lines = []
for d, g in [("TAF-v3", "main"), ("FoolsGold", "review_foolsgold"), ("FoolsGold+clip", "r2_fg_clip")]:
    cells = [cell(get({"group": g, "dataset": "cifar10", "base": d, "attack": a}), "share") for a, _ in ATK6[1:]]
    lines.append(NAME.get(d, d) + " & " + " & ".join(cells) + r" \\")
write("tab_fg_share.tex", "\n".join(lines))
_R = RUNS[(RUNS.dataset == "cifar10") & (RUNS.seed <= 2)]
_t = _R[(_R.group == "main") & (_R.base == "TAF-v3")]
tests = []
for d, g in [("FoolsGold", "review_foolsgold"), ("FoolsGold+clip", "r2_fg_clip")]:
    _b = _R[(_R.group == g) & (_R.base == d)]
    for a, lab in ATK6:
        met = "asr" if a in ("LF", "BD", "AD") else "acc"
        x = _t[_t.attack == a].set_index("seed")[met]; y = _b[_b.attack == a].set_index("seed")[met]
        idx = x.index.intersection(y.index)
        diff = 100 * (x[idx] - y[idx])
        p = _st.ttest_rel(x[idx], y[idx]).pvalue if diff.std() > 0 else float("nan")
        tests.append([NAME.get(d, d), lab, "ASR" if met == "asr" else "accuracy", diff.mean(), p])
_ps = np.array([1.0 if not np.isfinite(t[4]) else t[4] for t in tests]); _o = np.argsort(_ps); _a = np.empty(len(_ps)); _r = 0.0
for _k, _i in enumerate(_o):
    _r = max(_r, (len(_ps) - _k) * _ps[_i]); _a[_i] = min(1.0, _r)
_pf = lambda p: "n/a" if not np.isfinite(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")
lines = [f"{n} & {lab} & {met} & {fmt_signed(m)} & {_pf(p)} & {_pf(a)}" + r" \\" for (n, lab, met, m, p), a in zip(tests, _a)]
write("tab_fg_sig.tex", "\n".join(lines))

# Worst case over the six attacks, CIFAR-10, nine defences (seeds 0 to 2)
c = RUNS[(RUNS.dataset == "cifar10") & (RUNS.group.isin(["main", "main_deepsight", "review_foolsgold", "r2_fg_clip"])) & (RUNS.seed <= 2)]
g = c.groupby(["base", "attack"]).agg(acc=("acc", "mean"), asr=("asr", "mean")).reset_index()
ATTN = {"none": "no attack", "LF": "LF", "BD": "BD", "GS": "GS", "CO": "CO", "AD": "AD"}
rows = []
for d, s in g.groupby("base"):
    ia, ib = s.acc.idxmin(), s.asr.idxmax()
    rows.append((d, s.loc[ia, "acc"], s.loc[ia, "attack"], s.loc[ib, "asr"], s.loc[ib, "attack"]))
rows.sort(key=lambda x: -x[1])
lines = [f"{NAME.get(d, d)} & {100*a:.1f} & {ATTN[aa]} & {100*b:.1f} & {ATTN[bb]}" + r" \\" for d, a, aa, b, bb in rows]
write("tab_worst.tex", "\n".join(lines))

# Five seeds, key comparisons, 95% t intervals
c = RUNS[(RUNS.dataset == "cifar10") & (RUNS.group.isin(["main", "main_deepsight", "review_seeds"]))
         & (RUNS.base.isin(["TAF-v3", "FLAME", "FLTrust", "DeepSight"])) & (RUNS.attack.isin(["BD", "CO", "AD"]))]
lines = []
for d in ["FLAME", "FLTrust", "DeepSight", "TAF-v3"]:
    cells = []
    for a, met in [("BD", "asr"), ("CO", "acc"), ("AD", "asr")]:
        x = 100 * c[(c.base == d) & (c.attack == a)].sort_values("seed")[met].values
        m, sd = x.mean(), x.std(ddof=1)
        h = _st.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x))
        cells.append(f"{m:.1f} [{max(0, m-h):.1f}, {min(100, m+h):.1f}]")
    lines.append(NAME.get(d, d) + " & " + " & ".join(cells) + r" \\")
write("tab_seeds5.tex", "\n".join(lines))
# paired differences TAF minus each, five seeds
lines = []
for a, met, lab in [("BD", "asr", "BD, ASR"), ("CO", "acc", "CO, accuracy"), ("AD", "asr", "AD, ASR")]:
    t = c[(c.base == "TAF-v3") & (c.attack == a)].set_index("seed")[met]
    for d in ["FLAME", "FLTrust", "DeepSight"]:
        b = c[(c.base == d) & (c.attack == a)].set_index("seed")[met]
        diff = 100 * (t - b).dropna()
        h = _st.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
        p = _st.ttest_rel(t[diff.index], b[diff.index]).pvalue
        lines.append([lab, d, diff.mean(), h, p])
# Holm correction over all nine five seed tests
_ps = np.array([x[4] for x in lines]); _ord = np.argsort(_ps); _adj = np.empty(len(_ps)); _run = 0.0
for _k, _i in enumerate(_ord):
    _run = max(_run, (len(_ps) - _k) * _ps[_i]); _adj[_i] = min(1.0, _run)
_pf = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
lines = [f"{lab} & {d} & {fmt_signed(m)} & [{fmt_signed(m-h)}, {fmt_signed(m+h)}] & {_pf(p)} & {_pf(a)}" + r" \\"
         for (lab, d, m, h, p), a in zip(lines, _adj)]
write("tab_seeds5_diff.tex", "\n".join(lines))
# per seed appendix
lines = []
for d in ["FLAME", "FLTrust", "DeepSight", "TAF-v3"]:
    for a, met in [("BD", "asr"), ("CO", "acc"), ("AD", "asr")]:
        x = 100 * c[(c.base == d) & (c.attack == a)].sort_values("seed")[met].values
        lines.append(f"{NAME.get(d, d)} & {a} & {'ASR' if met == 'asr' else 'accuracy'} & " + " & ".join(f"{v:.1f}" for v in x) + r" \\")
write("tab_perseed.tex", "\n".join(lines))

# Partial participation
lines = []
for d in ["FedAvg", "FLAME", "TAF-v3"]:
    cells = []
    for a, sfx in [("BD", ""), ("CO", ""), ("AD", "@match")]:
        base = (d if d != "TAF-v3" else "TAF-v3") + sfx
        rf = get({"group": "main", "dataset": "cifar10", "tag": base, "attack": a})
        rp = get({"group": "review_partial", "tag": base + "@q0.5", "attack": a})
        met = "acc" if a == "CO" else "asr"
        cells += [cell(rf, met, sd=False), cell(rp, met)]
    lines.append(NAME.get(d, d) + " & " + " & ".join(cells) + r" \\")
write("tab_partial.tex", "\n".join(lines))

# Thresholds
TH = [("Default ($z_A = 3$, $z_C = 2$, $k = 10\\%$)", None), ("$z_A = 2$", "zA2"), ("$z_A = 4$", "zA4"),
      ("$z_C = 1$", "zC1"), ("$z_C = 3$", "zC3"), ("$k = 5\\%$", "k05"), ("$k = 20\\%$", "k20")]
lines = []
for lab, key in TH:
    if key is None:
        rc = get({"group": "main", "dataset": "cifar10", "tag": "TAF-v3", "attack": "CO"})
        ra = get({"group": "main", "dataset": "cifar10", "tag": "TAF-v3@match", "attack": "AD"})
    else:
        rc = get({"group": "review_thresholds", "tag": f"thr_{key}", "attack": "CO"})
        ra = get({"group": "review_thresholds", "tag": f"thr_{key}@match", "attack": "AD"})
    lines.append(lab + " & " + " & ".join([cell(rc, "acc"), f"{100*rc['share_mean']:.1f}", cell(ra, "asr"),
                                            f"{100*ra['share_mean']:.1f}"]) + r" \\")
write("tab_thresholds.tex", "\n".join(lines))

# Scaling (aggregation only, synthetic updates)
if RES is not None and (RES / "overhead_scaling.json").exists():
    S = json.loads((RES / "overhead_scaling.json").read_text())["rows"]
    ns = sorted({r["n"] for r in S})
    lines = []
    for d in ["FedAvg", "Krum", "FLAME", "FoolsGold", "TAF"]:
        v = {r["n"]: r["mean_s"] * 1000 for r in S if r["defense"] == d}
        lines.append(("TAF (ours)" if d == "TAF" else d) + " & " + " & ".join(f"{v[n]:.1f}" for n in ns) + r" \\")
    write("tab_scaling.tex", "\n".join(lines))
print("review tables written")
