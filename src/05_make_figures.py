"""
05_make_figures.py - Figures R1-R4 (PNG, 200 dpi). Colours follow one fixed entity->colour map:
  LSTM blue, MLP orange, linear models aqua, RF / XGB greys (validated 3-slot categorical palette + neutrals).
"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = dict(LSTM="#2a78d6", MLP="#eb6834", OLS="#1baf7a", RF="#9a9993", XGB="#5f5e59", NULL="#0b0b0b")
mpl.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": "#c9c8c3",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
                     "axes.axisbelow": True, "legend.frameon": False, "font.family": "DejaVu Sans"})

def stamp(fig):
    """Placeholder (no-op). Kept for call-site compatibility."""
    pass

d = np.load(os.path.join(RES, "oof_main.npz")); y = d["y"]; oof = {k: d[k] for k in d.files if k != "y"}
n = len(y); B = 2000; idx = np.random.default_rng(2026).integers(0, n, (B, n))
dd = np.load(os.path.join(RES, "oof_decoupled.npz")) if os.path.exists(os.path.join(RES, "oof_decoupled.npz")) else None
KN = json.load(open(os.path.join(RES, "key_numbers.json")))

def r2ci(P, yy, ix=None):
    b = boot_metrics(yy, P, B=B, seed=5, idx=ix)["R2"]; return float(np.mean(metrics_matrix(yy, P)["R2"])), *np.percentile(b, [2.5, 97.5])

def dot(ax, yy, est, color, ring=True, hollow=False, marker="o", size=7.5, lw=1.8, z=3):
    m, lo, hi = est
    ax.plot([lo, hi], [yy, yy], color=color, lw=lw, solid_capstyle="round", zorder=z)
    ax.plot([m], [yy], marker=marker, ms=size, mfc=(SURF if hollow else color), mec=color, mew=1.8 if hollow else 1.2, zorder=z + 1)

# ------------------------------------------------------------------ Figure R1
def panel(ax, rows, dec_keys, pub, title):
    ys = np.arange(len(rows))[::-1]
    if pub:
        ax.axvspan(pub[1], pub[2], color="#e9e8e4", zorder=0)
        ax.axvline(pub[0], color="#a9a8a2", lw=1, ls=(0, (3, 2)), zorder=1)
    for yy, (lab, key, col) in zip(ys, rows):
        dot(ax, yy + (0.14 if key in dec_keys else 0), r2ci(oof[key], y, idx), col)
        if dd is not None and key in dec_keys and key in dd.files:
            yi = dd["y"]; ii = np.random.default_rng(7).integers(0, len(yi), (B, len(yi)))
            dot(ax, yy - 0.18, r2ci(dd[key], yi, ii), col, hollow=True, marker="D", size=6.5, lw=1.2)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(-0.12, 1.02); ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("Cross-validated R² (mean of repeats; 95% patient-bootstrap CI)"); ax.grid(axis="y", visible=False)

fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw=dict(wspace=0.62))
rowsA = [("Intercept only", "Null|-", C["NULL"]), ("7 covariates only (linear)", "OLS-cov|-", C["OLS"]), ("C0 only (linear)", "OLS-C0|C0", C["OLS"]),
         ("C0 + dose (linear)", "OLS-C0+dose|C0", C["OLS"]), ("C0 + 7 cov. (linear)", "OLS|C0", C["OLS"]), ("C0 + 7 cov.: RF", "RF|C0", C["RF"]),
         ("C0 + 7 cov.: XGB", "XGB|C0", C["XGB"]), ("C0 + 7 cov.: MLP", "MLP|C0", C["MLP"]), ("C0 + 7 cov.: LSTM", "LSTM|C0", C["LSTM"])]
rowsB = [("Concentrations only (linear)", "OLS-noCov|C0+C0.5+C6", C["OLS"]), ("3 points + 7 cov. (linear)", "OLS|C0+C0.5+C6", C["OLS"]),
         ("3 points + 7 cov.: RF", "RF|C0+C0.5+C6", C["RF"]), ("3 points + 7 cov.: XGB", "XGB|C0+C0.5+C6", C["XGB"]),
         ("3 points + 7 cov.: MLP", "MLP|C0+C0.5+C6", C["MLP"]), ("3 points + 7 cov.: LSTM", "LSTM|C0+C0.5+C6", C["LSTM"])]
decA = {"Null|-", "OLS-C0|C0", "OLS-C0+dose|C0", "OLS|C0", "MLP|C0", "LSTM|C0"}; decB = {"OLS|C0+C0.5+C6", "MLP|C0+C0.5+C6", "LSTM|C0+C0.5+C6"}
panel(axs[0], rowsA, decA, (0.818, 0.527, 0.914), "A  C0 strategy (one sample)")
panel(axs[1], rowsB, decB, (0.887, 0.693, 0.954), "B  C0 + C0.5 + C6 strategy (three samples)")
from matplotlib.lines import Line2D
leg = [Line2D([], [], marker="o", color=INK2, mfc=INK2, ms=7, lw=1.6, label="vs original reference AUC (all samples)"),
       Line2D([], [], marker="D", color=INK2, mfc=SURF, ms=6, lw=1.2, label="vs independent reference AUC (no input sample inside it)"),
       mpl.patches.Patch(color="#e9e8e4", label="published single-split 95% CI (n = 20; dashed = point estimate)")]
fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Figure R1  Cross-validated R\u00b2 of reference and machine-learning models against the original and an independent reference AUC", x=0.01, ha="left", fontsize=10.5, y=1.0)
fig.subplots_adjust(top=0.84, bottom=0.2)
stamp(fig); fig.savefig(os.path.join(FIG, "figR1_reference_models.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Figure R2: agreement of the C0 LSTM
fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.4), gridspec_kw=dict(wspace=0.42, width_ratios=[1, 1, 1.25]))
k = "LSTM|C0"; P = oof[k].mean(axis=0)
ax = axs[0]; lim = [0, max(y.max(), P.max()) * 1.04]
ax.fill_between(lim, [0.8 * v for v in lim], [1.2 * v for v in lim], color="#e9e8e4", zorder=0, label="±20% of observed")
ax.plot(lim, lim, color="#a9a8a2", lw=1.2, zorder=1)
ax.scatter(y, P, s=34, color=C["LSTM"], edgecolor=SURF, linewidth=1.0, zorder=3)
sl = KN["R1|LSTM|C0"] if False else None
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("Observed AUC0–24 (mg·h/L)"); ax.set_ylabel("Predicted (out-of-fold, mean of repeats)")
ax.set_title("A  Observed vs predicted", loc="left", fontsize=10); ax.legend(loc="upper left", fontsize=8.5)
ax = axs[1]; pe = 100 * (P - y) / y; m, s = pe.mean(), pe.std(ddof=1)
ax.axhspan(-20, 20, color="#e9e8e4", zorder=0)
ax.scatter(y, pe, s=34, color=C["LSTM"], edgecolor=SURF, linewidth=1.0, zorder=3)
for v, ls_, lb in ((m, "-", f"mean {m:+.1f}%"), (m + 1.96 * s, (0, (4, 3)), f"+1.96 SD {m+1.96*s:+.0f}%"), (m - 1.96 * s, (0, (4, 3)), f"−1.96 SD {m-1.96*s:+.0f}%")):
    ax.axhline(v, color=INK2, lw=1.2, ls=ls_, zorder=2)
ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 40]); ax.set_xticklabels(["5", "10", "20", "40"])
ax.xaxis.set_minor_locator(mpl.ticker.NullLocator()); ax.set_xlim(y.min() * 0.9, y.max() * 1.05)
ax.set_xlabel("Observed AUC0\u201324 (mg\u00b7h/L, log scale)"); ax.set_ylabel("Prediction error (% of observed)")
ax.set_title("B  Percentage error (shaded: \u00b120%)", loc="left", fontsize=10)
ax.legend(handles=[Line2D([], [], color=INK2, lw=1.2, label=f"mean error {m:+.1f}%"),
                   Line2D([], [], color=INK2, lw=1.2, ls=(0, (4, 3)), label=f"95% limits {m-1.96*s:+.0f}% to {m+1.96*s:+.0f}%")], loc="upper right", fontsize=8.2)
ax = axs[2]
t5 = pd.read_csv(os.path.join(RES, "table_R5_target_attainment.csv"))
tg = list(dict.fromkeys(t5.iloc[:, 0])); mods = [("C0-LSTM", C["LSTM"], "o", False), ("C0-MLP", C["MLP"], "o", False), ("C0+dose linear", C["OLS"], "o", False), ("3-point LSTM", C["LSTM"], "D", True)]
yy = np.arange(len(tg))[::-1]
for j, (nm, col, mk, hol) in enumerate(mods):
    for yv, t_ in zip(yy, tg):
        r = t5[(t5.iloc[:, 0] == t_) & (t5["Model"] == nm)].iloc[0]
        a = r["Accuracy, % (95% CI)"]; mm, rest = a.split(" ("); lo, hi = rest.rstrip(")").split("–")
        off = (1.5 - j) * 0.17
        ax.plot([float(lo), float(hi)], [yv + off] * 2, color=col, lw=1.4, zorder=2)
        ax.plot([float(mm)], [yv + off], marker=mk, ms=6.5, mfc=(SURF if hol else col), mec=col, mew=1.5, zorder=3, ls="none")
def _lab(t_):
    a_, b_ = t_.rsplit(" (\u2265 ", 1)
    return "AUC \u2265 " + b_.rstrip(")") + "\n" + a_.replace(", MIC ", " MIC ").replace("; ", ";\n")
ax.set_yticks(yy); ax.set_yticklabels([_lab(t_) for t_ in tg], fontsize=7.8); ax.set_xlim(40, 102)
ax.set_xlabel("Classification accuracy for target attainment, % (95% CI)"); ax.set_title("C  Target attainment", loc="left", fontsize=10); ax.grid(axis="y", visible=False)
hs = [Line2D([], [], marker=mk, color=col, mfc=(SURF if hol else col), ms=6, lw=1.4, label=nm) for nm, col, mk, hol in mods]
ax.legend(handles=hs, loc="lower left", fontsize=7.8, ncol=1, bbox_to_anchor=(0.0, 0.0))
fig.suptitle("Figure R2  Agreement and clinical classification of the recommended C0-based LSTM (repeated 5-fold cross-validation, n = 98)", x=0.01, ha="left", fontsize=10.5, y=1.02)
stamp(fig); fig.savefig(os.path.join(FIG, "figR2_agreement_target_attainment.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Figure R3: multi-seed spread + capacity
raw = pd.read_csv(os.path.join(RES, "multiseed_raw.csv")); cap_path = os.path.join(RES, "table_R6c_capacity.csv")
fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.3), gridspec_kw=dict(wspace=0.22, width_ratios=[1.5, 1]))
ax = axs[0]; S = list(STRATEGIES); rng = np.random.default_rng(1)
for i, s in enumerate(S):
    for j, (kd, col, lab) in enumerate((("lstm", C["LSTM"], "per-time-point LSTM"), ("mlp", C["MLP"], "capacity-matched MLP"))):
        v = raw[(raw.kind == kd) & (raw.strategy == s)].R2.to_numpy(); x0 = i + (j - 0.5) * 0.36
        ax.scatter(x0 + rng.uniform(-0.07, 0.07, len(v)), v, s=14, color=col, alpha=0.55, edgecolor="none", zorder=2)
        q1, q2, q3 = np.percentile(v, [25, 50, 75]); ax.plot([x0, x0], [q1, q3], color=col, lw=3.2, solid_capstyle="round", zorder=3); ax.plot([x0 - 0.09, x0 + 0.09], [q2, q2], color=INK, lw=1.6, zorder=4)
ax.set_xticks(range(len(S))); ax.set_xticklabels(S); ax.set_ylabel("Test-set R² (20 patients, split seed 42)")
ax.set_title("A  Same split, 30 network seeds: spread from initialisation alone", loc="left", fontsize=10)
ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=C["LSTM"], ms=6, label="per-time-point LSTM (1 layer, 32 units)"), Line2D([], [], marker="o", ls="none", color=C["MLP"], ms=6, label="capacity-matched MLP"),
                   Line2D([], [], color=INK, lw=1.6, label="median; bar = IQR")], loc="lower right", fontsize=8.2); ax.grid(axis="x", visible=False)
ax = axs[1]
if os.path.exists(cap_path):
    cp = pd.read_csv(cap_path); xs = cp["Parameters (LSTM)"].to_numpy()
    for col_, nm_, c_ in (("LSTM test R2, mean ± SD (10 seeds)", "LSTM", C["LSTM"]), ("Parameter-matched MLP test R2, mean ± SD", "matched MLP", C["MLP"])):
        mv = np.array([float(s_.split(" ± ")[0]) for s_ in cp[col_]]); sd = np.array([float(s_.split(" ± ")[1]) for s_ in cp[col_]])
        ax.fill_between(xs, mv - sd, mv + sd, color=c_, alpha=0.16, lw=0); ax.plot(xs, mv, color=c_, lw=2, marker="o", ms=6, mec=SURF, mew=1.2)
        ax.text(xs[-1] * 1.12, mv[-1], nm_, color=c_ if c_ != C["OLS"] else INK, va="center", fontsize=8.5, fontweight="bold")
    ax.set_xscale("log"); ax.set_xlim(xs[0] * 0.8, xs[-1] * 3.2); ax.set_xlabel("Trainable parameters (log scale)"); ax.set_ylabel("Test-set R² (mean ± SD, 10 seeds)")
ax.set_title("B  Capacity curve, C0 strategy", loc="left", fontsize=10)
fig.suptitle("Figure R3  Test-set R\u00b2 of the per-time-point LSTM and the capacity-matched MLP: network seeds on the original split, and model capacity", x=0.01, ha="left", fontsize=10.5, y=1.02)
stamp(fig); fig.savefig(os.path.join(FIG, "figR3_multiseed_capacity.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Figure R4: importance + LASSO stability
info = json.load(open(os.path.join(RES, "cv_info_main.json")))
fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.2), gridspec_kw=dict(wspace=0.55))
ax = axs[0]; imps = info["imp"]["LSTM|C0"]; feats = list(imps[0].keys())
mv = {f: np.mean([i[f] for i in imps]) for f in feats}; se = {f: np.std([i[f] for i in imps], ddof=1) / np.sqrt(len(imps)) for f in feats}
order = sorted(feats, key=lambda z: mv[z]); yy = np.arange(len(order))
ax.barh(yy, [mv[f] for f in order], height=0.55, color=[C["LSTM"] if f == "C0" else "#9ec5f4" for f in order], edgecolor="none", zorder=2)
ax.errorbar([mv[f] for f in order], yy, xerr=[1.96 * se[f] for f in order], fmt="none", ecolor=INK2, elinewidth=1.1, capsize=2.5, zorder=3)
ax.set_yticks(yy); ax.set_yticklabels(order); ax.axvline(0, color="#a9a8a2", lw=1); ax.grid(axis="y", visible=False)
ax.set_xlabel("Increase in out-of-fold RMSE when the input is permuted (mg·h/L)"); ax.set_title("A  Permutation importance, C0-LSTM", loc="left", fontsize=10)
ax = axs[1]; lf = KN["lasso_freq"]; names = sorted(lf, key=lambda z: lf[z]); yy = np.arange(len(names))
ax.barh(yy, [100 * lf[z] for z in names], height=0.55, color=[C["LSTM"] if z in COV7 else "#c9c8c3" for z in names], edgecolor="none", zorder=2)
ax.set_yticks(yy); ax.set_yticklabels(names); ax.set_xlim(0, 100); ax.grid(axis="y", visible=False)
ax.set_xlabel("Selected by LASSO in bootstrap resamples (%)"); ax.set_title("B  Stability of covariate selection (200 bootstraps)", loc="left", fontsize=10)
ax.legend(handles=[mpl.patches.Patch(color=C["LSTM"], label="in the published 7-covariate set"), mpl.patches.Patch(color="#c9c8c3", label="other candidates")], loc="center right", fontsize=8.2)
fig.suptitle("Figure R4  What the C0-LSTM relies on, and how stable the covariate selection is", x=0.01, ha="left", fontsize=10.5, y=1.02)
stamp(fig); fig.savefig(os.path.join(FIG, "figR4_importance_selection.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
print("figures written:", sorted(os.listdir(FIG)))
