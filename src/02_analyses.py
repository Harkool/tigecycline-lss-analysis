"""
02_analyses.py - turns the out-of-fold predictions into every summary table
(Tables R1-R5, R7-R9). Also writes results/key_numbers.json (every headline number of the analysis).

usage: python src/02_analyses.py [--data data/patients.csv]
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--data", default=os.path.join(DATA, "patients.csv")); a = ap.parse_args()
df = load_patients(a.data)
d = np.load(os.path.join(RES, "oof_main.npz")); y = d["y"]
oof = {k: d[k] for k in d.files if k != "y"}
n = len(y); R = next(iter(oof.values())).shape[0]
B = 2000
idx = np.random.default_rng(2026).integers(0, n, (B, n))          # shared (paired) bootstrap indices
idx0 = np.arange(n)[None, :]
KN = {}                                                            # key numbers of the analysis

def pt(key, yy=None, mask=None):
    yy = y if yy is None else yy
    P = oof[key] if mask is None else oof[key][:, mask]
    return {k: float(np.mean(v)) for k, v in metrics_matrix(yy, P).items()}

_cache = {}
def bt(key):
    if key not in _cache:
        _cache[key] = boot_metrics(y, oof[key], idx=idx)
    return _cache[key]

def ci(key, m):
    lo, hi = np.percentile(bt(key)[m], [2.5, 97.5]); return float(lo), float(hi)

def cell(key, m, dg=3):
    p = pt(key)[m]; lo, hi = ci(key, m)
    return f"{p:.{dg}f} ({lo:.{dg}f}–{hi:.{dg}f})"

def calib(key):
    P = oof[key]; sl = []
    for r in range(P.shape[0]):
        sl.append(np.cov(y, P[r])[0, 1] / np.var(P[r], ddof=1))
    return float(np.mean(sl))

def md(df_, path):
    df_.to_csv(path.replace(".md", ".csv"), index=False)
    cols = list(df_.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df_.iterrows():
        lines.append("| " + " | ".join((str(int(r[c])) if isinstance(r[c], float) and float(r[c]).is_integer() and not str(df_[c].dtype).startswith("object") else str(r[c])) for c in cols) + " |")
    open(path, "w").write("\n".join(lines) + "\n")

PRETTY = {"RF": "RF", "XGB": "XGB", "MLP": "MLP (capacity-matched)", "LSTM": "LSTM (per-time-point)", "OLS": "Linear regression"}

# ------------------------------------------------------------------ Table R1: reference models vs ML (C0 strategy)
rows = [("Intercept only (training mean)", "Null|-"), ("7 covariates only (linear)", "OLS-cov|-"),
        ("C0 only (linear)", "OLS-C0|C0"), ("C0 + dose (linear)", "OLS-C0+dose|C0"),
        ("log C0 + log dose (log-linear)", "LogLin-C0+dose|C0"), ("C0 + 7 covariates (linear)", "OLS|C0"),
        ("C0 + 7 covariates: RF", "RF|C0"), ("C0 + 7 covariates: XGB", "XGB|C0"),
        ("C0 + 7 covariates: MLP (capacity-matched)", "MLP|C0"), ("C0 + 7 covariates: LSTM (per-time-point)", "LSTM|C0"),
        ("C0 only, no covariates: MLP", "MLP-noCov|C0"), ("C0 only, no covariates: LSTM", "LSTM-noCov|C0")]
t = []
for lab, k in rows:
    t.append({"Model (C0 strategy)": lab, "R2 (95% CI)": cell(k, "R2"), "RMSE, mg·h/L (95% CI)": cell(k, "RMSE", 2),
              "MAPE, % (95% CI)": cell(k, "MAPE", 1), "MDPE, %": f"{pt(k)['MDPE']:.1f}", "F20, %": f"{pt(k)['F20']:.0f}",
              "F30, %": f"{pt(k)['F30']:.0f}", "Calib. slope": ("\u2013" if k.startswith("Null") else f"{calib(k):.2f}")})
md(pd.DataFrame(t), os.path.join(RES, "table_R1_reference_models.md"))
for lab, k in rows:
    KN["R1|" + k] = pt(k)

# ------------------------------------------------------------------ Table R3: 4 algorithms x 5 strategies + rank probabilities
S = list(STRATEGIES)
algs = ["RF", "XGB", "MLP", "LSTM"]
keys20 = [f"{m}|{s}" for s in S for m in algs]
keys25 = keys20 + [f"OLS|{s}" for s in S]
def best_prob(keys, m="RMSE"):
    M_ = np.stack([bt(k)[m] for k in keys])          # (K, B)
    win = np.argmin(M_, axis=0)
    return {k: float(np.mean(win == i)) for i, k in enumerate(keys)}
p20, p25 = best_prob(keys20), best_prob(keys25)
t = []
for s in S:
    for m in algs + ["OLS"]:
        k = f"{m}|{s}"
        t.append({"Strategy": s, "Model": PRETTY[m], "R2 (95% CI)": cell(k, "R2"), "RMSE (95% CI)": cell(k, "RMSE", 2),
                  "MAPE, %": f"{pt(k)['MAPE']:.1f}", "MDPE, %": f"{pt(k)['MDPE']:.1f}", "F20, %": f"{pt(k)['F20']:.0f}",
                  "Best of the 20 ML combinations, %": (f"{100*p20[k]:.0f}" if k in p20 else "–")})
md(pd.DataFrame(t), os.path.join(RES, "table_R3_all_combinations.md"))
best20 = max(p20, key=p20.get)
KN["best20_key"] = best20; KN["best20_prob"] = p20[best20]; KN["best25_prob_ols"] = {k: v for k, v in p25.items() if k.startswith("OLS")}
KN["n_combinations_with_prob_gt_5pct"] = int(sum(v > 0.05 for v in p20.values()))
KN["R3"] = {k: pt(k) for k in keys25}

# paired contrasts (same patients, same bootstrap draws)
def diff(k1, k2, m):
    d1 = bt(k1)[m] - bt(k2)[m]; p = pt(k1)[m] - pt(k2)[m]
    lo, hi = np.percentile(d1, [2.5, 97.5]); return p, float(lo), float(hi)
contr = [("LSTM|C0", "OLS-C0+dose|C0"), ("LSTM|C0", "OLS|C0"), ("LSTM|C0", "MLP|C0"), ("LSTM|C0", "RF|C0"), ("LSTM|C0", "XGB|C0"),
         ("LSTM|C0+C0.5+C6", "OLS|C0+C0.5+C6"), ("LSTM|C0+C0.5+C6", "MLP|C0+C0.5+C6"), ("LSTM|C0+C0.5+C6", "LSTM|C0"),
         ("OLS|C0+C0.5+C6", "OLS|C0"), ("LSTM|C0.5+C6", "LSTM|C0")]
t = []
for k1, k2 in contr:
    r = {"Contrast (A − B)": f"{k1.replace('|',' [')}] − {k2.replace('|',' [')}]"}
    for m, dg in (("RMSE", 2), ("MAPE", 1), ("F20", 0)):
        p, lo, hi = diff(k1, k2, m); r[f"Δ{m} (95% CI)"] = f"{p:+.{dg}f} ({lo:+.{dg}f} to {hi:+.{dg}f})"
    t.append(r); KN["contrast|" + k1 + "-" + k2] = {m: diff(k1, k2, m) for m in ("RMSE", "MAPE", "F20")}
md(pd.DataFrame(t), os.path.join(RES, "table_R3b_paired_contrasts.md"))

# ------------------------------------------------------------------ Table R2: decoupled reference
if os.path.exists(os.path.join(RES, "oof_decoupled.npz")):
    dd = np.load(os.path.join(RES, "oof_decoupled.npz")); yi = dd["y"]; keep = dd["keep"]; yref_sub = dd["y_ref_subset"]
    dk = [k for k in dd.files if "|" in k]
    nn_ = len(yi); idi = np.random.default_rng(7).integers(0, nn_, (B, nn_))
    t = []
    lab = {"Null|-": "Intercept only", "OLS-C0|C0": "C0 only (linear)", "OLS-C0+dose|C0": "C0 + dose (linear)",
           "OLS|C0": "C0 + 7 covariates (linear)", "MLP|C0": "C0 + 7 covariates: MLP", "LSTM|C0": "C0 + 7 covariates: LSTM",
           "OLS|C0+C0.5+C6": "C0+C0.5+C6 + 7 covariates (linear)", "MLP|C0+C0.5+C6": "C0+C0.5+C6 + 7 covariates: MLP",
           "LSTM|C0+C0.5+C6": "C0+C0.5+C6 + 7 covariates: LSTM"}
    for k in dk:
        Pk = dd[k]; m_ind = {q: float(np.mean(v)) for q, v in metrics_matrix(yi, Pk).items()}
        b = boot_metrics(yi, Pk, idx=idi); lo, hi = np.percentile(b["R2"], [2.5, 97.5]); lo2, hi2 = np.percentile(b["RMSE"], [2.5, 97.5])
        kk = k if k in oof else None
        r2_orig = (float(np.mean(metrics_matrix(y[keep], oof[kk][:, keep])["R2"])) if kk else np.nan)
        t.append({"Model": lab.get(k, k), "R2, independent reference (95% CI)": f"{m_ind['R2']:.3f} ({lo:.3f}–{hi:.3f})",
                  "RMSE, independent ref.": f"{m_ind['RMSE']:.2f} ({lo2:.2f}–{hi2:.2f})", "MAPE, %": f"{m_ind['MAPE']:.1f}",
                  "F20, %": f"{m_ind['F20']:.0f}", "R2, original reference (same patients)": ("–" if np.isnan(r2_orig) else f"{r2_orig:.3f}")})
        KN["R2dec|" + k] = dict(R2=m_ind["R2"], R2_orig=r2_orig, RMSE=m_ind["RMSE"], MAPE=m_ind["MAPE"], F20=m_ind["F20"], CI_R2=(float(lo), float(hi)))
    md(pd.DataFrame(t), os.path.join(RES, "table_R2_decoupled_reference.md"))
    ratio = 100 * (yi / yref_sub - 1)
    KN["dec_n"] = int(keep.sum()); KN["dec_bias_pct_mean"] = float(ratio.mean()); KN["dec_bias_pct_sd"] = float(ratio.std())
    KN["dec_corr"] = float(np.corrcoef(yi, yref_sub)[0, 1])

# ------------------------------------------------------------------ Table R5: target attainment
def class_stats(P, T, ix):
    yb = y[ix]; Pb = P[:, ix]; A = (yb >= T)[None]; Pp = Pb >= T
    tp = (Pp & A).sum(-1); tn = (~Pp & ~A).sum(-1); fp = (Pp & ~A).sum(-1); fn = (~Pp & A).sum(-1)
    nn_ = ix.shape[1]; acc = (tp + tn) / nn_
    sens = tp / np.maximum(tp + fn, 1); spec = tn / np.maximum(tn + fp, 1)
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / nn_ ** 2
    kap = (acc - pe) / (1 - pe + 1e-12)
    return dict(acc=acc.mean(0), sens=sens.mean(0), spec=spec.mean(0), kappa=kap.mean(0))
thr = {}
for site, tg in sorted(((s_, t_) for s_, t_ in TARGETS.items()), key=lambda z: z[1]):
    for mic in MICS:
        T_ = round(tg * mic, 2)
        near = [k_ for k_ in thr if abs(k_ - T_) < 0.2]
        if near:                                   # merge thresholds that differ by < 0.2 mg*h/L (8.95 vs 9.0)
            labs_ = thr.pop(near[0]); thr[max(T_, near[0])] = labs_ + [f"{site}, MIC {mic:g}"]
        else:
            thr[T_] = [f"{site}, MIC {mic:g}"]
rows5 = []
mods5 = [("LSTM|C0", "C0-LSTM"), ("MLP|C0", "C0-MLP"), ("OLS-C0+dose|C0", "C0+dose linear"), ("LSTM|C0+C0.5+C6", "3-point LSTM")]
for T, labs in sorted(thr.items()):
    prev = float(np.mean(y >= T))
    if not (0.10 <= prev <= 0.90):
        KN.setdefault("thr_trivial", []).append((T, prev)); continue
    for k, nm in mods5:
        p0 = class_stats(oof[k], T, idx0); b = class_stats(oof[k], T, idx)
        f = lambda m: f"{100*p0[m][0]:.0f} ({100*np.percentile(b[m],2.5):.0f}–{100*np.percentile(b[m],97.5):.0f})" if m != "kappa" else f"{p0[m][0]:.2f} ({np.percentile(b[m],2.5):.2f}–{np.percentile(b[m],97.5):.2f})"
        rows5.append({"Target (AUC0–24 threshold, mg·h/L)": f"{'; '.join(labs)} (≥ {T:g})", "Attained, %": f"{100*prev:.0f}", "Model": nm,
                      "Accuracy, % (95% CI)": f("acc"), "Sensitivity, %": f("sens"), "Specificity, %": f("spec"), "Kappa": f("kappa")})
        KN[f"TA|{T}|{k}"] = {m: float(p0[m][0]) for m in p0}
md(pd.DataFrame(rows5), os.path.join(RES, "table_R5_target_attainment.md"))

# Bland-Altman / agreement for the recommended C0 model
for k in ("LSTM|C0", "OLS-C0+dose|C0"):
    P = oof[k].mean(axis=0); dpe = 100 * (P - y) / y
    KN["BA|" + k] = dict(mean_pe=float(dpe.mean()), sd_pe=float(dpe.std(ddof=1)), loa_lo=float(dpe.mean() - 1.96 * dpe.std(ddof=1)),
                         loa_hi=float(dpe.mean() + 1.96 * dpe.std(ddof=1)), mdpe_ci=[float(x) for x in ci(k, "MDPE")])

# ------------------------------------------------------------------ Table R7a: counts of measured vs imputed values + restricted evaluation
cnt = []
for c in CONC:
    nm = int(df[c].isna().sum()); cnt.append({"Time point": c, "Measured, n": n - nm, "Missing/imputed, n": nm, "Missing, %": f"{100*nm/n:.1f}"})
md(pd.DataFrame(cnt), os.path.join(RES, "table_R7a_missingness.md"))
t = []
for s in S:
    mask = df[STRATEGIES[s]].notna().all(axis=1).to_numpy()
    for m in ("OLS", "MLP", "LSTM"):
        k = f"{m}|{s}"
        full = pt(k); sub = pt(k, y[mask], mask)
        bsub = boot_metrics(y[mask], oof[k][:, mask], B=1000, seed=3)
        lo, hi = np.percentile(bsub["R2"], [2.5, 97.5])
        t.append({"Strategy": s, "Model": PRETTY[m], "Patients with all inputs measured, n": int(mask.sum()),
                  "R2, all patients": f"{full['R2']:.3f}", "R2, measured inputs only (95% CI)": f"{sub['R2']:.3f} ({lo:.3f}–{hi:.3f})",
                  "RMSE, all": f"{full['RMSE']:.2f}", "RMSE, measured only": f"{sub['RMSE']:.2f}"})
        KN[f"meas|{k}"] = dict(n=int(mask.sum()), R2_all=full["R2"], R2_meas=sub["R2"], RMSE_all=full["RMSE"], RMSE_meas=sub["RMSE"])
md(pd.DataFrame(t), os.path.join(RES, "table_R7b_measured_only.md"))

# ------------------------------------------------------------------ Table R9: importance / ablation
info = json.load(open(os.path.join(RES, "cv_info_main.json")))
t = []
for k, nm in (("LSTM|C0", "C0-LSTM"), ("LSTM|C0+C0.5+C6", "3-point LSTM")):
    imps = info["imp"].get(k, [])
    if not imps: continue
    feats = list(imps[0].keys())
    m_ = {f: np.mean([i[f] for i in imps]) for f in feats}
    sd_ = {f: np.std([i[f] for i in imps], ddof=1) / np.sqrt(len(imps)) for f in feats}
    for f in sorted(feats, key=lambda z: -m_[z]):
        t.append({"Model": nm, "Feature permuted": f, "Increase in RMSE, mg·h/L (mean ± SE over folds)": f"{m_[f]:+.2f} ± {sd_[f]:.2f}"})
        KN[f"perm|{k}|{f}"] = float(m_[f])
md(pd.DataFrame(t), os.path.join(RES, "table_R9a_permutation_importance.md"))
t = []
for base, tag in (("LSTM|C0", "LSTM"), ("OLS|C0", "Linear")):
    for dropk, lab in (("-dropSOFA", "without SOFA"), ("-dropPT", "without PT"), ("-dropSOFA+PT", "without SOFA and PT")):
        k = base.replace("|", dropk + "|")
        p, lo, hi = diff(k, base, "RMSE"); p2, lo2, hi2 = diff(k, base, "R2")
        t.append({"Model (C0 strategy)": tag, "Covariate set": lab, "ΔRMSE vs full 7-covariate model (95% CI)": f"{p:+.2f} ({lo:+.2f} to {hi:+.2f})",
                  "ΔR2 (95% CI)": f"{p2:+.3f} ({lo2:+.3f} to {hi2:+.3f})"})
        KN[f"drop|{k}"] = dict(dRMSE=(p, lo, hi), dR2=(p2, lo2, hi2))
md(pd.DataFrame(t), os.path.join(RES, "table_R9b_drop_covariates.md"))

# LASSO selection stability (bootstrap)
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
yl = y; Xl = df[COV_CANDIDATES].to_numpy(float)
Xl = np.where(np.isnan(Xl), np.nanmedian(Xl, axis=0), Xl)          # LASSO needs complete data: median-fill for this stability check only
Bl = 200; rng = np.random.default_rng(11); cnts = np.zeros(len(COV_CANDIDATES))
for b in range(Bl):
    ii = rng.integers(0, n, n)
    Xs = StandardScaler().fit_transform(Xl[ii]); las = LassoCV(cv=5, random_state=b, max_iter=5000).fit(Xs, yl[ii])
    cnts += (np.abs(las.coef_) > 1e-8)
sel = pd.DataFrame({"Candidate covariate": COV_CANDIDATES, "Selected in bootstrap LASSO, %": (100 * cnts / Bl).round(0).astype(int),
                    "In the published 7-covariate set": ["yes" if c in COV7 else "" for c in COV_CANDIDATES]}).sort_values("Selected in bootstrap LASSO, %", ascending=False)
md(sel, os.path.join(RES, "table_R9c_lasso_stability.md"))
KN["lasso_freq"] = dict(zip(COV_CANDIDATES, (cnts / Bl).tolist()))

# ------------------------------------------------------------------ Table R4: sample size (Riley et al.)
mean_y, sd_y = float(y.mean()), float(y.std(ddof=1))
t = []
for P_, r2 in ((8, 0.80), (8, 0.60), (19, 0.80), (19, 0.60)):
    r = riley_continuous(P_, r2, mean_y, sd_y)
    t.append({"Candidate predictor parameters (P)": P_, "Anticipated adjusted R2": r2, "(i) shrinkage ≥ 0.90": r["n_shrinkage"],
              "(ii) optimism in R2 ≤ 0.05": r["n_r2_optimism"], "(iii) residual variance precision": r["n_resid_var"],
              "(iv) mean outcome precision": r["n_mean"], "Minimum n": r["n_min"]})
    KN[f"riley|{P_}|{r2}"] = r
_t4 = pd.DataFrame(t)
for _c in _t4.columns:
    if _c != "Anticipated adjusted R2": _t4[_c] = _t4[_c].astype(int)
md(_t4, os.path.join(RES, "table_R4_sample_size.md"))
KN["y_mean"], KN["y_sd"] = mean_y, sd_y

# ------------------------------------------------------------------ Table R8: internal-consistency audit of published RMSE/R2 pairs (published numbers)
def implied_var(r2, rmse, dr=0.0005, dm=0.0005):
    return rmse ** 2 / (1 - r2), (rmse - dm) ** 2 / (1 - r2 + dr), (rmse + dm) ** 2 / (1 - r2 - dr)
test_pairs = {  # Dai et al. 2026, Table 2 (ML) and Table 4 (PopPK), 20 test patients: (R2, RMSE)
 "C0 RF": (0.607, 4.648), "C0 XGB": (0.669, 4.267), "C0 MLP": (0.707, 4.017), "C0 LSTM": (0.818, 3.164),
 "C0+C0.5 RF": (0.803, 3.295), "C0+C0.5 XGB": (0.793, 3.377), "C0+C0.5 MLP": (0.664, 4.301), "C0+C0.5 LSTM": (0.885, 2.518),
 "C0+C6 RF": (0.773, 3.536), "C0+C6 XGB": (0.795, 3.361), "C0+C6 MLP": (0.807, 3.261), "C0+C6 LSTM": (0.819, 3.157),
 "C0.5+C6 RF": (0.571, 4.857), "C0.5+C6 XGB": (0.621, 4.567), "C0.5+C6 MLP": (0.678, 4.207), "C0.5+C6 LSTM": (0.736, 3.811),
 "C0+C0.5+C6 RF": (0.798, 3.331), "C0+C0.5+C6 XGB": (0.799, 3.324), "C0+C0.5+C6 MLP": (0.811, 3.225), "C0+C0.5+C6 LSTM": (0.887, 2.496),
 "PopPK A": (0.772, 3.541), "PopPK B": (0.826, 3.093), "PopPK C": (0.830, 3.062), "PopPK D": (0.662, 4.310)}
train_pairs = {  # training rows of Supplementary Table S4 (78 training patients): (R2, RMSE)
 "LSTM C0+C0.5": (0.898, 2.409), "LSTM C0.5+C6": (0.900, 2.409), "RF C0": (0.869, 2.762), "RF C0+C0.5": (0.868, 2.762), "RF C0.5+C6": (0.867, 2.762)}
def audit(pairs):
    out = []
    for kx, (r2, rm) in pairs.items():
        v, lo, hi = implied_var(r2, rm); out.append((kx, r2, rm, v, lo, hi))
    lo_max = max(o[4] for o in out); hi_min = min(o[5] for o in out)
    return out, lo_max, hi_min
ot, lo_t, hi_t = audit(test_pairs); otr, lo_tr, hi_tr = audit(train_pairs)
KN["audit_test"] = dict(n_pairs=len(ot), common_lo=lo_t, common_hi=hi_t, consistent=bool(lo_t <= hi_t),
                        V_mean=float(np.mean([o[3] for o in ot])), SD=float(np.sqrt(np.mean([o[3] for o in ot]))))
KN["audit_train"] = dict(common_lo=lo_tr, common_hi=hi_tr, consistent=bool(lo_tr <= hi_tr))
Vref = 58.0
t = []
for kx, r2, rm, v, lo, hi in otr:
    exp = float(np.sqrt((1 - r2) * Vref))
    t.append({"Training-set row (as printed in Table S4)": kx, "R2": f"{r2:.3f}", "RMSE printed": f"{rm:.3f}", "Implied outcome variance": f"{v:.1f} ({lo:.1f}–{hi:.1f})",
              f"Consistent with Var = {Vref:.0f}?": "yes" if lo <= Vref <= hi else "NO", f"RMSE expected if Var = {Vref:.0f}": f"{exp:.3f}"})
md(pd.DataFrame(t), os.path.join(RES, "table_R8a_tableS4_audit.md"))
json.dump(KN, open(os.path.join(RES, "key_numbers.json"), "w"), indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o))
print("tables written:", sorted(f for f in os.listdir(RES) if f.endswith(".md")))
