"""
03_imputation_pooling.py - Comment 5(c): what does 'average 20 imputations into one dataset' do to the CIs?

Compares, for a fast learner (linear regression, 5x repeated 5-fold CV, identical folds for every imputation):
  (a) the paper's approach: ONE completed dataset = mean of 20 stochastic imputations, patient-bootstrap CI
  (b) Rubin's rules over the 20 separately completed datasets: total variance = W + (1 + 1/m) B
      (W = mean within-imputation bootstrap variance of R2, B = between-imputation variance of R2)
Output: results/table_R7c_imputation_pooling.md
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from cv_pipeline import IMPUTE_COLS
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold
import models as M
import warnings; warnings.filterwarnings("ignore")

df = load_patients(); y = auc24_reference(df); n = len(y)
X = df[IMPUTE_COLS].to_numpy(float); cols = {c: i for i, c in enumerate(IMPUTE_COLS)}
m = 20
draws = []
for i in range(m):
    imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, max_iter=8, random_state=500 + i, initial_strategy="mean", skip_complete=True)
    draws.append(imp.fit_transform(X))
draws = np.stack(draws); avg = draws.mean(axis=0)
folds = []
for r in range(5):
    for tr, te in KFold(5, shuffle=True, random_state=900 + r).split(np.arange(n)):
        folds.append((r, tr, te))
def oof_ols(Xfull, strat):
    ix = [cols[c] for c in COV7 + STRATEGIES[strat]]
    P = np.full((5, n), np.nan)
    for r, tr, te in folds:
        Z = Xfull[:, ix]; mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        P[r, te] = M.ols_fit_predict((Z[tr] - mu) / sd, y[tr], (Z[te] - mu) / sd)
    return P
rows = []; KN = {}
for s in STRATEGIES:
    miss_inputs = int(df[STRATEGIES[s]].isna().any(axis=1).sum())
    r2s, ws = [], []
    for i in range(m):
        P = oof_ols(draws[i], s); r2s.append(np.mean(metrics_matrix(y, P)["R2"]))
        b = boot_metrics(y, P, B=400, seed=i)["R2"]; ws.append(np.var(b, ddof=1))
    r2s, ws = np.array(r2s), np.array(ws)
    Qbar, W, Bv = r2s.mean(), ws.mean(), r2s.var(ddof=1); T = W + (1 + 1 / m) * Bv
    Pa = oof_ols(avg, s); ba = boot_metrics(y, Pa, B=1000, seed=1)["R2"]; lo, hi = np.percentile(ba, [2.5, 97.5])
    ra = float(np.mean(metrics_matrix(y, Pa)["R2"]))
    half = 1.96 * np.sqrt(T)
    infl = 100 * (np.sqrt(T / W) - 1)                      # widening of the CI caused by imputation uncertainty
    rows.append({"Strategy": s, "Patients with ≥ 1 imputed input, n": miss_inputs,
                 "R2, averaged imputations (95% CI)": f"{ra:.3f} ({lo:.3f}–{hi:.3f})",
                 "R2, Rubin pooled (95% CI)": f"{Qbar:.3f} ({Qbar-half:.3f}–{Qbar+half:.3f})",
                 "Between-imputation SD of R2": f"{np.sqrt(Bv):.4f}", "95% CI widening from imputation uncertainty": f"{infl:.1f}%"})
    KN[s] = dict(n_imputed_patients=miss_inputs, R2_avg=ra, R2_rubin=float(Qbar), between_sd=float(np.sqrt(Bv)), width_change_pct=float(infl))
out = pd.DataFrame(rows); out.to_csv(os.path.join(RES, "table_R7c_imputation_pooling.csv"), index=False)
cols_ = list(out.columns)
open(os.path.join(RES, "table_R7c_imputation_pooling.md"), "w").write("\n".join(["| " + " | ".join(cols_) + " |", "| " + " | ".join(["---"] * len(cols_)) + " |"] + ["| " + " | ".join(str(r[c]) for c in cols_) + " |" for _, r in out.iterrows()]) + "\n")
json.dump(KN, open(os.path.join(RES, "imputation_pooling.json"), "w"), indent=1)
print(out.to_string())
