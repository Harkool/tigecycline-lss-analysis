"""
04_multiseed_single_split.py - what the ORIGINAL design (one 78/20 split, random_state = 42) can and cannot show.
  (1) 30 network seeds on the same split: spread of test R2 for the per-time-point LSTM and the
      capacity-matched MLP, per strategy (Comment 2 and Comment 4: multi-seed results)
  (2) time-order test (Comment 4): for strategies with >= 2 time points, evaluate the trained LSTM with the
      time steps reversed / shuffled. If the recurrent structure carried information, scrambling the order should hurt.
  (3) capacity curve (C0): LSTM and capacity-matched MLP for 8..256 hidden units, 10 seeds each
Outputs: results/table_R6a_multiseed.md, table_R6b_order_test.md, table_R6c_capacity.md, results/multiseed_raw.csv
usage: python src/04_multiseed_single_split.py [--seeds 30] [--jobs 2]
"""
import argparse, os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from cv_pipeline import prepare_fold, _zc
import models as M
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import warnings; warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=30); ap.add_argument("--jobs", type=int, default=2)
ap.add_argument("--data", default=os.path.join(DATA, "patients.csv")); a = ap.parse_args()
df = load_patients(a.data); y = auc24_reference(df)
tr, te = train_test_split(np.arange(len(df)), test_size=20, random_state=42)       # the paper's split design (78 / 20)
F = prepare_fold(df, tr, te, seed=42)
ytr, yte = y[tr], y[te]
rng0 = np.random.default_rng(0); val_mask = np.zeros(len(tr), bool); val_mask[rng0.choice(len(tr), 12, replace=False)] = True

def one(kind, strat, seed, hidden=32, perm=False):
    concs = STRATEGIES[strat]
    zc_tr, zc_te = _zc(F, COV7, "tr"), _zc(F, COV7, "te"); zk_tr, zk_te = _zc(F, concs, "tr"), _zc(F, concs, "te")
    pred, info, pack = M.fit_predict_nn(kind, zc_tr, zk_tr, ytr, zc_te, zk_te, val_mask, seed, configs=[dict(hidden=hidden, wd=1e-3)], return_model=True)
    out = dict(kind=kind, strategy=strat, seed=seed, hidden=hidden, epochs=info["epochs"], **metrics(yte, pred))
    if perm and kind == "lstm" and len(concs) > 1:
        rr = np.random.default_rng(seed)
        rev = M.predict_with(pack, kind, zc_te, zk_te[:, ::-1]); out["R2_reversed"] = metrics(yte, rev)["R2"]
        sh = [metrics(yte, M.predict_with(pack, kind, zc_te, zk_te[:, rr.permutation(len(concs))]))["R2"] for _ in range(4)]
        out["R2_shuffled"] = float(np.mean(sh))
    return out

t0 = time.time()
jobs = [(k, s, sd) for s in STRATEGIES for k in ("lstm", "mlp") for sd in range(a.seeds)]
res = Parallel(n_jobs=a.jobs)(delayed(one)(k, s, sd, 32, True) for k, s, sd in jobs)
raw = pd.DataFrame(res); raw.to_csv(os.path.join(RES, "multiseed_raw.csv"), index=False)
print("multiseed done", round(time.time() - t0), "s")

def q(x): return f"{np.median(x):.3f} ({np.percentile(x,25):.3f}–{np.percentile(x,75):.3f}) [{np.min(x):.3f}, {np.max(x):.3f}]"
rows = []
for s in STRATEGIES:
    l = raw[(raw.kind == "lstm") & (raw.strategy == s)]; m = raw[(raw.kind == "mlp") & (raw.strategy == s)]
    rows.append({"Strategy": s, "LSTM test R2: median (IQR) [min, max]": q(l.R2), "MLP test R2: median (IQR) [min, max]": q(m.R2),
                 "LSTM test MAPE, % (median)": f"{l.MAPE.median():.1f}", "MLP test MAPE, % (median)": f"{m.MAPE.median():.1f}",
                 "Seeds in which LSTM R2 > MLP R2, %": f"{100*np.mean(l.R2.to_numpy() > m.R2.to_numpy()):.0f}"})
t6a = pd.DataFrame(rows)
rows = []
for s in STRATEGIES:
    if len(STRATEGIES[s]) < 2: continue
    l = raw[(raw.kind == "lstm") & (raw.strategy == s)]
    rows.append({"Strategy": s, "Test R2, correct order": f"{l.R2.median():.3f}", "Test R2, reversed order": f"{l.R2_reversed.median():.3f}",
                 "Test R2, shuffled order": f"{l.R2_shuffled.median():.3f}", "ΔR2 (reversed − correct), median [IQR]": f"{(l.R2_reversed-l.R2).median():+.3f} [{(l.R2_reversed-l.R2).quantile(.25):+.3f}, {(l.R2_reversed-l.R2).quantile(.75):+.3f}]"})
t6b = pd.DataFrame(rows)

# capacity curve for C0
cap = Parallel(n_jobs=a.jobs)(delayed(one)(k, "C0", sd, h, False) for h in (8, 16, 32, 64, 128) for k in ("lstm", "mlp") for sd in range(10))
cap = pd.DataFrame(cap)
rows = []
for h in (8, 16, 32, 64, 128):
    c = cap[cap.hidden == h]; l = c[c.kind == "lstm"]; mm = c[c.kind == "mlp"]
    rows.append({"LSTM hidden units (1 layer)": h, "Parameters (LSTM)": M.lstm_param_count(8, h), "LSTM test R2, mean ± SD (10 seeds)": f"{l.R2.mean():.3f} ± {l.R2.std():.3f}",
                 "Parameter-matched MLP test R2, mean ± SD": f"{mm.R2.mean():.3f} ± {mm.R2.std():.3f}"})
t6c = pd.DataFrame(rows)
for nm, tb in (("table_R6a_multiseed", t6a), ("table_R6b_order_test", t6b), ("table_R6c_capacity", t6c)):
    tb.to_csv(os.path.join(RES, nm + ".csv"), index=False); cc = list(tb.columns)
    open(os.path.join(RES, nm + ".md"), "w").write("\n".join(["| " + " | ".join(cc) + " |", "| " + " | ".join(["---"] * len(cc)) + " |"] + ["| " + " | ".join(str(r[c]) for c in cc) + " |" for _, r in tb.iterrows()]) + "\n")
    print(tb.to_string(), "\n")
