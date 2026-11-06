"""
01_run_resampling.py - repeated nested CV for ALL comparisons (main) and for the decoupled-reference
sensitivity analysis. Writes results/oof_main.npz, results/oof_decoupled.npz and results/cv_meta.json.

usage:  python src/01_run_resampling.py [--data data/patients.csv] [--repeats 10] [--folds 5] [--jobs 2]
(for production use --repeats 20 or more; run time scales linearly)
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv_pipeline import *

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(DATA, "patients.csv"))
ap.add_argument("--repeats", type=int, default=10)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--jobs", type=int, default=2)
ap.add_argument("--only", default="both", choices=["both", "main", "decoupled"])
a = ap.parse_args()

df = load_patients(a.data)
y_ref = auc24_reference(df)
meta = dict(n=len(df), repeats=a.repeats, folds=a.folds, data=os.path.basename(a.data))

# ------------------------------------------------------------------ main comparison
S = list(STRATEGIES)
main = [("Null", "-"), ("OLS-cov", "-"), ("OLS-C0", "C0"), ("OLS-C0+dose", "C0"), ("LogLin-C0+dose", "C0")]
for s in S:
    main += [("OLS", s), ("OLS-noCov", s), ("RF", s), ("XGB", s), ("MLP", s), ("LSTM", s)]
for s in ("C0", "C0+C0.5+C6"):
    main += [("MLP-noCov", s), ("LSTM-noCov", s)]
for d in ("-dropSOFA", "-dropPT", "-dropSOFA+PT"):
    main += [("LSTM" + d, "C0"), ("OLS" + d, "C0")]
imp_for = [("LSTM", "C0"), ("LSTM", "C0+C0.5+C6")]

if a.only in ("both", "main"):
    t0 = time.time()
    res = run_cv(df, y_ref, main, n_repeats=a.repeats, n_splits=a.folds, n_jobs=a.jobs, imp_for=imp_for, log="main")
    np.savez_compressed(os.path.join(RES, "oof_main.npz"), y=y_ref, **{k: v for k, v in res["oof"].items()})
    json.dump(dict(info=res["info"], imp=res["imp"]), open(os.path.join(RES, "cv_info_main.json"), "w"))
    meta["main_seconds"] = time.time() - t0

# ------------------------------------------------------------------ decoupled reference (no input concentration inside the AUC)
if a.only in ("both", "decoupled"):
    y_ind_all = auc24_independent(df)
    keep = ~np.isnan(y_ind_all)
    d2 = df.loc[keep].reset_index(drop=True); y_ind = y_ind_all[keep]
    dec = [("Null", "-"), ("OLS-C0", "C0"), ("OLS-C0+dose", "C0")]
    for s in ("C0", "C0+C0.5+C6"):
        dec += [("OLS", s), ("MLP", s), ("LSTM", s)]
    t0 = time.time()
    res = run_cv(d2, y_ind, dec, n_repeats=a.repeats, n_splits=a.folds, n_jobs=a.jobs, log="decoupled")
    np.savez_compressed(os.path.join(RES, "oof_decoupled.npz"), y=y_ind, keep=keep, y_ref_subset=y_ref[keep],
                        **{k: v for k, v in res["oof"].items()})
    meta["decoupled_n"] = int(keep.sum()); meta["decoupled_seconds"] = time.time() - t0
json.dump(meta, open(os.path.join(RES, "cv_meta.json"), "w"), indent=1)
print("done", meta)
