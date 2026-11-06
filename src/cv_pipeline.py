"""
cv_pipeline.py - repeated, nested cross-validation shared by all analyses.

Design (cross-validation protocol)
  * outer loop  : R x K-fold CV over PATIENTS (every patient is tested R times, never trained on when tested)
  * inside each outer-training fold, with NO access to the outer-test fold:
        - MICE (BayesianRidge, 20 stochastic imputations averaged) fitted on the training fold only
        - StandardScaler fitted on the training fold only
        - an inner 85/15 split used for (a) early stopping of the networks and (b) choosing among
          4 pre-specified hyper-parameter sets - the SAME budget for RF, XGB, MLP and LSTM
        - the winning configuration is refit on the whole training fold
  * networks use FULL-BATCH Adam; MLP width is matched to the LSTM parameter count (capacity matching)
Out-of-fold predictions are stored as (R, n) arrays; every table/figure is computed from them.
"""
from __future__ import annotations
import os, sys, time, json, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import models as M

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------------------------
def mice_mean(Xtr: np.ndarray, Xte: np.ndarray, m: int = 20, seed: int = 0, max_iter: int = 8):
    """MICE as in the paper (scikit-learn IterativeImputer, BayesianRidge, sample_posterior=True, m draws,
    draws averaged). Fitted on the training fold only, then applied to the test fold."""
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    if not np.isnan(Xtr).any() and not np.isnan(Xte).any():
        return Xtr.copy(), Xte.copy()
    tr_acc = np.zeros_like(Xtr); te_acc = np.zeros_like(Xte)
    for i in range(m):
        imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, max_iter=max_iter,
                               random_state=seed * 100 + i, initial_strategy="mean", skip_complete=True)
        tr_acc += imp.fit_transform(Xtr)
        te_acc += imp.transform(Xte)
    return tr_acc / m, te_acc / m


IMPUTE_COLS = COV_CANDIDATES + ["daily_dose"] + CONC


def prepare_fold(df, tr, te, seed, m_imp=20):
    X = df[IMPUTE_COLS].to_numpy(float)
    Xtr, Xte = mice_mean(X[tr], X[te], m=m_imp, seed=seed)
    cols = {c: i for i, c in enumerate(IMPUTE_COLS)}
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0) + 1e-9
    Ztr = (Xtr - mu) / sd; Zte = (Xte - mu) / sd
    return dict(cols=cols, Xtr=Xtr, Xte=Xte, Ztr=Ztr, Zte=Zte)


def _zc(F, names, which):
    ix = [F["cols"][c] for c in names]
    return F["Z" + which][:, ix]


def run_combo(model, strat, F, ytr, val_mask, seed, want_importance=False):
    """Train one (model, strategy) on the fold and return (test predictions, info dict)."""
    cov_all = COV7
    drop = []
    base = model
    for tag, cols in (("-dropSOFA+PT", ["SOFA", "PT"]), ("-dropSOFA", ["SOFA"]), ("-dropPT", ["PT"])):
        if model.endswith(tag):
            base, drop = model[: -len(tag)], cols
    cov = [c for c in cov_all if c not in drop]
    concs = STRATEGIES.get(strat, ["C0"])
    zc_tr, zc_te = _zc(F, cov, "tr"), _zc(F, cov, "te")
    zk_tr, zk_te = _zc(F, concs, "tr"), _zc(F, concs, "te")
    info = {}
    if base == "Null":
        return np.full(F["Zte"].shape[0], ytr.mean()), info
    if base == "OLS-cov":
        return M.ols_fit_predict(_zc(F, COV7, "tr"), ytr, _zc(F, COV7, "te")), info
    if base == "OLS-C0":
        return M.ols_fit_predict(_zc(F, ["C0"], "tr"), ytr, _zc(F, ["C0"], "te")), info
    if base == "OLS-C0+dose":
        c = ["C0", "daily_dose"]
        return M.ols_fit_predict(_zc(F, c, "tr"), ytr, _zc(F, c, "te")), info
    if base == "LogLin-C0+dose":
        ix = [F["cols"]["C0"], F["cols"]["daily_dose"]]
        a, b = np.log(np.clip(F["Xtr"][:, ix], 1e-3, None)), np.log(np.clip(F["Xte"][:, ix], 1e-3, None))
        return M.loglin_fit_predict(a, ytr, b), info
    if base == "OLS":
        return M.ols_fit_predict(np.c_[zc_tr, zk_tr], ytr, np.c_[zc_te, zk_te]), info
    if base == "OLS-noCov":
        return M.ols_fit_predict(zk_tr, ytr, zk_te), info
    if base in ("RF", "XGB"):
        return M.tree_fit_predict(base, np.c_[zc_tr, zk_tr], ytr, np.c_[zc_te, zk_te], val_mask, seed), info
    kind = "lstm" if base.startswith("LSTM") else "mlp"
    if base.endswith("-noCov"):
        zc_tr = zc_tr[:, :0]; zc_te = zc_te[:, :0]
    pred, info, pack = M.fit_predict_nn(kind, zc_tr, zk_tr, ytr, zc_te, zk_te, val_mask, seed, return_model=True)
    if want_importance:
        rng = np.random.default_rng(seed)
        base_rmse = float(np.sqrt(np.mean((pred - 0) ** 2)))  # placeholder, replaced by caller (needs y_te)
        info["_pack"] = (pack, kind, zc_te, zk_te)
    return pred, info


def permutation_importance(pack, kind, zc_te, zk_te, y_te, cov_names, conc_names, seed, n_perm=5):
    rng = np.random.default_rng(seed)
    base = M.predict_with(pack, kind, zc_te, zk_te)
    base_rmse = np.sqrt(np.mean((base - y_te) ** 2))
    out = {}
    for j, nme in enumerate(cov_names):
        d = []
        for _ in range(n_perm):
            z = zc_te.copy(); z[:, j] = z[rng.permutation(len(z)), j]
            d.append(np.sqrt(np.mean((M.predict_with(pack, kind, z, zk_te) - y_te) ** 2)) - base_rmse)
        out[nme] = float(np.mean(d))
    for j, nme in enumerate(conc_names):
        d = []
        for _ in range(n_perm):
            z = zk_te.copy(); z[:, j] = z[rng.permutation(len(z)), j]
            d.append(np.sqrt(np.mean((M.predict_with(pack, kind, zc_te, z) - y_te) ** 2)) - base_rmse)
        out[nme] = float(np.mean(d))
    return out


def _fold_job(args):
    df, y, tr, te, rep, fold, combos, imp_for, m_imp, base_seed = args
    t0 = time.time()
    seed = base_seed + rep * 100 + fold
    F = prepare_fold(df, tr, te, seed, m_imp=m_imp)
    rng = np.random.default_rng(seed)
    n_tr = len(tr)
    nv = max(8, int(round(0.15 * n_tr)))
    val_mask = np.zeros(n_tr, bool); val_mask[rng.choice(n_tr, nv, replace=False)] = True
    ytr = y[tr]; yte = y[te]
    out = {"preds": {}, "info": {}, "imp": {}, "time": 0.0, "rep": rep, "fold": fold, "te": te}
    for (model, strat) in combos:
        want = (model, strat) in imp_for
        pred, info = run_combo(model, strat, F, ytr, val_mask, seed, want_importance=want)
        key = f"{model}|{strat}"
        out["preds"][key] = pred
        if "_pack" in info:
            pack, kind, zc_te, zk_te = info.pop("_pack")
            out["imp"][key] = permutation_importance(pack, kind, zc_te, zk_te, yte, COV7, STRATEGIES[strat], seed)
        out["info"][key] = info
    out["time"] = time.time() - t0
    return out


def run_cv(df: pd.DataFrame, y: np.ndarray, combos, n_repeats=10, n_splits=5, base_seed=1000, n_jobs=2,
           imp_for=(), m_imp=20, log=None):
    """Repeated K-fold CV. Returns dict(oof={key:(R,n)}, info={key:[...]}, imp={key:[...]})."""
    from sklearn.model_selection import KFold
    from joblib import Parallel, delayed
    n = len(df)
    jobs = []
    for r in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=base_seed + r)
        for f, (tr, te) in enumerate(kf.split(np.arange(n))):
            jobs.append((df, y, tr, te, r, f, combos, set(imp_for), m_imp, base_seed))
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(delayed(_fold_job)(j) for j in jobs)
    oof = {f"{m}|{s}": np.full((n_repeats, n), np.nan) for (m, s) in combos}
    info, imp = {k: [] for k in oof}, {k: [] for k in oof}
    for o in res:
        for k, p in o["preds"].items():
            oof[k][o["rep"], o["te"]] = p
            if o["info"].get(k):
                info[k].append(o["info"][k])
            if k in o["imp"]:
                imp[k].append(o["imp"][k])
    if log:
        print(f"[{log}] {len(jobs)} folds, {time.time()-t0:.0f}s, mean fold time {np.mean([o['time'] for o in res]):.1f}s", flush=True)
    return dict(oof=oof, info=info, imp=imp)
