"""
common.py - shared helpers for the analysis pipeline
(tigecycline AUC0-24h limited-sampling study, Dai et al. DDDT 2026).

All analysis scripts import from here. Change `data/patients.csv` as needed
(see README.md for the column schema) - nothing in this file needs editing.
"""
from __future__ import annotations
import math, os, json, warnings
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
for _d in (DATA, RES, FIG):
    os.makedirs(_d, exist_ok=True)

# ----------------------------------------------------------------------------------------------
# Study design constants (taken from the paper: Methods + Figure 2)
# ----------------------------------------------------------------------------------------------
TIMES = {"C0": 0.0, "C0.5": 0.5, "C1": 1.0, "C2": 2.0, "C3": 3.0, "C4": 4.0, "C6": 6.0, "C12": 12.0}
CONC = list(TIMES)                         # all sampled time points (h after the 7th dose start)
COV7 = ["TBIL", "CrCl", "SCr", "BMI", "Age", "SOFA", "PT"]          # LASSO-retained covariates (paper)
COV_CANDIDATES = ["Age", "Sex", "BMI", "APACHE", "SOFA", "BUN", "SCr", "CrCl", "ALT", "AST",
                  "ALP", "GGT", "TBIL", "DBIL", "IBIL", "TP", "ALB", "PT"]
STRATEGIES = {                                   # the five limited-sampling strategies (paper)
    "C0": ["C0"],
    "C0+C0.5": ["C0", "C0.5"],
    "C0+C6": ["C0", "C6"],
    "C0.5+C6": ["C0.5", "C6"],
    "C0+C0.5+C6": ["C0", "C0.5", "C6"],
}
INPUT_POINTS = ["C0", "C0.5", "C6"]              # every concentration that is a model input in some strategy
METRICS = ["R2", "RMSE", "MAPE", "MDPE", "F20", "F30"]

# AUC0-24/MIC targets quoted in the paper's Introduction (site specific)
TARGETS = {"HAP": 4.5, "cIAI": 6.96, "cSSTI": 17.9}
MICS = [0.5, 1.0, 2.0]


# ----------------------------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------------------------
def load_patients(path: str | None = None) -> pd.DataFrame:
    """One row per patient. Required columns: id, daily_dose, the 18 candidate covariates
    (at least the 7 in COV7) and the 8 concentrations (NaN = not measured)."""
    path = path or os.path.join(DATA, "patients.csv")
    df = pd.read_csv(path)
    need = ["id", "daily_dose"] + COV7 + CONC
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"patients.csv is missing columns: {miss}")
    return df


# ----------------------------------------------------------------------------------------------
# AUC
# ----------------------------------------------------------------------------------------------
def _trapz(t, c) -> float:
    t = np.asarray(t, float); c = np.asarray(c, float)
    return float(np.sum((c[1:] + c[:-1]) * np.diff(t) / 2.0))


def auc24_reference(df: pd.DataFrame) -> np.ndarray:
    """Reference AUC0-24 exactly as in the paper: linear trapezoid over ALL observed samples
    (a missing interior point is bridged by its neighbours), AUC0-24 = 2 x AUC0-12.
    ASSUMPTION (confirm against the original code): if C12 is missing, C0 stands in for the
    12-h value (steady-state periodicity)."""
    out = np.full(len(df), np.nan)
    for i, (_, r) in enumerate(df.iterrows()):
        t, c = [], []
        for k in CONC:
            v = r[k]
            if not np.isnan(v):
                t.append(TIMES[k]); c.append(v)
        if np.isnan(r["C12"]) and not np.isnan(r["C0"]):
            t.append(12.0); c.append(r["C0"])
        out[i] = 2.0 * _trapz(t, c) if len(t) >= 2 else np.nan
    return out


def auc24_independent(df: pd.DataFrame, inputs=INPUT_POINTS) -> np.ndarray:
    """'Decoupled' reference AUC0-24 that contains NONE of the model-input concentrations
    (C0, C0.5, C6): built from C1, C2, C3, C4 and C12 only, with C12 also standing in for the
    t = 0 value (steady-state periodicity). Needs C12 measured (else NaN)."""
    use = [k for k in ["C1", "C2", "C3", "C4"] if k not in inputs]
    out = np.full(len(df), np.nan)
    for i, (_, r) in enumerate(df.iterrows()):
        if np.isnan(r["C12"]):
            continue
        t, c = [0.0], [r["C12"]]
        for k in use:
            if not np.isnan(r[k]):
                t.append(TIMES[k]); c.append(r[k])
        t.append(12.0); c.append(r["C12"])
        out[i] = 2.0 * _trapz(t, c)
    return out


# ----------------------------------------------------------------------------------------------
# Metrics (definitions as in the paper)
# ----------------------------------------------------------------------------------------------
def metrics(y, p) -> dict:
    y = np.asarray(y, float); p = np.asarray(p, float)
    e = p - y
    pe = 100.0 * e / y
    sst = np.sum((y - y.mean()) ** 2)
    return dict(R2=1.0 - np.sum(e ** 2) / sst, RMSE=float(np.sqrt(np.mean(e ** 2))),
                MAPE=float(np.mean(np.abs(pe))), MDPE=float(np.median(pe)),
                F20=float(100 * np.mean(np.abs(pe) <= 20)), F30=float(100 * np.mean(np.abs(pe) <= 30)))


def metrics_matrix(y, P) -> dict:
    """Vectorised metrics. y: (n,), P: (R, n) predictions of R repeats -> dict of (R,) arrays."""
    y = np.asarray(y, float)[None, :]; P = np.asarray(P, float)
    e = P - y; pe = 100.0 * e / y
    sst = np.sum((y - y.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return dict(R2=1.0 - np.sum(e ** 2, axis=1) / sst, RMSE=np.sqrt(np.mean(e ** 2, axis=1)),
                MAPE=np.mean(np.abs(pe), axis=1), MDPE=np.median(pe, axis=1),
                F20=100 * np.mean(np.abs(pe) <= 20, axis=1), F30=100 * np.mean(np.abs(pe) <= 30, axis=1))


def boot_metrics(y, P, B=1000, seed=0, idx=None) -> dict:
    """Patient-level bootstrap of the repeat-averaged metric. y (n,), P (R, n).
    Returns dict metric -> (B,) array. Pass the same `idx` (B, n) to get PAIRED bootstraps."""
    y = np.asarray(y, float); P = np.atleast_2d(np.asarray(P, float))
    n = len(y)
    if idx is None:
        idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
    yb = y[idx]                                   # (B, n)
    Pb = P[:, idx]                                # (R, B, n)
    e = Pb - yb[None]; pe = 100.0 * e / yb[None]
    sst = np.sum((yb - yb.mean(axis=1, keepdims=True)) ** 2, axis=1)            # (B,)
    out = dict(
        R2=(1.0 - np.sum(e ** 2, axis=2) / sst[None]).mean(axis=0),
        RMSE=np.sqrt(np.mean(e ** 2, axis=2)).mean(axis=0),
        MAPE=np.mean(np.abs(pe), axis=2).mean(axis=0),
        MDPE=np.median(pe, axis=2).mean(axis=0),
        F20=(100 * np.mean(np.abs(pe) <= 20, axis=2)).mean(axis=0),
        F30=(100 * np.mean(np.abs(pe) <= 30, axis=2)).mean(axis=0))
    return out


def summarize(y, P, B=1000, seed=0) -> dict:
    """Point estimate (mean over repeats) + 95% patient-bootstrap CI for every metric."""
    P = np.atleast_2d(P)
    pt = {k: float(np.mean(v)) for k, v in metrics_matrix(y, P).items()}
    bt = boot_metrics(y, P, B=B, seed=seed)
    res = {}
    for k in METRICS:
        lo, hi = np.percentile(bt[k], [2.5, 97.5])
        res[k] = (pt[k], float(lo), float(hi))
    return res


def fmt(est, digits=3, pct=False) -> str:
    m, lo, hi = est
    d = 1 if pct else digits
    return f"{m:.{d}f} ({lo:.{d}f}–{hi:.{d}f})"


# ----------------------------------------------------------------------------------------------
# Riley et al. sample-size criteria for a continuous outcome (re-implemented; cross-checked
# against the `pmsampsize` package)
# ----------------------------------------------------------------------------------------------
def _shrink(n, P, r2a):
    return 1 + (P - 2) / (n * math.log(1 - ((r2a * (n - P - 1)) + P) / (n - 1)))


def riley_continuous(P, r2a, mean, sd, shrinkage=0.9, mmoe=1.1) -> dict:
    from scipy.stats import t as tdist
    n1 = P + 2
    while True:
        try:
            es = _shrink(n1, P, r2a)
            if es >= shrinkage:
                break
        except (ValueError, ZeroDivisionError):
            pass
        n1 += 1
    n2 = math.ceil(1 + (P * (1 - r2a)) / 0.05)
    n3 = 234 + P
    n4 = P + 2                                    # criterion (iv) evaluated on its own
    while True:
        df_ = n4 - P - 1
        uci = mean + math.sqrt(sd ** 2 * (1 - r2a) / n4) * tdist.ppf(0.975, df_)
        if uci / mean <= mmoe:
            break
        n4 += 1
    return dict(P=P, r2a=r2a, n_shrinkage=n1, n_r2_optimism=n2, n_resid_var=n3, n_mean=n4,
                n_min=max(n1, n2, n3, n4))
