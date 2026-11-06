"""
02b_audit_published_tables.py - consistency audit of published (R2, RMSE) pairs; uses ONLY
published numbers (no data needed). Checks every (R2, RMSE) pair that shares an outcome
vector with the identity  RMSE^2 = (1 - R2) * Var(y).
Writes results/table_R8a_tableS4_audit.md (training rows of Supplementary Table S4) and
table_R8b_test_set_audit.md (Tables 2 and 4).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES

def implied(r2, rm, dr=0.0005, dm=0.0005):
    return rm ** 2 / (1 - r2), (rm - dm) ** 2 / (1 - r2 + dr), (rm + dm) ** 2 / (1 - r2 - dr)

test_pairs = {  # Dai et al. 2026, Table 2 (ML) and Table 4 (PopPK): 20 test patients; (R2, RMSE)
 "C0 RF": (0.607, 4.648), "C0 XGB": (0.669, 4.267), "C0 MLP": (0.707, 4.017), "C0 LSTM": (0.818, 3.164),
 "C0+C0.5 RF": (0.803, 3.295), "C0+C0.5 XGB": (0.793, 3.377), "C0+C0.5 MLP": (0.664, 4.301), "C0+C0.5 LSTM": (0.885, 2.518),
 "C0+C6 RF": (0.773, 3.536), "C0+C6 XGB": (0.795, 3.361), "C0+C6 MLP": (0.807, 3.261), "C0+C6 LSTM": (0.819, 3.157),
 "C0.5+C6 RF": (0.571, 4.857), "C0.5+C6 XGB": (0.621, 4.567), "C0.5+C6 MLP": (0.678, 4.207), "C0.5+C6 LSTM": (0.736, 3.811),
 "C0+C0.5+C6 RF": (0.798, 3.331), "C0+C0.5+C6 XGB": (0.799, 3.324), "C0+C0.5+C6 MLP": (0.811, 3.225), "C0+C0.5+C6 LSTM": (0.887, 2.496),
 "PopPK model A": (0.772, 3.541), "PopPK model B": (0.826, 3.093), "PopPK model C": (0.830, 3.062), "PopPK model D": (0.662, 4.310)}
train_pairs = {  # training rows of Supplementary Table S4 (78 training patients)
 "LSTM C0+C0.5": (0.898, 2.409), "LSTM C0.5+C6": (0.900, 2.409), "RF C0": (0.869, 2.762), "RF C0+C0.5": (0.868, 2.762), "RF C0.5+C6": (0.867, 2.762)}

def table(pairs, vref):
    rows, lo_max, hi_min = [], -1e9, 1e9
    for k, (r2, rm) in pairs.items():
        v, lo, hi = implied(r2, rm); lo_max, hi_min = max(lo_max, lo), min(hi_min, hi)
        rows.append({"Row": k, "R2": f"{r2:.3f}", "RMSE printed": f"{rm:.3f}", "Implied Var(y)": f"{v:.2f} ({lo:.2f}–{hi:.2f})",
                     f"Compatible with Var = {vref}?": "yes" if lo <= vref <= hi else "NO", f"RMSE implied by Var = {vref}": f"{np.sqrt((1 - r2) * vref):.3f}"})
    return pd.DataFrame(rows), lo_max, hi_min

def write(df, name):
    df.to_csv(os.path.join(RES, name + ".csv"), index=False); cols = list(df.columns)
    open(os.path.join(RES, name + ".md"), "w").write("\n".join(["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"] + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]) + "\n")

ta, lo, hi = table(train_pairs, 58.0); write(ta, "table_R8a_tableS4_audit")
print("training rows: common interval", round(lo, 2), round(hi, 2), "-> consistent:", lo <= hi); print(ta.to_string())
tb, lo, hi = table(test_pairs, 55.0); write(tb, "table_R8b_test_set_audit")
vs = [implied(*v)[0] for v in test_pairs.values()]
print("test rows: common interval", round(lo, 3), round(hi, 3), "-> consistent:", lo <= hi, "| mean Var", round(np.mean(vs), 2), "SD", round(np.sqrt(np.mean(vs)), 2))
print("range of rounding intervals:", round(min(implied(*v)[1] for v in test_pairs.values()), 2), round(max(implied(*v)[2] for v in test_pairs.values()), 2))
print(tb.to_string())
