"""00_make_synthetic_data.py

Generate a fully synthetic dataset with the same schema as the study data so
that the pipeline in this repository can be executed end-to-end without any
patient information.

The synthetic values are drawn independently from parametric distributions
whose ranges are informed only by the aggregate summary statistics reported in
the article (cohort-level min/max/mean). No real patient record was used,
permuted or perturbed. Missingness is introduced at exactly the counts
reported in Table R7a: C0.5/C1/C3/C4/C6/C12 missing in 6/4/4/3/5/8
of 98 patients; C0 and C2 are never missing.

Usage:  python src/00_make_synthetic_data.py          # writes data/synthetic_patients.csv
"""
import os
import numpy as np

RNG = np.random.default_rng(20261107)
N = 98
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "synthetic_patients.csv")

COLS = ["id", "Age", "Sex", "BMI", "APACHE", "SOFA", "BUN", "SCr", "CrCl", "ALT", "AST",
        "ALP", "GGT", "TBIL", "DBIL", "IBIL", "TP", "ALB", "PT", "daily_dose",
        "C0", "C0.5", "C1", "C2", "C3", "C4", "C6", "C12"]


def clip(x, lo, hi, nd=None):
    x = np.clip(x, lo, hi)
    return np.round(x, nd) if nd is not None else x


def main():
    age = clip(RNG.normal(68, 14, N), 29, 96, 0)
    sex = (RNG.random(N) < 0.62).astype(int)
    bmi = clip(RNG.normal(22.6, 3.4, N), 15.0, 34.0, 1)
    apache = clip(RNG.normal(25, 7, N), 9, 40, 0)
    sofa = clip(np.round(RNG.normal(6.7, 2.8, N)), 0, 13, 0)
    bun = clip(RNG.lognormal(np.log(12), 0.55, N), 2.9, 53.4, 2)
    scr = clip(RNG.lognormal(np.log(70), 0.5, N), 14, 319, 0)
    crcl = clip(RNG.normal(94, 45, N), 13.5, 300, 1)
    alt = clip(RNG.lognormal(np.log(18), 0.7, N), 3, 114, 0)
    ast = clip(RNG.lognormal(np.log(22), 0.7, N), 5, 97, 0)
    alp = clip(RNG.normal(112, 45, N), 35, 271, 0)
    ggt = clip(RNG.lognormal(np.log(50), 0.9, N), 3, 532, 0)
    tbil = clip(RNG.lognormal(np.log(16), 0.6, N), 2.2, 66.4, 1)
    dbil = clip(RNG.lognormal(np.log(8), 0.7, N), 0.7, 52.7, 1)
    ibil = clip(RNG.normal(10.7, 6, N), 1.0, 40.8, 1)
    tp = clip(RNG.normal(51, 6, N), 34.1, 66.1, 1)
    alb = clip(RNG.normal(27.8, 3.6, N), 20.5, 37.8, 1)
    pt = clip(RNG.normal(16.5, 2.0, N), 12.2, 22.1, 1)

    dose = np.array([100] * 91 + [200] * 7, dtype=float)
    RNG.shuffle(dose)

    # pseudo-PK profile: lognormal peak scaled by dose, decaying after the infusion peak
    peak = RNG.lognormal(np.log(1.05), 0.5, N) * np.where(dose == 200, 1.5, 1.0)
    factor = {"C0": 0.32, "C0.5": 1.05, "C1": 1.00, "C2": 0.72,
              "C3": 0.62, "C4": 0.55, "C6": 0.42, "C12": 0.30}
    conc = {}
    for name, f in factor.items():
        conc[name] = clip(peak * f * RNG.lognormal(0, 0.12, N), 0.05, 5.0, 2)

    # missingness at the published counts (Table R7a); C0 and C2 are never missing
    for name, k in {"C0.5": 6, "C1": 4, "C3": 4, "C4": 3, "C6": 5, "C12": 8}.items():
        conc[name][RNG.choice(N, size=k, replace=False)] = np.nan

    rows = []
    for i in range(N):
        row = [str(i + 1), *(f"{v:g}" for v in
                            [age[i], sex[i], bmi[i], apache[i], sofa[i], bun[i], scr[i], crcl[i],
                             alt[i], ast[i], alp[i], ggt[i], tbil[i], dbil[i], ibil[i], tp[i],
                             alb[i], pt[i], dose[i]])]
        row += ["" if np.isnan(conc[c][i]) else f"{conc[c][i]:g}" for c in
                ["C0", "C0.5", "C1", "C2", "C3", "C4", "C6", "C12"]]
        rows.append(row)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(COLS) + "\n")
        fh.writelines(",".join(r) + "\n" for r in rows)

    print(f"wrote {OUT} ({N} rows, fully synthetic, seed 20261107)")
    for c in ["C0", "C0.5", "C1", "C2", "C3", "C4", "C6", "C12"]:
        v = conc[c]
        print(f"  {c:>5}: missing {int(np.isnan(v).sum()):2d}   median {np.nanmedian(v):.2f} mg/L")


if __name__ == "__main__":
    main()
