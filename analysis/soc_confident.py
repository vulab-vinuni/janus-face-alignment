"""Robustness of the confident alignment-type predictions to monolayer SOC corrections (main text, confidence map).
Run from the repository root:  python analysis/soc_confident.py"""
import numpy as np, pandas as pd
A = pd.read_csv("data/predictions/alignment_map7796.csv"); A = A[(A.Eg1 > .05) & (A.Eg2 > .05)].copy()
s = pd.read_csv("data/monolayers/soc_gap_reduction.csv").set_index("name")
def typ(Eg1, Eg2, dEp):
    v1, c1, v2, c2 = -Eg1 / 2, Eg1 / 2, dEp - Eg2 / 2, dEp + Eg2 / 2
    t = np.where(np.minimum(c1, c2) < np.maximum(v1, v2), "III", np.where(((v2 >= v1) & (c2 <= c1)) | ((v1 >= v2) & (c1 <= c2)), "I", "II"))
    return t, np.minimum.reduce([np.abs(v1 - v2), np.abs(c1 - c2), np.abs(np.minimum(c1, c2) - np.maximum(v1, v2))])
dEp = A.dE - A.eVh_hyb; t0, m0 = typ(A.Eg1, A.Eg2, dEp); conf = (A.p_local >= .3).values & (m0 > .1)
d1, d2 = A.L1.map(s.dEg_SOC_eV).values, A.L2.map(s.dEg_SOC_eV).values
ok = A.L1.map(s.Eg_SOC_eV > .05).values & A.L2.map(s.Eg_SOC_eV > .05).values; c = conf & ok
for f, nm in [(1, "VBM only"), (.5, "equal split"), (0, "CBM only")]:
    t, m = typ(A.Eg1 + d1, A.Eg2 + d2, dEp + (1 - 2 * f) * (d2 - d1) / 2)
    print(f"{nm}: {c.sum()} confident interfaces with SOC-gapped monolayers, type unchanged {100 * (t[c] == t0[c]).mean():.1f}%")
