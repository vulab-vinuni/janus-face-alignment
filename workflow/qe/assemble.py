#!/usr/bin/env python3
"""Assemble: interfaces.csv (het108) + mono_strain.csv (strained monolayers) + qe/<id>__<stack>/post/bilayer.json
   -> glr_table.csv (one row per configuration with DFT; with --all, every interface with model inputs only is added for prediction)
   python3 assemble.py --het het108 --mono mono_strain.csv [--all]
   Main columns: id, stack, L1,f1,L2,f2, contact(XX/XQ/QQ), a, dE, Eg1, Eg2, mv*, mc*, gv*, gc*, s1, s2, dEvac_sum, EN_f1, EN_f2,
              [DFT] Em1p, Em2p, layer VBM*/CBM*, dPhi_bi, eVh_A, eVh_B, Eg_bi, type_dft, d_int, Eb_meV_A2, w_min"""
import argparse, os, json, glob, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mono_collect import Mono
from glr import kind
EN = dict(S=2.58, Se=2.55, Te=2.10, P=2.19, As=2.18, Sb=2.05)     # Pauling, atom on the contact face
ap = argparse.ArgumentParser(); ap.add_argument("--het", default="het108"); ap.add_argument("--mono", default="mono_strain.csv")
ap.add_argument("--all", action="store_true"); ap.add_argument("--out", default="glr_table.csv"); ap.add_argument("--dmax", type=float, default=0.03)
a = ap.parse_args()
T = pd.read_csv(f"{a.het}/interfaces.csv"); T = T[T.mismatch <= a.dmax]
M = Mono(a.mono); have = set(M.D.name)
el = lambda name, face: [e for e in ("S", "Se", "Te", "P", "As", "Sb") if (name.startswith(e) if face == "X" else name.endswith(e))][-1]
rows = []
for r in T.itertuples():
    if r.L1 not in have or r.L2 not in have: continue
    m1, m2 = M.at(r.L1, r.a), M.at(r.L2, r.a)
    Em1 = -(m1[f"IP_{r.f1}"] + m1[f"EA_{r.f1}"]) / 2; Em2 = -(m2[f"IP_{r.f2}"] + m2[f"EA_{r.f2}"]) / 2
    s1 = m1["dEvac"] * (1 if r.f1 == "Q" else -1); s2 = m2["dEvac"] * (1 if r.f2 == "X" else -1)
    def M_(m, key, gkey):
        v = m.get({"m_v": "gm_v", "m_c": "gm_c"}[key], np.nan)
        return (v, "dos") if np.isfinite(v) else (m[key] * m[gkey] if np.isfinite(m[key]) else np.nan, "hess")
    (mv1, src), (mc1, _), (mv2, _), (mc2, _) = M_(m1, "m_v", "g_v"), M_(m1, "m_c", "g_c"), M_(m2, "m_v", "g_v"), M_(m2, "m_c", "g_c")
    base = dict(id=r.id, L1=r.L1, f1=r.f1, L2=r.L2, f2=r.f2, contact="".join(sorted(r.f1 + r.f2)), a=r.a, mismatch=r.mismatch,
                Em1=Em1, Em2=Em2, dE=Em2 - Em1, Eg1=m1["Eg"], Eg2=m2["Eg"], mv1=mv1, mc1=mc1, mv2=mv2, mc2=mc2, mass_src=src,   # (g·m*) DOS; g is included in the mass
                gv1=1, gc1=1, gv2=1, gc2=1, s1=s1, s2=s2, dEvac_sum=s1 + s2,
                EN_f1=EN[el(r.L1, r.f1)], EN_f2=EN[el(r.L2, r.f2)], E1=m1["Etot"], E2=m2["Etot"])
    v1, c1 = Em1 - m1["Eg"] / 2, Em1 + m1["Eg"] / 2; v2, c2 = Em2 - m2["Eg"] / 2, Em2 + m2["Eg"] / 2
    base["type_anderson"] = str(kind(np.array(v1), np.array(c1), np.array(v2), np.array(c2)))
    js = sorted(glob.glob(f"{a.het}/qe/{r.id}__*/post/bilayer.json"))
    if not js and a.all: rows.append(base)
    for j in js:
        B = json.load(open(j)); st = j.split("__")[1].split("/")[0]; x = dict(base, stack=st)
        x.update({k: B[k] for k in ("VBM1", "CBM1", "VBM2", "CBM2", "Em1p", "Em2p", "dPhi_bi", "Eg_bi", "d_int", "w_min", "Etot")})
        x["eVh_A"] = x["dE"] - (B["Em2p"] - B["Em1p"]); x["eVh_B"] = x["dEvac_sum"] - B["dPhi_bi"]
        x["type_dft"] = str(kind(*[np.array(B[k]) for k in ("VBM1", "CBM1", "VBM2", "CBM2")]))
        area = r.a ** 2 * np.sqrt(3) / 2; x["Eb_meV_A2"] = 1000 * (B["Etot"] - x["E1"] - x["E2"]) / area
        rows.append(x)
D = pd.DataFrame(rows)
if "stack" in D and D["stack"].notna().any():   # keep the lowest-energy stacking for each interface (one configuration per contact)
    D["best"] = D.groupby("id").Etot.transform("min") == D.Etot
D.to_csv(a.out, index=False); print(f"{len(D)} rows -> {a.out}")
