#!/usr/bin/env python3
"""Update the alignment predictions to HSE06+SOC using hse108_edges.csv (VASP).
   python3 hse_update.py hse108_edges.csv
   - Face-resolved corrections: d_IP_f = IP_f(HSE+SOC) − IP_f(PBE), likewise d_EA_f (same VASP setup) -> added to the QE PBE data
   - Model inputs: E_m,f shifted by −(d_IP_f + d_EA_f)/2 ; E_g shifted by (d_IP − d_EA) (average of both faces)
   - The frozen models (contact-resolved S + GBR residual + localization classifier) are applied unchanged to the new inputs
   - DFT heterostructures (training + blind): layer band edges shifted VBM −= d_IP, CBM −= d_EA (average of the two faces of that compound)
   Ra: hse_shifts.csv, pred7796_hse.csv, hse_eval.txt"""
import sys, json, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor as GBR, GradientBoostingClassifier as GBC
from glr import eVh_glr, build_inputs, kind
E = pd.read_csv(sys.argv[1]); C = pd.read_csv("../hse_rudra/hse108/compounds.csv") if len(sys.argv) < 3 else pd.read_csv(sys.argv[2])
Xel = lambda n: [e for e in ("Se", "Te", "S") if n.startswith(e)][0]
rows = []
for n, g in E.groupby("name"):
    g = g.set_index("calc")
    if not {"1_pbe", "3_hse_soc"} <= set(g.index): continue
    b = g.loc["1_pbe", "bottom"]; bot = "X" if b == Xel(n) else "Q"; top = "Q" if bot == "X" else "X"
    r = dict(name=n)
    for lvl in ("2_pbe_soc", "3_hse_soc"):
        if lvl not in g.index: continue
        tag = "soc" if lvl == "2_pbe_soc" else "hse"
        for side, f in (("bot", bot), ("top", top)):
            r[f"dIP_{f}_{tag}"] = g.loc[lvl, f"IP_{side}"] - g.loc["1_pbe", f"IP_{side}"]
            r[f"dEA_{f}_{tag}"] = g.loc[lvl, f"EA_{side}"] - g.loc["1_pbe", f"EA_{side}"]
        r[f"dEg_{tag}"] = (g.loc[lvl, "Eg"] - g.loc["1_pbe", "Eg"])
    rows.append(r)
H = pd.DataFrame(rows).set_index("name"); H.to_csv("hse_shifts.csv")
print(f"{len(H)} compounds with HSE+SOC | mean dEg {H.dEg_hse.mean():+.3f} eV | mean |dIP_X − dIP_Q| {np.abs(H.dIP_X_hse - H.dIP_Q_hse).mean():.3f} eV")
def shift_inputs(D):
    D = D.copy(); ok = D.L1.isin(H.index) & D.L2.isin(H.index); D = D[ok].copy()
    for i, L, f in ((1, "L1", "f1"), (2, "L2", "f2")):
        dIP = np.array([H.loc[n, f"dIP_{x}_hse"] for n, x in zip(D[L], D[f])]); dEA = np.array([H.loc[n, f"dEA_{x}_hse"] for n, x in zip(D[L], D[f])])
        D[f"Em{i}"] = D[f"Em{i}"] - (dIP + dEA) / 2 if f"Em{i}" in D else np.nan
        D[f"Eg{i}"] = D[f"Eg{i}"] + D[L].map(H.dEg_hse)
        D[f"dEm{i}"] = -(dIP + dEA) / 2
    D["dE"] = D.dE + D.dEm2 - D.dEm1; return D
def shift_dft(D):
    D = D.copy(); ok = D.L1.isin(H.index) & D.L2.isin(H.index); D = D[ok].copy()
    for i, L in ((1, "L1"), (2, "L2")):
        dIP = D[L].map((H.dIP_X_hse + H.dIP_Q_hse) / 2); dEA = D[L].map((H.dEA_X_hse + H.dEA_Q_hse) / 2)
        D[f"VBM{i}"] = D[f"VBM{i}"] - dIP; D[f"CBM{i}"] = D[f"CBM{i}"] - dEA
    return D
# ---- frozen models (trained on PBE)
T = pd.read_csv("train599.csv"); J = json.load(open("model_frozen.json"))
def glr(D): return np.array([eVh_glr(d, a + b, J["S"][c], m) for d, a, b, c, m in zip(D.dE, D.Eg1, D.Eg2, D.contact, build_inputs(D)(J["d_int"]))])
def X(D, b): return pd.DataFrame(dict(dE=D.dE, Eg1=D.Eg1, Eg2=D.Eg2, mv1=D.mv1, mc1=D.mc1, mv2=D.mv2, mc2=D.mc2, s1=D.s1, s2=D.s2, EN1=D.EN_f1, EN2=D.EN_f2, a=D.a,
    nQ=(D.f1 == "Q").astype(int) + (D.f2 == "Q").astype(int), xx=(D.contact == "XX").astype(int), qq=(D.contact == "QQ").astype(int), dEN=D.EN_f2 - D.EN_f1, glr=b))
bT = glr(T); RES = GBR(n_estimators=400, max_depth=3, learning_rate=.03, subsample=.8, random_state=0).fit(X(T, bT), T.eVh_B - bT)
def edges(D, v): return -D.Eg1 / 2, D.Eg1 / 2, D.dE - v - D.Eg2 / 2, D.dE - v + D.Eg2 / 2
def predict(D):
    D = D.copy(); D["eVh_glr"] = glr(D); D["eVh_hyb"] = D.eVh_glr + RES.predict(X(D, D.eVh_glr)); v1, c1, v2, c2 = edges(D, D.eVh_hyb)
    D["type_hyb"] = kind(v1, c1, v2, c2); D["margin"] = np.minimum.reduce([np.abs(v1 - v2), np.abs(c1 - c2), np.abs(np.minimum(c1, c2) - np.maximum(v1, v2))]); return D
out = []
# ---- all 7796 interfaces
A = pd.read_csv("pred7796.csv"); Ah = predict(shift_inputs(A)); Ah = Ah[(Ah.Eg1 > .05) & (Ah.Eg2 > .05)]
Ah["conf"] = (Ah.p_local >= .3) & (Ah.margin > .1); Ah.to_csv("pred7796_hse.csv", index=False)
Ah["pair"] = ["|".join(sorted([a, b])) for a, b in zip(Ah.L1, Ah.L2)]; nt = Ah.groupby("pair").type_hyb.nunique()
out.append(f"7796 (HSE+SOC): gapped {len(Ah)} | confident {Ah.conf.sum()} | types {Ah.type_hyb.value_counts().to_dict()} | face-switching pairs {(nt > 1).mean():.2f}")
# ---- rescoring: training set (shifted DFT) and blind set
Th = predict(shift_inputs(T)); Td = shift_dft(T).set_index("id").loc[Th.id]
okT = (Th.Eg1 > .05).values & (Th.Eg2 > .05).values
tT = np.asarray(kind(Td.VBM1, Td.CBM1, Td.VBM2, Td.CBM2)); acc = (np.asarray(Th.type_hyb) == tT)[okT]
out.append(f"Training (DFT + layer HSE+SOC correction): n={okT.sum()} | type accuracy {acc.mean():.2f} | localized {(np.asarray(Th.type_hyb) == tT)[okT & (Th.p_local.values >= .3)].mean():.2f}")
try:
    B = pd.read_csv("res/glr_table_blind.csv"); P = pd.read_csv("blind60_predictions_FROZEN.csv")
    Bi = A.set_index("id").loc[P.id].reset_index(); Bh = predict(shift_inputs(Bi)); Bd = shift_dft(B).set_index("id")
    Bh = Bh[Bh.id.isin(Bd.index)]; tb = np.asarray(kind(*[Bd.loc[Bh.id, k] for k in ("VBM1", "CBM1", "VBM2", "CBM2")]))
    ok = (Bh.Eg1 > .05).values & (Bh.Eg2 > .05).values; loc = Bh.p_local.values >= .3; hit = np.asarray(Bh.type_hyb) == tb
    out.append(f"Blind (HSE+SOC): n={ok.sum()} | type accuracy {hit[ok].mean():.2f} | localized {hit[ok & loc].mean():.2f} | localized & margin>0.1 eV {hit[ok & loc & (Bh.margin.values > .1)].mean():.2f} | fraction near a boundary {(Bh.margin.values[ok] < .1).mean():.2f}")
except FileNotFoundError: out.append("Blind set: res/glr_table_blind.csv missing")
open("hse_eval.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
