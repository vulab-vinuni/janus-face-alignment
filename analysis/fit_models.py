#!/usr/bin/env python3
"""Fit and evaluate the alignment models, including the Janus (contact-resolved) extensions.
   python3 fit_models.py glr_table.csv [--target eVh_A|eVh_B]  -> glr_fit.json, glr_eval.csv, prints an r² table
   Variants:  anderson (S=1) | midgap (S=0) | sLR (global S) | gLR (global S, d_int)
                  gLR+valley (DOS × valley degeneracy) | gLR-contact (separate S for XX/XQ/QQ)  <- Janus extension
   Evaluation: r² of eVh and of the heterostructure gap; type I/II/III accuracy vs DFT; with --cv: 5-fold grouped by material pair."""
import argparse, json, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from glr import fit, predict, build_inputs, r2
ap = argparse.ArgumentParser(); ap.add_argument("table"); ap.add_argument("--target", default="eVh_A"); ap.add_argument("--cv", action="store_true")
a = ap.parse_args()
D = pd.read_csv(a.table); D = D[D[a.target].notna()]
if "best" in D: D = D[D.best]
D = D[(D.Eg1 > 0.05) & (D.Eg2 > 0.05) & D[["mv1", "mc1", "mv2", "mc2"]].notna().all(axis=1)].reset_index(drop=True)
print(f"{len(D)} configurations; median |eVh_A - eVh_B| = {np.median(np.abs(D.eVh_A - D.eVh_B)):.3f} eV" if "eVh_B" in D else len(D))
V = dict(anderson=dict(model="slr", fixed=1.0), midgap=dict(model="slr", fixed=0.0), sLR=dict(model="slr"), gLR=dict(model="glr"),
         **{"gLR+valley": dict(model="glr", valley=True), "gLR-contact": dict(model="glr", groups="contact")})
out, fits, rows = {}, {}, []
for name, o in V.items():
    if "fixed" in o:
        S = o["fixed"]; v = (1 - S) * D.dE.values; p = dict(S={"None": S})
    else:
        p = fit(D, a.target, o["model"], o.get("groups"), o.get("valley", False)); v = p["pred"]
    Sv = np.array([p["S"].get(str(c), p["S"].get("None")) for c in (D[o["groups"]] if o.get("groups") else [None] * len(D))], float)
    gm = build_inputs(D, o.get("valley", False))(p["d_int"]) if o["model"] == "glr" else None
    dEp = D.dE.values - v; eg = np.minimum(D.Eg1 / 2, dEp + D.Eg2 / 2) - np.maximum(-D.Eg1 / 2, dEp - D.Eg2 / 2)
    from glr import kind
    ty = kind(-D.Eg1.values / 2, D.Eg1.values / 2, dEp - D.Eg2.values / 2, dEp + D.Eg2.values / 2)
    rec = dict(model=name, r2_eVh=r2(D[a.target], v), mae_eVh=np.abs(D[a.target] - v).mean(), r2_Eg=r2(D.Eg_bi, eg),
               type_acc=(ty == D.type_dft.values).mean(), S={k: round(float(x), 3) for k, x in p["S"].items()}, d_int=p.get("d_int"))
    if a.cv and "fixed" not in o:      # cross-validation grouped by material pair
        cvp = np.zeros(len(D))
        for tr, te in GroupKFold(5).split(D, groups=D.L1 + "|" + D.L2):
            q = fit(D.iloc[tr], a.target, o["model"], o.get("groups"), o.get("valley", False))
            Dt = D.iloc[te]; St = np.array([q["S"].get(str(c), q["S"].get("None")) for c in (Dt[o["groups"]] if o.get("groups") else [None] * len(Dt))])
            if o["model"] == "slr": cvp[te] = (1 - St) * Dt.dE.values
            else:
                from glr import eVh_glr
                gm_t = build_inputs(Dt, o.get("valley", False))(q["d_int"])
                cvp[te] = [eVh_glr(d, x + y, s, m) for d, x, y, s, m in zip(Dt.dE, Dt.Eg1, Dt.Eg2, St, gm_t)]
        rec["r2_eVh_cv"] = r2(D[a.target], cvp)
    rows.append(rec); fits[name.lower().replace("+", "_").replace("-", "_")] = dict(S=float(np.mean(list(p["S"].values()))), d_int=p.get("d_int"))
    D["eVh_" + name] = v
E = pd.DataFrame(rows); print(E.round(3).to_string(index=False)); E.to_csv("glr_eval.csv", index=False)
json.dump(dict(slr=fits["slr"], glr=fits["glr"]), open("glr_fit.json", "w"), indent=1, default=float)
D.to_csv("glr_pred.csv", index=False)
