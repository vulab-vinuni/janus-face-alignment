"""Generalization test: choose heterostructures in three further Janus families (BiTeI, L4, MXY) and record the
pre-registered prediction of the rule 'strong screening requires a pnictogen face' before any DFT calculation.
Face labels follow the S3 convention: 'X' = bottom face (first layer), 'Q' = top face (last layer)."""
import pandas as pd, numpy as np, ast, itertools, hashlib, json
S = pd.read_csv("S1a.csv"); S["lay"] = S.layers.apply(ast.literal_eval)
rng = np.random.default_rng(20261122)
PN = {"N", "P", "As", "Sb", "Bi"}; CH = {"O", "S", "Se", "Te"}; HA = {"F", "Cl", "Br", "I"}
kind = lambda el: "Pn" if el in PN else "Ch" if el in CH else "Ha" if el in HA else el
GAP = {"BiTeI": .5, "L4": .5, "MXY": .4}; NWANT = {"BiTeI": 30, "L4": 45, "MXY": 20}
rows = []
for fam in ("BiTeI", "L4", "MXY"):
    G = S[(S.family == fam) & (~S.metal) & S.intact & (S.mag.abs() < 1e-3) & (S.gap > GAP[fam])].reset_index(drop=True)
    for i, j in itertools.combinations(range(len(G)), 2):
        A, B = G.loc[i], G.loc[j]
        if fam in ("MXY", "BiTeI") and A.phase != B.phase: continue
        a = (A.a + B.a) / 2
        if abs(A.a - B.a) / a > .03: continue
        for f1, f2 in itertools.product("XQ", "XQ"):      # f1: face of layer 1 pointing up; f2: face of layer 2 pointing down
            e1, e2 = A.lay[0] if f1 == "X" else A.lay[-1], B.lay[0] if f2 == "X" else B.lay[-1]
            Em = lambda R, f: -((R.IP_bot + R.EA_bot) if f == "X" else (R.IP_top + R.EA_top)) / 2
            dE = Em(B, f2) - Em(A, f1); eg = (A.gap + B.gap) / 2
            rows.append(dict(family=fam, L1=A["name"], f1=f1, L2=B["name"], f2=f2, el1=e1, el2=e2,
                             contact="-".join(sorted([kind(e1), kind(e2)])), atoms="-".join(sorted([e1, e2])),
                             a=a, mismatch=abs(A.a - B.a) / a, dE=dE, Eg1=A.gap, Eg2=B.gap, gap_margin=eg - abs(dE)))
C = pd.DataFrame(rows); C["pair"] = C.L1 + "|" + C.L2
print(C.groupby(["family", "contact"]).size())
# usable: well-conditioned offset, layers stay gapped at separation (no broken gap at large d)
U = C[(C.dE.abs() >= .30) & (C.gap_margin >= .15)].copy()
print("usable:"); print(U.groupby(["family", "contact"]).size())
sel = []
for fam, g in U.groupby("family"):
    cls = sorted(g.contact.unique()); per = int(np.ceil(NWANT[fam] / len(cls)))
    for c in cls:
        h = g[g.contact == c].sample(frac=1, random_state=int(rng.integers(1e9)))
        # spread over |dE| (terciles) and avoid reusing a pair more than twice
        h["bin"] = pd.qcut(h.dE.abs().rank(method="first"), 3, labels=False) if len(h) >= 3 else 0
        take, used = [], {}
        for b in range(3):
            for r in h[h.bin == b].itertuples():
                if len(take) >= per * (b + 1) / 3 + .5: break
                if used.get(r.pair, 0) >= 2: continue
                take.append(r.Index); used[r.pair] = used.get(r.pair, 0) + 1
        sel.append(h.loc[take])
T = pd.concat(sel).reset_index(drop=True)
T.insert(0, "id", [f"G{k:03d}" for k in range(len(T))])
# pre-registered prediction
T["has_Pn_face"] = T.contact.str.contains("Pn")
T["S_pred"] = np.where(T.contact == "Pn-Pn", "<=0.65 (strong)", np.where(T.has_Pn_face, "intermediate (0.6-0.75)", ">=0.75 (weak)"))
T.to_csv("gen_selection_FROZEN.csv", index=False)
h = hashlib.sha256(open("gen_selection_FROZEN.csv", "rb").read()).hexdigest()
reg = {"file": "gen_selection_FROZEN.csv", "sha256": h, "n": len(T), "by_family_contact": {f"{a}/{b}": int(n) for (a, b), n in T.groupby(["family", "contact"]).size().items()},
       "rule": "strong screening requires a pnictogen face; coupling modulates but does not create it",
       "criteria": {"1": "S (through origin, per family and contact class) >= 0.72 for every class without a pnictogen face (BiTeI Ch-Ch, Ch-Ha, Ha-Ha; L4 Ch-Ch; MXY Ch-Ch)",
                    "2": "S <= 0.65 for Pn-Pn contacts (L4 IV-V layers); Ch-Pn contacts (L4) intermediate",
                    "3": "pair-bootstrap 95% CI of the Pn-Pn class does not overlap with that of any class without a pnictogen face"},
       "seed": 20261122, "date": "2026-11-22"}
json.dump(reg, open("gen_registration.json", "w"), indent=1)
print(T.groupby(["family", "contact"]).agg(n=("id", "size"), dE_med=("dE", lambda x: x.abs().median()), pairs=("pair", "nunique"), mism=("mismatch", "max")).round(3))
print("compounds:", len(set(T.L1) | set(T.L2)), "| sha256", h[:16])
