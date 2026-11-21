"""Blind test, round 2 (pre-registered): 40 interfaces from unseen material pairs, predictions taken unchanged
from the frozen models of round 1 (pred7796.csv, models trained on the 599 fitting interfaces, 12 Nov 2026).

Design
  P_switch (10): 5 material pairs with two confident configurations of different predicted type
  P_conf   (20): further confident interfaces, stratified by contact type
  C_bound   (5): predicted localized, predicted type margin <= 0.1 eV (control)
  C_hyb     (5): predicted hybridized (control)
  confident = predicted localized (p_local >= 0.3) AND predicted type margin > 0.1 eV

Pre-registered criteria (fixed before any DFT calculation of these interfaces)
  1. alignment-type accuracy on the 30 confident interfaces >= 0.85 (>= 26/30)
  2. at least 4 of the 5 confident switching pairs switch type in DFT
  3. MAE of the electrostatic dipole eV_h^(B) (ML-corrected model) <= 0.11 eV
  Controls are diagnostic: their accuracy is reported but has no pass criterion.
"""
import numpy as np, pandas as pd, hashlib, json, os, sys
B1 = "../blind"; EX = "../expand"
rng = np.random.default_rng(20261121)
A = pd.read_csv("../ms/pred7796_hyb.csv").drop(columns=["margin", "conf", "pair"], errors="ignore")  # frozen round-1 predictions incl. ML dipole (identical to round-1 frozen values)
pr = lambda a, b: "|".join(sorted([a, b]))
A["pair"] = [pr(a, b) for a, b in zip(A.L1, A.L2)]
seen = pd.concat([pd.read_csv(f) for f in (f"{EX}/pilot_list.csv", f"{EX}/expand_list.csv", f"{EX}/expand_list_b2.csv",
                                            f"{B1}/blind60_predictions_FROZEN.csv")], ignore_index=True)
seen_id = set(seen.id); seen_pair = {pr(a, b) for a, b in zip(seen.L1, seen.L2)}
# also exclude anything in the released 832 set
H = pd.read_csv("/home/claude/janus-face-alignment/data/heterostructures/het832_dft.csv")
seen_pair |= {pr(a, b) for a, b in zip(H.L1, H.L2)}
A = A[(A.Eg1 > .05) & (A.Eg2 > .05) & ~A.id.isin(seen_id) & ~A.pair.isin(seen_pair)].copy()
v1, c1 = -A.Eg1 / 2, A.Eg1 / 2; v2, c2 = A.dE - A.eVh_glr - A.Eg2 / 2, A.dE - A.eVh_glr + A.Eg2 / 2
A["margin"] = np.minimum.reduce([np.abs(v1 - v2), np.abs(c1 - c2), np.abs(np.minimum(c1, c2) - np.maximum(v1, v2))])
A["loc"] = A.p_local >= .3; A["conf"] = A["loc"] & (A.margin > .1)
print("candidates", len(A), "pairs", A.pair.nunique(), "| confident", int(A.conf.sum()), A[A.conf].contact.value_counts().to_dict())
sel = []; used_pairs = set()
# P_switch: pairs with >= 2 confident configurations of different predicted type
C = A[A.conf]; cand = []
for p, g in C.groupby("pair"):
    if g.type_glr.nunique() > 1:
        g = g.sample(frac=1, random_state=int(rng.integers(1e9)))
        a = g.iloc[0]; b = g[g.type_glr != a.type_glr].iloc[0]; cand.append((p, a.id, b.id))
cand = pd.DataFrame(cand, columns=["pair", "i1", "i2"]).sample(frac=1, random_state=int(rng.integers(1e9))).head(5)
used_pairs |= set(cand.pair); sel.append(A[A.id.isin(list(cand.i1) + list(cand.i2))].assign(part="P_switch"))
print("confident switching pairs available:", len(C.groupby("pair").filter(lambda g: g.type_glr.nunique() > 1).pair.unique()))
# P_conf: 20 confident interfaces, one per material pair, stratified by contact as available
R = C[~C.pair.isin(used_pairs)].sample(frac=1, random_state=int(rng.integers(1e9))).drop_duplicates("pair")
want = {"XX": 7, "QX": 7, "QQ": 6}; take = []
for c, n in want.items(): take.append(R[R.contact == c].head(n))
T = pd.concat(take)
if len(T) < 20: T = pd.concat([T, R[~R.id.isin(T.id)].head(20 - len(T))])
used_pairs |= set(T.pair); sel.append(T.assign(part="P_conf"))
# controls
Rr = A[~A.pair.isin(used_pairs)].sample(frac=1, random_state=int(rng.integers(1e9))).drop_duplicates("pair")
cb = Rr[Rr["loc"] & (Rr.margin <= .1)].head(5); used_pairs |= set(cb.pair); sel.append(cb.assign(part="C_bound"))
Rr = Rr[~Rr.pair.isin(used_pairs)]
ch = Rr[~Rr["loc"]].head(5); sel.append(ch.assign(part="C_hyb"))
S = pd.concat(sel).reset_index(drop=True)
keep = ["id", "part", "L1", "f1", "L2", "f2", "contact", "a", "mismatch", "dE", "Eg1", "Eg2", "eVh_glr", "type_glr", "Eg_glr",
        "VBO_glr", "CBO_glr", "p_local", "regime", "SBO_svm", "p_SBO", "margin"]
if "eVh_hyb" in A: keep += ["eVh_hyb"]
S[keep].to_csv("blind2_predictions_FROZEN.csv", index=False)
S[["id", "L1", "f1", "L2", "f2", "a"]].to_csv("blind2_list.csv", index=False)
h = hashlib.sha256(open("blind2_predictions_FROZEN.csv", "rb").read()).hexdigest()
json.dump({"file": "blind2_predictions_FROZEN.csv", "sha256": h, "n": len(S), "parts": S.part.value_counts().to_dict(),
           "criteria": {"type_accuracy_confident": ">=26/30", "switching_pairs": ">=4/5", "dipole_MAE_eV": "<=0.11"},
           "models": "frozen round-1 models (model_frozen.json, pred7796.csv)", "seed": 20261121}, open("blind2_registration.json", "w"), indent=1)
print(len(S), S.part.value_counts().to_dict(), S.contact.value_counts().to_dict(), S.type_glr.value_counts().to_dict(), "sha256", h[:16])
