"""Freeze the models on the 599 DFT heterostructures and predict all 7796 interfaces: contact-resolved S, localization probability (GBC), SBO probability (SVM-RBF)."""
import json, numpy as np, pandas as pd
from sklearn.svm import SVC; from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.ensemble import GradientBoostingClassifier as GBC
from glr import fit, eVh_glr, build_inputs, kind
T = pd.read_csv("data/heterostructures/het599_training_set.csv"); A = pd.read_csv("data/predictions/interfaces7796_descriptors.csv")
# 1) linear response with contact-resolved S
P = fit(T, "eVh_B", "glr", "contact"); S = {k: float(v) for k, v in P["S"].items()}; dint = float(P["d_int"])
def glr_pred(D):
    gm = build_inputs(D)(dint); v = np.array([eVh_glr(d, a + b, S[c], m) for d, a, b, c, m in zip(D.dE, D.Eg1, D.Eg2, D.contact, gm)])
    v1, c1, v2, c2 = -D.Eg1 / 2, D.Eg1 / 2, D.dE - v - D.Eg2 / 2, D.dE - v + D.Eg2 / 2
    return v, kind(v1, c1, v2, c2), np.minimum(c1, c2) - np.maximum(v1, v2), v1, c1, v2, c2
def feat_loc(D, v1, c1, v2, c2):
    return pd.DataFrame(dict(Eg_sum=D.Eg1 + D.Eg2, Eg_min=np.minimum(D.Eg1, D.Eg2), adE=D.dE.abs(), ENs=D.EN_f1 + D.EN_f2, ENd=(D.EN_f1 - D.EN_f2).abs(),
        mv=D.mv1 + D.mv2, mc=D.mc1 + D.mc2, ss=D.s1 + D.s2, a=D.a, nQ=(D.f1 == "Q").astype(int) + (D.f2 == "Q").astype(int), dV=(v1 - v2).abs(), dC=(c1 - c2).abs()))
def feat_svm(D):
    f = {}
    for k, (a_, b_) in dict(Em=("Em1", "Em2"), Eg=("Eg1", "Eg2"), mc=("mc1", "mc2"), mv=("mv1", "mv2"), EN=("EN_f1", "EN_f2"), s=("s1", "s2")).items():
        f[f"{k}_sum"] = D[a_].values + D[b_].values; f[f"|{k}_diff|"] = np.abs(D[a_].values - D[b_].values)
    return pd.DataFrame(f)
out = {}
for nm, D in (("T", T), ("A", A)):
    v, ty, eg, v1, c1, v2, c2 = glr_pred(D); D["eVh_glr"] = v; D["type_glr"] = ty; D["Eg_glr"] = eg
    D["VBO_glr"] = np.abs(v1 - v2); D["CBO_glr"] = np.abs(c1 - c2); out[nm] = (v1, c1, v2, c2)
# 2) localization probability
FT = feat_loc(T, *out["T"]); FA = feat_loc(A, *out["A"])
loc = GBC(n_estimators=300, max_depth=3, learning_rate=.05, subsample=.8, random_state=0).fit(FT, T.w_min >= .6)
A["p_local"] = loc.predict_proba(FA)[:, 1]
# 3) SVM SBO
y = ((T.type_dft != "I") & (T.Eg_bi.abs() < .15)).astype(int)
F = feat_svm(T); C = F.corr().abs(); drop = set()
for i, c in enumerate(F.columns):
    for d in F.columns[i + 1:]:
        if C.loc[c, d] > .6 and c not in drop and d not in drop and {c, d} != {"Eg_sum", "|Em_diff|"}: drop.add(d)
cols = [c for c in F.columns if c not in drop]
g = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=0)),
                 dict(svc__C=10.0 ** np.arange(-2, 5), svc__gamma=2.0 ** np.arange(-9, 3, 2)), cv=list(GroupKFold(5).split(F[cols], y, T.L1 + "|" + T.L2)),
                 scoring="balanced_accuracy", n_jobs=-1).fit(F[cols], y)
A["SBO_svm"] = g.predict(feat_svm(A)[cols]); A["p_SBO"] = g.predict_proba(feat_svm(A)[cols])[:, 1]
A["regime"] = np.where(A.p_local >= .3, "localized", "hybridized")
A.to_csv("data/predictions/alignment_map7796_rebuilt.csv", index=False)
json.dump(dict(S=S, d_int=dint, svm=g.best_params_, svm_cols=cols, loc_thr=0.3, n_train=len(T), date="2026-11-12"), open("models/model_frozen_rebuilt.json", "w"), indent=1, default=float)
ok = (A.Eg1 > .05) & (A.Eg2 > .05)
print("S", S, "d_int", round(dint, 3), "SVM", g.best_params_)
print("7796: Eg ok", ok.sum(), "| type gLR", A[ok].type_glr.value_counts().to_dict(), "| localized", (A[ok].regime == "localized").sum(), "| SBO_svm", A[ok].SBO_svm.sum())
A["pair"] = ["|".join(sorted([a, b])) for a, b in zip(A.L1, A.L2)]
sw = A[ok].groupby("pair").type_glr.nunique(); print("gapped pairs that switch type with the contact face (model):", (sw > 1).sum(), "/", len(sw))
