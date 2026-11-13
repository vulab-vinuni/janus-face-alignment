#!/usr/bin/env python3
"""Small-band-offset (SBO) classification with an SVM:
   SBO = (type II or III) and |Eg_bi| < 0.15 eV; monolayer features combined as sum and |difference| (layer-order invariant);
   drop feature pairs with |r|>0.6; standardize; linear & RBF SVM; 5-fold CV for C, γ by NA; balanced class weights;
   70/30 stratified, 50 repeats; NA = (TPR+TNR)/2. Compared with Anderson / sLR / gLR applied to all data.
   Janus extension: (i) FACE-resolved features (E_m, EN of the contact atoms) + intrinsic dipole steps s_i;
                  (ii) --split group: split by material pair (all contacts and stackings of a pair stay in the same set).
   python3 ml_sbo.py glr_table.csv [--split group|random] [--n 50] [--full-grid]"""
import argparse, json, numpy as np, pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, GroupShuffleSplit, StratifiedKFold, GroupKFold
from sklearn.metrics import balanced_accuracy_score, recall_score
from glr import predict, build_inputs
ap = argparse.ArgumentParser(); ap.add_argument("table"); ap.add_argument("--split", default="group"); ap.add_argument("--n", type=int, default=50)
ap.add_argument("--full-grid", action="store_true"); ap.add_argument("--fit", default="glr_fit.json"); ap.add_argument("--thr", type=float, default=0.15)
a = ap.parse_args()
D = pd.read_csv(a.table); D = D[D.Eg_bi.notna()]
if "best" in D: D = D[D.best]
D = D[(D.Eg1 > 0.05) & (D.Eg2 > 0.05)]                                    # drop layers with gaps < 0.05 eV
y = ((D.type_dft != "I") & (D.Eg_bi.abs() < a.thr)).astype(int).values
f = {}
for k, (c1, c2) in dict(Em=("Em1", "Em2"), Eg=("Eg1", "Eg2"), mc=("mc1", "mc2"), mv=("mv1", "mv2"), EN=("EN_f1", "EN_f2"), s=("s1", "s2")).items():
    f[f"{k}_sum"] = D[c1].values + D[c2].values; f[f"|{k}_diff|"] = np.abs(D[c1].values - D[c2].values)
F = pd.DataFrame(f); C = F.corr().abs(); drop = set()
for i, c in enumerate(F.columns):
    for d in F.columns[i + 1:]:
        if C.loc[c, d] > 0.6 and c not in drop and d not in drop and {c, d} != {"Eg_sum", "|Em_diff|"}: drop.add(d)
F = F.drop(columns=sorted(drop)); X = F.values; groups = (D.L1 + "|" + D.L2).values
print(f"{len(D)} samples, {y.sum()} SBO ({y.mean():.1%}); features: {list(F.columns)} (dropped {sorted(drop)})")
# --- physical alignment models (no training, applied to all data)
res = {}
fit = json.load(open(a.fit)) if a.fit and __import__("os").path.exists(a.fit) else {}
for m in ("anderson", "slr", "glr"):
    if m != "anderson" and m not in fit: continue
    S = fit.get(m, {}).get("S", 1.0); gm = build_inputs(D)(fit[m]["d_int"]) if m == "glr" else None
    _, eg, ty = predict(D.dE.values, D.Eg1.values, D.Eg2.values, m, S, gm)
    p = ((ty != "I") & (np.abs(eg) < a.thr)).astype(int); res[m] = dict(NA=balanced_accuracy_score(y, p), TPR=recall_score(y, p))
# --- SVM
grid_full = dict(lin=dict(svc__C=10.0 ** np.arange(-10, 11)), rbf=dict(svc__C=10.0 ** np.arange(-10, 11), svc__gamma=2.0 ** np.arange(-15, 5)))
grid_fast = dict(lin=dict(svc__C=10.0 ** np.arange(-3, 4)), rbf=dict(svc__C=10.0 ** np.arange(-2, 5), svc__gamma=2.0 ** np.arange(-9, 3, 2)))
grid = grid_full if a.full_grid else grid_fast
spl = (GroupShuffleSplit(n_splits=a.n, test_size=0.3, random_state=0).split(X, y, groups) if a.split == "group"
       else StratifiedShuffleSplit(n_splits=a.n, test_size=0.3, random_state=0).split(X, y))
acc = {k: [] for k in ("lin", "rbf")}; coefs = []
for tr, te in spl:
    for k, kern in (("lin", "linear"), ("rbf", "rbf")):
        pipe = make_pipeline(StandardScaler(), SVC(kernel=kern, class_weight="balanced"))
        cv = GroupKFold(5).split(X[tr], y[tr], groups[tr]) if a.split == "group" else StratifiedKFold(5, shuffle=True, random_state=1).split(X[tr], y[tr])
        g = GridSearchCV(pipe, grid[k], cv=list(cv), scoring="balanced_accuracy", n_jobs=-1).fit(X[tr], y[tr])
        p = g.predict(X[te]); acc[k].append((balanced_accuracy_score(y[te], p), recall_score(y[te], p, zero_division=0)))
        if k == "lin": coefs.append(np.abs(g.best_estimator_[-1].coef_[0]))
for k in acc: res["SVM_" + k] = dict(NA=float(np.mean([x[0] for x in acc[k]])), NA_sd=float(np.std([x[0] for x in acc[k]])), TPR=float(np.mean([x[1] for x in acc[k]])))
imp = np.mean(coefs, 0); imp /= imp.max()
res["importance_linear"] = dict(sorted(zip(F.columns, imp.round(3)), key=lambda t: -t[1]))
print(json.dumps(res, indent=1, default=float)); json.dump(res, open("ml_sbo_result.json", "w"), indent=1, default=float)
