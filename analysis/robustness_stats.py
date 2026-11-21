"""Statistics of Supplementary Notes 1-4: data-set overlap, spillage-threshold counts,
pair-bootstrap confidence intervals of the contact-specific screening factors, charge-transfer
direction within mixed contacts, continuous screening descriptors, face-switching breakdown,
confusion matrices and coverage-accuracy of the alignment type.

Run from the repository root:  python analysis/robustness_stats.py
"""
import itertools, math, sys
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import balanced_accuracy_score as BA
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "analysis")
from glr import fit  # noqa: E402

D = pd.read_csv("data/heterostructures/het599_training_set.csv")
H = pd.read_csv("data/heterostructures/het832_dft.csv")
B = pd.read_csv("data/blind_test/blind_eval.csv"); B = B[B.type_dft.notna()]
pair = lambda a, b: "|".join(sorted([a, b]))
D["pair"] = [pair(a, b) for a, b in zip(D.L1, D.L2)]


def wilson(x, n, z=1.96):
    p = x / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


# Note 1: blind set vs main set
kH = set(zip(H.L1, H.f1, H.L2, H.f2)) | set(zip(H.L2, H.f2, H.L1, H.f1))
print("blind interfaces in 832:", sum(k in kH for k in zip(B.L1, B.f1, B.L2, B.f2)),
      "| shared material pairs:", len({pair(a, b) for a, b in zip(B.L1, B.L2)} & {pair(a, b) for a, b in zip(H.L1, H.L2)}))

# Spillage threshold: bare vs screened offset (DFT electrostatic dipole)
Es = (D.Eg1 + D.Eg2) / 2
print("beyond threshold, bare |dE_m|:", int((D.dE.abs() > Es).sum()), "| screened |dE'|:", int(((D.dE - D.eVh_B).abs() > Es).sum()))

# Note 2: pair-bootstrap CIs of S (generalized model, contact-dependent S)
G0 = D.copy(); G0["contact"] = G0.contact.replace({"QX": "XQ"})
rng = np.random.default_rng(1); pairs = G0.pair.unique(); idx = {p: np.where(G0.pair.values == p)[0] for p in pairs}
bs = []
for _ in range(1000):
    G = G0.iloc[np.concatenate([idx[p] for p in rng.choice(pairs, len(pairs))])].reset_index(drop=True)
    o = fit(G, target="eVh_B", model="glr", groups="contact"); bs.append([o["S"][c] for c in ("XX", "XQ", "QQ")])
bs = np.array(bs)
print("S 95% CI (XX, XQ, QQ):", np.percentile(bs, [2.5, 97.5], axis=0).round(3).tolist())
M = G0.copy(); dq = np.where(M.f1 == "Q", M.Em1 - M.Em2, M.Em2 - M.Em1)
M["grp"] = M.contact; M.loc[(M.contact == "XQ") & (dq > 0), "grp"] = "XQ_Q_higher"; M.loc[(M.contact == "XQ") & (dq <= 0), "grp"] = "XQ_Q_lower"
print("charge-transfer direction:", {k: round(v, 3) for k, v in fit(M, target="eVh_B", model="glr", groups="grp")["S"].items()})

std = lambda v: ((v - v.mean()) / v.std()).values
feat = {"chi": std(D.EN_f1 + D.EN_f2), "d": std(D.d_int), "Eb": std(D.Eb_meV_A2), "nQ": ((D.f1 == "Q").astype(int) + (D.f2 == "Q").astype(int)).values}
y = D.eVh_B.values
for cols in [[], ["nQ"], ["d"], ["Eb"], ["chi"], ["chi", "nQ"], ["chi", "d"], ["chi", "Eb"], ["chi", "d", "Eb", "nQ"]]:
    X = np.column_stack([np.ones(len(D))] + [feat[c] for c in cols]); f = np.zeros(len(D))
    pred = lambda p, ii: (lambda a: a / (1 + a))(np.exp(X[ii] @ p)) * D.dE.values[ii]
    for tr, te in GroupKFold(5).split(D, groups=D.pair):
        f[te] = pred(least_squares(lambda p: pred(p, tr) - y[tr], np.zeros(X.shape[1])).x, te)
    print("descriptors", cols, "out-of-fold r2 %.3f" % (1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()))

# Note 3: face switching
vbo, cbo = D.VBM2 - D.VBM1, D.CBM2 - D.CBM1
D["margin"] = np.minimum.reduce([vbo.abs(), cbo.abs(), (np.minimum(D.CBM1, D.CBM2) - np.maximum(D.VBM1, D.VBM2)).abs()])
def switching(G):
    g = G.groupby("pair"); m = g.size() >= 2; return int((g.type_dft.nunique()[m] > 1).sum()), int(m.sum())
for nm, G in [("all", D), ("margin>0.1", D[D.margin > .1]), ("no QQ", D[D.contact != "QQ"]), ("no QQ, margin>0.1", D[(D.contact != "QQ") & (D.margin > .1)])]:
    x, n = switching(G); print("switching", nm, f"{x}/{n}", "Wilson CI %.2f-%.2f" % wilson(x, n))

# Note 4: confusion matrices and coverage-accuracy (out-of-fold predictions in het599_training_set.csv)
loc = D.p_local >= .3
for nm, m in [("CV all", np.ones(len(D), bool)), ("CV localized", loc), ("CV hybridized", ~loc)]:
    t, p = D.type_dft[m], D.type_glr[m]
    print(nm, "acc %.2f BA %.2f majority %.2f" % ((t == p).mean(), BA(t, p), t.value_counts(normalize=True).max())); print(pd.crosstab(t, p))
for nm, m in [("blind all", np.ones(len(B), bool)), ("blind localized", B.regime == "localized"), ("blind hybridized", B.regime == "hybridized")]:
    t, p = B.type_dft[m], B.type_glr[m]
    print(nm, "acc %.2f BA %.2f majority %.2f" % ((t == p).mean(), BA(t, p), t.value_counts(normalize=True).max())); print(pd.crosstab(t, p))
dEp = D.dE - D["eVh_gLR-contact"]; v1, c1, v2, c2 = -D.Eg1 / 2, D.Eg1 / 2, dEp - D.Eg2 / 2, dEp + D.Eg2 / 2
pm = np.minimum.reduce([(v2 - v1).abs(), (c2 - c1).abs(), (np.minimum(c1, c2) - np.maximum(v1, v2)).abs()])
ok = D.type_dft == D.type_glr; okb = B.type_dft == B.type_glr; lb = B.regime == "localized"
for t in (0, .05, .1, .15, .2):
    s = loc & (pm > t); sb = lb & (B.margin > t)
    print("margin > %.2f eV: CV coverage %.1f%% acc %.2f | blind %d/%d correct" % (t, 100 * s.mean(), ok[s].mean(), okb[sb].sum(), sb.sum()))

# Note 2 (continued): controlled comparison Te-Te (XX) vs Sb-containing QQ contacts
import re
at = lambda n, f: re.findall(r"[A-Z][a-z]?", n)[0 if f == "X" else 3]
D["atoms"] = ["-".join(sorted([at(a, f), at(b, g)])) for a, f, b, g in zip(D.L1, D.f1, D.L2, D.f2)]
rng = np.random.default_rng(0)
for atoms, c in [("Te-Te", "XX"), ("As-Sb", "QQ"), ("P-Sb", "QQ"), ("S-S", "XX")]:
    G = D[(D.atoms == atoms) & (D.contact == c)]; pp = G.pair.unique()
    Sfun = lambda H: 1 - (H.dE * H.eVh_B).sum() / (H.dE ** 2).sum()
    bs = [Sfun(pd.concat([G[G.pair == p] for p in rng.choice(pp, len(pp))])) for _ in range(1000)]
    print(f"{c} {atoms}: N={len(G)} S={Sfun(G):.2f} CI {np.percentile(bs, 2.5):.2f}-{np.percentile(bs, 97.5):.2f}")

# Note 2 (continued): screening refitted with SOC-corrected midgap levels, bracketing the VBM/CBM split.
# Same 543 interfaces as Supplementary Table 3: both monolayers keep a PBE+SOC gap > 0.05 eV (12x12 mesh).
socd = pd.read_csv("data/monolayers/soc_gap_reduction.csv").set_index("name")
soc, ok = socd.dEg_SOC_eV, socd.Eg_SOC_eV > .05
m = (D.L1.map(ok) & D.L2.map(ok)).values
d1, d2 = D.L1.map(soc), D.L2.map(soc)
o = fit(G0[m].reset_index(drop=True), target="eVh_B", model="glr", groups="contact")
print("no SOC, same set:", int(m.sum()), {k: round(v, 3) for k, v in o["S"].items()}, "r2 %.3f" % o["r2"])
for f_vbm in (0.0, 0.5, 1.0):  # fraction of the (negative) gap change taken by the VBM
    G = G0.copy(); G["dE"] = G.dE + (1 - 2 * f_vbm) * (d2 - d1).values / 2; G["Eg1"] = G.Eg1 + d1.values; G["Eg2"] = G.Eg2 + d2.values
    G = G[m].reset_index(drop=True); G["Eg1"] = G.Eg1.clip(lower=.01); G["Eg2"] = G.Eg2.clip(lower=.01)
    o = fit(G, target="eVh_B", model="glr", groups="contact")
    print("SOC, VBM share %.1f:" % f_vbm, len(G), {k: round(v, 3) for k, v in o["S"].items()}, "r2 %.3f" % o["r2"])

# Note 2 (continued): per-contact fits with and without an intercept (push-back test) and origin estimator
for c in ("XX", "QX", "QQ"):
    g = D[D.contact == c]; X = np.column_stack([np.ones(len(g)), g.dE]); b = np.linalg.lstsq(X, g.eVh_B, rcond=None)[0]
    pp = g.pair.unique(); ix = {p: np.where(g.pair.values == p)[0] for p in pp}; bb = []
    r = np.random.default_rng(2)
    for _ in range(1000):
        h = g.iloc[np.concatenate([ix[p] for p in r.choice(pp, len(pp))])]; bb.append(np.linalg.lstsq(np.column_stack([np.ones(len(h)), h.dE]), h.eVh_B, rcond=None)[0][0])
    print(f"{c}: origin S {1 - (g.dE * g.eVh_B).sum() / (g.dE ** 2).sum():.3f} | with intercept S {1 - b[1]:.3f}, intercept {b[0]:+.3f} eV CI {np.percentile(bb, 2.5):+.3f} to {np.percentile(bb, 97.5):+.3f}")

# Interface-dipole / intrinsic-step check: eV_h^(B) regressed on s1+s2 alone
X = np.column_stack([np.ones(len(D)), D.s1 + D.s2]); f = X @ np.linalg.lstsq(X, D.eVh_B, rcond=None)[0]
print("r2(eVh_B ~ s1+s2) = %.3f" % (1 - ((D.eVh_B - f) ** 2).sum() / ((D.eVh_B - D.eVh_B.mean()) ** 2).sum()))

# Localization classifier: precision and recall for w_min >= 0.6
for nm, t, p in [("CV", D.w_min >= .6, D.p_local >= .3), ("blind", B.w_min >= .6, B.regime == "localized")]:
    print(f"{nm}: predicted localized {int(p.sum())}, truly localized {int(t.sum())}, precision {(t & p).sum() / p.sum():.2f}, recall {(t & p).sum() / t.sum():.2f}")


# Note 2 (continued): further single descriptors (distance relative to Bondi vdW radii; band-edge weight w_min)
rvdw = {"S": 1.80, "Se": 1.90, "Te": 2.06, "P": 1.80, "As": 1.85, "Sb": 2.06}
rsum = np.array([rvdw[at(a, f)] + rvdw[at(b, g)] for a, f, b, g in zip(D.L1, D.f1, D.L2, D.f2)])
for nm, v in [("d/r_vdW", D.d_int.values / rsum), ("w_min", D.w_min.values)]:
    z = (v - v.mean()) / v.std(); X = np.column_stack([np.ones(len(D)), z]); f = np.zeros(len(D))
    pred = lambda p, ii: (lambda a: a / (1 + a))(np.exp(X[ii] @ p)) * D.dE.values[ii]
    for tr, te in GroupKFold(5).split(D, groups=D.pair):
        f[te] = pred(least_squares(lambda p: pred(p, tr) - y[tr], np.zeros(2)).x, te)
    print("descriptor", nm, "out-of-fold r2 %.3f" % (1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()))

# Supplementary Table 9: |eVh_A - eVh_B| by contact and w_min bin
b = pd.cut(D.w_min, [0, .4, .6, .8, 1.01], labels=["<=0.4", "0.4-0.6", "0.6-0.8", ">0.8"])
print(pd.crosstab(D.contact, b, values=(D.eVh_A - D.eVh_B).abs(), aggfunc="median").round(2))

# Supplementary Table 5: CIs of the two charge-transfer directions within mixed contacts (300 pair-bootstrap resamples)
rng = np.random.default_rng(5); bs = []
for _ in range(300):
    Gb = M.iloc[np.concatenate([np.where(M.pair.values == p)[0] for p in rng.choice(pairs, len(pairs))])].reset_index(drop=True)
    o = fit(Gb, target="eVh_B", model="glr", groups="grp"); bs.append([o["S"]["XQ_Q_higher"], o["S"]["XQ_Q_lower"]])
print("direction CIs (Q higher, Q lower):", np.percentile(bs, [2.5, 97.5], axis=0).round(2).tolist())
