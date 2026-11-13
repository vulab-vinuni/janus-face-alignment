"""Select the 60 blind-test interfaces (no DFT yet, material pairs absent from the training set) from the frozen predictions."""
import numpy as np, pandas as pd
rng = np.random.default_rng(1112)
A = pd.read_csv("pred7796.csv"); E = pd.read_csv("../expand/expand_list.csv"); P = pd.read_csv("../expand/pilot_list.csv"); T = pd.read_csv("train599.csv")
seen_id = set(E.id) | set(P.id)
pr = lambda a, b: "|".join(sorted([a, b]))
A["pair"] = [pr(a, b) for a, b in zip(A.L1, A.L2)]
seen_pair = {pr(a, b) for a, b in zip(E.L1, E.L2)} | {pr(a, b) for a, b in zip(P.L1, P.L2)}
A = A[(A.Eg1 > .05) & (A.Eg2 > .05) & ~A.id.isin(seen_id) & ~A.pair.isin(seen_pair)].copy()
print("candidates (unseen pairs):", len(A), "pairs", A.pair.nunique())
cnt = pd.concat([T.L1, T.L2]).value_counts(); A["rarity"] = A.L1.map(cnt).fillna(0) + A.L2.map(cnt).fillna(0)
# distance to the type boundary (eV): closest pair of edges
v1, c1 = -A.Eg1 / 2, A.Eg1 / 2; v2, c2 = A.dE - A.eVh_glr - A.Eg2 / 2, A.dE - A.eVh_glr + A.Eg2 / 2
A["margin"] = np.minimum.reduce([np.abs(v1 - v2), np.abs(c1 - c2), np.abs(np.minimum(c1, c2) - np.maximum(v1, v2))])
sel = []
def take(df, n, part, w=None):
    df = df[~df.id.isin([s for x in sel for s in x.id])]
    if w is not None: df = df.sort_values(w, ascending=False).head(max(n * 4, n))
    x = df.sample(min(n, len(df)), random_state=int(rng.integers(1e9))).assign(part=part); sel.append(x); return x
# F: 10 face-switch pairs predicted to change type; prefer both configurations in the localized regime (5 pairs) + 5 arbitrary pairs
used_pairs = set()
def flip_pairs(df, n, part, need_local):
    out = []
    for p, g in df.groupby("pair"):
        if p in used_pairs: continue
        g = g.sort_values("p_local", ascending=False)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g.iloc[i], g.iloc[j]
                if a.type_glr != b.type_glr and (not need_local or min(a.p_local, b.p_local) >= .3): out.append((p, a.id, b.id, min(a.p_local, b.p_local))); break
            else: continue
            break
    out = pd.DataFrame(out, columns=["pair", "i1", "i2", "pl"]).sample(frac=1, random_state=int(rng.integers(1e9))).head(n)
    for r in out.itertuples(): used_pairs.add(r.pair)
    x = df[df.id.isin(list(out.i1) + list(out.i2))].assign(part=part); sel.append(x); return x
flip_pairs(A, 5, "F_flip_local", True); flip_pairs(A, 5, "F_flip_any", False)
Ar = A[~A.pair.isin(used_pairs)]
take(Ar[Ar.margin < .1], 15, "B_boundary")
take(Ar, 15, "X_extrap", w=-Ar.rarity) if False else take(Ar.assign(r=-Ar.rarity), 15, "X_extrap", w="r")
take(Ar, 10, "R_random")
S = pd.concat(sel).drop_duplicates("id")
# top up so that the localized regime has ≥ 25 interfaces
nl = (S.p_local >= .3).sum()
if nl < 25: take(Ar[Ar.p_local >= .3], 25 - nl, "L_local_extra"); S = pd.concat(sel).drop_duplicates("id")
S = S.reset_index(drop=True)
print(len(S), S.part.value_counts().to_dict(), "| localized", (S.p_local >= .3).sum(), "| contact", S.contact.value_counts().to_dict(), "| type_glr", S.type_glr.value_counts().to_dict())
keep = ["id", "part", "L1", "f1", "L2", "f2", "contact", "a", "mismatch", "dE", "Eg1", "Eg2", "eVh_glr", "type_glr", "Eg_glr", "VBO_glr", "CBO_glr", "p_local", "regime", "SBO_svm", "p_SBO", "margin", "rarity"]
S[keep].to_csv("blind60_predictions_FROZEN.csv", index=False); S[["id", "L1", "f1", "L2", "f2", "a"]].to_csv("blind_list.csv", index=False)
