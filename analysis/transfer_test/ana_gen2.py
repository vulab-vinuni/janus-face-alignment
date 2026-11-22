import numpy as np, pandas as pd
D = pd.read_csv("gen_table.csv").dropna(subset=["eVh_B"]); D["metal"] = D[["Eg1m", "Eg2m"]].min(axis=1) <= 0
def Sest(g, y="eVh_B"): return 1 - (g.dE_dft * g[y]).sum() / (g.dE_dft ** 2).sum()
def boot(g, y="eVh_B", n=5000):
    rng = np.random.default_rng(20261122); P = g.pair.unique(); G = {p: g[g.pair == p] for p in P}
    b = [Sest(pd.concat([G[p] for p in rng.choice(P, len(P))]), y) for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])
rows = []
for lab, sub in [("all", D), ("semiconducting", D[~D.metal])]:
    for c, g in sub.groupby("cls"):
        lo, hi = boot(g); rows.append(dict(set=lab, cls=c, n=len(g), S=Sest(g), lo=lo, hi=hi, S_A=Sest(g, "eVh_A"),
                                           med_ratio=np.median(1 - g.eVh_B / g.dE_dft), wmin=g.w_min.median()))
R = pd.DataFrame(rows); print(R.round(3).to_string()); R.to_csv("gen_S_final.csv", index=False)
# pooled: Pn-face vs no Pn face
for lab, sub in [("all", D), ("semi", D[~D.metal])]:
    for k, g in sub.groupby("has_Pn_face"):
        lo, hi = boot(g); print(lab, "Pn face" if k else "no Pn face", len(g), round(Sest(g), 3), np.round([lo, hi], 3))
