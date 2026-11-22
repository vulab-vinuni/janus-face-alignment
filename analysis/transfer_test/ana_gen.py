import json, glob, sys, numpy as np, pandas as pd
sys.path.insert(0, "../gen/pkg/gen/glr_gen")
from mono_collect import Mono
from glr import kind
S = pd.read_csv("gen_selection_FROZEN.csv"); M = Mono("gen/mono_strain_gen.csv")
rows = []
for r in S.itertuples():
    m1, m2 = M.at(r.L1, r.a), M.at(r.L2, r.a)
    Em1 = -(m1[f"IP_{r.f1}"] + m1[f"EA_{r.f1}"]) / 2; Em2 = -(m2[f"IP_{r.f2}"] + m2[f"EA_{r.f2}"]) / 2
    s1 = m1["dEvac"] * (1 if r.f1 == "Q" else -1); s2 = m2["dEvac"] * (1 if r.f2 == "X" else -1)
    x = dict(r._asdict()); x.pop("Index"); x.update(Em1=Em1, Em2=Em2, dE_dft=Em2 - Em1, Eg1m=m1["Eg"], Eg2m=m2["Eg"], s1=s1, s2=s2,
                                                    extrap=m1.get("extrap", False) or m2.get("extrap", False))
    js = glob.glob(f"gen/het/qe/{r.id}__*/post/bilayer.json")
    if js:
        B = json.load(open(js[0])); x["stack"] = js[0].split("__")[1].split("/")[0]
        x.update({k: B[k] for k in ("VBM1", "CBM1", "VBM2", "CBM2", "Em1p", "Em2p", "dPhi_bi", "Eg_bi", "d_int", "w_min", "flat_bot", "flat_top")})
        x["eVh_A"] = x["dE_dft"] - (B["Em2p"] - B["Em1p"]); x["eVh_B"] = s1 + s2 - B["dPhi_bi"]
        x["type_dft"] = str(kind(*[np.array(B[k]) for k in ("VBM1", "CBM1", "VBM2", "CBM2")]))
    rows.append(x)
D = pd.DataFrame(rows); D["cls"] = D.family + "/" + D.contact
D.to_csv("gen_table.csv", index=False)
print(len(D), "rows;", D.eVh_B.notna().sum(), "with DFT; extrap", D.extrap.sum())
print("flat (max)", D[["flat_bot", "flat_top"]].max().round(4).to_dict(), " d_int", D.d_int.describe().round(2).to_dict())
def Sest(g): return 1 - (g.dE_dft * g.eVh_B).sum() / (g.dE_dft ** 2).sum()
rng = np.random.default_rng(20261122); out = {}
for c, g in D.dropna(subset=["eVh_B"]).groupby("cls"):
    bs = [Sest(g.iloc[rng.integers(0, len(g), len(g))]) for _ in range(5000)]
    out[c] = dict(n=len(g), S=Sest(g), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                  SA=1 - (g.dE_dft * g.eVh_A).sum() / (g.dE_dft ** 2).sum(), dErms=np.sqrt((g.dE_dft ** 2).mean()),
                  wmin=g.w_min.median(), pnface=bool(g.has_Pn_face.iloc[0]))
R = pd.DataFrame(out).T; print(R.round(3).to_string()); R.to_csv("gen_S_by_class.csv")
