import os, sys, numpy as np, pandas as pd
os.environ["S3"] = "../expand/S3.extxyz"; sys.path.insert(0, "../expand")
import mace_scan as M
from ase.io import write
G = np.arange(1.6, 4.41, 0.2)
part, npart = int(sys.argv[1]), int(sys.argv[2])
ids = [l.strip() for l in open("bad_ids.txt") if l.strip()]
L = pd.read_csv("gen_list.csv"); L = L[L.id.isin(ids)].iloc[part::npart]
K = M.calc(); rows = []
for r in L.itertuples():
    best = None
    for st in M.STACK:
        E = []
        for d in G:
            at, _ = M.build(r.L1, r.f1, r.L2, r.f2, r.a, st, d); at.calc = K; E.append(at.get_potential_energy())
        E = np.array(E); j = int(E.argmin()); i = int(np.clip(j, 1, len(E) - 2))
        c = np.polyfit(G[i-1:i+2], E[i-1:i+2], 2)
        if c[0] > 0 and G[i-1] <= -c[1]/(2*c[0]) <= G[i+1]:
            dmin = -c[1]/(2*c[0]); Emin = np.polyval(c, dmin)
        else:
            dmin, Emin = G[j], E[j]
        edge = j in (0, len(E)-1)
        rows.append(dict(id=r.id, stack=st, d_mace=dmin, E_mace=Emin, edge=edge))
        if best is None or Emin < best[1]: best = (st, Emin, dmin)
    for f in os.listdir("geom"):
        if f.startswith(r.id + "__"): os.remove("geom/" + f)
    at, _ = M.build(r.L1, r.f1, r.L2, r.f2, r.a, best[0], best[2]); write(f"geom/{r.id}__{best[0]}.extxyz", at)
    pd.DataFrame(rows).to_csv(f"rescan_{part}.csv", index=False)
print("RESCAN_DONE")
