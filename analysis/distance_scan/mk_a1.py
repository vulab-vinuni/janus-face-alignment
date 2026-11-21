"""A1: rigid interlayer-distance scan. Builds a1/qe/<id>__<stack>__d<+x.xx>/relax.in from the DFT-relaxed
geometries of the 832 set (repository extxyz); layer 2 (upper layer, atoms 5-8) is shifted rigidly along z.
Only static SCF + electrostatic post-processing is run on these inputs (task_bilayer.sh), no relaxation."""
import sys, re, os, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])                    # path to glr_janus (for qeio.write_pw)
from qeio import write_pw
XYZ = sys.argv[2]; OUT = "a1"
IDS = ["H20992", "H21356", "H03396", "H03523", "H15111", "H14018", "H21905"]
DELTA = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.3, 0.7, 1.5]
DMIN = 2.25                                         # skip points whose shortest interlayer atom-atom distance < DMIN (Å)
fr = {}; L = open(XYZ).read().splitlines(); i = 0
while i < len(L):
    n = int(L[i]); h = L[i + 1]; kv = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', h))
    if kv.get("in_dft832") == "T" and kv["id"] in IDS:
        cell = np.array(kv["Lattice"].strip('"').split(), float).reshape(3, 3)
        sym = [l.split()[0] for l in L[i + 2:i + 2 + n]]; pos = np.array([l.split()[1:4] for l in L[i + 2:i + 2 + n]], float)
        fr[kv["id"]] = (kv, cell, sym, pos)
    i += n + 2
assert set(fr) == set(IDS), set(IDS) - set(fr)
rows = []; lst = []
for hid in IDS:
    kv, cell, sym, pos = fr[hid]; n = len(sym) // 2
    assert pos[:n, 2].mean() < pos[n:, 2].mean()   # layer 1 = first half = lower layer
    d0 = pos[n:, 2].min() - pos[:n, 2].max()
    for dl in DELTA:
        p = pos.copy(); p[n:, 2] += dl
        dmin = min(np.linalg.norm(p[j] - p[k] + a * cell[0] + b * cell[1]) for j in range(n) for k in range(n, 2 * n) for a in (-1, 0, 1) for b in (-1, 0, 1))
        if dmin < DMIN: print(f"skip {hid} {dl:+.1f} (shortest {dmin:.2f} A)"); continue
        name = f"{hid}__{kv['stack']}__d{dl:+.2f}"; d = f"{OUT}/qe/{name}"; os.makedirs(d, exist_ok=True)
        write_pw(f"{d}/relax.in", "scf", sym, p, cell, prefix="b", pseudo_dir="../../../pseudo_xamq")
        lst.append(d); rows.append(dict(dir=name, id=hid, stack=kv["stack"], L1=kv["L1"], f1=kv["f1"], L2=kv["L2"], f2=kv["f2"],
                                        contact=kv["contact"], delta=dl, d_plane=d0 + dl, dmin_atom=dmin))
open(f"{OUT}/list_a1.txt", "w").write("\n".join(lst) + "\n"); pd.DataFrame(rows).to_csv(f"{OUT}/a1_points.csv", index=False)
print(len(lst), "points")
