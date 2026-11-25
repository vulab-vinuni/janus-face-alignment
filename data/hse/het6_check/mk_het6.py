"""Build VASP inputs for the direct HSE06 (and HSE06+SOC) check on six DFT heterostructures.
Run from the repository root: python data/hse/het6_check/mk_het6.py  -> data/hse/het6_check/calc/<id>/POSCAR, layers.txt
Each layer keeps its own species block in POSCAR (types 1-4 = layer 1, 5-8 = layer 2), so layer projections need no z cut."""
import re, os, numpy as np
SEL = ["H03752", "H10072", "H00877", "H03913", "H00855", "H23023"]   # XX loc, XX hyb, XQ loc, XQ hyb, QQ (w 0.59), QQ hyb
SOC = ["H10072", "H00877", "H23023"]
src = open("data/heterostructures/het832_relaxed.extxyz").read().split("\n")
out = "data/hse/het6_check"; i = 0; found = {}
while i < len(src) and src[i].strip():
    n = int(src[i]); h = src[i + 1]; at = src[i + 2:i + 2 + n]; i += 2 + n
    kv = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', h)); hid = kv["id"]
    if hid in SEL and kv.get("lowest_stack") == "T": found[hid] = (kv, at)
assert set(found) == set(SEL), set(SEL) - set(found)
with open(f"{out}/het6.txt", "w") as f: f.write("\n".join(SEL) + "\n")
with open(f"{out}/soc3.txt", "w") as f: f.write("\n".join(SOC) + "\n")
for hid in SEL:
    kv, at = found[hid]; L = np.array([float(x) for x in kv["Lattice"].strip('"').split()]).reshape(3, 3)
    sp = [a.split()[0] for a in at]; r = np.array([[float(x) for x in a.split()[1:4]] for a in at])
    o = np.argsort(r[:, 2]); sp = [sp[j] for j in o]; r = r[o]            # bottom to top: layer 1 = lowest four atoms
    r[:, 2] += L[2, 2] / 2 - (r[:, 2].min() + r[:, 2].max()) / 2           # centre the bilayer at z = c/2 (DIPOL 0.5 0.5 0.5)
    d = f"{out}/calc/{hid}"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/POSCAR", "w") as f:
        f.write(f"{hid} {kv['L1']}({kv['f1']})/{kv['L2']}({kv['f2']}) {kv['contact']} stack {kv['stack']}\n1.0\n")
        for v in L: f.write(" ".join(f"{x:14.8f}" for x in v) + "\n")
        f.write(" ".join(sp) + "\n" + " ".join(["1"] * 8) + "\nCartesian\n")
        for p in r: f.write(" ".join(f"{x:14.8f}" for x in p) + "\n")
    with open(f"{d}/layers.txt", "w") as f: f.write(f"{kv['L1']} {kv['L2']}\n")
    gap = r[4, 2] - r[3, 2]
    print(hid, kv["L1"], kv["f1"], kv["L2"], kv["f2"], kv["contact"], kv["stack"], f"a={float(kv['a']):.4f}", f"plane gap={gap:.2f}", " ".join(sp))
