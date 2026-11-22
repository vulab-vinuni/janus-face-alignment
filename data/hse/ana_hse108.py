"""Run on the login node after the array: face-resolved IP/EA for PBE and HSE, and the HSE shifts of VBM/CBM.
X face = side of the chalcogen atom. Shifts use the mean of the two vacuum levels (independent of the dipole step)."""
import os, sys, json, numpy as np
R = os.path.dirname(os.path.abspath(__file__)); meta = json.load(open(f"{R}/meta.json"))
def edges(d):
    L = open(f"{d}/EIGENVAL").read().split("\n"); nel, nk, nb = map(int, L[5].split()[:3])
    E = []; i = 7
    for k in range(nk):
        E.append([float(L[i + 1 + b].split()[1]) for b in range(nb)]); i += nb + 2
    E = np.array(E); nv = int(round(float(open(f"{d}/OUTCAR").read().split("NELECT =")[1].split()[0]))) // 2
    return E[:, nv - 1].max(), E[:, nv].min()
def vac(d, Xidx, Qidx):
    t = open(f"{d}/LOCPOT").read().split("\n"); s = float(t[1]); cell = np.array([[float(x) for x in t[i].split()] for i in (2, 3, 4)]) * s
    nat = sum(map(int, t[6].split())); i0 = 8 + nat + 1; ng = list(map(int, t[i0].split()))
    v = np.array(" ".join(t[i0 + 1:i0 + 1 + int(np.ceil(np.prod(ng) / 5)) + 1]).split()[:np.prod(ng)], float)
    v = v.reshape(ng[2], ng[1], ng[0]).mean(axis=(1, 2)); c = cell[2, 2]; z = np.linspace(0, c, ng[2], endpoint=False)
    pz = np.array([[float(x) for x in t[8 + j].split()[:3]] for j in range(nat)])[:, 2] * c
    lo, hi = pz.min(), pz.max()
    vlo = v[(z > lo - 7) & (z < lo - 4)].mean(); vhi = v[(z > hi + 4) & (z < hi + 7)].mean()
    flat = max(np.ptp(v[(z > lo - 7) & (z < lo - 4)]), np.ptp(v[(z > hi + 4) & (z < hi + 7)]))
    return (vlo, vhi, flat) if pz[Xidx] < pz[Qidx] else (vhi, vlo, flat)   # (vacuum on X side, on Q side)
rows = []
for n in [l.strip() for l in open(f"{R}/list.txt") if l.strip()]:
    m = meta[n]; out = dict(name=n)
    for tag, d in (("pbe", "1_pbe"), ("hse", "2_hse")):
        p = f"{R}/calc/{n}/{d}"
        try:
            vb, cb = edges(p); vX, vQ, fl = vac(p, 0, 3)
            out.update({f"Eg_{tag}": cb - vb, f"IP_X_{tag}": vX - vb, f"EA_X_{tag}": vX - cb, f"IP_Q_{tag}": vQ - vb, f"EA_Q_{tag}": vQ - cb,
                        f"dEvac_{tag}": vQ - vX, f"flat_{tag}": fl, f"VBM_vm_{tag}": vb - (vX + vQ) / 2, f"CBM_vm_{tag}": cb - (vX + vQ) / 2})
        except Exception as e: out[f"err_{tag}"] = str(e)[:60]
    if "VBM_vm_hse" in out and "VBM_vm_pbe" in out:
        out["dVBM_hse"] = out["VBM_vm_hse"] - out["VBM_vm_pbe"]; out["dCBM_hse"] = out["CBM_vm_hse"] - out["CBM_vm_pbe"]
    rows.append(out)
keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "name", k))
with open(f"{R}/hse108_vasp.csv", "w") as f:
    f.write(",".join(keys) + "\n")
    for r in rows: f.write(",".join(f"{r[k]:.5f}" if isinstance(r.get(k), float) else str(r.get(k, "")) for k in keys) + "\n")
ok = [r for r in rows if "dVBM_hse" in r]
print(f"{len(ok)}/{len(rows)} complete -> hse108_vasp.csv")
if ok:
    a = np.array([[r["Eg_pbe"], r["Eg_hse"], r["dVBM_hse"], r["dCBM_hse"]] for r in ok])
    print("median Eg PBE %.2f HSE %.2f | dVBM %.2f dCBM %+.2f eV" % tuple(np.median(a, 0)))
