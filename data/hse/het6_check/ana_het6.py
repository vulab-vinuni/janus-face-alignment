"""Direct HSE06 (+SOC) of six heterostructures vs the monolayer-derived corrections used in Supplementary Note 8.
Run from het6_check/ after run_het6.sh:  python ana_het6.py  -> het6_direct_vs_monolayer.csv
Layer-resolved band edges from PROCAR (LORBIT=11) with the cumulative-weight rule of the QE workflow (bilayer_post.py):
at each k, states sorted by energy, C_L(n) = sum_{m<=n} W_L(m); the layer VBM/CBM is the first state with C_L >= nv_L -/+ 1/2.
Energies relative to the mean of the two vacuum levels (planar-averaged LOCPOT)."""
import numpy as np, pandas as pd, re, os
H = pd.read_csv("hse108_vasp.csv").set_index("name"); S = pd.read_csv("soc_gap_reduction.csv").set_index("name")
def zvals(pot): return [float(x) for x in re.findall(r"ZVAL\s*=\s*([\d.]+)", open(pot).read())]
def procar(fn, nion):
    E, W1, k = [], [], -1
    with open(fn) as f: lines = f.readlines()
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.lstrip().startswith("k-point") and "weight" in l: E.append([]); W1.append([]); k += 1
        elif l.lstrip().startswith("band") and "energy" in l:
            e = float(re.search(r"energy\s+(-?[\d.]+)", l).group(1)); j = i + 1
            while not lines[j].lstrip().startswith("ion"): j += 1
            tot = np.array([float(lines[j + 1 + a].split()[-1]) for a in range(nion)])
            E[k].append(e); W1[k].append(tot[:4].sum() / tot.sum() if tot.sum() > 0 else 0.5); i = j + nion
        i += 1
    return np.array(E), np.array(W1)
def vac(fn):
    with open(fn) as f: L = f.readlines()
    sc = float(L[1]); cell = np.array([[float(x) for x in L[k].split()] for k in (2, 3, 4)]) * sc
    nat = sum(int(x) for x in L[6].split()); z = np.array([float(L[8 + a].split()[2]) for a in range(nat)])
    cart = L[7].strip()[0] in "cCkK"; z = z if cart else z * cell[2, 2]
    i = 8 + nat + 1; nx, ny, nz = [int(x) for x in L[i].split()]; vals = []
    for l in L[i + 1:]:
        vals += [float(x) for x in l.split()]
        if len(vals) >= nx * ny * nz: break
    v = np.array(vals[:nx * ny * nz]).reshape(nz, ny, nx).mean(axis=(1, 2)); zg = np.arange(nz) / nz * cell[2, 2]
    bot = (zg > z.min() - 7) & (zg < z.min() - 4); top = (zg > z.max() + 4) & (zg < z.max() + 7)
    return np.median(v[bot]), np.median(v[top])
def edges(d, nion, nv, spin):
    E, W1 = procar(f"{d}/PROCAR", nion); vb, vt = vac(f"{d}/LOCPOT"); ref = (vb + vt) / 2; r = {}
    for L, W in ((1, W1), (2, 1 - W1)):
        V, C = -1e9, 1e9
        for k in range(len(E)):
            o = np.argsort(E[k]); cw = np.cumsum(W[k][o]) * spin; e = E[k][o]   # cumulative electrons of layer L (spin = electrons per band)
            V = max(V, e[np.argmax(cw >= nv[L - 1] - spin / 2)]); C = min(C, e[np.argmax(cw >= nv[L - 1] + spin / 2)])
        r[f"VBM{L}"], r[f"CBM{L}"] = V - ref, C - ref
    r["dPhi_bi"] = vt - vb; return r
def typ(v1, c1, v2, c2):
    t = "III" if min(c1, c2) < max(v1, v2) else ("I" if (v2 >= v1 and c2 <= c1) or (v1 >= v2 and c1 <= c2) else "II")
    return t, min(abs(v1 - v2), abs(c1 - c2), abs(min(c1, c2) - max(v1, v2)))
rows = []
for hid in open("het6.txt").read().split():
    d = f"calc/{hid}"; l1, l2 = open(f"{d}/layers.txt").read().split(); z = zvals(f"{d}/POTCAR")
    if not os.path.exists(f"{d}/DONE"): print(hid, "not done"); continue
    nv = [sum(z[:4]), sum(z[4:])]   # valence electrons of each layer
    P = edges(f"{d}/1_pbe", 8, nv, 2.0); Hh = edges(f"{d}/2_hse", 8, nv, 2.0)
    r = dict(id=hid, L1=l1, L2=l2, dPhi_bi_pbe=P["dPhi_bi"], dPhi_bi_hse=Hh["dPhi_bi"])
    pred = {}
    for L, n in ((1, l1), (2, l2)):
        for e, col in (("VBM", "dVBM_hse"), ("CBM", "dCBM_hse")):
            r[f"d{e}{L}_het"] = Hh[f"{e}{L}"] - P[f"{e}{L}"]; r[f"d{e}{L}_mono"] = H.loc[n, col]
            r[f"err_{e}{L}"] = r[f"d{e}{L}_mono"] - r[f"d{e}{L}_het"]; pred[f"{e}{L}"] = P[f"{e}{L}"] + H.loc[n, col]
    r["type_pbe"], r["m_pbe"] = typ(P["VBM1"], P["CBM1"], P["VBM2"], P["CBM2"])
    r["type_hse_direct"], r["m_hse_direct"] = typ(Hh["VBM1"], Hh["CBM1"], Hh["VBM2"], Hh["CBM2"])
    r["type_hse_mono"], r["m_hse_mono"] = typ(pred["VBM1"], pred["CBM1"], pred["VBM2"], pred["CBM2"])
    if os.path.exists(f"{d}/DONE_SOC"):
        Ps = edges(f"{d}/3_pbesoc", 8, nv, 1.0); Hs = edges(f"{d}/4_hsesoc", 8, nv, 1.0)
        ad = {}
        for L, n in ((1, l1), (2, l2)):   # monolayer-derived: PBE + HSE06 shift + SOC gap reduction on the VBM (share 1.0)
            ad[f"VBM{L}"] = P[f"VBM{L}"] + H.loc[n, "dVBM_hse"] - S.loc[n, "dEg_SOC_eV"]; ad[f"CBM{L}"] = P[f"CBM{L}"] + H.loc[n, "dCBM_hse"]
            for e in ("VBM", "CBM"): r[f"err_soc_{e}{L}"] = ad[f"{e}{L}"] - Hs[f"{e}{L}"]
        r["type_hsesoc_direct"], r["m_hsesoc_direct"] = typ(Hs["VBM1"], Hs["CBM1"], Hs["VBM2"], Hs["CBM2"])
        r["type_hsesoc_mono"], r["m_hsesoc_mono"] = typ(ad["VBM1"], ad["CBM1"], ad["VBM2"], ad["CBM2"])
        r["dPhi_bi_hsesoc"] = Hs["dPhi_bi"]
    rows.append(r)
R = pd.DataFrame(rows); pd.set_option("display.width", 250); print(R.round(3).T.to_string()); R.to_csv("het6_direct_vs_monolayer.csv", index=False)
e = R.filter(regex=r"^err_(VBM|CBM)").abs().values
print(f"\nmonolayer HSE06 shift vs direct, |err| max {np.nanmax(e):.3f} median {np.nanmedian(e):.3f} eV; type unchanged in "
      f"{(R.type_hse_mono == R.type_hse_direct).sum()}/{len(R)}; dipole change PBE->HSE (dPhi_bi) max {abs(R.dPhi_bi_hse - R.dPhi_bi_pbe).max():.3f} eV")
