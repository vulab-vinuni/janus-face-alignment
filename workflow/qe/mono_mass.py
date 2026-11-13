#!/usr/bin/env python3
"""Run in MONO/<name>/e<ε>/ after relaxation.  Two modes:
   python3 mono_mass.py write   -> read tmp/m.save/data-file-schema.xml, locate the VBM/CBM k-points on the SCF mesh, write nscf_mass.in
                                   (7 points around each edge: centre, ±x, ±y, ±(x+y)/√2; δ = 0.02 Å^-1)
   python3 mono_mass.py parse   -> read the NSCF output, fit the 2D Hessian tensor -> m*_DOS = ħ²/√det(H) (units of m_e), take the heaviest branch
                                   among degenerate bands (≤5 meV); g_v from the k-point weights of the IBZ mesh
   Uses the full tensor (independent of the x,y axes) and also records the valley degeneracy g_v."""
import sys, json, os, shutil, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qeio import read_xml, write_pw, SET
H2M = 7.619964  # ħ²/m_e  (eV Å²)
DK = 0.02; DEG = 0.005
XML = "tmp/m.save/data-file-schema.xml"
mode = sys.argv[1]
if mode == "write":
    X = read_xml(XML); nv = int(round(X["nel"])) // 2; e = X["e"]
    meta = dict(l.split() for l in open("meta.txt")); kg = int(meta.get("kgrid", SET["k"]))
    iv, ic = e[:, nv - 1].argmax(), e[:, nv].argmin()
    gv = lambda i: int(round(X["w"][i] * kg * kg / X["w"].sum()))         # number of equivalent k-points (≈ valley degeneracy)
    info = dict(nv=nv, k_vbm=X["k"][iv].tolist(), k_cbm=X["k"][ic].tolist(), g_v=gv(iv), g_c=gv(ic),
                vbm_grid=float(e[iv, nv - 1]), cbm_grid=float(e[ic, nv]), ef=X["ef"], etot=X["etot"], alat=X["alat"])
    dirs = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]], float)
    dirs[5:] /= np.sqrt(2)
    kp = []
    for k0 in (X["k"][iv], X["k"][ic]):
        for d in dirs: kp.append(np.r_[k0[:2] + DK * d, 0.0] * X["alat"] / (2 * np.pi))     # tpiba
    kp = [list(k) + [1.0] for k in kp]
    sym = [s for s, _ in X["atoms"]]; pos = np.array([p for _, p in X["atoms"]])
    pd_ = [l.split("=")[1].strip().strip("'") for l in open("relax.in") if "pseudo_dir" in l][0]
    ec = float([l.split("=")[1] for l in open("relax.in") if "ecutwfc" in l][0])
    write_pw("nscf_mass.in", "nscf", sym, pos, X["cell"], prefix="m", pseudo_dir=pd_, nbnd=nv + 8, kpts=kp, s=dict(SET, ecut=ec))
    json.dump(info, open("edges.json", "w"), indent=1)
elif mode == "parse":
    info = json.load(open("edges.json")); X = read_xml(XML); e = X["e"]; nv = info["nv"]
    def mass(E7, band_idx, sign):
        out = []
        for b in band_idx:
            E = E7[:, b]; Exx = (E[1] + E[2] - 2 * E[0]) / DK ** 2; Eyy = (E[3] + E[4] - 2 * E[0]) / DK ** 2
            Euu = (E[5] + E[6] - 2 * E[0]) / DK ** 2; Exy = Euu - (Exx + Eyy) / 2
            H = sign * np.array([[Exx, Exy], [Exy, Eyy]]); det = np.linalg.det(H)
            out.append(H2M / np.sqrt(det) if det > 0 and np.trace(H) > 0 else np.nan)
        return out
    Ev, Ec = e[:7], e[7:14]
    vb = [b for b in range(nv) if Ev[0, nv - 1] - Ev[0, b] < DEG]; cb = [b for b in range(nv, e.shape[1]) if Ec[0, b] - Ec[0, nv] < DEG]
    mv, mc = mass(Ev, vb, -1), mass(Ec, cb, +1)
    info.update(m_v=float(np.nanmax(mv)) if np.isfinite(mv).any() else None, m_c=float(np.nanmax(mc)) if np.isfinite(mc).any() else None,
                m_v_all=mv, m_c_all=mc, n_deg_v=len(vb), n_deg_c=len(cb))
    json.dump(info, open("edges.json", "w"), indent=1, default=float)
    print("m_v=%s m_c=%s (deg %d/%d) g_v=%d g_c=%d" % (info["m_v"], info["m_c"], len(vb), len(cb), info["g_v"], info["g_c"]))
