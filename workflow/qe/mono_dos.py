#!/usr/bin/env python3
"""DOS masses from counting states near the band edges (instead of a parabolic fit; also valid for the 'Mexican-hat' VBM of the Se/Te–In compounds).
   Run in a relaxed MONO/<name>/e<ε>/ (needs tmp/m.save/charge-density).
   python3 mono_dos.py write [--k 48]  -> nscf_dos.in (dense k-mesh, symmetry on, diagonalization='cg')
   python3 mono_dos.py parse           -> dos_mass.json:
        n(ΔE) = Σ_k w_k · #{bands in [E_edge-ΔE, E_edge]} / A   (per spin, Å^-2)
        (g·m*)_DOS / m_e = 2π (ħ²/m_e) n / ΔE                     (equals g_v m* for a parabolic band)
   written for ΔE = 0.05, 0.10, 0.20 eV; the model uses ΔE = 0.10 eV (the energy scale of charge spillage)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qeio import read_xml, write_pw, SET
H2M = 7.619964
XML = "tmp/m.save/data-file-schema.xml"; WINDOWS = (0.05, 0.10, 0.20)
mode = sys.argv[1]
if mode == "write":
    k = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 48
    X = read_xml(XML); nv = int(round(X["nel"])) // 2
    sym = [s for s, _ in X["atoms"]]; pos = np.array([p for _, p in X["atoms"]])
    pd_ = [l.split("=")[1].strip().strip("'") for l in open("relax.in") if "pseudo_dir" in l][0]
    ec = float([l.split("=")[1] for l in open("relax.in") if "ecutwfc" in l][0])
    write_pw("nscf_dos.in", "nscf", sym, pos, X["cell"], prefix="m", pseudo_dir=pd_, nbnd=nv + 8, kgrid=k, s=dict(SET, ecut=ec))
    t = open("nscf_dos.in").read().replace("    nosym = .true.\n", "")          # uniform mesh: keep symmetry (IBZ + weights)
    open("nscf_dos.in", "w").write(t)                                       # qeio already adds diagonalization='cg' for nscf
    json.dump(dict(nv=nv, kdos=k), open("dos_meta.json", "w"))
elif mode == "parse":
    X = read_xml(XML); nv = json.load(open("dos_meta.json"))["nv"]; e = X["e"]; w = X["w"] / X["w"].sum()
    A = np.linalg.norm(np.cross(X["cell"][0], X["cell"][1]))
    vbm, cbm = e[:, nv - 1].max(), e[:, nv:].min()
    out = dict(vbm=float(vbm), cbm=float(cbm), Eg=float(cbm - vbm), k_vbm=X["k"][e[:, nv - 1].argmax()].tolist(),
               k_cbm=X["k"][e[:, nv].argmin()].tolist(), nk=len(w))
    for dE in WINDOWS:
        nvs = (w[:, None] * ((e[:, :nv] >= vbm - dE))).sum() / A
        ncs = (w[:, None] * ((e[:, nv:] <= cbm + dE))).sum() / A
        out[f"gm_v_{dE:.2f}"] = float(2 * np.pi * H2M * nvs / dE); out[f"gm_c_{dE:.2f}"] = float(2 * np.pi * H2M * ncs / dE)
    json.dump(out, open("dos_mass.json", "w"), indent=1)
    print("Eg=%.3f  (g·m)_v=%.2f  (g·m)_c=%.2f  [ΔE=0.10 eV]" % (out["Eg"], out["gm_v_0.10"], out["gm_c_0.10"]))
