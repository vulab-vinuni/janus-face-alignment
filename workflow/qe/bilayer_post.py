#!/usr/bin/env python3
"""Step B: post-processing of a relaxed heterostructure (directory het108/qe/<id>__<stack>/ containing relax.out).
   python3 bilayer_post.py write  [--test]  -> post/scf.in (final relaxed coordinates), post/projwfc.in, post/pp.in, post/avg.in
   python3 bilayer_post.py parse            -> post/bilayer.json:
       - LAYER-resolved band edges: assign each state (k,n) by its layer weight and count the valence electrons
         of each layer (z_valence in the UPF) -> VBM_L, CBM_L even with charge spillage (type III)
       - E'_m,L = (VBM_L+CBM_L)/2 ; vacuum levels below/above the bilayer ; ΔΦ_bi = Evac_top - Evac_bot ; Etot
       - w_min: smallest layer weight of the 4 band-edge states (hybridization index)
   Run in the configuration directory (e.g. het108/qe/H00360__AB)."""
import sys, os, re, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qeio import write_pw, pp_input, avg_input, read_xml, planar_avg, vac_levels, SET


def scf_etot(fn, fallback):
    """SCF E_tot (eV) from scf.out — the XML may have been overwritten by the NSCF run."""
    import re
    try:
        v = re.findall(r"^!\s+total energy\s+=\s+([-0-9.]+) Ry", open(fn).read(), re.M)
        return float(v[-1]) * 13.605693122994 if v else fallback
    except OSError:
        return fallback


def final_coords(fn):
    """Coordinates + cell from relax.out ('Begin final coordinates' block; otherwise the last ATOMIC_POSITIONS)."""
    t = open(fn).read(); blk = t.split("Begin final coordinates")[-1] if "Begin final coordinates" in t else t
    i = blk.rfind("ATOMIC_POSITIONS"); unit = blk[i:].split("\n")[0]; L = blk[i:].split("\n")[1:]
    sym, pos = [], []
    for l in L:
        p = l.split()
        if len(p) < 4 or not re.match(r"^[A-Z][a-z]?$", p[0]): break
        sym.append(p[0]); pos.append([float(x) for x in p[1:4]])
    return sym, np.array(pos), unit


def read_input_cell(fn):
    L = open(fn).read().split("CELL_PARAMETERS angstrom\n")[1].split("\n")[:3]
    return np.array([[float(x) for x in l.split()] for l in L])


def zval(pseudo_dir, el):
    t = open(os.path.join(pseudo_dir, f"{el}.upf")).read()
    return float(re.search(r'z_valence="\s*([0-9.Ee+-]+)"', t).group(1))


def layer_weights_wfc(save, zcut, c):
    """Layer-1 weight of every state from the QE wavefunction files (wfcN.dat): ρ̄(z) = Σ_(h,k) |FFT_z c(h,k,Gz)|²;
       W1 = ∫_{z<zcut} ρ̄ dz (slab does not cross the cell boundary). No PP_PSWFC needed (absent in SG15) -> replaces projwfc."""
    from scipy.io import FortranFile
    nz = 256; zg = np.arange(nz) / nz * c; m1 = zg < zcut
    E, W = [], []
    X = read_xml(os.path.join(save, "data-file-schema.xml")); E = X["e"]
    for ik in range(1, len(E) + 1):
        f = FortranFile(os.path.join(save, f"wfc{ik}.dat"), "r")
        f.read_record(np.uint8)   # ik, xk, ispin, gamma_only, scalef
        ngw, igwx, npol, nbnd = f.read_ints(np.int32); f.read_reals(np.float64)          # b1,b2,b3 (one 3×3 record)
        mill = f.read_ints(np.int32).reshape(igwx, 3)
        hk = mill[:, 0] * 10000 + mill[:, 1]; uhk, inv = np.unique(hk, return_inverse=True); gz = mill[:, 2] % nz
        w = []
        for b in range(nbnd):
            cf = f.read_reals(np.complex128)[:igwx]
            A = np.zeros((len(uhk), nz), complex); A[inv, gz] = cf
            rho = (np.abs(np.fft.ifft(A, axis=1)) ** 2).sum(0)
            w.append(rho[m1].sum() / rho.sum())
        W.append(w); f.close()
    return np.array(E), np.array(W)


def parse_projwfc(fn, layer_of_atom):
    """-> E (nk,nb), W1 (nk,nb): layer-1 weight (normalized to the total projection)."""
    t = open(fn).read()
    st2atom = {int(m.group(1)): int(m.group(2)) for m in re.finditer(r"state #\s*(\d+): atom\s+(\d+)", t)}
    E, W = [], []
    for kb in re.split(r"\n\s*k =\s+", t)[1:]:
        e_k, w_k = [], []
        for m in re.finditer(r"==== e\(\s*\d+\) =\s+([-0-9.]+) eV ====(.*?)\|psi\|\^2 =", kb, re.S):
            e_k.append(float(m.group(1))); s = np.zeros(2)
            for c, st in re.findall(r"([0-9.]+)\*\[#\s*(\d+)\]", m.group(2)):
                s[layer_of_atom[st2atom[int(st)]]] += float(c)
            w_k.append(s[0] / s.sum() if s.sum() > 0 else 0.5)
        if e_k: E.append(e_k); W.append(w_k)
    return np.array(E), np.array(W)


mode = sys.argv[1]; test = "--test" in sys.argv
if mode == "write":
    if os.path.exists("relax.out") and "--from-input" not in sys.argv:
        sym, pos, unit = final_coords("relax.out")
    else:   # pipeline test only: use the initial coordinates
        sym, pos, unit = final_coords("relax.in"); print("WARNING: using relax.in coordinates (not relaxed)")
    cell = read_input_cell("relax.in")
    # 2D cutoff: the potential is exact only within c/2 of ALL charges. Bilayers are ~13–15 Å thick -> c = 35 Å leaves no flat vacuum region.
    # Rebuild a cell of height CPOST (default 45 Å) with the slab centred for the potential SCF (relaxation at c = 35 Å remains valid).
    CPOST = float(os.environ.get("CPOST", 45.0))
    if cell[2, 2] < CPOST:
        cell = cell.copy(); cell[2] = [0, 0, CPOST]
    pos = pos.copy(); pos[:, 2] += cell[2, 2] / 2 - (pos[:, 2].max() + pos[:, 2].min()) / 2
    assert "angstrom" in unit, unit
    pd_ = [l.split("=")[1].strip().strip("'") for l in open("relax.in") if "pseudo_dir" in l][0]
    s = dict(SET, ecut=40.0, k=6, conv=1e-7) if test else SET
    nel = sum(zval(pd_, e) for e in sym) if os.path.isdir(pd_) else None
    os.makedirs("post", exist_ok=True)
    write_pw("post/scf.in", "scf", sym, pos, cell, prefix="b", pseudo_dir=pd_, nbnd=int(nel // 2 + 16) if nel else None, s=s)
    open("post/projwfc.in", "w").write("&PROJWFC\n  prefix='b'\n  outdir='./tmp'\n  filpdos='b.pdos'\n  lsym=.false.\n/\n")
    pp_input("post/pp.in", "b"); avg_input("post/avg.in", "vtot.dat", 1500, cell[2, 2] / 0.529177)
    print("ghi post/ (nel=%s)" % nel)
elif mode == "parse":
    X = read_xml("post/tmp/b.save/data-file-schema.xml"); sym = [s for s, _ in X["atoms"]]; n = len(sym) // 2
    lay = {i + 1: (0 if i < n else 1) for i in range(len(sym))}
    pd_ = [l.split("=")[1].strip().strip("'") for l in open("post/scf.in") if "pseudo_dir" in l][0]
    nv = [int(round(sum(zval(pd_, e) for e in sym[:n]))) // 2, int(round(sum(zval(pd_, e) for e in sym[n:]))) // 2]
    zat_ = np.array([p[2] for _, p in X["atoms"]]); zcut = (zat_[:n].max() + zat_[n:].min()) / 2
    if os.path.exists("post/layerw.npz") and "--reweight" not in sys.argv:
        Z = np.load("post/layerw.npz"); E, W1 = Z["E"], Z["W1"]
    elif os.path.exists("post/projwfc.out") and "JOB DONE" in open("post/projwfc.out").read():
        E, W1 = parse_projwfc("post/projwfc.out", lay)
    else:
        E, W1 = layer_weights_wfc("post/tmp/b.save", zcut, X["cell"][2, 2])
        np.savez_compressed("post/layerw.npz", E=E, W1=W1, k=X["k"], w=X["w"])   # saved: later analysis does not need the wavefunctions
    res = {}
    # Layer band edges from CUMULATIVE (fractional) weights: at each k, sort states by energy, C_L(n) = Σ_{m≤n} W_L(m).
    # Top valence band of layer L = first state with C_L ≥ nv_L - 1/2; bottom conduction band = first state with C_L ≥ nv_L + 1/2.
    # Robust against accidental degeneracies between the layers (mixed ~50/50), which break a simple majority assignment.
    for L in (0, 1):
        vb, cb, wv, wc = [], [], [], []
        for k in range(E.shape[0]):
            o = np.argsort(E[k]); WL = (W1[k] if L == 0 else 1 - W1[k])[o]; C = np.cumsum(WL)
            iv = int(np.argmax(C >= nv[L] - 0.5)); ic = int(np.argmax(C >= nv[L] + 0.5))
            if C[-1] < nv[L] + 0.5: raise RuntimeError("not enough empty bands for layer %d – increase nbnd" % (L + 1))
            vb.append(E[k, o[iv]]); cb.append(E[k, o[ic]]); wv.append(WL[iv]); wc.append(WL[ic])
        iv, ic = int(np.argmax(vb)), int(np.argmin(cb))
        res[f"VBM{L + 1}"], res[f"CBM{L + 1}"] = vb[iv], cb[ic]; res[f"wv{L + 1}"], res[f"wc{L + 1}"] = wv[iv], wc[ic]
        res[f"Em{L + 1}p"] = (vb[iv] + cb[ic]) / 2
    z, V = planar_avg("post/avg.dat"); zat = [p[2] for _, p in X["atoms"]]
    vl = vac_levels(z, V, zat, X["cell"][2, 2])
    z1, z2 = np.array(zat[:n]), np.array(zat[n:])
    res.update(Evac_bot=vl["bot"], Evac_top=vl["top"], dPhi_bi=vl["top"] - vl["bot"], flat_bot=vl["flat_bot"], flat_top=vl["flat_top"],
               Etot=scf_etot("post/scf.out", X["etot"]), Ef=X["ef"], a=float(np.linalg.norm(X["cell"][0])), d_int=float(z2.min() - z1.max()),
               w_min=float(min(res["wv1"], res["wc1"], res["wv2"], res["wc2"])), nv=nv)
    t1 = max(res["VBM1"], res["VBM2"]); t2 = min(res["CBM1"], res["CBM2"]); res["Eg_bi"] = t2 - t1
    json.dump(res, open("post/bilayer.json", "w"), indent=1, default=float)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}))
