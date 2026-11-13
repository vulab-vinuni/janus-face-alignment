"""Shared QE utilities for the Janus alignment pipeline (numpy + standard library only, runs on clusters).
   - write_pw(): generate pw.x input (PBE+D3(BJ), 2D cutoff, SG15) as used for the monolayers/heterostructures
   - read_xml(): read data-file-schema.xml -> eigenvalues (eV), k (Å^-1, cartesian), weights, E_F, nelec, cell
   - planar_avg(): read avg.dat from average.x -> z (Å), V (eV)
   - vac_levels(): vacuum levels on both sides (window 4–7 Å from the outermost atom) + flatness
"""
import numpy as np, xml.etree.ElementTree as ET
RY = 13.605693122994; HA = 2 * RY; BOHR = 0.529177210903
MASS = dict(S=32.06, Se=78.971, Te=127.6, Al=26.9815, Ga=69.723, In=114.818, Si=28.0855, Ge=72.63, Sn=118.71, Pb=207.2,
            P=30.97376, As=74.9216, Sb=121.76)
SET = dict(ecut=80.0, k=12, conv=1e-8, degauss=0.005)          # production settings; reduced for local tests


def write_pw(fn, calc, sym, pos, cell, prefix="x", outdir="./tmp", pseudo_dir="PSEUDO_DIR", nbnd=None, kpts=None,
             kgrid=None, s=SET, extra_sys=""):
    """sym: list of elements; pos: (N,3) Å cartesian; cell: (3,3) Å. kpts: list (kx,ky,kz,w) in tpiba for nscf."""
    sp = list(dict.fromkeys(sym)); kg = kgrid or s["k"]
    L = ["&CONTROL", f"    calculation = '{calc}'", f"    prefix = '{prefix}'", f"    outdir = '{outdir}'",
         f"    pseudo_dir = '{pseudo_dir}'", "    tprnfor = .true.", "    forc_conv_thr = 5.0d-4", "    etot_conv_thr = 1.0d-5", "/",
         "&SYSTEM", "    ibrav = 0", f"    nat = {len(sym)}", f"    ntyp = {len(sp)}", f"    ecutwfc = {s['ecut']}",
         "    occupations = 'smearing'", "    smearing = 'gauss'", f"    degauss = {s['degauss']}", "    assume_isolated = '2D'",
         "    vdw_corr = 'dft-d3'", "    dftd3_version = 4"]
    if nbnd: L.append(f"    nbnd = {nbnd}")
    if calc == "nscf": L.append("    nosym = .true.")
    L += [extra_sys] if extra_sys else []
    L += ["/", "&ELECTRONS", f"    conv_thr = {s['conv']:.1e}".replace("e", "d"), "    mixing_beta = 0.3", "    mixing_mode = 'local-TF'"]
    if calc == "nscf": L += ["    diago_full_acc = .true.", "    diagonalization = 'cg'"]
    L += ["/"]
    if calc in ("relax",): L += ["&IONS", "/"]
    L += ["ATOMIC_SPECIES"] + [f"{e} {MASS[e]} {e}.upf" for e in sp]
    L += ["CELL_PARAMETERS angstrom"] + ["  %.10f %.10f %.10f" % tuple(v) for v in cell]
    L += ["ATOMIC_POSITIONS angstrom"] + ["%s %.10f %.10f %.10f" % (e, *p) for e, p in zip(sym, pos)]
    if kpts is None:
        L += ["K_POINTS automatic", f"{kg} {kg} 1 0 0 0"]
    else:
        L += ["K_POINTS tpiba", str(len(kpts))] + ["%.10f %.10f %.10f %.6f" % tuple(k) for k in kpts]
    open(fn, "w").write("\n".join(L) + "\n")


def read_xml(fn):
    r = ET.parse(fn).getroot(); o = r.find("output")
    alat = float(o.find("atomic_structure").get("alat")) * BOHR
    cell = np.array([[float(x) for x in o.find(f"atomic_structure/cell/{a}").text.split()] for a in ("a1", "a2", "a3")]) * BOHR
    bs = o.find("band_structure"); nel = float(bs.find("nelec").text)
    ef = bs.find("fermi_energy"); ef = float(ef.text) * HA if ef is not None else None
    K, W, E = [], [], []
    for ks in bs.findall("ks_energies"):
        kp = ks.find("k_point"); K.append([float(x) for x in kp.text.split()]); W.append(float(kp.get("weight")))
        E.append([float(x) * HA for x in ks.find("eigenvalues").text.split()])
    K = np.array(K) * 2 * np.pi / alat                                   # Å^-1 cartesian
    at = [(a.get("name"), [float(x) * BOHR for x in a.text.split()]) for a in o.findall("atomic_structure/atomic_positions/atom")]
    etot = float(o.find("total_energy/etot").text) * HA
    return dict(alat=alat, cell=cell, nel=nel, ef=ef, k=K, w=np.array(W), e=np.array(E), atoms=at, etot=etot)


def planar_avg(fn="avg.dat"):
    d = np.loadtxt(fn); return d[:, 0] * BOHR, d[:, 1] * RY        # z (Å), V (eV) [average.x: bohr, Ry]


def vac_levels(z, V, zat, c, w0=4.0, w1=7.0):
    """Vacuum levels below (bot) and above (top) a slab with atomic z coordinates zat (Å) in a cell of height c (Å)."""
    zmin, zmax = min(zat), max(zat); out = {}
    for side, lo, hi in (("top", zmax + w0, zmax + w1), ("bot", zmin - w1, zmin - w0)):
        m = ((z - lo) % c <= (hi - lo))
        out[side] = V[m].mean(); out["flat_" + side] = V[m].std()
    return out


def avg_input(fn, filplot, nz, c_bohr):
    open(fn, "w").write(f"1\n{filplot}\n1.D0\n{nz}\n3\n1.0\n")      # awin = 1 bohr (macroscopic average not used)


def pp_input(fn, prefix, outdir="./tmp", filplot="vtot.dat", plot_num=11):
    open(fn, "w").write(f"&INPUTPP\n  prefix='{prefix}'\n  outdir='{outdir}'\n  filplot='{filplot}'\n  plot_num={plot_num}\n/\n")
