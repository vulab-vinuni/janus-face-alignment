#!/usr/bin/env python3
"""Dynamical-stability screening of the 108 Janus XAMQ monolayers with the fine-tuned MACE potential (B330) + phonopy.
   conda activate mace
   python3 phon108_mace.py --model ~/mace_pilot_pc/xamq_B330.model --s3 S3.extxyz --names names108.txt [--device cuda] [--sc 6]
   Per compound: MACE relaxation of the atoms (fixed cell, fmax 1e-4 eV/Å) -> phonopy 6x6 supercell (contains M and K), 0.01 Å displacements
   -> dispersion along Γ–M–K–Γ (with eigenvectors) + 24x24 mesh.
   Classification of imaginary frequencies (stable / ZA_noise / CHECK / UNSTABLE):
     - near Γ and mainly out-of-plane (ZA, z fraction > 0.8): 'ZA_noise' (numerical artefact)
     - anywhere else (M, K, mid-zone; or in-plane modes) below −1 cm⁻¹: 'UNSTABLE'
   Output: phon108_summary.csv, bands/<compound>.npz (for the SI figures)"""
import argparse, os, numpy as np, pandas as pd
from ase.io import read
from ase.optimize import BFGS

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--s3", default="S3.extxyz"); ap.add_argument("--names", default=None)
ap.add_argument("--device", default="cuda"); ap.add_argument("--sc", type=int, default=6); ap.add_argument("--disp", type=float, default=0.01); ap.add_argument("--out", default="phon108")
ap.add_argument("--fmax", type=float, default=1e-4); ap.add_argument("--rG", type=float, default=0.15, help="radius around Γ in units of |b| (|ΓM| = 0.5) within which a ZA pocket is treated as numerical noise"); ap.add_argument("--only", default=None, help="comma-separated compound names, for testing")
a = ap.parse_args()
from mace.calculators import MACECalculator
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
THZ2CM = 33.35641
calc = MACECalculator(model_paths=a.model, device=a.device, default_dtype="float64")
S3 = {x.info["name"]: x for x in read(a.s3, ":")}
names = [l.split()[0] for l in open(a.names)] if a.names else sorted(n for n in S3 if len(n) <= 9 and n[:1] in "ST")
if a.only: names = a.only.split(",")
os.makedirs(f"{a.out}/bands", exist_ok=True); rows = []
G, M, K = np.zeros(3), np.array([.5, 0, 0]), np.array([1 / 3, 1 / 3, 0])
def path(n=80):
    seg = [(G, M), (M, K), (K, G)]; return [[p + (q - p) * t for t in np.linspace(0, 1, n)] for p, q in seg]
for nm in names:
    at = S3[nm].copy(); at.calc = calc
    f0 = np.abs(at.get_forces()).max(); BFGS(at, logfile=None).run(fmax=a.fmax, steps=500); f1 = np.abs(at.get_forces()).max()
    u = PhonopyAtoms(symbols=at.get_chemical_symbols(), cell=at.cell[:], scaled_positions=at.get_scaled_positions())
    ph = Phonopy(u, supercell_matrix=np.diag([a.sc, a.sc, 1]), primitive_matrix=np.eye(3))
    ph.generate_displacements(distance=a.disp, is_plusminus=True); F = []
    for sc in ph.supercells_with_displacements:
        s = at.__class__(symbols=sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True); s.calc = calc; F.append(s.get_forces())
    ph.forces = np.array(F); ph.produce_force_constants(); ph.symmetrize_force_constants()
    P = path(); ph.run_band_structure(P, with_eigenvectors=True)
    bs = ph.get_band_structure_dict(); q = np.vstack(bs["qpoints"]); w = np.vstack(bs["frequencies"]) * THZ2CM; ev = np.vstack(bs["eigenvectors"])
    nat = len(at); zfrac = np.stack([np.sum(np.abs(ev[i].reshape(nat, 3, -1)[:, 2, :]) ** 2, axis=0) for i in range(len(q))])   # (nq, nbranch)
    ph.run_mesh([24, 24, 1], with_eigenvectors=True, is_mesh_symmetry=False); md = ph.get_mesh_dict(); wm = md["frequencies"] * THZ2CM; qm = md["qpoints"]; evm = md["eigenvectors"]
    zm = np.stack([np.sum(np.abs(evm[i].reshape(nat, 3, -1)[:, 2, :]) ** 2, axis=0) for i in range(len(qm))])
    def classify(qq, ww, zz):
        dq = qq[:, :2] - np.round(qq[:, :2]); rq = np.sqrt(dq[:, 0] ** 2 + dq[:, 1] ** 2 + dq[:, 0] * dq[:, 1])   # |q| with the hexagonal metric, in units of |b| (|ΓM| = 0.5)
        nr = (rq < a.rG)[:, None]
        neg = ww < -1.0; za = neg & nr & (zz > .8) & (np.arange(ww.shape[1]) == 0)[None, :]
        atG = (rq < 0.03)[:, None] & (np.arange(ww.shape[1]) < 3)[None, :] & (ww > -5.0)   # at Γ: three mixed acoustic branches, acoustic-sum-rule error
        za = za | (neg & atG); return za, neg & ~za
    za, bad = classify(q, w, zfrac); zam, badm = classify(qm, wm, zm)
    worst_far = float(min(np.where(bad, w, 0).min(), np.where(badm, wm, 0).min()))
    imin = np.unravel_index(np.argmin(np.where(bad, w, np.inf)), w.shape) if bad.any() else None
    # 'CHECK': only the tail of a ZA pocket (lowest branch, z > 0.8) outside the radius, shallower than 3 cm⁻¹ -> confirm with DFPT; in-plane or M/K modes -> UNSTABLE
    def tail_only(bb, ww, zz): return bool(bb.any()) and np.all(zz[bb] > .8) and np.all(np.argwhere(bb)[:, 1] == 0) and ww[bb].min() > -3.0
    if not (bad.any() or badm.any()): status = "ZA_noise" if (za.any() or zam.any()) else "stable"
    elif (not bad.any() or tail_only(bad, w, zfrac)) and (not badm.any() or tail_only(badm, wm, zm)): status = "CHECK"
    else: status = "UNSTABLE"
    rows.append(dict(name=nm, status=status, wmin_all=float(min(w.min(), wm.min())), wmin_notZA=worst_far, wmin_ZA_nearG=float(min(w[za].min() if za.any() else 0.0, wm[zam].min() if zam.any() else 0.0)),
                     q_worst=None if imin is None else "%.3f,%.3f" % tuple(q[imin[0], :2]), zfrac_worst=None if imin is None else float(zfrac[imin]),
                     fmax_QE=f0, fmax_relaxed=f1))
    xs = np.concatenate([np.linspace(0, 1, 80), np.linspace(1, 1 + np.linalg.norm(K - M) / np.linalg.norm(M), 80), np.linspace(1 + np.linalg.norm(K - M) / np.linalg.norm(M), 1 + (np.linalg.norm(K - M) + np.linalg.norm(K)) / np.linalg.norm(M), 80)])
    np.savez_compressed(f"{a.out}/bands/{nm}.npz", x=xs, w=w, zfrac=zfrac)
    print(f"{nm:10s} {status:9s} min(excluding ZA pocket) {worst_far:7.2f} cm-1 | ZA near Γ {rows[-1]['wmin_ZA_nearG']:7.2f} | fmax QE→MACE {f0:.3f}→{f1:.1e}", flush=True)
pd.DataFrame(rows).to_csv(f"{a.out}/phon108_summary.csv", index=False)
D = pd.DataFrame(rows); print(D.status.value_counts().to_dict())
