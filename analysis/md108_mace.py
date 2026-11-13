#!/usr/bin/env python3
"""NVT (Langevin) MD with the fine-tuned MACE potential (B330) for the Janus XAMQ monolayers: thermal-stability check.
   conda activate mace
   python3 md108_mace.py --model ~/mace_pilot_pc/xamq_B330.model --s3 S3.extxyz --T 300 --ps 5            # all 108 compounds
   python3 md108_mace.py ... --only SeInPbP,TeAlPbSb,SAlSiP --T 500 --ps 10                             # a few compounds, high temperature
Per compound: relax atoms (fixed cell) -> 6x6 supercell (144 atoms) -> Langevin, 1 fs step, friction 0.01/fs.
Output md<T>K/summary.csv: energy fluctuation, mean T, mean-square displacement, number of broken bonds (relative to the initial structure, 1.25× threshold),
and traj/<compound>.extxyz (every 50 steps) for E(t), T(t) plots."""
import argparse, os, time, numpy as np, pandas as pd
from ase.io import read, write
from ase.optimize import BFGS
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.neighborlist import neighbor_list
from ase import units
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--s3", default="S3.extxyz"); ap.add_argument("--only", default=None)
ap.add_argument("--T", type=float, default=300); ap.add_argument("--ps", type=float, default=5); ap.add_argument("--dt", type=float, default=1.0); ap.add_argument("--sc", type=int, default=6)
ap.add_argument("--device", default="cuda"); a = ap.parse_args()
from mace.calculators import MACECalculator
calc = MACECalculator(model_paths=a.model, device=a.device, default_dtype="float32")   # float32 is sufficient for MD, ~2x faster
S3 = {x.info["name"]: x for x in read(a.s3, ":")}
names = a.only.split(",") if a.only else sorted(n for n in S3 if len(n) <= 9 and n[:1] in "ST")
out = f"md{int(a.T)}K"; os.makedirs(f"{out}/traj", exist_ok=True); rows = []
if os.path.exists(f"{out}/summary.csv"): rows = pd.read_csv(f"{out}/summary.csv").to_dict("records")
done = {r["name"] for r in rows}
for nm in names:
    if nm in done: continue
    t0 = time.time(); at = S3[nm].copy(); at.calc = calc; BFGS(at, logfile=None).run(fmax=1e-3, steps=300)
    sc = at.repeat((a.sc, a.sc, 1)); sc.calc = calc; ref = sc.positions.copy()
    i0, j0, d0 = neighbor_list("ijd", sc, 3.3); bonds = np.c_[i0, j0][i0 < j0]  # initial bonds (< 3.3 Å)
    MaxwellBoltzmannDistribution(sc, temperature_K=a.T, rng=np.random.default_rng(7)); Stationary(sc); ZeroRotation(sc)
    dyn = Langevin(sc, a.dt * units.fs, temperature_K=a.T, friction=0.01 / units.fs, rng=np.random.default_rng(11))
    nstep = int(a.ps * 1000 / a.dt); E, T = [], []; tr = f"{out}/traj/{nm}.extxyz"; open(tr, "w").close()
    def rec():
        E.append(sc.get_potential_energy() / len(sc)); T.append(sc.get_temperature())
        if len(E) % 50 == 0: write(tr, sc, append=True)
    dyn.attach(rec, interval=1); dyn.run(nstep)
    d1 = sc.get_distances(bonds[:, 0], bonds[:, 1], mic=True)
    sc0 = at.repeat((a.sc, a.sc, 1)); dI = sc0.get_distances(bonds[:, 0], bonds[:, 1], mic=True)
    broken = int(np.sum(d1 > 1.25 * dI)); half = len(E) // 2; Eh = np.array(E[half:])
    disp = sc.positions - ref; disp -= disp.mean(0); msd = float(np.mean(np.sum(disp ** 2, 1)))
    rows.append(dict(name=nm, T_set=a.T, T_mean=float(np.mean(T[half:])), E_std_meV_at=1000 * float(Eh.std()), E_drift_meV_at=1000 * float(np.polyfit(np.arange(len(Eh)), Eh, 1)[0] * len(Eh)),
                     msd_A2=msd, dz_max_A=float(np.abs(disp[:, 2]).max()), broken_bonds=broken, n_bonds=len(bonds), stable=broken == 0, minutes=(time.time() - t0) / 60))
    pd.DataFrame(rows).to_csv(f"{out}/summary.csv", index=False); print(rows[-1], flush=True)
