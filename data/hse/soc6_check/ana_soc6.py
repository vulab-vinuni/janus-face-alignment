"""Compare HSE06+SOC (noncollinear) with HSE06 + (PBE+SOC - PBE), all from VASP on the same mesh. Run in hse108_vasp/ on the login node."""
import numpy as np
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
def edges(d):
    L = open(f"{d}/EIGENVAL").read().split("\n"); nel, nk, nb = map(int, L[5].split()[:3]); E = []; i = 7
    for k in range(nk):
        E.append([float(L[i + 1 + b].split()[1]) for b in range(nb)]); i += nb + 2
    E = np.array(E); ne = int(round(float(open(f"{d}/OUTCAR").read().split("NELECT =")[1].split()[0])))
    nv = ne if "LNONCOLLINEAR =      T" in open(f"{d}/OUTCAR").read() or "LSORBIT" in open(f"{d}/INCAR").read() else ne // 2
    return E[:, nv - 1].max(), E[:, nv].min()
def lev(d):  # VBM, CBM relative to the mean vacuum level
    vb, cb = edges(d); vX, vQ, _ = vac(d, 0, 3); m = (vX + vQ) / 2; return vb - m, cb - m, vQ - vX
print(f"{'name':10s} {'Eg PBE':>7s} {'PBE+SOC':>7s} {'HSE':>6s} {'HSE+SOC':>8s} {'additive':>8s} | dVBM_SOC dCBM_SOC (PBE) | err VBM CBM Eg (additive - full)")
for n in [l.strip() for l in open("soc6.txt") if l.strip()]:
    try:
        P, PS, Hh, HS = (lev(f"calc/{n}/{d}") for d in ("1_pbe", "3_pbesoc", "2_hse", "4_hsesoc"))
        av, ac = Hh[0] + PS[0] - P[0], Hh[1] + PS[1] - P[1]
        print(f"{n:10s} {P[1]-P[0]:7.3f} {PS[1]-PS[0]:7.3f} {Hh[1]-Hh[0]:6.3f} {HS[1]-HS[0]:8.3f} {ac-av:8.3f} | {PS[0]-P[0]:+.3f} {PS[1]-P[1]:+.3f} | {av-HS[0]:+.3f} {ac-HS[1]:+.3f} {(ac-av)-(HS[1]-HS[0]):+.3f}")
    except Exception as e: print(n, "not ready:", str(e)[:60])
