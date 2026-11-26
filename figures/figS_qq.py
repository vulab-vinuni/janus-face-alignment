import numpy as np, pandas as pd, pyvista as pv, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from matplotlib.patches import Polygon, FancyArrowPatch
pv.OFF_SCREEN = True
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "mathtext.default": "regular"})
B = 0.529177210903
ZC = {13: "Al", 31: "Ga", 49: "In", 16: "S", 34: "Se", 52: "Te", 14: "Si", 32: "Ge", 50: "Sn", 82: "Pb", 15: "P", 33: "As", 51: "Sb"}
ROLE = {"S": "X", "Se": "X", "Te": "X", "Al": "A", "Ga": "A", "In": "A", "Si": "M", "Ge": "M", "Sn": "M", "Pb": "M", "P": "Q", "As": "Q", "Sb": "Q"}
COL = {"X": "#E08E2B", "A": "#6F9FD8", "M": "#4F4F6F", "Q": "#C04A7A"}; RAD = {"X": .55, "A": .45, "M": .58, "Q": .52}
def cube(fn):
    L = open(fn).read().split("\n"); nat = abs(int(L[2].split()[0])); o = np.array(L[2].split()[1:4], float) * B
    ax = [np.array(L[3 + i].split(), float) for i in range(3)]; n = [int(a[0]) for a in ax]; v = np.array([a[1:] for a in ax]) * B
    at = np.array([L[6 + i].split() for i in range(nat)], float); data = np.array(" ".join(L[6 + nat:]).split(), float).reshape(n)
    return data, o, v, at[:, 0].astype(int), at[:, 2:5] * B
S = pd.read_csv("qqcheck/qq_summary.csv").set_index("id"); prof = np.load("qqcheck/qq_profiles.npz")
pick = ["H01855", "H10295", "H19643"]; HZ = 11.0; WPX, HPX = 700, 1400; ASP = WPX / HPX
out = {}
for idn in pick:
    data, o, v, Z, P = cube(f"qqcheck/{idn}/bi/elf.cube"); n = np.array(data.shape); cell = v * n[:, None]
    zc = np.median(P[:, 2]); low, up = P[P[:, 2] < zc], P[P[:, 2] > zc]
    tl = np.where((P[:, 2] < zc) & (P[:, 2] > low[:, 2].max() - .3))[0]; bu = np.where((P[:, 2] > zc) & (P[:, 2] < up[:, 2].min() + .3))[0]; best = None
    for i in tl:
        for j in bu:
            for sx in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    qq = P[j] + sx * cell[0] + sy * cell[1]; d = np.linalg.norm(qq - P[i])
                    if best is None or d < best[0]: best = (d, P[i].copy(), qq, i, j)
    d, q1, q2, i1, j2 = best; mid = (q1 + q2) / 2
    u = q2 - q1; u[2] = 0; u /= np.linalg.norm(u); zz = np.array([0, 0, 1.]); nrm = np.cross(zz, u)   # nrm: hướng nhìn (vuông góc mặt chứa Q–Q và z)
    # ---- 3D: nguyên tử + liên kết + mặt đẳng trị ELF, chiếu song song, nhìn dọc nrm ----
    W = HZ * ASP + 1.0; D = 2.4
    def inbox(r): s = r - mid; return abs(s @ u) < W and abs(s @ nrm) < D and abs(s[2]) < HZ
    atoms = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            for k in range(len(P)):
                r = P[k] + a * cell[0] + b * cell[1]
                if inbox(r): atoms.append((r, ROLE[ZC[Z[k]]], P[k][2] > zc))
    pl = pv.Plotter(off_screen=True, window_size=(WPX, HPX)); pl.set_background("white")
    # ELF lát 3x3 rồi cắt hộp
    sub = 2; dd = data[::sub, ::sub, :]; m = np.array(dd.shape)
    kz = np.where(np.abs(o[2] + np.arange(m[2]) * v[2, 2] - mid[2]) < HZ + .5)[0]; dd = dd[:, :, kz]
    T = np.tile(dd, (5, 5, 1)); I, J, K = np.meshgrid(np.arange(T.shape[0]) - 2 * m[0], np.arange(T.shape[1]) - 2 * m[1], kz, indexing="ij")
    X = o[None, None, None, :] + I[..., None] * sub * v[0] + J[..., None] * sub * v[1] + K[..., None] * v[2]
    g = pv.StructuredGrid(X[..., 0], X[..., 1], X[..., 2]); g["elf"] = T.ravel(order="F")
    iso = g.contour([0.75], scalars="elf")
    for sgn, ax_ in ((1, u), (-1, u), (1, nrm), (-1, nrm)):
        lim = W if ax_ is u else D
        iso = iso.clip(normal=sgn * ax_, origin=mid + sgn * ax_ * lim, invert=True)
    pl.add_mesh(iso, color="#F3E21B", opacity=.55, smooth_shading=True, specular=.3)
    for r, role, _ in atoms: pl.add_mesh(pv.Sphere(radius=RAD[role] * .62, center=r, theta_resolution=40, phi_resolution=40), color=COL[role], smooth_shading=True, specular=.5, specular_power=30)
    for x in range(len(atoms)):
        for y in range(x + 1, len(atoms)):
            ra, rb = atoms[x][0], atoms[y][0]
            if atoms[x][2] == atoms[y][2] and np.linalg.norm(ra - rb) < 3.25:
                mm = (ra + rb) / 2
                for p0, role in ((ra, atoms[x][1]), (rb, atoms[y][1])):
                    pl.add_mesh(pv.Cylinder(center=(p0 + mm) / 2, direction=mm - p0, radius=.11, height=np.linalg.norm(mm - p0)), color=COL[role], smooth_shading=True)
    pl.add_mesh(pv.Tube(pointa=q1, pointb=q2, radius=.07), color="black")
    pl.enable_parallel_projection(); pl.camera.focal_point = mid; pl.camera.position = mid - 60 * nrm; pl.camera.up = (0, 0, 1); pl.camera.parallel_scale = HZ
    img = pl.screenshot(transparent_background=False, return_img=True); pl.close()
    # kiểm tra hướng ngang ảnh: right = up × (−view) 
    right = np.cross(nrm, zz); sgn_r = np.sign(right @ u)
    # ---- lát ELF hình thoi qua hai nguyên tử gần nhất ----
    e1 = (q2 - q1) / d; e2 = np.cross(nrm, e1); e2 /= np.linalg.norm(e2); Hd = 1.5 * d; Wd = .45 * Hd
    s1, s2 = np.meshgrid(np.linspace(-Hd, Hd, 600), np.linspace(-Wd, Wd, 300), indexing="ij")
    R = mid + s1[..., None] * e1 + s2[..., None] * e2 - o; A = np.linalg.inv(v.T)
    idx = np.tensordot(R, A.T, axes=([2], [0])).transpose(2, 0, 1); elf = map_coordinates(data, idx.reshape(3, -1), order=1, mode="wrap").reshape(s1.shape)
    elf = np.where(np.abs(s1) / Hd + np.abs(s2) / Wd <= 1.0005, elf, np.nan)
    V = s1[..., None] * e1 + s2[..., None] * e2; XX, YY = V @ right, V[..., 2]
    rh = np.array([[Hd, 0], [0, Wd], [-Hd, 0], [0, -Wd]]); RH = np.array([[(a * e1 + b * e2) @ right, (a * e1 + b * e2)[2]] for a, b in rh])
    fr = np.linalg.solve(cell[:2, :2].T, right[:2]); fr = fr / np.abs(fr).max()
    line = map_coordinates(data, (np.tensordot(mid + np.linspace(-d/2, d/2, 200)[:, None] * e1 - o, A.T, axes=([1], [0]))).T, order=1, mode="wrap")
    out[idn] = dict(XX=XX, YY=YY, RH=RH, fr=fr, q1p=(-d/2*e1) @ right, q1z=-d/2*e1[2], img=img, elf=elf, Hd=Hd, Wd=Wd, d=d, e1=e1, e2=e2, cell=cell, sgn=sgn_r, u=u, nrm=nrm, zmid=mid[2], q1=ZC[Z[i1]], q2=ZC[Z[j2]], elfmin=line.min(), right=right)
np.save("v2/_dummy.npy", 0)
# ---- ghép hình ----
fig = plt.figure(figsize=(7.2, 8.6)); gs = fig.add_gridspec(len(pick), 3, width_ratios=[1.0, 1.15, 0.85], wspace=.10, hspace=.30)
for r, idn in enumerate(pick):
    D_ = out[idn]; s = S.loc[idn]
    a0 = fig.add_subplot(gs[r, 0]); a0.imshow(D_["img"], extent=[-HZ * ASP, HZ * ASP, -HZ, HZ], aspect="equal", interpolation="lanczos")
    a0.set_ylim(-HZ, HZ); a0.set_xticks([]); a0.set_ylabel("z (Å)"); a0.text(-0.34, 1.0, 'abc'[r], transform=a0.transAxes, va="bottom", ha="left", fontweight="bold", fontsize=10)
    # bộ chỉ hướng a, b, c chiếu lên mặt phẳng ảnh (ngang = right, đứng = z)
    ox, oy, Lr = -HZ * ASP + 2.6, -HZ + 1.4, 1.4
    for vec, lab, c in ((D_["cell"][0], "a", "#C0392B"), (D_["cell"][1], "b", "#27AE60"), (np.array([0, 0, 1.]), "c", "#2F5DA8")):
        e = vec / np.linalg.norm(vec); px, py = e @ D_["right"], e[2]
        if np.hypot(px, py) < .15: a0.plot(ox, oy, "o", ms=3, c=c); a0.text(ox + .25, oy + .25, lab, color=c, fontsize=7, fontweight="bold"); continue
        a0.annotate("", (ox + Lr * px, oy + Lr * py), (ox, oy), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1, mutation_scale=7))
        a0.text(ox + 1.25 * Lr * px, oy + 1.25 * Lr * py, lab, color=c, fontsize=7, fontweight="bold", ha="center", va="center")
    # lát hình thoi: trục dọc = e1 (Q1→Q2), trục ngang = e2
    a1 = fig.add_subplot(gs[r, 1]); Hd, Wd = D_["Hd"], D_["Wd"]; XX, YY, RH = D_["XX"], D_["YY"], D_["RH"]
    im = a1.pcolormesh(XX, YY, D_["elf"], cmap="jet", vmin=0, vmax=1, shading="gouraud", rasterized=True)
    a1.add_patch(Polygon(RH, closed=True, fc="none", ec="k", lw=.8)); a1.set_aspect("equal"); a1.axis("off")
    xr = np.abs(RH[:, 0]).max(); yr = np.abs(RH[:, 1]).max(); a1.set_xlim(-xr * 1.08, xr * 1.08); a1.set_ylim(-yr * 1.75, yr * 1.12)
    e1 = D_["e1"]; p1 = np.array([(-D_["d"] / 2 * e1) @ D_["right"], -D_["d"] / 2 * e1[2]])
    for p_, lab in ((p1, D_["q1"]), (-p1, D_["q2"])): a1.text(p_[0] + .35, p_[1] - .05, lab, fontsize=7.5, va="center", ha="left", color="w", fontweight="bold")
    tilt = np.degrees(np.arccos(abs(e1[2])))
    a1.set_title(f"{s.L1}/{s.L2} (QQ)", fontsize=7.5, pad=2)
    a1.text(xr * 1.05, -yr * 1.05, f"{D_['q1']}–{D_['q2']}: {D_['d']:.2f} Å\nELF$_{{min}}$ = {s.ELF_min:.2f}\n∠(Q–Q, c) = {tilt:.0f}°", fontsize=6.8, va="top", ha="right")
    # chỉ hướng: c đứng, ngang = hướng trong mặt phẳng ghi theo chỉ số tinh thể
    fr = D_["fr"]
    for mlt in (1, 2, 3, 4):
        if np.all(np.abs(fr * mlt - np.round(fr * mlt)) < .08): fr = np.round(fr * mlt).astype(int); break
    lab = "[" + " ".join(str(int(x)) for x in (*fr, 0)) + "]"
    ox, oy, L = -xr * 1.0, -yr * 1.6, 1.0
    a1.annotate("", (ox, oy + L), (ox, oy), arrowprops=dict(arrowstyle="-|>", color="#2F5DA8", lw=1.0, mutation_scale=7)); a1.text(ox, oy + L * 1.2, "c", color="#2F5DA8", fontsize=7, fontweight="bold", ha="center")
    a1.annotate("", (ox + L, oy), (ox, oy), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0, mutation_scale=7)); a1.text(ox + L * 1.15, oy, lab.replace("-", "−"), fontsize=6.5, va="center")
    # Δρ(z)
    a2 = fig.add_subplot(gs[r, 2], sharey=a0); zp, dr = prof[idn][:, 0] - D_["zmid"], prof[idn][:, 1] * 1e3
    a2.fill_betweenx(zp, 0, dr, where=dr > 0, color="#C0392B", alpha=.55, lw=0); a2.fill_betweenx(zp, 0, dr, where=dr < 0, color="#2F5DA8", alpha=.55, lw=0)
    a2.plot(dr, zp, c="k", lw=.7); a2.axvline(0, c="k", lw=.5); a2.axhline(0, c="#888", lw=.5, ls=":"); a2.tick_params(labelleft=False)
    mm = np.abs(dr[np.abs(zp) < HZ]).max(); a2.set_xlim(-1.15 * mm, 1.15 * mm); a2.set_xlabel(r"$\Delta\bar\rho$ (10$^{-3}$ e/bohr$^3$)")
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], marker="o", ls="", mfc=COL[k], mec="none", ms=6) for k in "XAMQ"] + [Line2D([], [], marker="s", ls="", mfc="#F3E21B", mec="none", alpha=.7, ms=7)],
           ["X (S/Se/Te)", "A (Al/Ga/In)", "M (Si/Ge/Sn/Pb)", "Q (P/As/Sb)", "ELF = 0.75"], loc="upper left", bbox_to_anchor=(0.10, 0.955), ncol=3, frameon=False, fontsize=7, handletextpad=.2, columnspacing=.7)
cax = fig.add_axes([0.66, 0.925, 0.22, 0.011]); cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label("ELF", fontsize=7.5, labelpad=1); cb.ax.tick_params(labelsize=6.5, top=True, labeltop=True, bottom=False, labelbottom=False)
fig.savefig("figS_qq.pdf", bbox_inches="tight", dpi=300); fig.savefig("figS_qq.png", bbox_inches="tight", dpi=180)
for k in pick: print(k, out[k]["sgn"], round(out[k]["d"], 3), round(out[k]["elfmin"], 3))
