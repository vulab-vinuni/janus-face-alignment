"""Supplementary Fig. 1: (a,b) SeInPbP MACE vs DFPT; (c) cost; (d–i) six Al–Sb compounds MACE vs VASP 3x3."""
import yaml, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.linewidth":1,"xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True,"mathtext.default":"regular"})
RED, BLU = "#C0392B", "#2E6DB4"; DSH = (3.2, 1.8)
r2 = lambda a, b: 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
def mace(c):
    w = np.load(f"../phon/f2/phon108/bands/{c}.npz")["w"]; fm = np.concatenate([np.linspace(k, k + 1, 80) for k in range(3)])
    o = np.argsort(fm, kind="stable"); fm, w = fm[o], w[o]; _, u = np.unique(fm, return_index=True); return fm[u], w[u]
def panel(ax, X, Wd, F, tk, c, lw=1.1):
    fm, w = mace(c); fs = np.linspace(0, 3, 300); xs = np.interp(fs, [0, 1, 2, 3], tk)
    for j in range(12): ax.plot(X, Wd[:, j], "--", c=BLU, lw=lw, dashes=DSH, zorder=4)
    for j in range(12): ax.plot(xs, np.interp(fs, fm, w[:, j]), "-", c=RED, lw=lw, zorder=3)
    for v in tk[1:-1]: ax.axvline(v, c="k", lw=.7)
    ax.axhline(0, c="0.6", lw=.5, ls=":"); ax.set_xticks(tk); ax.set_xticklabels(["Γ", "M", "K", "Γ"]); ax.set_xlim(0, tk[-1])
    Wm = np.array([np.interp(F, fm, w[:, j]) for j in range(12)]).T; return np.sort(Wd, 1), np.sort(Wm, 1)

import phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_CONSTANTS
GRN = "#2E8B57"
_ph = phonopy.Phonopy(read_vasp("../mlff/POSCAR_unit"), supercell_matrix=np.diag([6, 6, 1])); _ph.force_constants = parse_FORCE_CONSTANTS("../mlff/FORCE_CONSTANTS_2ND"); _ph.symmetrize_force_constants()
def mlff_freq(qs): _ph.run_qpoints(qs); return _ph.qpoints.frequencies * 33.35641
fig = plt.figure(figsize=(7.3, 7.6))
gs = fig.add_gridspec(3, 3, hspace=.55, wspace=.32)
lab = lambda ax, s, dx=-.2: ax.text(dx, 1.06, f"({s})", transform=ax.transAxes, fontweight="bold", fontsize=9.5)
# (a,b) SeInPbP vs DFPT
g = np.loadtxt("../dfpt/InSePbP.freq.gp"); W = g[:, 1:]
q = np.array([l.split() for l in open("../dfpt/matdyn.in").read().split("/")[1].split("\n")[2:142]], float)
Bm = np.array([[1, 1/np.sqrt(3)], [0, 2/np.sqrt(3)]]); cart = lambda f: f[:, :2] @ Bm.T
segs = [np.arange(59, 111), np.arange(110, 140), np.arange(0, 60)]; X, Wd, F, x0, tk = [], [], [], 0., [0.]
for k, s in enumerate(segs):
    cc = cart(q[s]); d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(cc, axis=0), axis=1))]); sl = slice(None) if k == 0 else slice(1, None)
    X.append((x0 + d)[sl]); Wd.append(W[s][sl]); F.append((k + d / d[-1])[sl]); x0 += d[-1]; tk.append(x0)
X, Wd, F = np.concatenate(X), np.vstack(Wd), np.concatenate(F)
a = fig.add_subplot(gs[0, 0]); dd, mm = panel(a, X, Wd, F, tk, "SeInPbP"); a.set_ylim(-35, 365)
qs = np.vstack([q[s] if k == 0 else q[s][1:] for k, s in enumerate(segs)]); Wl = mlff_freq(qs)
for j in range(12): a.plot(X, Wl[:, j], ":", c=GRN, lw=1.3, zorder=5)
dl = np.sort(Wl, 1); a.set_ylabel("Frequency (cm$^{-1}$)")
a.set_title("SeInPbP (excluded from training)", fontsize=7.3, pad=3); a.text(-.2, 1.2, "(a)", transform=a.transAxes, fontweight="bold", fontsize=9.5)
from matplotlib.lines import Line2D
print("SeInPbP", r2(dd.ravel(), mm.ravel()), r2(dd[:, :3].ravel(), mm[:, :3].ravel()), np.abs(dd-mm).mean(), "MLFF", r2(dd.ravel(), dl.ravel()), r2(dd[:, :3].ravel(), dl[:, :3].ravel()), np.abs(dd-dl).mean())
fig.legend([Line2D([], [], c=BLU, ls="--", dashes=DSH, lw=1.4), Line2D([], [], c=RED, lw=1.4), Line2D([], [], c=GRN, ls=":", lw=1.6)], ["DFT", "MACE (fine-tuned, family)", "VASP MLFF (SeInPbP only)"], loc="upper center", bbox_to_anchor=(.5, .975), ncol=3, frameon=False, fontsize=7.8)
b = fig.add_subplot(gs[0, 1])
sM = f"MACE  {r2(dd.ravel(), mm.ravel()):.3f} ({r2(dd[:, :3].ravel(), mm[:, :3].ravel()):.2f}), {np.abs(dd - mm).mean():.1f}"
sL = f"MLFF  {r2(dd.ravel(), dl.ravel()):.3f} ({r2(dd[:, :3].ravel(), dl[:, :3].ravel()):.2f}), {np.abs(dd - dl).mean():.1f}"
b.scatter(dd.ravel(), mm.ravel(), s=9, facecolors="none", edgecolors=RED, lw=.6, zorder=2, label=sM)
b.scatter(dd.ravel(), dl.ravel(), s=9, c=GRN, marker="x", lw=.6, alpha=.8, zorder=1, label=sL)
b.plot([0, 360], [0, 360], "k-", lw=1); b.set(xlim=(0, 325), ylim=(0, 360)); b.set_xlabel("DFT (cm$^{-1}$)"); b.set_ylabel("model (cm$^{-1}$)", labelpad=1)
b.set_title("SeInPbP, model vs DFT", fontsize=7.3, pad=3)
lg = b.legend(loc="upper left", title="$R^2$ (acoustic), MAE", title_fontsize=6.0, alignment="left", fontsize=6.0, frameon=False, handletextpad=.1, labelspacing=.35, borderaxespad=.2, markerscale=1.4)
b.text(-.2, 1.2, "(b)", transform=b.transAxes, fontweight="bold", fontsize=9.5)
# (c–h) 6 Al–Sb vs VASP 3x3
C = ["TeAlPbSb", "TeAlSnSb", "SeAlSnSb", "SAlSnSb", "SeAlPbSb", "SAlPbSb"]; GP = {"TeAlPbSb": "gapped", "TeAlSnSb": "gapped, excluded"}
for i, c in enumerate(C):
    ax = fig.add_subplot(gs[1 + i // 3, i % 3])
    bb = yaml.safe_load(open(f"../her/{c}/01_phon3/band.yaml")); ph = bb["phonon"]
    f = np.array([[x["frequency"] for x in p["band"]] for p in ph]) * 33.35641; d = np.array([p["distance"] for p in ph])
    s1, s2, s3 = slice(0, 101), slice(101, 202), slice(202, 303); L1, L2, L3 = d[100] - d[0], d[201] - d[101], d[302] - d[202]
    Xv = np.concatenate([L1 - (d[s1] - d[0]), L1 + (L3 - (d[s3] - d[202])), L1 + L3 + (L2 - (d[s2] - d[101]))])
    t = np.linspace(0, 1, 101); Fv = np.concatenate([1 - t, 2 - t, 3 - t]); Wv = np.vstack([f[s1], f[s3], f[s2]])
    dd, mm = panel(ax, Xv, Wv, Fv, [0, L1, L1 + L3, L1 + L3 + L2], c); ax.set_ylim(-35, 365)
    ax.set_title(f"{c} ({GP.get(c, 'semimetal')}), VASP\n$R^2$ = {r2(dd.ravel(), mm.ravel()):.3f}, MAE = {np.abs(dd - mm).mean():.1f} cm$^{{-1}}$", fontsize=7.3, pad=3)
    print(c, r2(dd.ravel(), mm.ravel()), np.abs(dd - mm).mean())
    if i % 3 == 0: ax.set_ylabel("Frequency (cm$^{-1}$)")
    else: ax.set_yticklabels([])
    ax.text(-.2, 1.2, f"({'defghi'[i]})", transform=ax.transAxes, fontweight="bold", fontsize=9.5)
# (i,j) cost
dfpt1 = 4.483 * 224 + 43 / 3600 * 224; sc330, uc2500, small240, st1_384, strain640 = 18.8, 7.5, 19.6, 12.7, 7.5 * 640 / 2500
full = (sc330 + uc2500 + small240 + st1_384 + strain640) * 480; C_D, C_M = "#2B2B2B", RED
a = fig.add_subplot(gs[0, 2]); val = [dfpt1, 108 * dfpt1, full]
a.bar(range(3), val, .6, color=[C_D, C_D, C_M], edgecolor="k", lw=.6)
for i, v in enumerate(val): a.text(i, v * 1.25, f"{v/1000:.1f}k" if v >= 1000 else f"{v:.0f}", ha="center", fontsize=6.8)
a.set_yscale("log"); a.set_ylim(300, 6e5); a.set_xticks(range(3)); a.set_xticklabels(["DFPT\n×1", "DFPT\n×108", "MACE\ntraining"], fontsize=7); a.tick_params(axis="x", which="both", top=False, bottom=False)
a.set_ylabel("CPU cost (core-hours)", labelpad=1); a.set_title("Phonon cost", fontsize=7.3, pad=3); a.text(-.2, 1.2, "(c)", transform=a.transAxes, fontweight="bold", fontsize=9.5)
print("cost", dfpt1, 108*dfpt1, full, full/dfpt1, 108*dfpt1/full)
fig.savefig("figS_mace.pdf", bbox_inches="tight", dpi=300); fig.savefig("figS1_all.png", bbox_inches="tight", dpi=150)
