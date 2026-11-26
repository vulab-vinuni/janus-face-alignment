import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Ellipse, FancyArrowPatch, Wedge
from matplotlib.colors import to_rgb
import matplotlib.patheffects as pe
from ase.io import read
INK, MUTE, ARR = "k", "k", "#E8A86B"
EL = {"Se": "#E08E2B", "Al": "#6F9FD8", "Sn": "#4F4F6F", "As": "#C04A7A"}   # trùng màu X/A/M/Q của Hình 2a   # màu kiểu VESTA (Se vàng-lục, As tím)
RAD = {"Se": .55, "Al": .48, "Sn": .62, "As": .55}
TY = {"I": "#4C72B0", "II": "#DD8452", "III": "#8172B3"}
FH = 5.6; YT = 100 * FH / 7.2
fig = plt.figure(figsize=(7.2, FH)); W = fig.add_axes([0, 0, 1, 1]); W.set_xlim(0, 100); W.set_ylim(0, YT); W.axis("off")
def FA(x0, y0, x1, y1): return fig.add_axes([x0 / 100, y0 / YT, (x1 - x0) / 100, (y1 - y0) / YT])
BLUE, ORG, PUR, GREY = "#1F4E9C", "#F28C28", "#1F4E9C", "#8A8F98"
def panel(y0, h, lab, title):
    W.add_patch(FancyBboxPatch((1, y0), 98, h, boxstyle="round,pad=0,rounding_size=1.5", fc="#F3F5F8", ec="#4B5563", lw=.7, zorder=0))
    W.text(2.6, y0 + h - 2.4, lab, fontsize=12, fontweight="bold", va="center"); W.text(5.6, y0 + h - 2.4, title, fontsize=11, va="center")
def arrow(p1, p2, color=GREY, lw=.9):
    W.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>,head_length=4,head_width=2.4", lw=lw, color=color, zorder=3, shrinkA=0, shrinkB=0))
def line(pts, color=GREY, lw=.9):
    xs, ys = zip(*pts); W.plot(xs, ys, c=color, lw=lw, zorder=2, solid_joinstyle="miter")
def box(x, y, w, h, ec, fc="white", ls="-", r=1.0):
    W.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=.8, ls=ls, zorder=3))
def sphere(ax, x, y, r, col, z=3):
    c = np.array(to_rgb(col)); ax.add_patch(Circle((x, y), r, fc=c * .72, ec="none", zorder=z))
    for k in range(1, 9):
        f = k / 9; cc = c * (.72 + .28 * f) + (1 - c) * .55 * f ** 2.2
        ax.add_patch(Circle((x - .32 * r * f, y + .32 * r * f), r * (1 - .82 * f), fc=np.clip(cc, 0, 1), ec="none", zorder=z))
def bond(ax, p, q, c1, c2):
    m = (np.array(p) + np.array(q)) / 2
    for a, b, cc in ((p, m, c1), (m, q, c2)):
        ax.plot([a[0], b[0]], [a[1], b[1]], c=cc, lw=3.0, solid_capstyle="butt", zorder=2, path_effects=[pe.Stroke(linewidth=3.8, foreground="#555"), pe.Normal()])

BT, OT = "#E4ECF8", "#FDEBD7"           # tints
PA0, PAH = 41.0, YT - 42.5
panel(PA0, PAH, "a", "DFT data")
panel(1.5, 38.0, "b", "Models, frozen and tested")
xs0, bw, gap = 2.4, 22.4, 1.95; CX = [xs0 + k * (bw + gap) for k in range(4)]
CY0, CH = PA0 + 7.0, 6.6                  # chevrons
IY0, IY1 = CY0 + CH + 3.0, PA0 + PAH - 6.6
SH = ["#7FA3D9", "#5584C8", "#3767B4", "#1F4E9C"]
LAB = [("108", "monolayers"), ("7 796", "interfaces"), ("832", "heterostructures"), ("599", "fitting set")]
from matplotlib.patches import Polygon
for k in range(4):
    x = CX[k]; tip = 2.2; x1 = x + bw + (gap - .3 if k < 3 else 0)
    pts = [(x, CY0), (x1 - tip, CY0), (x1, CY0 + CH / 2), (x1 - tip, CY0 + CH), (x, CY0 + CH)] + ([(x + tip, CY0 + CH / 2)] if k else [])
    W.add_patch(Polygon(pts, closed=True, fc=SH[k], ec="white", lw=1.2, zorder=3))
    W.text(x + (tip if k else 0) + 1.6, CY0 + CH / 2, LAB[k][0], color="white", fontweight="bold", va="center", zorder=4)
    W.text(x + (tip if k else 0) + 1.6 + 1.25 * len(LAB[k][0]) + 1.2, CY0 + CH / 2, LAB[k][1], color="white", va="center", zorder=4)
TK = "#374151"
def tile(x0, y0, x1, y1, z=1):
    W.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec=TK, lw=.7, zorder=z))
for k in range(4):
    tile(CX[k], IY0 - 1.4, CX[k] + bw, IY1 + 1.6)
    W.add_patch(Circle((CX[k] + 1.9, IY1 + 1.6 - 1.9), 1.15, fc=BLUE, ec="none", zorder=6)); W.text(CX[k] + 1.9, IY1 + 1.6 - 1.9, str(k + 1), ha="center", va="center", color="white", fontweight="bold", zorder=7)
# ---- pictogram 1: monolayer + face levels (no axes)
at = [a for a in read("../expand/S3.extxyz", ":") if a.info["name"] == "SeAlSnAs"][0]
c = at.cell; z0 = at.positions[:, 2].min()
a1 = FA(CX[0] + 1.5, IY0, CX[0] + 11.0, IY1); a1.axis("off"); a1.set_aspect("equal")
sc = at.repeat((4, 4, 1)); R = sc.positions.copy(); R[:, 0] -= c[0, 0] * 1.5 + c[1, 0] * 1.5; R[:, 1] -= c[1, 1] * 1.5; R[:, 2] -= z0; SY = sc.get_chemical_symbols()
keep = (R[:, 1] > -0.2) & (R[:, 1] < c[1, 1] - 0.2) & (R[:, 0] > -0.6) & (R[:, 0] < 2 * c[0, 0] - 0.4)
from scipy.spatial.distance import cdist
Dm = cdist(R, R); seen = set()
for i in np.where(keep)[0]:
    for j in np.where((Dm[i] > 2.0) & (Dm[i] < 3.05))[0]:
        if (j, i) in seen or not keep[j]: continue
        seen.add((i, j)); bond(a1, (R[i, 0], R[i, 2]), (R[j, 0], R[j, 2]), EL[SY[i]], EL[SY[j]])
for i in np.argsort(R[:, 1])[::-1]:
    if keep[i]: sphere(a1, R[i, 0], R[i, 2], RAD[SY[i]], EL[SY[i]])
a1.set_xlim(-1.4, 2 * c[0, 0] - 0.2); a1.set_ylim(-2.4, 7.4)
a1.text(-1.3, 5.2, "Q", fontsize=8, color="#C04A7A", fontweight="bold"); a1.text(-1.3, 0, "X", fontsize=8, color="#B86A12", fontweight="bold")
M = pd.read_csv("../res4/mono_strain.csv"); m0 = M[(M.name == "SeAlSnAs") & (M.strain == 0)].iloc[0]
a2 = FA(CX[0] + 12.2, IY0, CX[0] + bw - .4, IY1); a2.axis("off"); a2.set_xlim(-.6, 2.3); a2.set_ylim(-7.3, -2.4)
for k, f in enumerate(("X", "Q")):
    vb, cb = -m0[f"IP_{f}"], -m0[f"EA_{f}"]
    a2.add_patch(Rectangle((k - .32, vb - 1.1), .64, 1.1, fc=BLUE, ec="none")); a2.add_patch(Rectangle((k - .32, cb), .64, 1.1, fc=ORG, ec="none"))
    a2.text(k, -7.2, f, ha="center", va="bottom")
a2.plot([.32, 1.62], [-m0.IP_X] * 2, c="k", lw=.5, ls=":"); a2.plot([1.32, 1.62], [-m0.IP_Q] * 2, c="k", lw=.5, ls=":")
a2.annotate("", xy=(1.55, -m0.IP_Q), xytext=(1.55, -m0.IP_X), arrowprops=dict(arrowstyle="-|>", lw=.6, mutation_scale=5, shrinkA=0, shrinkB=0)); a2.text(1.7, (-m0.IP_Q - m0.IP_X) / 2 + .9, r"$\Delta E_{vac}$", va="center")
# ---- pictogram 2: four contacts
EC = ["Se", "Al", "Sn", "As"]
a3 = FA(CX[1] + 1.2, IY0, CX[1] + bw - 1.0, IY1); a3.axis("off"); a3.set_aspect("equal"); a3.set_xlim(0, 4.4); a3.set_ylim(-.75, 2.55)
for i, nm in enumerate(("XX", "XQ", "QX", "QQ")):
    for (y0, up) in ((0, nm[0]), (1.25, "Q" if nm[1] == "X" else "X")):
        seq = EC[::-1] if up == "X" else EC
        for k, e in enumerate(seq):
            for n in range(3):
                if n == 2 and k % 2: continue
                sphere(a3, i * 1.1 + .22 + n * .3 + (.15 if k % 2 else 0), y0 + .1 + k * .26, .11 if e != "Al" else .09, EL[e])
    a3.add_patch(Rectangle((i * 1.1 + .05, 1.0), .95, .3, fc="none", ec=ORG, lw=.9))
    a3.text(i * 1.1 + .52, -.55, nm, ha="center")
# ---- pictogram 3: potential step (no ticks)
a4 = FA(CX[2] + 4.0, IY0 + 1.6, CX[2] + bw - 1.6, IY1 - .4); a4.spines["top"].set_visible(False); a4.spines["right"].set_visible(False); z = np.linspace(0, 45, 600)
V = 1.3 * np.tanh((z - 22.5) / .6) - 7 * (np.exp(-((z - 17) / 1.4) ** 2) + np.exp(-((z - 28) / 1.4) ** 2)) - 3.5 * (np.exp(-((z - 19.5) / .9) ** 2) + np.exp(-((z - 25.5) / .9) ** 2))
a4.fill_between(z, V, -8.6, color=BT, lw=0); a4.plot(z, V, c=BLUE, lw=1.0); a4.set_xlim(0, 45); a4.set_ylim(-8.6, 3.4)
a4.set_xticks([]); a4.set_yticks([]); a4.set_xlabel("z", labelpad=1); a4.set_ylabel(r"$\bar V$", labelpad=1)
a4.plot([1, 43], [-1.3, -1.3], c="k", lw=.5, ls=":"); a4.annotate("", xy=(40.5, 1.3), xytext=(40.5, -1.3), arrowprops=dict(arrowstyle="<->", lw=.8, color=ORG, shrinkA=0, shrinkB=0)); a4.text(39.0, 0, r"$\Delta\Phi_{bi}$", color=ORG, ha="right", va="center")
# ---- pictogram 4: three alignment types
a5 = FA(CX[3] + 1.2, IY0, CX[3] + bw - .8, IY1); a5.axis("off"); a5.set_xlim(0, 3); a5.set_ylim(-1.4, 3.0)
ALN = {"I": ((0, 2.2), (.5, 1.7)), "II": ((0, 1.6), (.7, 2.4)), "III": ((0, .9), (1.2, 2.2))}
for k, t in enumerate(("I", "II", "III")):
    (v1, c1), (v2, c2) = ALN[t]; x = k + .5
    for dx, v, cc in ((-.38, v1, c1), (.04, v2, c2)):
        a5.add_patch(Rectangle((x + dx, v - .55), .34, .55, fc=BLUE, ec="none")); a5.add_patch(Rectangle((x + dx, cc), .34, .55, fc=ORG, ec="none"))
    a5.text(x, -1.05, t, ha="center")
# ---- MACE (auxiliary) tags and independent test sets
def gtag(x, y, txt):
    W.text(x, y, txt, ha="center", va="center", color="0.35", fontsize=6.5, zorder=5,
           bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.6", fc="white", ec="0.55", lw=.6, ls=(0, (2, 1.5))))
gtag(CX[0] + bw / 2, PA0 + 3.6, "MACE: phonon check")
gtag(CX[2] + 5.5, PA0 + 3.6, "MACE: stacking scan")
W.add_patch(FancyArrowPatch((CX[2] + bw - 2, CY0), (CX[3] + 5.0, PA0 + 3.6 + 1.3), connectionstyle="arc3,rad=0.25", arrowstyle="-|>,head_length=4,head_width=2.4", lw=.9, color=BLUE, ls=(0, (3, 2)), zorder=3))
W.text(CX[3] + 13.5, PA0 + 3.6, "59 blind + 102 transfer", ha="center", va="center", color=BLUE, fontsize=6.5, zorder=5,
       bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.6", fc=BT, ec=BLUE, lw=.7, ls=(0, (3, 2))))

# ================= panel b
tile(2.2, 2.8, 17.0, 35.4); tile(19.4, 2.8, 30.6, 35.4); tile(32.8, 2.8, 63.2, 35.4); tile(65.0, 2.8, 98.6, 35.4)
T = pd.read_csv("train_oof.csv"); D = pd.read_csv("../ana2/all/pred_final.csv")
chips = [r"$\Delta E_m$", r"$E_{g,1}+E_{g,2}$", r"$m_v,\ m_c$", r"$s_1,\ s_2$", r"$\chi_1+\chi_2$"]
W.text(9.6, 33.3, "monolayer inputs", ha="center")
ys = [28.9 - k * 5.4 for k in range(5)]
for y, ch in zip(ys, chips):
    box(3.0, y - 1.8, 13.0, 3.6, BLUE, fc=BT, r=1.6); W.text(9.5, y, ch, ha="center", va="center", zorder=4)
    line([(16.0, y), (18.0, y)])
line([(18.0, ys[0]), (18.0, ys[-1])])
MXc, MR = 25.0, 3.6
MOD = [(26.0, "S(contact)"), (18.0, "ML"), (8.6, "classifier")]
for y, t in MOD:
    W.add_patch(Circle((MXc, y), MR, fc=ORG, ec="none", zorder=3)); W.text(MXc, y, t, ha="center", va="center", color="white", fontweight="bold", zorder=4)
    arrow((18.0, y), (MXc - MR, y))
W.text(MXc, 33.3, "models", ha="center")
# mini plots
pA = FA(36.0, 18.6, 47.0, 30.4)
for cc in ("XX", "QX", "QQ"):
    d = T[T.contact == cc]; pA.scatter(d.oof_hyb, d.eVh_B, s=1.2, c=BLUE, lw=0, alpha=.55)
pA.plot([-.8, .8], [-.8, .8], c=ORG, lw=.8); pA.set_xlim(-.8, .8); pA.set_ylim(-.8, .8); pA.set_xticks([]); pA.set_yticks([])
pA.set_xlabel("predicted", labelpad=1); pA.set_ylabel("DFT", labelpad=2)
W.text(41.5, 33.3, r"dipole  $r^2$ = 0.94", ha="center")
pB = FA(36.0, 6.0, 47.0, 14.4)
bins = np.linspace(0, 1, 21); pB.hist(D.w_min[D.w_min < .6], bins=bins, color="0.7"); pB.hist(D.w_min[D.w_min >= .6], bins=bins, color=BLUE)
pB.axvline(.6, c=ORG, lw=.8); pB.set_xticks([]); pB.set_yticks([]); pB.set_xlabel(r"$w_{min}$", labelpad=1)
W.text(41.5, 15.6, "regime  AUC = 0.87", ha="center")
for y0_, y1_ in ((26.0, 22.0), (18.0, 22.0)): line([(MXc + MR, y0_), (31.7, y0_), (31.7, y1_)])
arrow((31.7, 22.0), (36.0, 22.0)); arrow((MXc + MR, 8.6), (36.0, 8.6))
# type node
TXc, TY0 = 55.5, 17.0
line([(47.0, 25.5), (51.0, 25.5), (51.0, TY0 + 1.5)]); line([(47.0, 8.6), (51.0, 8.6), (51.0, TY0 - 1.5)]); arrow((51.0, TY0), (52.0, TY0))
box(52.0, TY0 - 7.0, 9.5, 14.0, ORG, fc=OT, r=1.2)
for dx, (v, cc) in ((-2.1, (TY0 - 1.8, TY0 + .4)), (.2, (TY0 - .4, TY0 + 1.9))):
    W.add_patch(Rectangle((TXc + dx + .6, v - 2.4), 1.8, 2.4, fc=BLUE, ec="none", zorder=4)); W.add_patch(Rectangle((TXc + dx + .6, cc), 1.8, 2.4, fc=ORG, ec="none", zorder=4))
W.text(56.75, TY0 + 8.6, "type + margin", ha="center", va="center", zorder=5)
W.text(56.75, TY0 - 8.6, "1 105 confident", ha="center", va="center", zorder=5, fontsize=6.5)
arrow((61.5, TY0), (65.5, TY0))
# frozen-model tests as bars
pC = FA(70.5, 10.0, 97.2, 29.6)
cat = ["CV", "pair\nout", "mono.\nout", "shuffled", "blind"]; val = [.94, .93, .91, -.20, .88]; col = [BLUE, BLUE, BLUE, "0.7", ORG]
pC.bar(range(5), val, .62, color=col)
for i, v in enumerate(val): pC.text(i, v + (.03 if v > 0 else -.1), f"{v:.2f}".replace("-", "−"), ha="center", va="bottom")
pC.axhline(0, c="k", lw=.5); pC.set_xticks(range(5)); pC.set_xticklabels(cat); pC.set_ylim(-.35, 1.12); pC.set_yticks([0, .5, 1])
pC.set_ylabel(r"dipole $r^2$", labelpad=1); pC.tick_params(axis="x", length=0)
W.text(81.8, 33.3, "frozen-model tests", ha="center")
W.text(81.8, 4.4, "type accuracy (localized): 0.90 CV, 0.71 blind", ha="center", fontsize=6.5, color="0.3")
fig.savefig("fig0_workflow7.pdf"); fig.savefig("fig/fig0_workflow7.png", dpi=220)
