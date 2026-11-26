import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Ellipse, FancyArrowPatch, Wedge
from matplotlib.colors import to_rgb
import matplotlib.patheffects as pe
from ase.io import read
INK, MUTE, ARR = "k", "k", "#E8A86B"
EL = {"Se": "#E08E2B", "Al": "#6F9FD8", "Sn": "#4F4F6F", "As": "#C04A7A"}   # trùng màu X/A/M/Q của Hình 2a   # màu kiểu VESTA (Se vàng-lục, As tím)
RAD = {"Se": .55, "Al": .48, "Sn": .62, "As": .55}
TY = {"I": "#4C72B0", "II": "#DD8452", "III": "#8172B3"}
FH = 6.5; YT = 100 * FH / 7.2
fig = plt.figure(figsize=(7.2, FH)); W = fig.add_axes([0, 0, 1, 1]); W.set_xlim(0, 100); W.set_ylim(0, YT); W.axis("off")
def FA(x0, y0, x1, y1): return fig.add_axes([x0 / 100, y0 / YT, (x1 - x0) / 100, (y1 - y0) / YT])
BLUE, ORG, PUR, GREY = "#2F5DA8", "#E08E2B", "#8E44AD", "#7A8494"
def panel(y0, h, lab, title):
    W.add_patch(FancyBboxPatch((1, y0), 98, h, boxstyle="round,pad=0,rounding_size=1.5", fc="#FAFBFC", ec="#8A94A6", lw=.9, ls=(0, (4, 3)), zorder=0))
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

PA0, PAH = 40.0, YT - 41.5          # panel a
panel(PA0, PAH, "a", "Data generation (DFT)")
panel(1.5, 37.0, "b", "Regression models: fitted on the 599 interfaces, frozen, then tested")
# legend for box colours (top right of panel a)
LY = PA0 + PAH - 2.4
box(60.0, LY - 1.1, 3.0, 2.2, BLUE); W.text(63.8, LY, "DFT step", va="center", fontsize=7)
box(72.0, LY - 1.1, 3.0, 2.2, ORG, fc="#FDF2E9", ls=(0, (3, 2))); W.text(75.8, LY, "MACE, auxiliary only", va="center", fontsize=7)
box(88.5, LY - 1.1, 3.0, 2.2, PUR, fc="#F4EEF9", ls=(0, (3, 2))); W.text(92.3, LY, "test sets", va="center", fontsize=7)

xs0, bw, gap = 2.4, 22.4, 1.95; CX = [xs0 + k * (bw + gap) for k in range(4)]
SY0, bh = PA0 + 6.0, 8.2           # step boxes
IY0, IY1 = SY0 + bh + 6.6, PA0 + PAH - 5.0   # illustration band
# ---- column 1: monolayer side view + face-resolved levels
at = [a for a in read("../expand/S3.extxyz", ":") if a.info["name"] == "SeAlSnAs"][0]
c = at.cell; z0 = at.positions[:, 2].min()
a1 = FA(CX[0] + .2, IY0 - 1, CX[0] + 10.2, IY1 + 1); a1.axis("off"); a1.set_aspect("equal")
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
a1.set_xlim(-1.4, 2 * c[0, 0] - 0.2); a1.set_ylim(-2.2, 7.2)
a1.text(c[0, 0] / 2, 6.3, "Q face", ha="center", fontsize=8, color="#C04A7A"); a1.text(c[0, 0] / 2, -1.7, "X face", ha="center", fontsize=8, color="#B86A12")
M = pd.read_csv("../res4/mono_strain.csv"); m0 = M[(M.name == "SeAlSnAs") & (M.strain == 0)].iloc[0]
a2 = FA(CX[0] + 14.6, IY0 + 1.5, CX[0] + bw - .6, IY1)
for k, f in enumerate(("X", "Q")):
    vb, cb = -m0[f"IP_{f}"], -m0[f"EA_{f}"]
    a2.add_patch(Rectangle((k - .3, vb - 1.2), .6, 1.2, fc="#9DB7DA", ec="k", lw=.4)); a2.add_patch(Rectangle((k - .3, cb), .6, 1.2, fc="#F4C6A5", ec="k", lw=.4))
a2.set_xlim(-.6, 2.25); a2.set_ylim(-6.6, -2.6); a2.set_xticks([0, 1]); a2.set_xticklabels(["X", "Q"])
a2.set_ylabel(r"$E-E_{vac}$ (eV)", labelpad=1)
a2.annotate("", xy=(1.3, -m0.IP_Q), xytext=(1.3, -m0.IP_X), arrowprops=dict(arrowstyle="<->", lw=.6)); a2.text(1.85, (-m0.IP_Q - m0.IP_X) / 2, r"$\Delta E_{vac}$", rotation=90, va="center", ha="center")
# ---- column 2: four contacts
EC = ["Se", "Al", "Sn", "As"]
a3 = FA(CX[1] + .6, IY0, CX[1] + bw - .6, IY1); a3.axis("off"); a3.set_aspect("equal"); a3.set_xlim(0, 4.4); a3.set_ylim(-.75, 2.55)
for i, nm in enumerate(("XX", "XQ", "QX", "QQ")):
    for (y0, up) in ((0, nm[0]), (1.25, "Q" if nm[1] == "X" else "X")):
        seq = EC[::-1] if up == "X" else EC
        for k, e in enumerate(seq):
            for n in range(3):
                if n == 2 and k % 2: continue
                sphere(a3, i * 1.1 + .22 + n * .3 + (.15 if k % 2 else 0), y0 + .1 + k * .26, .11 if e != "Al" else .09, EL[e])
    a3.add_patch(Rectangle((i * 1.1 + .05, 1.0), .95, .3, fc="none", ec="#C0392B", lw=.8, ls=":"))
    a3.text(i * 1.1 + .52, -.5, nm, ha="center", fontsize=8)
# ---- column 3: DFT heterostructure potential
a4 = FA(CX[2] + 4.6, IY0 + 1.5, CX[2] + bw - .6, IY1); z = np.linspace(0, 45, 600)
V = .6 * np.tanh((z - 22.5) / .6) - 7 * (np.exp(-((z - 17) / 1.4) ** 2) + np.exp(-((z - 28) / 1.4) ** 2)) - 3.5 * (np.exp(-((z - 19.5) / .9) ** 2) + np.exp(-((z - 25.5) / .9) ** 2))
a4.plot(z, V, c="#2E9E62", lw=1.0); a4.set_xlim(0, 45); a4.set_ylim(-8.5, 2.6)
a4.set_xlabel("z (Å)", labelpad=1); a4.set_ylabel(r"$\bar V$ (eV)", labelpad=1)
a4.plot([2, 41], [-.6, -.6], c="k", lw=.4, ls=":"); a4.annotate("", xy=(40, .6), xytext=(40, -.6), arrowprops=dict(arrowstyle="<->", lw=.6, shrinkA=0, shrinkB=0)); a4.text(28, 1.1, r"$\Delta\Phi_{bi}$")
# ---- column 4: alignment types in the fitting set
a5 = FA(CX[3] + .4, IY0, CX[3] + bw - .4, IY1); a5.axis("off"); a5.set_xlim(0, 3); a5.set_ylim(-1.7, 3.0)
ALN = {"I": ((0, 2.2), (.5, 1.7)), "II": ((0, 1.6), (.7, 2.4)), "III": ((0, .9), (1.2, 2.2))}   # (VBM, CBM) layer 1, layer 2
CNT = {"I": 161, "II": 369, "III": 69}
for k, t in enumerate(("I", "II", "III")):
    (v1, c1), (v2, c2) = ALN[t]; x = k + .5
    a5.add_patch(Rectangle((x - .38, v1 - .55), .34, .55, fc="#9DB7DA", ec="k", lw=.4)); a5.add_patch(Rectangle((x - .38, c1), .34, .55, fc="#F4C6A5", ec="k", lw=.4))
    a5.add_patch(Rectangle((x + .04, v2 - .55), .34, .55, fc="#9DB7DA", ec="k", lw=.4)); a5.add_patch(Rectangle((x + .04, c2), .34, .55, fc="#F4C6A5", ec="k", lw=.4))
    a5.text(x, -1.0, f"type {t}", ha="center", fontsize=7); a5.text(x, -1.5, f"n = {CNT[t]}", ha="center", fontsize=7)
# ---- step boxes
STEPS = [("1", "108 monolayers", "X–A–M–Q, DFT relaxed;\nIP and EA on each face"),
         ("2", "7 796 interfaces", "4 contacts per pair; lattice-\nmatched (≤ 3%) of 23 112"),
         ("3", "832 heterostructures", "DFT relaxed; interface dipole,\nlayer-resolved band edges"),
         ("4", "599 fitting set", "both layers gapped;\nalignment type known")]
for k, (n, ttl, sub) in enumerate(STEPS):
    x = CX[k]; box(x, SY0, bw, bh, BLUE)
    W.add_patch(Circle((x + 2.3, SY0 + bh - 2.2), 1.5, fc=BLUE, ec="none", zorder=4))
    W.text(x + 2.3, SY0 + bh - 2.2, n, ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=5)
    W.text(x + 4.5, SY0 + bh - 2.2, ttl, va="center", fontsize=8.5, fontweight="bold", zorder=5)
    W.text(x + 1.3, SY0 + 2.5, sub, va="center", fontsize=6.6, linespacing=1.15, zorder=5)
    if k < 3: arrow((x + bw + .15, SY0 + bh / 2), (x + bw + gap - .15, SY0 + bh / 2), color=ORG, lw=1.1)
def tag(x, txt):
    box(x, SY0 + bh + .8, bw, 4.6, ORG, fc="#FDF2E9", ls=(0, (3, 2)), r=.8)
    W.text(x + bw / 2, SY0 + bh + 3.1, txt, linespacing=1.1, ha="center", va="center", fontsize=6.4, color="#9A5512", zorder=4)
tag(CX[0], "fine-tuned MACE: phonons of all 108\n(stability check, no compound removed)")
tag(CX[2], "MACE-MP-0 rigid scan: stacking\npre-selection, then DFT relaxation")
# ---- independent test sets
TX0 = CX[1] + 6.5; box(TX0, PA0 + .9, CX[3] + bw - TX0, 4.4, PUR, fc="#F4EEF9", ls=(0, (3, 2)), r=.8)
W.text(TX0 + (CX[3] + bw - TX0) / 2, PA0 + 3.0, "independent DFT test sets, same protocol, never fitted:\n59 blind interfaces of unseen pairs; 102 interfaces of three other Janus families",
       ha="center", va="center", fontsize=6.4, color="#5E2C78", zorder=4)
W.text(CX[0] + 1.0, PA0 + 3.1, "All data in steps 1–4 are DFT; MACE neither\nremoves compounds nor supplies training data.", va="center", fontsize=6.4, style="italic", linespacing=1.1)

# ================= panel b
chips = [r"$\Delta E_m$ (face midgap)", r"$E_{g,1}+E_{g,2}$", r"$m_v, m_c$ (DOS)", r"$s_1, s_2$ (intrinsic steps)", r"$\chi_1+\chi_2$ (contact atoms)"]
W.text(11.8, 32.6, "monolayer inputs", ha="center", fontsize=8.5, fontweight="bold")
ys = [28.6 - k * 5.3 for k in range(5)]
for y, ch in zip(ys, chips):
    box(3.0, y - 1.7, 17.6, 3.4, "#3A86C8", r=1.2); W.text(11.8, y, ch, ha="center", va="center", fontsize=7.6, zorder=4)
    line([(20.6, y), (22.6, y)])
line([(22.6, ys[0]), (22.6, ys[-1])])
MX, MW, MH = 25.0, 22.5, 6.4
MODELS = [(28.6, "Linear response, S(contact)", "dipole and band offsets"),
          (21.0, "ML correction", "gradient boosting on residual of model 1"),
          (8.6, "Localization classifier", r"predicts $w_{min}\geq0.6$ (rigid-band regime)")]
for k, (y, t1, t2) in enumerate(MODELS):
    box(MX, y - MH / 2, MW, MH, "#2E9E62")
    W.add_patch(Circle((MX + 1.9, y + 1.2), 1.2, fc="#2E9E62", ec="none", zorder=4)); W.text(MX + 1.9, y + 1.2, str(k + 1), ha="center", va="center", color="white", fontsize=7.5, fontweight="bold", zorder=5)
    W.text(MX + 3.7, y + 1.2, t1, va="center", fontsize=8, fontweight="bold", zorder=5); W.text(MX + 1.2, y - 1.5, t2, va="center", fontsize=6.6, zorder=5)
    arrow((22.6, y), (MX, y))
W.text(MX + MW / 2, 32.6, "models (monolayer data only)", ha="center", fontsize=8.5, fontweight="bold")
OX, OW = 51.5, 20.0
OUTS = {"dip": (24.8, "Interface dipole", r"$r^2$ = 0.94, MAE 0.043 eV (CV)"),
        "typ": (16.7, "Alignment type + margin", "1 105 confident predictions\namong the 7 796 interfaces"),
        "reg": (8.6, "Rigid-band regime", "AUC = 0.87 (CV)")}
for key, (y, t1, t2) in OUTS.items():
    hh = 6.4 if key != "typ" else 7.0
    box(OX, y - hh / 2, OW, hh, "#C0392B" if key == "typ" else BLUE, fc="#FCEDEB" if key == "typ" else "white")
    W.text(OX + 1.2, y + 1.6, t1, va="center", fontsize=8, fontweight="bold", zorder=5); W.text(OX + 1.2, y - 1.3, t2, va="center", fontsize=6.6, linespacing=1.1, zorder=5)
W.text(OX + OW / 2, 32.6, "predictions", ha="center", fontsize=8.5, fontweight="bold")
line([(MX + MW, 28.6), (49.6, 28.6), (49.6, 21.0), (MX + MW, 21.0)]); arrow((49.6, 24.8), (OX, 24.8))
arrow((MX + MW, 8.6), (OX, 8.6))
arrow((OX + OW / 2, 24.8 - 3.2), (OX + OW / 2, 16.7 + 3.5)); arrow((OX + OW / 2, 8.6 + 3.2), (OX + OW / 2, 16.7 - 3.5))
W.text(OX + OW / 2 + .6, 20.85, "+ monolayer gaps", fontsize=6.4, va="center", ha="left")
# ---- validation card
x0, y0, w, h = 73.2, 3.3, 25.3, 30.6
W.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc="white", ec=PUR, lw=.9, zorder=3))
W.add_patch(FancyBboxPatch((x0, y0 + h - 3.4), w, 3.4, boxstyle="round,pad=0,rounding_size=1.0", fc=PUR, ec=PUR, lw=.9, zorder=4))
W.add_patch(Rectangle((x0, y0 + h - 3.4), w, 1.2, fc=PUR, ec="none", zorder=4))
W.text(x0 + w / 2, y0 + h - 1.7, "FROZEN-MODEL TESTS", ha="center", va="center", color="white", fontsize=8.2, fontweight="bold", zorder=5)
C1, C2 = x0 + 17.6, x0 + w - .8
W.text(x0 + .8, y0 + h - 5.5, "test", fontsize=7, color="0.35", zorder=5, va="center")
W.text(C1, y0 + h - 5.5, r"dipole $r^2$", fontsize=7, color="0.35", ha="right", zorder=5, va="center"); W.text(C2, y0 + h - 5.5, "type acc.*", fontsize=7, color="0.35", ha="right", zorder=5, va="center")
rows = [("5-fold CV (by pair)", "0.94", "0.90"), ("pair hold-out", "0.93", "–"), ("monolayer hold-out", "0.91", "–"),
        ("shuffled (control)", "−0.20", "–"), ("blind (59)", "0.88", "0.71"), ("   type margin > 0.1 eV", "–", "1.00")]
for k, (a, b, cc) in enumerate(rows):
    yy = y0 + h - 8.4 - k * 2.6
    if k % 2 == 0: W.add_patch(Rectangle((x0 + .3, yy - 1.2), w - .6, 2.4, fc="#F4EEF9", ec="none", zorder=4))
    col = "0.35" if k == 3 else PUR
    W.text(x0 + .8, yy, a, fontsize=6.9, va="center", zorder=5)
    W.text(C1, yy, b, fontsize=6.9, va="center", ha="right", color=col, fontweight="bold", zorder=5); W.text(C2, yy, cc, fontsize=6.9, va="center", ha="right", color=col, fontweight="bold", zorder=5)
W.text(x0 + .8, y0 + 5.1, "transfer (102): pnictogen-face rule holds\nin direction, not in magnitude", fontsize=6.9, va="center", linespacing=1.15, zorder=5)
W.text(x0 + .8, y0 + 1.7, "*localized regime; blind MAE 0.048 eV", fontsize=6.0, color="0.35", va="center", zorder=5)
arrow((OX + OW, 16.7), (x0, 16.7))
arrow((90.0, PA0 + .9), (90.0, y0 + h), color=PUR)
fig.savefig("fig0_workflow5.pdf"); fig.savefig("fig/fig0_workflow5.png", dpi=220)
