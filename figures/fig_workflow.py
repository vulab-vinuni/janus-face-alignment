import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Ellipse, FancyArrowPatch, Wedge
from matplotlib.colors import to_rgb
import matplotlib.patheffects as pe
from ase.io import read
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.0, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "axes.linewidth": .8, "mathtext.default": "regular"})
plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 10, "axes.labelweight": "bold", "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5, "mathtext.fontset": "cm", "mathtext.default": "it"})
import re as _re
def _bm(s):
    if "$" not in s: return s
    parts=s.split("$")
    for i in range(1,len(parts),2):
        m=parts[i]; out=""; j=0
        toks=_re.findall(r"\\[A-Za-z]+|[A-Za-z]+|[0-9.]+|\{[^{}]*\}|.",m)
        k=0
        while k<len(toks):
            t=toks[k]
            if t in ("\\mathrm","\\mathbf","\\text"):
                out+="\\mathbf"; k+=1
                if k<len(toks): out+=toks[k]
            elif t.startswith("\\") and t[1:2].isupper(): out+="\\mathbf{"+t+"}"
            elif t.startswith("\\") and t[1:] in ("alpha","beta","gamma","delta","chi","rho","sigma","phi","psi","omega","epsilon","mu","nu","eta","theta","tau","lambda","kappa","xi","pi"): out+="\\mathbfit{"+t+"}"
            elif t.startswith("\\"): out+=t
            elif t[0].isalpha(): out+="\\mathbfit{"+t+"}"
            elif t[0].isdigit(): out+="\\mathbf{"+t+"}"
            elif t.startswith("{"): out+="{"+_bm("$"+t[1:-1]+"$")[1:-1]+"}"
            else: out+=t
            k+=1
        parts[i]=out
    return "$".join(parts)
def _boldticks(fig):
    for _a in fig.axes:
        _a.set_xlabel(_bm(_a.get_xlabel())); _a.set_ylabel(_bm(_a.get_ylabel()))
    fig.canvas.draw()
    for _a in fig.axes:
        for _t in _a.get_xticklabels() + _a.get_yticklabels(): _t.set_fontweight("bold")

INK, MUTE, ARR = "k", "k", "#E8A86B"
EL = {"Se": "#E08E2B", "Al": "#6F9FD8", "Sn": "#4F4F6F", "As": "#C04A7A"}   # trùng màu X/A/M/Q của Hình 2a   # màu kiểu VESTA (Se vàng-lục, As tím)
RAD = {"Se": .55, "Al": .48, "Sn": .62, "As": .55}
TY = {"I": "#7B3FA0", "II": "#12938F", "III": "#A61B1B"}
fig = plt.figure(figsize=(7.2, 6.0)); W = fig.add_axes([0, 0, 1, 1]); W.set_xlim(0, 100); W.set_ylim(0, 83.333); W.axis("off")
def panel(y0, h, lab, title):
    W.add_patch(FancyBboxPatch((1, y0), 98, h, boxstyle="round,pad=0,rounding_size=1.5", fc="#FAFBFC", ec="#8A94A6", lw=.9, ls=(0, (4, 3)), zorder=0))
    W.text(2.6, y0 + h - 2.6, lab, fontsize=12, fontweight="bold", va="center", color=INK); W.text(5.6, y0 + h - 2.6, title, fontsize=11.5, fontweight="bold", va="center", color=INK)
def curved(p1, p2, rad):
    W.add_patch(FancyArrowPatch(p1, p2, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>,head_length=5,head_width=3", lw=1.6, color=ARR, zorder=5))
def elbow(pts, color="#7A8494", lw=.9, head=True):
    xs, ys = zip(*pts); W.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), c=color, lw=lw, zorder=2, solid_joinstyle="miter")
    if head: W.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>,head_length=4,head_width=2.4", lw=lw, color=color, zorder=3))
def sphere(ax, x, y, r, col, z=3):
    c = np.array(to_rgb(col)); ax.add_patch(Circle((x, y), r, fc=c * .72, ec="none", zorder=z))
    for k in range(1, 9):
        f = k / 9; cc = c * (.72 + .28 * f) + (1 - c) * .55 * f ** 2.2
        ax.add_patch(Circle((x - .32 * r * f, y + .32 * r * f), r * (1 - .82 * f), fc=np.clip(cc, 0, 1), ec="none", zorder=z))
def bond(ax, p, q, c1, c2):
    m = (np.array(p) + np.array(q)) / 2
    for a, b, cc in ((p, m, c1), (m, q, c2)):
        ax.plot([a[0], b[0]], [a[1], b[1]], c=cc, lw=3.0, solid_capstyle="butt", zorder=2, path_effects=[pe.Stroke(linewidth=3.8, foreground="#555"), pe.Normal()])
panel(38.5, 43.33, "a", "Data generation: monolayers, interfaces and DFT heterostructures")
panel(1.5, 35.5, "b", "Regression models fitted on the 599 interfaces, frozen, then tested")
from matplotlib.patches import Polygon
_TX = tuple((2.4 + k * 24.35, 2.4 + k * 24.35 + 22.4) for k in range(4))
for k_ in range(3):
    xa_, xb_ = _TX[k_][1], _TX[k_ + 1][0]
    W.add_patch(FancyArrowPatch((xa_ - 2.2, 55.4), (xb_ + 2.2, 55.4), connectionstyle="arc3,rad=0.55", arrowstyle="-|>,head_length=4,head_width=2.5", lw=1.4, color=ARR, zorder=6))
for x0_, x1_ in _TX:
    W.add_patch(FancyBboxPatch((x0_, 54.6), x1_ - x0_, 23.0, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec="#4B5563", lw=.7, zorder=1))
# ---------- (a1) VESTA-like side view
at = {a.info["name"]: a for a in read("../expand/S3.extxyz", ":") if a.info["name"] == "SeAlSnAs"}["SeAlSnAs"]
c = at.cell; pos = at.positions; sym = at.get_chemical_symbols(); z0 = pos[:, 2].min()
a1 = fig.add_axes([0.03, 0.80, 0.20, 0.125]); a1.set_visible(False); a1.axis("off"); a1.set_aspect("equal")
sc = at.repeat((4, 4, 1)); R = sc.positions.copy(); R[:, 0] -= c[0, 0] * 1.5 + c[1, 0] * 1.5; R[:, 1] -= c[1, 1] * 1.5; R[:, 2] -= z0; SY = sc.get_chemical_symbols()
keep = (R[:, 1] > -0.2) & (R[:, 1] < c[1, 1] - 0.2) & (R[:, 0] > -0.6) & (R[:, 0] < 2 * c[0, 0] - 0.4)
from scipy.spatial.distance import cdist
Dm = cdist(R, R); seen = set()
for i in np.where(keep)[0]:
    for j in np.where((Dm[i] > 2.0) & (Dm[i] < 3.05))[0]:
        if (j, i) in seen: continue
        seen.add((i, j)); keep_j = keep[j]
        if not keep_j: continue
        bond(a1, (R[i, 0], R[i, 2]), (R[j, 0], R[j, 2]), EL[SY[i]], EL[SY[j]])
order = np.argsort(R[:, 1])[::-1]
for i in order:
    if keep[i]: sphere(a1, R[i, 0], R[i, 2], RAD[SY[i]], EL[SY[i]])
a1.set_xlim(-1.4, 2 * c[0, 0] - 0.2); a1.set_ylim(-2.2, 7.2)
a1.text(c[0, 0] / 2, 6.3, "Q", ha="center", fontsize=9.4, fontweight="bold", color="#C04A7A"); a1.text(c[0, 0] / 2, -1.6, "X", ha="center", fontsize=9.4, fontweight="bold", color="#E08E2B")

# ---------- (a2) face-resolved levels: Origin frame
M = pd.read_csv("../res4/mono_strain.csv"); m0 = M[(M.name == "SeAlSnAs") & (M.strain == 0)].iloc[0]
a2 = fig.add_axes([0.034, 0.668, 0.198, 0.252]); a2.axis("off")
from matplotlib.colors import to_rgb as _rgb
def _t(c, f): r = np.array(_rgb(c)); return tuple(r + (1 - r) * f)
cX, cQ, LC = "#1F4E9C", "#E07B1F", "#6B7280"
dV = m0.IP_Q - m0.IP_X; vbm, cbm = -m0.IP_X, -m0.EA_X; L0, L1 = 1.5, 2.3
a2.add_patch(Rectangle((L0, vbm - 1.1), L1 - L0, 1.1, fc=LC, ec="k", lw=.4)); a2.add_patch(Rectangle((L0, cbm), L1 - L0, 1.1, fc=_t(LC, .65), ec="k", lw=.4))
a2.plot([L0 - .05, L1 + .05], [(vbm + cbm) / 2] * 2, c="k", lw=.8); a2.text(L0 - .1, (vbm + cbm) / 2, r"$\mathrm{E_m}$", ha="right", va="center", fontsize=7.5)
a2.text((L0 + L1) / 2, vbm - 1.35, "monolayer", ha="center", va="top", fontsize=7.5)
a2.plot([-.45, L0], [0, 0], c=cX, ls="--", lw=.9); a2.text(-.45, .2, r"$\mathrm{E_{vac,X}}$", ha="left", va="bottom", fontsize=7.5, color=cX)
a2.plot([L1, 4.25], [dV, dV], c=cQ, ls="--", lw=.9); a2.text(3.1, dV + .2, r"$\mathrm{E_{vac,Q}}$", ha="center", va="bottom", fontsize=7.5, color=cQ)
def _ar(x, y0_, y1_, col): a2.annotate("", xy=(x, y1_), xytext=(x, y0_), arrowprops=dict(arrowstyle="<->", mutation_scale=5, lw=.7, shrinkA=0, shrinkB=0, color=col))
_ar(.32, 0, vbm, cX); a2.text(.2, -3.0, r"$\mathrm{IP_X}$", ha="right", va="center", fontsize=7.5, color=cX)
_ar(1.08, 0, cbm, cX); a2.text(.96, -1.6, r"$\mathrm{EA_X}$", ha="right", va="center", fontsize=7.5, color=cX)
_ar(2.72, dV, cbm, cQ); a2.text(2.84, -1.3, r"$\mathrm{EA_Q}$", ha="left", va="center", fontsize=7.5, color=cQ)
_ar(3.5, dV, vbm, cQ); a2.text(3.62, -2.9, r"$\mathrm{IP_Q}$", ha="left", va="center", fontsize=7.5, color=cQ)
a2.plot([L1, 4.25], [0, 0], c=cX, ls=":", lw=.6); _ar(4.2, 0, dV, "k")
a2.text(4.3, dV / 2, r"$\mathrm{\Delta E_{vac}}$", ha="left", va="center", fontsize=7.5)
a2.set_xlim(-.5, 5.05); a2.set_ylim(-6.5, 1.9)

# ---------- (a3) four contacts
ROW = [("X", EL["Se"]), ("A", EL["Al"]), ("M", EL["Sn"]), ("Q", EL["As"])]
a3 = fig.add_axes([0.2775, 0.668, 0.21, 0.245]); a3.axis("off"); a3.set_aspect("equal"); a3.set_xlim(0, 4.4); a3.set_ylim(-.65, 2.6)
ROW = ["Se", "Al", "Sn", "As"]
for i, nm in enumerate(("XX", "XQ", "QX", "QQ")):
    for (y0, up) in ((0, nm[0]), (1.25, "Q" if nm[1] == "X" else "X")):
        seq = ROW[::-1] if up == "X" else ROW
        for k, e in enumerate(seq):
            for n in range(3):
                xx = i * 1.1 + .22 + n * .3 + (.15 if k % 2 else 0); yy = y0 + .1 + k * .26
                if n == 2 and k % 2: continue
                sphere(a3, xx, yy, .11 if e != "Al" else .09, EL[e])
    a3.add_patch(Rectangle((i * 1.1 + .05, 1.0), .95, .3, fc="none", ec="#C0392B", lw=.8, ls=":"))
    a3.text(i * 1.1 + .52, -.45, nm, ha="center", fontsize=8.5, fontweight="bold", color=INK)

# ---------- (a4) database
def _db(cx, y0, w, col, h=2.6, g=.7, e=2.3, pad=0.0, fc_override=None, z=3):
    tt = np.linspace(0, np.pi, 40); hw_ = w / 2 + pad; c_ = fc_override or col
    ytop = y0 + 3 * h + 2 * g
    W.add_patch(Rectangle((cx - hw_, y0), 2 * hw_, ytop - y0, fc=c_, ec="none", zorder=z))
    W.add_patch(Ellipse((cx, y0), 2 * hw_, e + 2 * pad, fc=c_, ec="none", zorder=z))
    W.add_patch(Ellipse((cx, ytop), 2 * hw_, e + 2 * pad, fc=c_, ec="none", zorder=z))
    if fc_override: return
    W.add_patch(Ellipse((cx, ytop), .78 * w, .62 * e, fc="white", ec="none", zorder=z + .2))
    for k in (1, 2):
        yb = y0 + k * h + (k - 1) * g
        lo = [(cx + w / 2 * np.cos(t), yb - e / 2 * np.sin(t)) for t in tt]
        hi = [(cx + w / 2 * np.cos(t), yb + g - e / 2 * np.sin(t)) for t in tt[::-1]]
        W.add_patch(Polygon(lo + hi, closed=True, fc="white", ec="none", zorder=z + .1))
for cx_ in (82.2, 91.1): _db(cx_, 64.6, 7.6, "#2B63D0", z=3)
_db(86.65, 60.4, 7.6, "#161B57", pad=.7, fc_override="white", z=4)
_db(86.65, 60.4, 7.6, "#161B57", z=4.5)
W.text(86.65, 58.3, "599 of 832 with known type", ha="center", va="top", fontsize=7.4, color=INK)
for x_, t_, n_ in ((80.6, "I", 161), (86.65, "II", 369), (92.7, "III", 69)): W.text(x_, 55.9, f"{t_}: {n_}", ha="center", va="center", fontsize=7.2, color=TY[t_], fontweight="bold")
adb = fig.add_axes([64.2 / 100, 56.4 / 83.333, 10.8 / 100, 14.6 / 83.333]); adb.axis("off")
# ---------- (a5) DFT potential: Origin frame
a4 = fig.add_axes([0.552, 0.705, 0.175, 0.205]); z = np.linspace(0, 45, 600)
V = 4.4 + .6 * np.tanh((z - 22.5) / .6) - 7 * (np.exp(-((z - 17) / 1.4) ** 2) + np.exp(-((z - 28) / 1.4) ** 2)) - 3.5 * (np.exp(-((z - 19.5) / .9) ** 2) + np.exp(-((z - 25.5) / .9) ** 2))
V = V - 4.4 + .0
a4.plot(z, V, c="#2E9E62", lw=1.0); a4.set_xlim(0, 45); a4.set_ylim(-8.5, 3.2); a4.minorticks_on()
a4.set_xlabel("z (Å)", fontsize=8.1, labelpad=1); a4.set_ylabel(r"$\bar V$ (eV)", fontsize=8.1, labelpad=1); a4.tick_params(labelsize=7.5, length=2.5)
a4.plot([1, 44], [-.6, -.6], c="k", lw=.4, ls=":"); a4.annotate("", xy=(41.5, .6), xytext=(41.5, -.6), arrowprops=dict(arrowstyle="<->", mutation_scale=5, lw=.6, shrinkA=0, shrinkB=0)); a4.text(40.3, .9, r"$\mathrm{\Delta\Phi_{bi}}$", ha="right", va="bottom", fontsize=7.5)

# curved arrows (over / under alternating)
pass  # connectors removed: curved((17.5, 76.61), (26.5, 76.61), -0.35); curved((36.8, 54.61), (43.5, 54.61), 0.45); curved((62.8, 72.61), (66.0, 70.11), -0.45); curved((73.5, 70.11), (79.5, 72.61), -0.45)

# ---------- (a) numbered pipeline with counts and the two roles of MACE
STEPS = [("1", "108 monolayers", "all X–A–M–Q, DFT relaxed;\nIP and EA on each face"),
         ("2", "7 796 interfaces", "4 contacts per pair, lattice-\nmatched (≤ 3%) of 23 112"),
         ("3", "832 heterostructures", "lowest stacking, DFT relaxed;\ndipole, layer band edges"),
         ("4", "599 fitting set", "both layers gapped; all\nvalues in steps 1–4 from DFT")]
SY0 = 39.6; bw, bh, gap = 22.4, 8.2, 1.95; xs0 = 2.4
SHD = ["#7FA3D9", "#5584C8", "#3767B4", "#1F4E9C"]; HH = 3.3
from matplotlib.patches import Polygon
for k, (n, ttl, sub) in enumerate(STEPS):
    x = xs0 + k * (bw + gap); cH = SHD[k]
    W.add_patch(FancyBboxPatch((x, SY0), bw, bh, boxstyle="round,pad=0,rounding_size=0.9", fc="white", ec=cH, lw=.9, zorder=3))
    W.add_patch(FancyBboxPatch((x, SY0 + bh - HH), bw, HH, boxstyle="round,pad=0,rounding_size=0.9", fc=cH, ec=cH, lw=.9, zorder=4))
    W.add_patch(Rectangle((x, SY0 + bh - HH), bw, HH / 2, fc=cH, ec="none", zorder=4))
    W.add_patch(Circle((x + 1.9, SY0 + bh - HH / 2), 1.15, fc="white", ec="none", zorder=5))
    W.text(x + 1.9, SY0 + bh - HH / 2, n, ha="center", va="center", color=cH, fontweight="bold", zorder=6)
    W.text(x + 3.7, SY0 + bh - HH / 2, ttl, va="center", color="white", fontweight="bold", zorder=6)
    W.text(x + 1.2, SY0 + (bh - HH) / 2, sub, va="center", fontsize=6.6, color="#374151", linespacing=1.15, zorder=5)
    if k < 3:
        xa, ym = x + bw + .35, SY0 + bh - HH / 2
        W.add_patch(Polygon([(xa, ym - 1.1), (xa + gap - .7, ym), (xa, ym + 1.1)], closed=True, fc=SHD[k + 1], ec="none", zorder=5))
def tag(x, y, txt):
    W.add_patch(FancyBboxPatch((x, y), bw, 5.0, boxstyle="round,pad=0,rounding_size=0.8", fc="#FDF2E9", ec="#E08E2B", lw=.7, ls=(0, (3, 2)), zorder=3))
    W.text(x + bw / 2, y + 2.5, txt, linespacing=1.1, ha="center", va="center", fontsize=6.4, color="#9A5512", zorder=4)
tag(xs0, SY0 + bh + .7, "fine-tuned MACE: phonons of\nall 108 (check, no filtering)")
tag(xs0 + 2 * (bw + gap), SY0 + bh + .7, "MACE-MP-0 rigid scan:\nstacking pre-selection")
W.add_patch(FancyBboxPatch((xs0 + 3 * (bw + gap), SY0 + bh + .7), bw, 5.0, boxstyle="round,pad=0,rounding_size=0.8", fc="#EAF0FA", ec="#2F5DA8", lw=.7, ls=(0, (3, 2)), zorder=3))
W.text(xs0 + 3 * (bw + gap) + bw / 2, SY0 + bh + 3.2, "+ 59 blind, 102 transfer interfaces\n(same DFT protocol, never fitted)", linespacing=1.1, ha="center", va="center", fontsize=6.4, color="#2F5DA8", zorder=4)
# ---------- (b) chips -> hub (elbow), hub -> outputs (elbow)
chips = [r"$\Delta E_m$ (contact faces)", r"$E_{g,1}+E_{g,2}$", r"$m_v, m_c$ (DOS)", r"$s_1, s_2$ (dipole)", r"$\chi_1+\chi_2$ (contact)"]
ys = [27.5 - k * 5.0 for k in range(5)]; HX, HY, HR = 35, 17.5, 7.4
W.text(12.5, 30.7, "monolayer inputs", ha="center", fontsize=8.5, fontweight="bold", color=INK)
for y, ch in zip(ys, chips):
    W.add_patch(FancyBboxPatch((4, y - 1.6), 16.5, 3.2, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec="#3A86C8", lw=.8, zorder=3))
    W.text(12.25, y, ch, ha="center", va="center", fontsize=8.2, color=INK, zorder=4)
    W.plot([20.5, 24.5], [y, y], c="#7A8494", lw=.9, zorder=2)
W.plot([24.5, 24.5], [ys[0], ys[-1]], c="#7A8494", lw=.9, zorder=2)           # thanh góp (bus)
elbow([(24.5, HY), (HX - HR - .3, HY)])
cols = ["#2F5DA8", "#3A86C8", "#2E9E62", "#C8D93A", "#E08E2B", "#C0392B", "#8E44AD", "#5B6FD0"]
for k in range(8): W.add_patch(Wedge((HX, HY), HR, k * 45, (k + 1) * 45, width=.9, fc=cols[k], ec="none", alpha=.9, zorder=3))
W.add_patch(Circle((HX, HY), HR - .9, fc="white", ec="none", zorder=3))
W.text(HX, HY + 3.0, "models", ha="center", va="center", fontweight="bold", color=INK, zorder=4)
W.text(HX - 4.6, HY - .7, "1  S(contact)\n2  ML correction\n3  classifier", ha="left", va="center", linespacing=1.15, color=INK, zorder=4)
outs = [("Interface dipole  (1 + 2)", 27.5, "r² = 0.94 (CV)"), ("Hybridization regime  (3)", 17.5, "AUC = 0.87 (CV)"), ("Alignment type  (1–3 + gaps)", 7.5, "1 105 confident of 7 796")]
W.plot([HX + HR, 46.0], [HY, HY], c="#7A8494", lw=.9, zorder=2); W.plot([46.0, 46.0], [outs[0][1], outs[2][1]], c="#7A8494", lw=.9, zorder=2)
T = pd.read_csv("train_oof.csv"); D = pd.read_csv("../ana2/all/pred_final.csv"); A = pd.read_csv("pred7796_hyb.csv"); A = A[A.conf]
for k, (name, y, stat) in enumerate(outs):
    elbow([(46.0, y), (49.8, y)])
    ax = fig.add_axes([0.50, (y - 4.2) / 83.333, 0.085, 8.4 / 83.333]); ax.tick_params(labelsize=5.6, length=1.8, pad=1); ax.minorticks_on()
    if k == 0:
        for cc, col in (("XX", "#2F6DB5"), ("QX", "#D46A9F"), ("QQ", "#5A5A5A")):
            d = T[T.contact == cc]; ax.scatter(d.oof_hyb, d.eVh_B, s=.8, c=col, lw=0)
        ax.plot([-.8, .8], [-.8, .8], c="k", lw=.5); ax.set_xlim(-.8, .8); ax.set_ylim(-.8, .8)
    elif k == 1:
        ax.hist(D.w_min[D.contact == "XX"], bins=12, color="#2F6DB5", alpha=.6); ax.hist(D.w_min[D.contact == "QQ"], bins=12, color="#5A5A5A", alpha=.6)
    else:
        for t in ("I", "II", "III"):
            d = A[A.type_hyb == t]; ax.scatter(d.dE.abs(), d.Eg1 + d.Eg2, s=.5, c=TY[t], lw=0)
        ax.set_xlim(0, 2.5)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    W.text(59.3, y + 1.4, name, fontsize=8.0, fontweight="bold", color=INK, va="center"); W.text(59.3, y - 1.4, stat, fontsize=7.5, color=MUTE, va="center")
    elbow([(78.6, y), (79.8, y)], head=False)
W.plot([79.8, 79.8], [outs[0][1], outs[2][1]], c="#7A8494", lw=.9, zorder=2); elbow([(79.8, HY), (80.4, HY)])
# ---------- validation card (table)
x0, y0, w, h = 80.4, 3.0, 18.4, 28.4
W.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc="white", ec="#8E44AD", lw=.9, zorder=3))
W.add_patch(FancyBboxPatch((x0, y0 + h - 3.4), w, 3.4, boxstyle="round,pad=0,rounding_size=1.0", fc="#8E44AD", ec="#8E44AD", lw=.9, zorder=4))
W.add_patch(Rectangle((x0, y0 + h - 3.4), w, 1.2, fc="#8E44AD", ec="none", zorder=4))
W.text(x0 + w / 2, y0 + h - 1.7, "FROZEN-MODEL TESTS", ha="center", va="center", color="white", fontsize=8.2, fontweight="bold", zorder=5)
W.text(x0 + 1.0, y0 + h - 5.4, "test", fontsize=7.5, color=MUTE, zorder=5); W.text(x0 + w - 1.0, y0 + h - 5.4, "value", fontsize=7.5, color=MUTE, ha="right", zorder=5)
rows = [("CV (pair), r²", "0.94"), ("pair hold-out, r²", "0.93"), ("monolayer out, r²", "0.91"), ("shuffled, r²", "−0.20"), ("blind (59), r²", "0.88"), ("blind (59), type acc.*", "0.71"), ("transfer (102), Pn rule", "✓")]
for k, (a, b) in enumerate(rows):
    yy = y0 + h - 8.0 - k * 2.7
    if k % 2 == 0: W.add_patch(Rectangle((x0 + .3, yy - 1.3), w - .6, 2.6, fc="#F4EEF9", ec="none", zorder=4))
    W.text(x0 + .8, yy, a, fontsize=6.9, va="center", color=INK, zorder=5); W.text(x0 + w - .8, yy, b, fontsize=6.9, va="center", ha="right", color="#8E44AD" if k != 3 else MUTE, fontweight="bold", zorder=5)
W.text(x0 + .8, y0 + 1.2, "*localized regime (CV 0.90)", fontsize=6.0, color=MUTE, va="center", zorder=5)
_boldticks(fig); fig.savefig("fig0_workflow14.pdf"); fig.savefig("fig/fig0_workflow14.png", dpi=220)
