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
EL = {"Se": "#E08E2B", "Al": "#6F9FD8", "Sn": "#4F4F6F", "As": "#C04A7A"}   # same X/A/M/Q colours as Fig. 2a (VESTA-like)
RAD = {"Se": .55, "Al": .48, "Sn": .62, "As": .55}
TY = {"I": "#4C72B0", "II": "#DD8452", "III": "#8172B3"}
fig = plt.figure(figsize=(7.2, 5.4)); W = fig.add_axes([0, 0, 1, 1]); W.set_xlim(0, 100); W.set_ylim(0, 75); W.axis("off")
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
panel(38.5, 35, "a", "Face-resolved physics and data generation")
panel(1.5, 35.5, "b", "Learning, validation and screening")
# ---------- (a1) VESTA-like side view
at = {a.info["name"]: a for a in read("../expand/S3.extxyz", ":") if a.info["name"] == "SeAlSnAs"}["SeAlSnAs"]
c = at.cell; pos = at.positions; sym = at.get_chemical_symbols(); z0 = pos[:, 2].min()
a1 = fig.add_axes([0.03, 0.585, 0.17, 0.33]); a1.axis("off"); a1.set_aspect("equal")
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
for xc in (0, c[0, 0]): a1.plot([xc, xc], [-2.2, 7.2], c="k", lw=.6, zorder=1)
a1.set_xlim(-1.4, 2 * c[0, 0] - 0.2); a1.set_ylim(-2.2, 7.2)
a1.text(c[0, 0] / 2, 6.3, "Q", ha="center", fontsize=9.4, fontweight="bold", color="#C04A7A"); a1.text(c[0, 0] / 2, -1.6, "X", ha="center", fontsize=9.4, fontweight="bold", color="#E08E2B")
W.text(11.5, 41.2, "Janus X–A–M–Q (108)", ha="center", fontsize=8.0, color=INK, fontweight="bold")
# ---------- (a2) face-resolved levels: Origin frame
M = pd.read_csv("../res4/mono_strain.csv"); m0 = M[(M.name == "SeAlSnAs") & (M.strain == 0)].iloc[0]
a2 = fig.add_axes([0.26, 0.625, 0.125, 0.255])
for k, f in enumerate(("X", "Q")):
    vb, cb = -m0[f"IP_{f}"], -m0[f"EA_{f}"]
    a2.add_patch(Rectangle((k - .3, vb - 1.2), .6, 1.2, fc="#9DB7DA", ec="k", lw=.4)); a2.add_patch(Rectangle((k - .3, cb), .6, 1.2, fc="#F4C6A5", ec="k", lw=.4))
    a2.plot([k - .38, k + .38], [(vb + cb) / 2] * 2, c="k", ls="--", lw=.6)
a2.set_xlim(-.6, 2.0); a2.set_ylim(-6.6, -2.6); a2.set_xticks([0, 1]); a2.set_xticklabels(["X", "Q"]); a2.tick_params(axis="x", which="both", top=False)
a2.set_ylabel(r"E − E$_{vac}$ (eV)", fontsize=8.1, labelpad=1); a2.tick_params(labelsize=7.5, length=2.5); a2.minorticks_on(); a2.tick_params(axis="x", which="minor", bottom=False, top=False)
a2.annotate("", xy=(1.45, -m0.IP_Q), xytext=(1.45, -m0.IP_X), arrowprops=dict(arrowstyle="<->", lw=.6)); a2.text(1.75, (-m0.IP_Q - m0.IP_X) / 2, r"$\Delta E_{vac}$", fontsize=7.5, rotation=90, va="center", ha="center")
W.text(31, 41.2, "face-resolved IP / EA", ha="center", fontsize=8.0, color=INK, fontweight="bold")
# ---------- (a3) four contacts
ROW = [("X", EL["Se"]), ("A", EL["Al"]), ("M", EL["Sn"]), ("Q", EL["As"])]
a3 = fig.add_axes([0.435, 0.605, 0.19, 0.27]); a3.axis("off"); a3.set_aspect("equal"); a3.set_xlim(0, 4.4); a3.set_ylim(-.65, 2.6)
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
W.text(54, 41.2, "4 contact faces per pair", ha="center", fontsize=8.0, color=INK, fontweight="bold")
# ---------- (a4) database
W.add_patch(Rectangle((66.3, 52.5), 6.6, 7.5, fc="#CFDDF0", ec="#2F5DA8", lw=.8))
for yy in (52.5, 55, 57.5): W.add_patch(Ellipse((69.6, yy), 6.6, 1.7, fc="#CFDDF0", ec="#2F5DA8", lw=.8))
W.add_patch(Ellipse((69.6, 60), 6.6, 1.7, fc="#A9C3E8", ec="#2F5DA8", lw=.8))
W.text(69.6, 50.6, "7 796 interfaces\n832 DFT", ha="center", va="top", fontsize=7.8, color=INK)
# ---------- (a5) DFT potential: Origin frame
a4 = fig.add_axes([0.805, 0.64, 0.165, 0.245]); z = np.linspace(0, 45, 600)
V = 4.4 + .6 * np.tanh((z - 22.5) / .6) - 7 * (np.exp(-((z - 17) / 1.4) ** 2) + np.exp(-((z - 28) / 1.4) ** 2)) - 3.5 * (np.exp(-((z - 19.5) / .9) ** 2) + np.exp(-((z - 25.5) / .9) ** 2))
V = V - 4.4 + .0
a4.plot(z, V, c="#2E9E62", lw=1.0); a4.set_xlim(0, 45); a4.set_ylim(-8.5, 2.5); a4.minorticks_on()
a4.set_xlabel("z (Å)", fontsize=8.1, labelpad=1); a4.set_ylabel(r"$\bar V$ (eV)", fontsize=8.1, labelpad=1); a4.tick_params(labelsize=7.5, length=2.5)
a4.plot([2, 41], [-.6, -.6], c="k", lw=.4, ls=":"); a4.annotate("", xy=(40, .6), xytext=(40, -.6), arrowprops=dict(arrowstyle="<->", lw=.6, shrinkA=0, shrinkB=0)); a4.text(29, 1.1, r"$\Delta\Phi_{bi}$", fontsize=7.5)
W.text(87.3, 41.2, "DFT: dipole & band edges", ha="center", fontsize=8.0, color=INK, fontweight="bold")
# curved arrows (over / under alternating)
curved((17.5, 68.0), (26.5, 68.0), -0.35); curved((36.8, 45.0), (43.5, 45.0), 0.45); curved((62.8, 64.0), (66.0, 61.5), -0.45); curved((73.5, 61.5), (79.5, 64.0), -0.45)
# ---------- (b) chips -> hub (elbow), hub -> outputs (elbow)
chips = [r"$\Delta E_m$ (face)", r"$E_{g,1}+E_{g,2}$", r"$m_v, m_c$ (DOS)", r"$s_1, s_2$ (dipole)", r"$\chi_1+\chi_2$ (contact)"]
ys = [27.5 - k * 5.0 for k in range(5)]; HX, HY, HR = 35, 17.5, 6.6
W.text(12.5, 30.7, "monolayer descriptors", ha="center", fontsize=8.5, fontweight="bold", color=INK)
for y, ch in zip(ys, chips):
    W.add_patch(FancyBboxPatch((4, y - 1.6), 16.5, 3.2, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec="#3A86C8", lw=.8, zorder=3))
    W.text(12.25, y, ch, ha="center", va="center", fontsize=8.2, color=INK, zorder=4)
    W.plot([20.5, 24.5], [y, y], c="#7A8494", lw=.9, zorder=2)
W.plot([24.5, 24.5], [ys[0], ys[-1]], c="#7A8494", lw=.9, zorder=2)           # thanh góp (bus)
elbow([(24.5, HY), (HX - HR - .3, HY)])
cols = ["#2F5DA8", "#3A86C8", "#2E9E62", "#C8D93A", "#E08E2B", "#C0392B", "#8E44AD", "#5B6FD0"]
for k in range(8): W.add_patch(Wedge((HX, HY), HR, k * 45, (k + 1) * 45, width=.9, fc=cols[k], ec="none", alpha=.9, zorder=3))
W.add_patch(Circle((HX, HY), HR - .9, fc="white", ec="none", zorder=3))
W.text(HX, HY + 1.0, "linear\nresponse", ha="center", va="center", fontsize=9.6, linespacing=1.0, fontweight="bold", color=INK, zorder=4)
W.text(HX, HY - 1.9, "S(contact) + ML", ha="center", va="center", fontsize=6.4, color=MUTE, zorder=4)
outs = [("Interface dipole", 27.5, "r² = 0.94 (CV)"), ("Hybridization regime", 17.5, "AUC = 0.87"), ("Alignment map", 7.5, "1 105 confident")]
W.plot([HX + HR, 46.0], [HY, HY], c="#7A8494", lw=.9, zorder=2); W.plot([46.0, 46.0], [outs[0][1], outs[2][1]], c="#7A8494", lw=.9, zorder=2)
T = pd.read_csv("train_oof.csv"); D = pd.read_csv("../ana2/all/pred_final.csv"); A = pd.read_csv("pred7796_hyb.csv"); A = A[A.conf]
for k, (name, y, stat) in enumerate(outs):
    elbow([(46.0, y), (49.8, y)])
    ax = fig.add_axes([0.50, (y - 4.2) / 75, 0.085, 8.4 / 75]); ax.tick_params(labelsize=5.6, length=1.8, pad=1); ax.minorticks_on()
    if k == 0:
        for cc, col in (("XX", "#2F5DA8"), ("QX", "#2E9E62"), ("QQ", "#C0392B")):
            d = T[T.contact == cc]; ax.scatter(d.oof_hyb, d.eVh_B, s=.8, c=col, lw=0)
        ax.plot([-.8, .8], [-.8, .8], c="k", lw=.5); ax.set_xlim(-.8, .8); ax.set_ylim(-.8, .8)
    elif k == 1:
        ax.hist(D.w_min[D.contact == "XX"], bins=12, color="#2F5DA8", alpha=.6); ax.hist(D.w_min[D.contact == "QQ"], bins=12, color="#C0392B", alpha=.6)
    else:
        for t in ("I", "II", "III"):
            d = A[A.type_hyb == t]; ax.scatter(d.dE.abs(), d.Eg1 + d.Eg2, s=.5, c=TY[t], lw=0)
        ax.set_xlim(0, 2.5)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    W.text(59.3, y + 1.4, name, fontsize=8.0, fontweight="bold", color=INK, va="center"); W.text(59.3, y - 1.4, stat, fontsize=7.5, color=MUTE, va="center")
    elbow([(78.6, y), (79.8, y)], head=False)
W.plot([79.8, 79.8], [outs[0][1], outs[2][1]], c="#7A8494", lw=.9, zorder=2); elbow([(79.8, HY), (81.2, HY)])
# ---------- validation card (table)
x0, y0, w, h = 81.3, 5.0, 17.3, 25.5
W.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc="white", ec="#8E44AD", lw=.9, zorder=3))
W.add_patch(FancyBboxPatch((x0, y0 + h - 3.4), w, 3.4, boxstyle="round,pad=0,rounding_size=1.0", fc="#8E44AD", ec="#8E44AD", lw=.9, zorder=4))
W.add_patch(Rectangle((x0, y0 + h - 3.4), w, 1.2, fc="#8E44AD", ec="none", zorder=4))
W.text(x0 + w / 2, y0 + h - 1.7, "VALIDATION", ha="center", va="center", color="white", fontsize=8.2, fontweight="bold", zorder=5)
W.text(x0 + 1.0, y0 + h - 5.4, "test", fontsize=7.5, color=MUTE, zorder=5); W.text(x0 + w - 1.0, y0 + h - 5.4, "metric", fontsize=7.5, color=MUTE, ha="right", zorder=5)
rows = [("5-fold CV (pair)", "0.94"), ("pair hold-out", "0.93"), ("material hold-out", "0.91"), ("shuffled labels", "−0.20"), ("blind (59), MAE", "0.048"), ("blind (59), AUC", "0.88")]
for k, (a, b) in enumerate(rows):
    yy = y0 + h - 8.2 - k * 2.9
    if k % 2 == 0: W.add_patch(Rectangle((x0 + .3, yy - 1.3), w - .6, 2.6, fc="#F4EEF9", ec="none", zorder=4))
    W.text(x0 + .8, yy, a, fontsize=6.9, va="center", color=INK, zorder=5); W.text(x0 + w - .8, yy, b, fontsize=6.9, va="center", ha="right", color="#8E44AD" if k != 3 else MUTE, fontweight="bold", zorder=5)
_boldticks(fig); fig.savefig("fig0_workflow.pdf"); fig.savefig("fig/fig0_workflow.png", dpi=220)
