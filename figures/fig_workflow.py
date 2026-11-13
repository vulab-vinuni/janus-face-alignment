import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, Circle, Polygon
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
INK, MUTE = "#1F2A44", "#6B7280"
C = ["#2F5DA8", "#3A86C8", "#2E9E62", "#C0392B", "#8E44AD", "#E08E2B"]
fig, ax = plt.subplots(figsize=(7.2, 2.05)); ax.set_xlim(0, 120); ax.set_ylim(0, 34); ax.axis("off")
X = [10, 30, 50, 70, 90, 110]
TAG = ["01  INPUT", "02  ENUMERATE", "03  COMPUTE", "04  LEARN", "05  VALIDATE", "06  PREDICT"]
TIT = ["108", "7 796", "832", "r² 0.94", "59", "1 105"]
SUB = ["Janus monolayers", "lattice-matched faces", "DFT bilayers", "gLR + ML residual", "blind interfaces", "confident alignments"]
for i, x in enumerate(X):
    ax.add_patch(FancyBboxPatch((x - 8.6, 3), 17.2, 28, boxstyle="round,pad=0,rounding_size=1.6", fc="white", ec="#D0D5DD", lw=.8))
    ax.add_patch(Rectangle((x - 8.6, 28.2), 17.2, 2.8, fc=C[i], ec="none"))
    ax.text(x, 29.6, TAG[i], ha="center", va="center", color="white", fontsize=6.2, family="DejaVu Sans Mono", fontweight="bold")
    ax.text(x, 9.4, TIT[i], ha="center", va="center", color=C[i], fontsize=9.5, fontweight="bold", family="DejaVu Sans Mono")
    ax.text(x, 5.4, SUB[i], ha="center", va="center", color=MUTE, fontsize=5.7)
    if i < 5: ax.annotate("", xy=(X[i + 1] - 9.2, 18.5), xytext=(x + 9.2, 18.5), arrowprops=dict(arrowstyle="-|>", lw=1.0, color=INK, mutation_scale=8))
cy = 18.5
# 01 database cylinder
x = X[0]; w, h = 9, 9
ax.add_patch(Rectangle((x - w / 2, cy - h / 2), w, h, fc="#DCE6F4", ec=C[0], lw=.9))
for yy in (cy - h / 2, cy - h / 2 + 3, cy - h / 2 + 6):
    ax.add_patch(Ellipse((x, yy), w, 2.2, fc="#DCE6F4", ec=C[0], lw=.9))
ax.add_patch(Ellipse((x, cy + h / 2), w, 2.2, fc="#BFD2EE", ec=C[0], lw=.9))
# 02 grid of interfaces
x = X[1]; rng = np.random.default_rng(3)
for a in range(5):
    for b in range(5):
        on = rng.random() < .55
        ax.add_patch(Rectangle((x - 5 + 2.05 * a, cy - 5 + 2.05 * b), 1.75, 1.75, fc=C[1] if on else "#E5ECF5", ec="none"))
# 03 server rack
x = X[2]
for k in range(4):
    y0 = cy - 5 + 2.6 * k
    ax.add_patch(FancyBboxPatch((x - 5, y0), 10, 2.0, boxstyle="round,pad=0,rounding_size=.4", fc="#DDF0E2", ec=C[2], lw=.8))
    ax.add_patch(Circle((x + 3.6, y0 + 1.0), .38, fc=C[2], ec="none")); ax.plot([x - 3.8, x + 1.5], [y0 + 1.0] * 2, c=C[2], lw=.7)
# 04 neural network
x = X[3]; L = [(-4.5, [-3, 0, 3]), (0, [-4, -1.3, 1.3, 4]), (4.5, [0])]
for (xa, ya), (xb, yb) in zip(L[:-1], L[1:]):
    for p in ya:
        for q in yb: ax.plot([x + xa, x + xb], [cy + p, cy + q], c="#E3B3AD", lw=.5, zorder=1)
for xa, ys in L:
    for p in ys: ax.add_patch(Circle((x + xa, cy + p), .95, fc="white", ec=C[3], lw=.9, zorder=2))
# 05 shield with check
x = X[4]; sh = np.array([[0, 5.5], [4.4, 4.0], [4.0, -1.5], [0, -5.5], [-4.0, -1.5], [-4.4, 4.0]])
ax.add_patch(Polygon(sh + [x, cy], closed=True, fc="#EADDF2", ec=C[4], lw=.9))
ax.plot([x - 2.0, x - .4, x + 2.4], [cy + .2, cy - 1.6, cy + 2.2], c=C[4], lw=1.6, solid_capstyle="round")
# 06 mini scatter map
x = X[5]; ax.add_patch(Rectangle((x - 5, cy - 5), 10, 10, fc="#FFF4E6", ec=C[5], lw=.8))
pts = rng.random((40, 2)); col = np.where(pts[:, 0] + .3 * pts[:, 1] < .55, "#4C72B0", np.where(pts[:, 0] < .85, "#DD8452", "#8172B3"))
ax.scatter(x - 4.4 + 8.8 * pts[:, 0], cy - 4.4 + 8.8 * pts[:, 1], s=2.2, c=col, lw=0, zorder=3)
fig.savefig("fig0_workflow.pdf", bbox_inches="tight"); fig.savefig("fig/fig0_workflow.png", bbox_inches="tight", dpi=220)
