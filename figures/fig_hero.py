"""Hero figure: one Janus pair, four contacts -> three alignment types (SeAlSnAs/TeGaGeP)."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import to_rgb
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.linewidth":1,"xtick.direction":"in","ytick.direction":"in","ytick.right":True,"mathtext.fontset":"cm"})
tint = lambda c, f: tuple(np.array(to_rgb(c)) + (1 - np.array(to_rgb(c))) * f)
ROW = {"X": "#E08E2B", "A": "#6F9FD8", "M": "#4F4F6F", "Q": "#C04A7A"}
CC = {"XX": "#1F4E9C", "XQ": "#14946A", "QX": "#14946A", "QQ": "#C0392B"}
TT = {"I": "#7E3BA8", "II": "#F08C1E", "III": "#1FB0CC"}; TN = {"I": "straddling", "II": "staggered", "III": "broken gap"}
L1C, L2C = "#2B5BA8", "#D9661F"
T = pd.read_csv("train_oof.csv"); T["pair"] = ["|".join(sorted([a, b])) for a, b in zip(T.L1, T.L2)]
P = T[T.pair == "SeAlSnAs|TeGaGeP"].copy(); P["lab"] = P.f1 + P.f2; P = P.set_index("lab").loc[["XX", "XQ", "QX", "QQ"]]
fig = plt.figure(figsize=(7.3, 4.9))
gs = fig.add_gridspec(2, 4, height_ratios=[.9, 2.5], hspace=.08, wspace=.12, left=.08, right=.99, top=.95, bottom=.16)
for i, (lb, r) in enumerate(P.iterrows()):
    # --- bilayer sketch: layer 1 (SeAlSnAs) at the bottom, layer 2 (TeInSiAs) on top; contact faces in the middle
    a = fig.add_subplot(gs[0, i]); a.axis("off"); a.set_xlim(0, 1); a.set_ylim(-.05, 2.55)
    def stack(y0, contact_face, flip):
        seq = ["X", "A", "M", "Q"]                       # bottom to top when the X face is at the bottom
        if contact_face == "X": seq = seq[::-1] if not flip else seq
        else: seq = seq if not flip else seq[::-1]
        return [(y0 + k * .27, e) for k, e in enumerate(seq)]
    # bottom layer: contact face on top; top layer: contact face at the bottom
    low = ["X", "A", "M", "Q"] if r.f1 == "Q" else ["Q", "M", "A", "X"]
    up = ["X", "A", "M", "Q"] if r.f2 == "X" else ["Q", "M", "A", "X"]
    for k, e in enumerate(low): a.add_patch(Rectangle((.18, .05 + k * .26), .64, .21, fc=ROW[e], ec="none"))
    for k, e in enumerate(up): a.add_patch(Rectangle((.18, 1.35 + k * .26), .64, .21, fc=ROW[e], ec="none"))
    a.add_patch(Rectangle((.13, 1.0), .74, .4, fc="none", ec="0.35", lw=.8, ls=":"))
    a.add_patch(Rectangle((.06, .02), .05, 1.02, fc=L1C, ec="none")); a.add_patch(Rectangle((.06, 1.32), .05, 1.04, fc=L2C, ec="none"))
    a.text(.5, 2.52, lb, ha="center", va="bottom", fontsize=12, fontweight="bold", color=CC[lb])
    # --- layer-resolved band-edge diagram
    ax = fig.add_subplot(gs[1, i])
    for L, x0, col in ((1, .2, L1C), (2, .58, L2C)):
        v, c = r[f"VBM{L}"], r[f"CBM{L}"]
        ax.add_patch(Rectangle((x0, v - .75), .26, .75, fc=col, ec=col, lw=.8))
        ax.add_patch(Rectangle((x0, c), .26, .75, fc=tint(col, .62), ec=col, lw=1.0))
    vb = max(r.VBM1, r.VBM2); cb = min(r.CBM1, r.CBM2)
    ax.axhspan(min(vb, cb), max(vb, cb), xmin=.06, xmax=.94, color=TT[r.type_dft], alpha=.18, lw=0)
    ax.plot([.06, .94], [vb, vb], c="k", lw=.6, ls="--"); ax.plot([.06, .94], [cb, cb], c="k", lw=.6, ls="--")
    ax.text(.5, -2.47, r.type_dft, ha="center", va="center", fontsize=15, family="DejaVu Serif", fontweight="bold", color=TT[r.type_dft])
    ax.text(.5, -2.72, TN[r.type_dft], ha="center", va="center", fontsize=9.5, color=TT[r.type_dft])
    ax.text(.5, -6.6, r"$E_\mathrm{g}^\mathrm{het} = %s%.2f~\mathrm{eV}$" % ("+" if r.Eg_bi >= 0 else "-", abs(r.Eg_bi)), ha="center", fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(-6.75, -2.25); ax.set_yticks([-6, -5.5, -5, -4.5, -4, -3.5]); ax.set_xticks([])
    if i == 0:
        ax.set_ylabel("Layer band edges (eV)", fontsize=11, fontweight="bold")
        for t in ax.get_yticklabels(): t.set_fontweight("bold"); t.set_fontsize(10)
    else: ax.set_yticklabels([])
from matplotlib.patches import Patch
fig.legend([Patch(fc=ROW[e]) for e in "XAMQ"] + [Patch(fc=L1C), Patch(fc=L2C), Patch(fc="white", ec="k", lw=.6), Patch(fc="0.55")],
           ["X (S, Se, Te)", "A (Al, Ga, In)", "M (Si–Pb)", "Q (P, As, Sb)", "SeAlSnAs", "TeGaGeP", "CB (light)", "VB (solid)"],
           loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(.53, -0.01), handlelength=2.0, handleheight=1.2, columnspacing=1.6)
fig.savefig("fig_hero.pdf", bbox_inches="tight"); fig.savefig("fig_hero.png", bbox_inches="tight", dpi=200)
