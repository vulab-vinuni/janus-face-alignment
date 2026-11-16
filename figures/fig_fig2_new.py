import numpy as np, pandas as pd, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from glr import r2
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.0, "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True, "savefig.dpi": 300, "legend.frameon": False, "mathtext.default": "regular"})
plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 10, "axes.labelweight": "bold", "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5, "mathtext.fontset": "cm", "mathtext.default": "it"})
def _boldticks(fig):
    fig.canvas.draw()
    for _a in fig.axes:
        for _t in _a.get_xticklabels() + _a.get_yticklabels(): _t.set_fontweight("bold")

C = {"XX": "#2F5DA8", "QX": "#2E9E62", "QQ": "#C0392B"}; LB = {"XX": "XX", "QX": "XQ", "QQ": "QQ"}
TY = {"I": "#4C72B0", "II": "#DD8452", "III": "#8172B3"}
LAY = {"X": "#E08E2B", "Q": "#C04A7A"}
T = pd.read_csv("train_oof.csv"); D = pd.read_csv("../ana2/all/pred_final.csv"); J = json.load(open("../ana2/ml_sbo_result.json"))
def nomx(ax): ax.tick_params(axis="x", which="minor", bottom=False, top=False)
def lab(ax, s, dx=-.26): ax.text(dx, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=9.5)
# ================= Figure 2 (file fig3m.pdf): face matters, new palette =================
from matplotlib.colors import to_rgb
def tint(c, f): r = np.array(to_rgb(c)); return tuple(r + (1 - r) * f)
CC = {"XX": "#1F4E9C", "QX": "#14946A", "QQ": "#C0392B"}                      # contact type (same as the screening figure)
TT = {"I": "#7E3BA8", "II": "#F08C1E", "III": "#1FB0CC"}                      # alignment type
L1C, L2C = "#2B5BA8", "#D9661F"                                             # layer 1 / layer 2
FS = 7.2
fig = plt.figure(figsize=(6.4, 5.0)); gs = fig.add_gridspec(2, 2, wspace=.38, hspace=.42)
axs = [None] + [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
axs.insert(3, None)
for a_ in axs:
    if a_ is not None: a_.tick_params(labelsize=FS)
# (b) face-resolved levels of SeAlSnAs
ax = axs[1]; M = pd.read_csv("../res4/mono_strain.csv"); m0 = M[(M.name == "SeAlSnAs") & (M.strain == 0)].iloc[0]
VBc, CBc = "#2B5BA8", "#D9661F"
for k, f in enumerate(("X", "Q")):
    vb, cb = -m0[f"IP_{f}"], -m0[f"EA_{f}"]
    ax.add_patch(Rectangle((k - .3, vb - 1.2), .6, 1.2, fc=tint(VBc, .25), ec=VBc, lw=.9))
    ax.add_patch(Rectangle((k - .3, cb), .6, 1.2, fc=tint(CBc, .25), ec=CBc, lw=.9))
    ax.plot([k - .38, k + .38], [(vb + cb) / 2] * 2, c="k", ls="--", lw=.8)
ax.annotate("", xy=(1.5, -m0.IP_Q), xytext=(1.5, -m0.IP_X), arrowprops=dict(arrowstyle="<->", lw=.8, shrinkA=0, shrinkB=0))
ax.text(1.64, (-m0.IP_Q - m0.IP_X) / 2, r"$\Delta E_{vac}$", rotation=90, va="center", ha="center", fontsize=FS)
ax.text(.5, -2.75, "CB", color=CBc, fontsize=FS, ha="center", fontweight="bold"); ax.text(.5, -6.55, "VB", color=VBc, fontsize=FS, ha="center", fontweight="bold")
ax.set(xlim=(-.6, 1.85), ylim=(-6.7, -2.5)); ax.set_ylabel(r"E − E$_{vac}$ (eV)", fontsize=FS + .3); ax.set_xticks([0, 1]); ax.set_xticklabels(["X face", "Q face"]); nomx(ax)
ax.set_title("SeAlSnAs", fontsize=FS, pad=3)
# (c) data set
ax = axs[2]; cnt = T.groupby(["contact", "type_dft"]).size().unstack(fill_value=0).loc[["XX", "QX", "QQ"]]; bot = np.zeros(3)
for t in ("I", "II", "III"): ax.bar(range(3), cnt[t], .6, bottom=bot, color=TT[t], edgecolor="white", lw=.8, label=f"type {t}"); bot += cnt[t].values
for i, v in enumerate(bot): ax.text(i, v + 5, int(v), ha="center", fontsize=FS - .4)
ax.set_xticks(range(3)); ax.set_xticklabels(["XX", "XQ", "QQ"]); nomx(ax); ax.set(ylim=(0, 330)); ax.set_ylabel("DFT heterostructures", fontsize=FS + .3)
ax.legend(loc="upper right", fontsize=FS - .6, handlelength=1, handleheight=.8)
T["pair"] = ["|".join(sorted([a, b])) for a, b in zip(T.L1, T.L2)]
# (e) |ΔVBO|
m_ = T[T.pair.map(T.pair.value_counts()) >= 2]; x = []
T["vbo_s"] = [(r.VBM1 - r.VBM2) if r.L1 == sorted([r.L1, r.L2])[0] else (r.VBM2 - r.VBM1) for r in T.itertuples()]
for p, g in T[T.pair.isin(m_.pair)].groupby("pair"):
    v = g.vbo_s.values; x += [abs(v[i] - v[j]) for i in range(len(v)) for j in range(i + 1, len(v))]
sw = m_.groupby("pair").type_dft.nunique() > 1
ax = axs[4]; ax.hist(x, bins=np.linspace(0, 2, 21), color="#3A6FB8", edgecolor="white", lw=.7); ax.axvline(np.median(x), c="#C0392B", ls="--", lw=1)
ax.set_xlabel(r"|$\Delta$VBO| between faces (eV)", fontsize=FS + .3); ax.set_ylabel("face pairs", fontsize=FS + .3)
ax.text(.96, .95, f"{sw.sum()}/{len(sw)} pairs switch type\nmedian {np.median(x):.2f} eV", transform=ax.transAxes, ha="right", va="top", fontsize=FS - .4, bbox=dict(fc="white", ec="none", pad=1.5))
# (f) layer-gap renormalization
ax = axs[5]; T["dg1"] = T.CBM1 - T.VBM1 - T.Eg1; T["dg2"] = T.CBM2 - T.VBM2 - T.Eg2
R = pd.DataFrame([(r.contact, r.f1, r.dg1) for r in T.itertuples()] + [(r.contact, r.f2, r.dg2) for r in T.itertuples()], columns=["contact", "face", "dg"])
grp = [("XX", "X"), ("QX", "X"), ("QX", "Q"), ("QQ", "Q")]
bp = ax.boxplot([R[(R.contact == c) & (R.face == f)].dg for c, f in grp], widths=.55, patch_artist=True, showfliers=False)
for k_, (c, f) in enumerate(grp):
    col = CC[c]; bp["boxes"][k_].set(facecolor=tint(col, .55), edgecolor=col, lw=1.0)
    for w_ in bp["whiskers"][2 * k_:2 * k_ + 2] + bp["caps"][2 * k_:2 * k_ + 2]: w_.set(color=col, lw=1.0)
    bp["medians"][k_].set(color=col, lw=1.6)
ax.axhline(0, c="k", lw=.5, ls=":"); ax.set_xticks(range(1, 5)); ax.set_xticklabels(["XX", "XQ:X", "XQ:Q", "QQ"]); nomx(ax)
ax.set_ylabel(r"$E_g^{layer}-E_g^{mono}$ (eV)", fontsize=FS + .3)
for k, a in enumerate([x for x in axs if x is not None]): lab(a, "abcd"[k], -.25)
_boldticks(fig); fig.savefig("fig3m.pdf", bbox_inches="tight"); fig.savefig("fig/fig3m.png", bbox_inches="tight", dpi=200)
import sys; sys.exit(0)
