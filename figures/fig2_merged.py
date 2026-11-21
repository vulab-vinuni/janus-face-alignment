"""Hero figure: one Janus pair, four contacts -> three alignment types (SeAlSnAs/TeGaGeP)."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import to_rgb
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.linewidth":1,"xtick.direction":"in","ytick.direction":"in","ytick.right":True,"mathtext.fontset":"cm"})
tint = lambda c, f: tuple(np.array(to_rgb(c)) + (1 - np.array(to_rgb(c))) * f)
ROW = {"X": "#E3C02A", "A": "#6F9FD8", "M": "#4F4F6F", "Q": "#C04A7A"}
CC = {"XX": "#1F4E9C", "XQ": "#14946A", "QX": "#14946A", "QQ": "#C0392B"}
TT = {"I": "#7E3BA8", "II": "#F08C1E", "III": "#1FB0CC"}; TN = {"I": "straddling", "II": "staggered", "III": "broken gap"}
L1C, L2C = "#2B5BA8", "#D9661F"
T = pd.read_csv("train_oof.csv"); T["pair"] = ["|".join(sorted([a, b])) for a, b in zip(T.L1, T.L2)]
P = T[T.pair == "SeAlSnAs|TeGaGeP"].copy(); P["lab"] = P.f1 + P.f2; P = P.set_index("lab").loc[["XX", "XQ", "QX", "QQ"]]
fig = plt.figure(figsize=(7.3, 7.7))
gs = fig.add_gridspec(2, 4, height_ratios=[.9, 2.5], hspace=.08, wspace=.12, left=.08, right=.99, top=.965, bottom=.455)
for i, (lb, r) in enumerate(P.iterrows()):
    # --- sơ đồ hai lớp: lớp 1 (SeAlSnAs) ở dưới, lớp 2 (TeInSiAs) ở trên; mặt tiếp xúc ở giữa
    a = fig.add_subplot(gs[0, i]); a.axis("off"); a.set_xlim(0, 1); a.set_ylim(-.05, 2.55)
    def stack(y0, contact_face, flip):
        seq = ["X", "A", "M", "Q"]                       # từ dưới lên khi mặt X ở dưới
        if contact_face == "X": seq = seq[::-1] if not flip else seq
        else: seq = seq if not flip else seq[::-1]
        return [(y0 + k * .27, e) for k, e in enumerate(seq)]
    # lớp dưới: mặt tiếp xúc ở trên cùng; lớp trên: mặt tiếp xúc ở dưới cùng
    low = ["X", "A", "M", "Q"] if r.f1 == "Q" else ["Q", "M", "A", "X"]
    up = ["X", "A", "M", "Q"] if r.f2 == "X" else ["Q", "M", "A", "X"]
    for k, e in enumerate(low): a.add_patch(Rectangle((.18, .05 + k * .26), .64, .21, fc=ROW[e], ec="none"))
    for k, e in enumerate(up): a.add_patch(Rectangle((.18, 1.35 + k * .26), .64, .21, fc=ROW[e], ec="none"))
    a.add_patch(Rectangle((.13, 1.0), .74, .4, fc="none", ec="0.35", lw=.8, ls=":"))
    a.add_patch(Rectangle((.06, .02), .05, 1.02, fc=L1C, ec="none")); a.add_patch(Rectangle((.06, 1.32), .05, 1.04, fc=L2C, ec="none"))
    a.text(.5, 2.52, lb, ha="center", va="bottom", fontsize=12, fontweight="bold", color=CC[lb])
    # --- giản đồ cạnh vùng theo lớp
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
           loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(.53, .345), handlelength=2.0, handleheight=1.2, columnspacing=1.6)


# ===== bottom row: data set, VBO change between faces, layer-gap renormalization
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


from glr import r2
CC = {"XX": "#1F4E9C", "QX": "#14946A", "QQ": "#C0392B"}
def nomx(ax): ax.tick_params(axis="x", which="minor", bottom=False, top=False)
FS = 9
gb = fig.add_gridspec(1, 3, left=.08, right=.99, top=.255, bottom=.05, wspace=.45, width_ratios=[1, 1.4, 1])
figS = plt.figure(figsize=(5.2, 2.4)); gSb = figS.add_gridspec(1, 2, wspace=.45)
AX = [figS.add_subplot(gSb[0, 0]), fig.add_axes([.36, .05, .40, .205]), figS.add_subplot(gSb[0, 1])]
for a_ in AX: a_.tick_params(labelsize=FS, direction="in", top=True, right=True); a_.minorticks_on()
# (c) bộ dữ liệu
ax = AX[0]; cnt = T.groupby(["contact", "type_dft"]).size().unstack(fill_value=0).loc[["XX", "QX", "QQ"]]; bot = np.zeros(3)
for t in ("I", "II", "III"): ax.bar(range(3), cnt[t], .6, bottom=bot, color=TT[t], edgecolor="white", lw=.8, label=f"type {t}"); bot += cnt[t].values
for i, v in enumerate(bot): ax.text(i, v + 5, int(v), ha="center", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(["XX", "XQ", "QQ"]); nomx(ax); ax.set(ylim=(0, 400)); ax.set_ylabel("DFT heterostructures", fontsize=10, fontweight="bold")
ax.legend(loc="upper center", ncol=3, fontsize=8, handlelength=.8, handleheight=.8, handletextpad=.3, columnspacing=.6, borderaxespad=.2)
T["pair"] = ["|".join(sorted([a, b])) for a, b in zip(T.L1, T.L2)]
# (e) |ΔVBO|
m_ = T[T.pair.map(T.pair.value_counts()) >= 2]; x = []
T["vbo_s"] = [(r.VBM1 - r.VBM2) if r.L1 == sorted([r.L1, r.L2])[0] else (r.VBM2 - r.VBM1) for r in T.itertuples()]
for p, g in T[T.pair.isin(m_.pair)].groupby("pair"):
    v = g.vbo_s.values; x += [abs(v[i] - v[j]) for i in range(len(v)) for j in range(i + 1, len(v))]
sw = m_.groupby("pair").type_dft.nunique() > 1
ax = AX[1]; ax.hist(x, bins=np.linspace(0, 2, 21), color="#3A6FB8", edgecolor="white", lw=.7); ax.axvline(np.median(x), c="#C0392B", ls="--", lw=1)
ax.set_xlabel(r"|$\Delta$VBO| between faces (eV)", fontsize=10, fontweight="bold"); ax.set_ylabel("face pairs", fontsize=10, fontweight="bold")
ax.text(.96, .95, f"{sw.sum()}/{len(sw)} pairs switch type\nmedian {np.median(x):.2f} eV", transform=ax.transAxes, ha="right", va="top", fontsize=8, bbox=dict(fc="white", ec="none", pad=1.5))
# (f) co khe theo lớp
ax = AX[2]; T["dg1"] = T.CBM1 - T.VBM1 - T.Eg1; T["dg2"] = T.CBM2 - T.VBM2 - T.Eg2
R = pd.DataFrame([(r.contact, r.f1, r.dg1) for r in T.itertuples()] + [(r.contact, r.f2, r.dg2) for r in T.itertuples()], columns=["contact", "face", "dg"])
grp = [("XX", "X"), ("QX", "X"), ("QX", "Q"), ("QQ", "Q")]
bp = ax.boxplot([R[(R.contact == c) & (R.face == f)].dg for c, f in grp], widths=.55, patch_artist=True, showfliers=False)
for k_, (c, f) in enumerate(grp):
    col = CC[c]; bp["boxes"][k_].set(facecolor=tint(col, .55), edgecolor=col, lw=1.0)
    for w_ in bp["whiskers"][2 * k_:2 * k_ + 2] + bp["caps"][2 * k_:2 * k_ + 2]: w_.set(color=col, lw=1.0)
    bp["medians"][k_].set(color=col, lw=1.6)
ax.axhline(0, c="k", lw=.5, ls=":"); ax.set_xticks(range(1, 5)); ax.set_xticklabels(["XX", "XQ:X", "XQ:Q", "QQ"], fontsize=8.5); nomx(ax)
ax.set_ylabel(r"$E_g^{layer}-E_g^{mono}$ (eV)", fontsize=10, fontweight="bold")

fig.text(.005, .975, "a", fontweight="bold", fontsize=12)
AX[1].text(-.19, 1.06, "b", transform=AX[1].transAxes, fontweight="bold", fontsize=12)
for a, l_ in zip([AX[0], AX[2]], "ab"): a.text(-.33, 1.06, l_, transform=a.transAxes, fontweight="bold", fontsize=12)
_boldticks(fig); fig.savefig("fig2_merged.pdf", bbox_inches="tight"); fig.savefig("fig/fig2_merged.png", bbox_inches="tight", dpi=200)
_boldticks(figS); figS.savefig("figS_dataset_gap.pdf", bbox_inches="tight")
