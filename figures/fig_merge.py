import numpy as np, pandas as pd, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from glr import r2
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.0, "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True, "savefig.dpi": 300, "legend.frameon": False, "mathtext.default": "regular"})
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

C = {"XX": "#2F5DA8", "QX": "#2E9E62", "QQ": "#C0392B"}; LB = {"XX": "XX", "QX": "XQ", "QQ": "QQ"}
TY = {"I": "#4C72B0", "II": "#DD8452", "III": "#8172B3"}
LAY = {"X": "#E08E2B", "Q": "#C04A7A"}
T = pd.read_csv("train_oof.csv"); D = pd.read_csv("../ana2/all/pred_final.csv"); J = json.load(open("../ana2/ml_sbo_result.json"))
def nomx(ax): ax.tick_params(axis="x", which="minor", bottom=False, top=False)
def lab(ax, s, dx=-.26): ax.text(dx, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=9.5)
# ================= Figure 3: face matters (2 x 3), merges the old data-set figure =================
fig = plt.figure(figsize=(7.3, 5.0)); gs = fig.add_gridspec(2, 3, wspace=.42, hspace=.45)
axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
# (a) four contact configurations (schematic)
ax = axs[0]; ax.axis("off"); ROWC = [("X", "#E08E2B"), ("A", "#6F9FD8"), ("M", "#4F4F6F"), ("Q", "#C04A7A")]
for i, nm in enumerate(("XX", "XQ", "QX", "QQ")):
    for (y0, up) in ((0, nm[0]), (1.3, "Q" if nm[1] == "X" else "X")):
        seq = ROWC[::-1] if up == "X" else ROWC
        for k, (_, cc) in enumerate(seq): ax.add_patch(Rectangle((i * 1.1 + .1, y0 + k * .27), .85, .22, fc=cc, ec="none"))
    ax.add_patch(Rectangle((i * 1.1 + .04, 1.03), .97, .3, fc="none", ec="#C0392B", lw=.8, ls=":"))
    ax.text(i * 1.1 + .52, -.4, nm, ha="center", fontsize=7.5, fontweight="bold", color={"XX": C["XX"], "XQ": C["QX"], "QX": C["QX"], "QQ": C["QQ"]}[nm])
for k, (nmx, cc) in enumerate(ROWC): ax.add_patch(Rectangle((k * 1.1 + .1, 2.75), .18, .16, fc=cc, ec="none")); ax.text(k * 1.1 + .33, 2.83, nmx, fontsize=7, va="center")
ax.set_xlim(0, 4.45); ax.set_ylim(-.65, 3.05)
# (b) face-resolved levels (SeAlSnAs)
ax = axs[1]; M = pd.read_csv("../res4/mono_strain.csv"); m0 = M[(M.name == "SeAlSnAs") & (M.strain == 0)].iloc[0]
for k, f in enumerate(("X", "Q")):
    vb, cb = -m0[f"IP_{f}"], -m0[f"EA_{f}"]
    ax.add_patch(Rectangle((k - .3, vb - 1.2), .6, 1.2, fc="#9DB7DA", ec="k", lw=.5)); ax.add_patch(Rectangle((k - .3, cb), .6, 1.2, fc="#F4C6A5", ec="k", lw=.5))
    ax.plot([k - .38, k + .38], [(vb + cb) / 2] * 2, c="k", ls="--", lw=.7)
ax.annotate("", xy=(1.5, -m0.IP_Q), xytext=(1.5, -m0.IP_X), arrowprops=dict(arrowstyle="<->", lw=.7, shrinkA=0, shrinkB=0))
ax.text(1.62, (-m0.IP_Q - m0.IP_X) / 2, r"$\Delta E_{vac}$", rotation=90, va="center", ha="center", fontsize=8)
ax.set(xlim=(-.6, 1.85), ylim=(-6.7, -2.5), ylabel=r"E − E$_{vac}$ (eV)"); ax.set_xticks([0, 1]); ax.set_xticklabels(["X face", "Q face"]); nomx(ax)
ax.text(.04, .04, "SeAlSnAs", transform=ax.transAxes, ha="left", va="bottom", fontsize=8, style="italic")
# (c) data set by contact and type
ax = axs[2]; cnt = T.groupby(["contact", "type_dft"]).size().unstack(fill_value=0).loc[["XX", "QX", "QQ"]]; bot = np.zeros(3)
for t in ("I", "II", "III"): ax.bar(range(3), cnt[t], .6, bottom=bot, color=TY[t], edgecolor="k", lw=.5, label=f"type {t}"); bot += cnt[t].values
for i, v in enumerate(bot): ax.text(i, v + 5, int(v), ha="center", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(["XX", "XQ", "QQ"]); nomx(ax); ax.set(ylabel="DFT heterostructures", ylim=(0, 330)); ax.legend(loc="upper right", fontsize=8.0, handlelength=1)
# (d) one pair, four faces
ax = axs[3]; T["pair"] = ["|".join(sorted([a, b])) for a, b in zip(T.L1, T.L2)]
P = T[T.pair == "SeAlSnAs|TeInSiAs"].copy(); P["lab"] = P.f1 + P.f2; P = P.set_index("lab").loc[["XX", "XQ", "QX", "QQ"]]
for i, (lb, r) in enumerate(P.iterrows()):
    for L, dx, colr in ((1, -.17, "#5B8BD0"), (2, .17, "#E3A857")):
        v, c_ = r[f"VBM{L}"], r[f"CBM{L}"]
        ax.add_patch(Rectangle((i + dx - .14, v - .6), .28, .6, fc=colr, ec="k", lw=.4)); ax.add_patch(Rectangle((i + dx - .14, c_), .28, .6, fc=colr, ec="k", lw=.4, alpha=.45))
    ax.text(i, -3.13, r.type_dft, ha="center", fontsize=8, color=TY[r.type_dft], fontweight="bold")
ax.set(xlim=(-.6, 3.6), ylim=(-6.0, -2.95), ylabel="layer band edges (eV)"); ax.set_xticks(range(4)); ax.set_xticklabels(["XX", "XQ", "QX", "QQ"]); nomx(ax)
ax.text(.03, .03, "SeAlSnAs / TeInSiAs", transform=ax.transAxes, fontsize=8, style="italic")
# (e) VBO change between faces
m_ = T[T.pair.map(T.pair.value_counts()) >= 2]; x = []
T["vbo_s"] = [(r.VBM1 - r.VBM2) if r.L1 == sorted([r.L1, r.L2])[0] else (r.VBM2 - r.VBM1) for r in T.itertuples()]
for p, g in T[T.pair.isin(m_.pair)].groupby("pair"):
    v = g.vbo_s.values; x += [abs(v[i] - v[j]) for i in range(len(v)) for j in range(i + 1, len(v))]
sw = m_.groupby("pair").type_dft.nunique() > 1
ax = axs[4]; ax.hist(x, bins=np.linspace(0, 2, 21), color="#E3A857", edgecolor="k", lw=.4); ax.axvline(np.median(x), c="k", ls="--", lw=.7)
ax.set(xlabel=r"|$\Delta$VBO| between faces (eV)", ylabel="face pairs")
ax.text(.96, .95, f"{sw.sum()}/{len(sw)} pairs\nswitch type\nmedian {np.median(x):.2f} eV", transform=ax.transAxes, ha="right", va="top", fontsize=8)
# (f) gap renormalization
ax = axs[5]; T["dg1"] = T.CBM1 - T.VBM1 - T.Eg1; T["dg2"] = T.CBM2 - T.VBM2 - T.Eg2
R = pd.DataFrame([(r.contact, r.f1, r.dg1) for r in T.itertuples()] + [(r.contact, r.f2, r.dg2) for r in T.itertuples()], columns=["contact", "face", "dg"])
grp = [("XX", "X"), ("QX", "X"), ("QX", "Q"), ("QQ", "Q")]
bp = ax.boxplot([R[(R.contact == c) & (R.face == f)].dg for c, f in grp], widths=.55, patch_artist=True, showfliers=False, medianprops=dict(color="k"))
for b_, (c, f) in zip(bp["boxes"], grp): b_.set(facecolor=C[c], alpha=.7, lw=.6)
ax.axhline(0, c="k", lw=.5, ls=":"); ax.set_xticks(range(1, 5)); ax.set_xticklabels(["XX", "XQ:X", "XQ:Q", "QQ"], fontsize=8); nomx(ax)
ax.set(ylabel=r"$E_g^{layer}-E_g^{mono}$ (eV)")
for k, a in enumerate(axs): lab(a, "abcdef"[k], -.08 if k == 0 else -.3)
_boldticks(fig); fig.savefig("fig3m.pdf", bbox_inches="tight"); fig.savefig("fig/fig3m.png", bbox_inches="tight")
# ================= Figure 4: hybridization, ML and validation (2 x 3) =================
figM = plt.figure(figsize=(7.3, 6.0)); gM = figM.add_gridspec(2, 2, wspace=.30, hspace=.42)
figS = plt.figure(figsize=(7.3, 5.0)); gS = figS.add_gridspec(2, 6, wspace=1.45, hspace=.6)
axs = [figM.add_subplot(gM[0, 0]), figS.add_subplot(gS[0, 0:2]), figM.add_subplot(gM[0, 1]), figS.add_subplot(gS[0, 2:4]), figS.add_subplot(gS[0, 4:6]),
       figS.add_subplot(gS[1, 1:3]), figM.add_subplot(gM[1, 0]), figM.add_subplot(gM[1, 1]), figS.add_subplot(gS[1, 3:5])]
fig = figM
ax = axs[0]
for c in ("XX", "QX", "QQ"): ax.hist(D[D.contact == c].w_min, bins=np.linspace(0, 1, 21), color=C[c], alpha=.55, label=LB[c], edgecolor="k", lw=.3)
ax.set(xlabel=r"$w_{min}$", ylabel="count", ylim=(0, 60)); ax.legend(loc="upper right", fontsize=8.0)
th = np.arange(0, .86, .05); ax = axs[1]
ax.plot(th, [np.abs(D.vbo_dft - D.vbo_and)[D.w_min >= t].mean() for t in th], "s-", c="#7F7F7F", ms=3, mfc="w", label="Anderson")
ax.plot(th, [np.abs(D.vbo_dft - D.vbo_glr)[D.w_min >= t].mean() for t in th], "o-", c="k", ms=3, label="S(contact) model")
ax.set(xlabel=r"$w_{min}$ threshold", ylabel="VBO MAE (eV)", ylim=(0, .4)); ax.legend(loc="lower left", fontsize=8.0)
ax = axs[2]; bins = [0, .1, .3, .5, .7, 1.01]; acc, cnt_, lb = [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    s = D[(D.p_local >= lo) & (D.p_local < hi)]; acc.append((s.type_glr == s.type_dft).mean()); cnt_.append(len(s)); lb.append(f"{lo:.1f}–{min(hi, 1):.1f}")
ax.bar(range(5), acc, .65, color="#9DB7DA", edgecolor="k", lw=.5)
for j, (a_, n_) in enumerate(zip(acc, cnt_)): ax.text(j, a_ + .02, n_, ha="center", fontsize=8)
ax.set_xticks(range(5)); ax.set_xticklabels(lb, rotation=35, fontsize=8); nomx(ax); ax.set(ylim=(0, 1.12), xlabel=r"predicted P($w_{min}\geq0.6$)", ylabel="type accuracy")
ax = axs[3]
for c in ("XX", "QX", "QQ"):
    d = T[T.contact == c]; ax.scatter(d.oof_hyb, d.eVh_B, s=4, c=C[c], alpha=.7, lw=0, label=LB[c])
lim = (-.85, .85); ax.plot(lim, lim, c="k", lw=.6); ax.set(xlim=lim, ylim=lim, xlabel=r"$eV_h$ S(contact) + ML (eV)", ylabel=r"$eV_h$ DFT (eV)")
ax.text(.04, .96, f"$r^2$ = {r2(T.eVh_B, T.oof_hyb):.2f}\nMAE = {np.abs(T.eVh_B - T.oof_hyb).mean():.3f} eV", transform=ax.transAxes, va="top", fontsize=8)
ax.legend(loc="lower right", fontsize=8.0, markerscale=2, handletextpad=.1)
ax = axs[4]; V = [("5-fold CV", .757, .937), ("pair hold-out", .770, .930), ("material\nhold-out", .747, .909)]; xx = np.arange(3)
for j, v in enumerate(V): ax.plot([j, j], [v[1], v[2]], c="0.55", lw=1.0, zorder=1)
ax.plot(xx, [v[1] for v in V], "o", ms=7, mfc="w", mec="#D9822B", mew=1.6, label="S(contact)", zorder=3)
ax.plot(xx, [v[2] for v in V], "D", ms=6.5, c="#7B3FA0", label="S(contact) + ML", zorder=3)
for j, v in enumerate(V):
    ax.text(j, v[1] - .022, f"{v[1]:.2f}", ha="center", va="top", fontsize=8, color="#B35F12"); ax.text(j, v[2] + .022, f"{v[2]:.2f}", ha="center", va="bottom", fontsize=8, color="#7B3FA0")
ax.set_xticks(xx); ax.set_xticklabels([v[0] for v in V], fontsize=8, rotation=25); nomx(ax); ax.set(xlim=(-.5, 2.5), ylim=(.66, 1.14), ylabel=r"$r^2$ on unseen data", xlabel="validation split")
ax.set_yticks([.7, .8, .9, 1.0])
ax.legend(loc="upper left", fontsize=8.0, handletextpad=.3, borderaxespad=.3)
ax = axs[5]; ks = ["anderson", "slr", "glr", "SVM_lin", "SVM_rbf"]; nm = ["Anderson", "global S", "S(contact)", "SVM (lin.)", "SVM (RBF)"]
ax.bar(range(5), [J[k]["NA"] for k in ks], .62, yerr=[J[k].get("NA_sd", 0) for k in ks], capsize=2, color=["#A9D3C4"] * 3 + ["#2A8C6E"] * 2, edgecolor="k", lw=.5)
for j, k in enumerate(ks): ax.text(j, J[k]["NA"] + J[k].get("NA_sd", 0) + .02, f"{J[k]['NA']:.2f}", ha="center", fontsize=8)
ax.axhline(.5, c="k", ls=":", lw=.7); ax.set_xticks(range(5)); ax.set_xticklabels(nm, rotation=30, fontsize=8); nomx(ax); ax.set(ylim=(0, 1.32), ylabel="Balanced accuracy"); ax.set_yticks(np.arange(0, 1.01, .2))
from matplotlib.patches import Patch; ax.legend([Patch(fc="#A9D3C4", ec="k", lw=.5), Patch(fc="#2A8C6E", ec="k", lw=.5)], ["physical model", "SVM classifier"], loc="upper left", fontsize=8.0, ncol=1, handlelength=1.2, columnspacing=.8, borderaxespad=.25)
# ---- row 3: confidence-resolved alignment map of the lattice-matched interfaces (former Fig. 6)
TTY = {"I": "#7E3BA8", "II": "#F08C1E", "III": "#1FB0CC"}            # same type colours as Fig. 2
A = pd.read_csv("pred7796_hyb.csv"); A = A[(A.Eg1 > .05) & (A.Eg2 > .05)]
ax = axs[6]; Al = A[A.p_local >= .3]; Ah = A[A.p_local < .3]
ax.scatter(Ah.dE.abs(), Ah.Eg1 + Ah.Eg2, s=1.5, c="#D0D0D0", lw=0, label="hybridized")
for t in ("I", "II", "III"):
    d = Al[Al.type_hyb == t]; ax.scatter(d.dE.abs(), d.Eg1 + d.Eg2, s=3, c=TTY[t], lw=0, label=f"type {t}")
ax.set(xlabel=r"$|\Delta E_m|$ (eV)", ylabel=r"$E_{g,1}+E_{g,2}$ (eV)", xlim=(0, 2.5), ylim=(0, 7.4)); ax.set_yticks(range(0, 6))
ax.legend(loc="upper left", fontsize=7.5, markerscale=3, handletextpad=0, ncol=2, columnspacing=.3, borderaxespad=.15, handlelength=1.2)
# confident subset: predicted localized and type margin > 0.1 eV (ML-corrected dipole)
v1, c1 = -A.Eg1 / 2, A.Eg1 / 2; sh = A.dE - A.eVh_hyb; v2, c2 = sh - A.Eg2 / 2, sh + A.Eg2 / 2
A["margin"] = np.minimum.reduce([np.abs(v1 - v2), np.abs(c1 - c2), np.abs(np.minimum(c1, c2) - np.maximum(v1, v2))])
A["conf"] = (A.p_local >= .3) & (A.margin > .1); Ac = A[A.conf]; print("confident", len(Ac))
ax = axs[7]; w = .36
for k, (G, col, nm) in enumerate([(T, "type_dft", "DFT"), (Ac, "type_hyb", "pred., confident")]):
    fr = G.groupby("contact")[col].value_counts(normalize=True).unstack().reindex(["XX", "QX", "QQ"]).fillna(0); n = G.contact.value_counts().reindex(["XX", "QX", "QQ"]).fillna(0).astype(int)
    x = np.arange(3) + (k - .5) * (w + .04); bot = np.zeros(3)
    for t in ("I", "II", "III"): ax.bar(x, fr[t], w, bottom=bot, color=TTY[t], edgecolor="k", lw=.5, hatch=None if k == 0 else "////", alpha=1 if k == 0 else .85); bot += fr[t].values
    if k == 1: ax.text(x[2], 1.015, f"n={n.iloc[2]}", ha="center", fontsize=6.5, va="bottom")
ax.set_xticks(range(3)); ax.set_xticklabels(["XX", "XQ", "QQ"]); nomx(ax); ax.set(ylim=(0, 1.25), ylabel="type fraction"); ax.set_yticks(np.arange(0, 1.01, .2))
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc="w", ec="k", lw=.5, label="DFT (599)"), Patch(fc="w", ec="k", lw=.5, hatch="////", label="confident pred.")], loc="upper center", ncol=2, fontsize=6.6, bbox_to_anchor=(.5, 1.01), handlelength=1.2, columnspacing=.6, borderaxespad=.1)
ax = axs[8]; A["pair"] = ["|".join(sorted([a, b])) for a, b in zip(A.L1, A.L2)]; nt = A.groupby("pair").type_hyb.nunique()
Ac = A[A.conf]; g = Ac.groupby("pair"); ntc = g.type_hyb.nunique()[g.size() >= 2]
vals = [(nt == k).sum() for k in (1, 2, 3)]; valc = [(ntc == k).sum() for k in (1, 2, 3)]; print("pairs conf>=2", len(ntc), valc)
ax.bar(np.arange(3) - .2, vals, .38, color="#9A9A9A", edgecolor="k", lw=.5, label="all predictions")
ax.bar(np.arange(3) + .2, valc, .38, color="#4D4D4D", edgecolor="k", lw=.5, label="confident only")
for j, (v, u) in enumerate(zip(vals, valc)): ax.text(j - .2, v + 15, v, ha="center", fontsize=7); ax.text(j + .2, u + 15, u, ha="center", fontsize=7)
ax.set_xticks(range(3)); ax.set_xticklabels(["1", "2", "3"]); nomx(ax); ax.set(xlabel="distinct types over faces", ylabel="material pairs", ylim=(0, max(vals) * 1.45))
ax.legend(loc="upper right", fontsize=7, handlelength=1.2, borderaxespad=.2)
for k, l_ in zip([0, 2, 6, 7], "abcd"): lab(axs[k], l_, -.2)
for k, l_ in zip([1, 3, 4, 5, 8], "abcde"): lab(axs[k], l_, -.3)
_boldticks(figM); figM.savefig("fig45.pdf", bbox_inches="tight"); figM.savefig("fig/fig45.png", bbox_inches="tight")
_boldticks(figS); figS.savefig("figS_ml_extra.pdf", bbox_inches="tight")
