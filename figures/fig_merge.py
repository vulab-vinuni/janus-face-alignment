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
fig, axs = plt.subplots(2, 3, figsize=(7.3, 4.9), gridspec_kw=dict(wspace=.42, hspace=.52)); axs = axs.ravel()
ax = axs[0]
for c in ("XX", "QX", "QQ"): ax.hist(D[D.contact == c].w_min, bins=np.linspace(0, 1, 21), color=C[c], alpha=.55, label=LB[c], edgecolor="k", lw=.3)
ax.set(xlabel=r"$w_{min}$", ylabel="count", ylim=(0, 52)); ax.legend(loc="upper right", fontsize=8.0)
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
ax = axs[4]; V = [("5-fold", .757, .937), ("pairs", .770, .930), ("materials", .747, .909), ("shuffled", np.nan, -.196)]; xx = np.arange(4)
ax.bar(xx - .18, [v[1] for v in V], .34, color="#BFBFBF", edgecolor="k", lw=.5, label="S(contact)"); ax.bar(xx + .18, [v[2] for v in V], .34, color="#5B8BD0", edgecolor="k", lw=.5, label="S(contact) + ML")
ax.axhline(0, c="k", lw=.5); ax.set_xticks(xx); ax.set_xticklabels([v[0] for v in V], fontsize=8, rotation=25); nomx(ax); ax.set(ylim=(-.3, 1.35), ylabel=r"$r^2$ on unseen data", xlabel="validation split")
ax.legend(loc="upper right", fontsize=8.0)
ax = axs[5]; ks = ["anderson", "slr", "glr", "SVM_lin", "SVM_rbf"]; nm = ["Anderson", "global S", "S(contact)", "SVM (lin.)", "SVM (RBF)"]
ax.bar(range(5), [J[k]["NA"] for k in ks], .62, yerr=[J[k].get("NA_sd", 0) for k in ks], capsize=2, color=["#BFBFBF"] * 3 + ["#5B8BD0"] * 2, edgecolor="k", lw=.5)
for j, k in enumerate(ks): ax.text(j, J[k]["NA"] + J[k].get("NA_sd", 0) + .02, f"{J[k]['NA']:.2f}", ha="center", fontsize=8)
ax.axhline(.5, c="k", ls=":", lw=.7); ax.set_xticks(range(5)); ax.set_xticklabels(nm, rotation=30, fontsize=8); nomx(ax); ax.set(ylim=(0, 1.0), ylabel="Balanced accuracy")
for k, a in enumerate(axs): lab(a, "abcdef"[k], -.3)
_boldticks(fig); fig.savefig("fig45.pdf", bbox_inches="tight"); fig.savefig("fig/fig45.png", bbox_inches="tight")
