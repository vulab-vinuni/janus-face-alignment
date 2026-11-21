import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.model_selection import GroupKFold
from glr import r2, fit
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
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
T = pd.read_csv("train_oof.csv"); g = (T.L1 + "|" + T.L2).values
EN = dict(S=2.58, Se=2.55, Te=2.10, P=2.19, As=2.18, Sb=2.05)
el = lambda n, f: [e for e in ("S", "Se", "Te", "P", "As", "Sb") if (n.startswith(e) if f == "X" else n.endswith(e))][-1]
T["ENs"] = [EN[el(a, b)] + EN[el(c, d)] for a, b, c, d in zip(T.L1, T.f1, T.L2, T.f2)]
mu, sd = T.ENs.mean(), T.ENs.std(); z = (T.ENs - mu) / sd
mdl = lambda p, z, dE: (lambda a: a / (1 + a) * dE)(np.exp(p[0] + p[1] * z))
pr = np.zeros(len(T))
for tr, te in GroupKFold(5).split(T, groups=g):
    o = least_squares(lambda p: mdl(p, z.values[tr], T.dE.values[tr]) - T.eVh_B.values[tr], [-1, 0]); pr[te] = mdl(o.x, z.values[te], T.dE.values[te])
T["oof_en"] = pr; P = least_squares(lambda p: mdl(p, z.values, T.dE.values) - T.eVh_B.values, [-1, 0]).x
figM = plt.figure(figsize=(7.3, 5.6)); gM = figM.add_gridspec(2, 2, wspace=.30, hspace=.38)
figS = plt.figure(figsize=(7.3, 2.45)); gS = figS.add_gridspec(1, 3, wspace=.42)
A = np.array([figM.add_subplot(gM[0, 0]), figS.add_subplot(gS[0, 0]), figS.add_subplot(gS[0, 1]), figS.add_subplot(gS[0, 2]),
              figM.add_subplot(gM[0, 1]), figM.add_subplot(gM[1, 0])], dtype=object)
AXD = figM.add_subplot(gM[1, 1]); fig = figM
full = fit(T, "eVh_B", "glr", "contact")
# (a) dữ liệu DFT thô: eV_h theo |ΔE_m|, độ dốc = 1 − S
x = T.dE.abs(); y = T.eVh_B * np.sign(T.dE); h = x - (T.Eg1 + T.Eg2) / 2; ax = A[0]; xx = np.linspace(0, x.max(), 10)
for c in ("XX", "QX", "QQ"):
    m = T.contact == c; ax.scatter(x[m], y[m], s=5, c=C[c], alpha=.6, lw=0, label=f"{LB[c]} (S = {full['S'][c]:.2f})"); ax.plot(xx, (1 - full["S"][c]) * xx, c=C[c], lw=1.3)
ax.plot(xx, xx, "k:", lw=.7); ax.text(1.66, 1.08, "S = 0", fontsize=8.0, ha="right", va="top", rotation=0)
ax.set(xlabel=r"$|\Delta E_m|$ (eV)", ylabel=r"$eV_h\,\mathrm{sgn}(\Delta E_m)$ DFT (eV)", ylim=(-.3, 1.7)); ax.legend(loc="upper left", fontsize=8.0, markerscale=2, handletextpad=.1, borderaxespad=.2)
# (b) kiểm tra ngưỡng tràn điện tích của gLR
ax = A[1]
for c in ("XX", "QX", "QQ"):
    m = T.contact == c; ax.scatter(h[m], y[m], s=5, c=C[c], alpha=.6, lw=0)
ax.axvline(0, c="k", lw=.7, ls="--"); ax.text(.04, .96, "no kink at\nbare threshold", transform=ax.transAxes, va="top", fontsize=8)
ax.set(xlabel=r"$|\Delta E_m|-E_g^\mathrm{sum}/2$ (eV)", ylim=(-.3, .75))
axs = A[2:]
for k, (col_, t) in enumerate((("oof_slr", "global S"), ("oof_glrc", "S by contact type"))):
    ax = axs[k]
    for c in ("XX", "QX", "QQ"):
        d = T[T.contact == c]; ax.scatter(d[col_], d.eVh_B, s=5, c=C[c], alpha=.7, lw=0, label=LB[c])
    lim = (-.85, .85); ax.plot(lim, lim, c="k", lw=.6); ax.set(xlim=lim, ylim=lim, xlabel=r"$eV_h$ model (eV)")
    ax.text(.04, .96, f"{t}\n$r^2$ = {r2(T.eVh_B, T[col_]):.2f}", transform=ax.transAxes, va="top", fontsize=8)
axs[0].set_ylabel(r"$eV_h$ DFT (eV)"); axs[1].set_ylabel(r"$eV_h$ DFT (eV)"); axs[1].legend(loc="lower right", fontsize=8.0, markerscale=2, handletextpad=.1)
Ss = {c: [] for c in ("XX", "QX", "QQ")}
for tr, te in GroupKFold(5).split(T, groups=g):
    q = fit(T.iloc[tr].reset_index(drop=True), "eVh_B", "glr", "contact")
    for c in Ss: Ss[c].append(q["S"][c])
ax = axs[2]
for j, c in enumerate(("XX", "QX", "QQ")):
    ax.bar(j, full["S"][c], .6, color=C[c], edgecolor="k", lw=.5, yerr=np.std(Ss[c]), capsize=2); ax.text(j, full["S"][c] + .03, f"{full['S'][c]:.2f}", ha="center", fontsize=8)
ax.axhline(fit(T, "eVh_B", "slr")["S"]["None"], c="k", ls="--", lw=.7)
ax.set_xticks(range(3)); ax.set_xticklabels(["XX", "XQ", "QQ"]); ax.tick_params(axis="x", which="minor", bottom=False, top=False); ax.set(ylim=(0, 1.05), ylabel="screening factor S")
# (d) S vs contact electronegativity: slope in bins
ax = axs[3]; bins = np.unique(np.round(np.quantile(T.ENs, np.linspace(0, 1, 7)), 3)); xs, ys, es = [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    d = T[(T.ENs >= lo) & (T.ENs <= hi)]
    if len(d) < 15: continue
    b = np.sum(d.dE * d.eVh_B) / np.sum(d.dE ** 2); res = d.eVh_B - b * d.dE; se = np.sqrt(np.sum(res ** 2) / (len(d) - 1) / np.sum(d.dE ** 2))
    xs.append(d.ENs.mean()); ys.append(1 - b); es.append(se)
ax.errorbar(xs, ys, yerr=es, fmt="o", c="k", ms=3.5, capsize=2, label="binned DFT")
xx = np.linspace(T.ENs.min(), T.ENs.max(), 100); a = np.exp(P[0] + P[1] * (xx - mu) / sd); ax.plot(xx, 1 / (1 + a), c="#C0392B", lw=1.2, label="fit")
import re as _re2
_at = lambda n, f: _re2.findall(r"[A-Z][a-z]?", n)[0 if f == "X" else 3]
_pa = np.array(["-".join(sorted([_at(a_, f_), _at(b_, g_)])) for a_, f_, b_, g_ in zip(T.L1, T.f1, T.L2, T.f2)])
def _sb(m):
    d = T[m]; b = np.sum(d.dE * d.eVh_B) / np.sum(d.dE ** 2); res = d.eVh_B - b * d.dE
    return d.ENs.mean(), 1 - b, np.sqrt(np.sum(res ** 2) / (len(d) - 1) / np.sum(d.dE ** 2))
_x, _y, _e = _sb((_pa == "Te-Te") & (T.contact == "XX").values)
ax.errorbar([_x], [_y], yerr=[_e], fmt="D", mfc="w", mec=C["XX"], ecolor=C["XX"], mew=1.3, ms=5.5, capsize=2, label="Te–Te (XX)", zorder=5)
_q = [_sb((_pa == p) & (T.contact == "QQ").values) for p in ("As-Sb", "P-Sb")]
ax.errorbar([v[0] for v in _q], [v[1] for v in _q], yerr=[v[2] for v in _q], fmt="s", mfc="w", mec=C["QQ"], ecolor=C["QQ"], mew=1.3, ms=5, capsize=2, label="As–Sb, P–Sb (QQ)", zorder=5)
print("TeTe", round(_x,2), round(_y,3), "QQ-Sb", [(round(v[0],2), round(v[1],3)) for v in _q])

for c, xr in (("QQ", 4.36), ("QX", 4.73), ("XX", 5.10)): ax.axvline(xr, c=C[c], lw=.6, ls=":"); ax.text(xr, .965, LB[c], color=C[c], fontsize=8, ha="center", va="top")
ax.set(xlabel=r"$\chi_1+\chi_2$ of contact atoms", ylabel="S", ylim=(.3, 1.0)); ax.legend(loc="lower right", fontsize=7.0, bbox_to_anchor=(1.03, -0.02), handlelength=1.0, labelspacing=.25)
# (d, main) rigid interlayer-distance scan at fixed chemistry (Supplementary Note 5)
ax = AXD; Fd = pd.read_csv("../a1/res/a1_final.csv"); Dh = T.set_index("id") if "id" in T else None
sty = [("H20992", "Te–Te (XX)", C["XX"], "o"), ("H21905", "P–Te (XQ)", C["QX"], "^"), ("H03523", "P–Sb (QQ)", C["QQ"], "s"), ("H15111", "As–Sb (QQ)", C["QQ"], "D")]
for hid, nm, col, mk in sty:
    gq = Fd[Fd.id == hid].sort_values("d"); ok = gq.Eg_bi > .05
    ax.plot(gq.d, gq.S, "-", c=col, lw=1.1, alpha=.85)
    ax.plot(gq.d[ok], gq.S[ok], mk, c=col, mec="k", mew=.4, ms=4.5, label=nm)
    ax.plot(gq.d[~ok], gq.S[~ok], mk, mfc="w", mec=col, mew=1, ms=4.5)
    e = gq[gq.delta == 0]; ax.plot(e.d, e.S, mk, mfc="none", mec="k", mew=1.1, ms=9)
ax.axvspan(2.2, 2.6, color=C["QQ"], alpha=.08, lw=0); ax.text(2.4, .985, "QQ\nequil.", ha="center", va="top", fontsize=7.5, color=C["QQ"])
ax.set(xlabel=r"interlayer separation $d$ (Å)", ylabel="S", xlim=(1.5, 4.8), ylim=(.25, 1.02))
ax.legend(loc="lower right", fontsize=7.0, handletextpad=.2, labelspacing=.25, borderaxespad=.2)
for a_, l_ in zip([A[0], A[4], A[5], AXD], "abcd"): a_.text(-.20, 1.05, l_, transform=a_.transAxes, fontweight="bold", fontsize=10)
for a_, l_ in zip([A[1], A[2], A[3]], "abc"): a_.text(-.32, 1.07, l_, transform=a_.transAxes, fontweight="bold", fontsize=10)
_boldticks(figM); figM.savefig("fig2.png", bbox_inches="tight", dpi=170); figM.savefig("fig2.pdf", bbox_inches="tight")
_boldticks(figS); figS.savefig("figS_screening_extra.pdf", bbox_inches="tight")
