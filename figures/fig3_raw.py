import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.model_selection import GroupKFold
from glr import r2, fit
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True, "savefig.dpi": 300, "legend.frameon": False, "mathtext.default": "regular"})
plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 10, "axes.labelweight": "bold", "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5, "mathtext.fontset": "cm", "mathtext.default": "it"})
def _boldticks(fig):
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
fig, A = plt.subplots(2, 3, figsize=(7.3, 4.75), gridspec_kw=dict(wspace=.42, hspace=.42)); A = A.ravel()
full = fit(T, "eVh_B", "glr", "contact")
# (a) raw DFT data: eV_h vs |ΔE_m|, slope = 1 − S
x = T.dE.abs(); y = T.eVh_B * np.sign(T.dE); h = x - (T.Eg1 + T.Eg2) / 2; ax = A[0]; xx = np.linspace(0, x.max(), 10)
for c in ("XX", "QX", "QQ"):
    m = T.contact == c; ax.scatter(x[m], y[m], s=5, c=C[c], alpha=.6, lw=0, label=f"{LB[c]} (S = {full['S'][c]:.2f})"); ax.plot(xx, (1 - full["S"][c]) * xx, c=C[c], lw=1.3)
ax.plot(xx, xx, "k:", lw=.7); ax.text(1.66, 1.08, "S = 0", fontsize=8, ha="right", va="top", rotation=0)
ax.set(xlabel=r"$|\Delta E_m|$ (eV)", ylabel=r"$eV_h\,\mathrm{sgn}(\Delta E_m)$ DFT (eV)", ylim=(-.3, 1.7)); ax.legend(loc="upper left", fontsize=8, markerscale=2, handletextpad=.1, borderaxespad=.2)
# (b) test of the charge-spillage threshold
ax = A[1]
for c in ("XX", "QX", "QQ"):
    m = T.contact == c; ax.scatter(h[m], y[m], s=5, c=C[c], alpha=.6, lw=0)
ax.axvline(0, c="k", lw=.7, ls="--"); ax.text(.04, .96, "no kink at the\nspillage onset", transform=ax.transAxes, va="top", fontsize=6.8)
ax.set(xlabel=r"$|\Delta E_m|-E_g^\mathrm{sum}/2$ (eV)", ylim=(-.3, .75))
axs = A[2:]
for k, (col_, t) in enumerate((("oof_slr", "global S"), ("oof_glrc", "S by contact type"))):
    ax = axs[k]
    for c in ("XX", "QX", "QQ"):
        d = T[T.contact == c]; ax.scatter(d[col_], d.eVh_B, s=5, c=C[c], alpha=.7, lw=0, label=LB[c])
    lim = (-.85, .85); ax.plot(lim, lim, c="k", lw=.6); ax.set(xlim=lim, ylim=lim, xlabel=r"$eV_h$ model (eV)")
    ax.text(.04, .96, f"{t}\n$r^2$ = {r2(T.eVh_B, T[col_]):.2f}", transform=ax.transAxes, va="top", fontsize=6.8)
axs[0].set_ylabel(r"$eV_h$ DFT (eV)"); axs[1].set_ylabel(r"$eV_h$ DFT (eV)"); axs[1].legend(loc="lower right", fontsize=8, markerscale=2, handletextpad=.1)
Ss = {c: [] for c in ("XX", "QX", "QQ")}
for tr, te in GroupKFold(5).split(T, groups=g):
    q = fit(T.iloc[tr].reset_index(drop=True), "eVh_B", "glr", "contact")
    for c in Ss: Ss[c].append(q["S"][c])
ax = axs[2]
for j, c in enumerate(("XX", "QX", "QQ")):
    ax.bar(j, full["S"][c], .6, color=C[c], edgecolor="k", lw=.5, yerr=np.std(Ss[c]), capsize=2); ax.text(j, full["S"][c] + .03, f"{full['S'][c]:.2f}", ha="center", fontsize=6.8)
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
for c, xr in (("QQ", 4.36), ("QX", 4.73), ("XX", 5.10)): ax.axvline(xr, c=C[c], lw=.6, ls=":"); ax.text(xr, .965, LB[c], color=C[c], fontsize=6.5, ha="center", va="top")
ax.set(xlabel=r"$\chi_1+\chi_2$ of contact atoms", ylabel="S", ylim=(.3, 1.0)); ax.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.03, -0.02), handlelength=1.2)
for k, a_ in enumerate(A): a_.text(-.32, 1.07, "abcdef"[k], transform=a_.transAxes, fontweight="bold", fontsize=10)
_boldticks(fig); fig.savefig("fig2.png", bbox_inches="tight", dpi=170); fig.savefig("fig2.pdf", bbox_inches="tight")
print("ENs oof r2", round(r2(T.eVh_B, pr), 3), "P", P.round(3), list(zip(np.round(xs, 2), np.round(ys, 2))))
