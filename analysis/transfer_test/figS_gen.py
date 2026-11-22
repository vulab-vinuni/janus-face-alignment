import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True, "savefig.dpi": 300, "legend.frameon": False,
    "axes.labelsize": 9.5, "axes.labelweight": "bold", "mathtext.fontset": "cm", "mathtext.default": "it"})
D = pd.read_csv("gen_table.csv"); D["metal"] = D[["Eg1m", "Eg2m"]].min(axis=1) <= 0
fam = {"BiTeI": (["Ch-Ch", "Ch-Ha", "Ha-Ha"], ["#1f5fa8", "#3fa34d", "#8e5bb5"]), "L4": (["Ch-Ch", "Ch-Pn", "Pn-Pn"], ["#1f5fa8", "#e08a1e", "#c8302a"]), "MXY": (["Ch-Ch"], ["#1f5fa8"])}
fig, ax = plt.subplots(1, 4, figsize=(10, 2.8)); x = np.linspace(-1.5, 1.5, 5)
for k, (f, (cl, cols)) in enumerate(fam.items()):
    a = ax[k]
    for c, col in zip(cl, cols):
        g = D[(D.family == f) & (D.contact == c) & D.eVh_B.notna()]; S = 1 - (g.dE_dft * g.eVh_B).sum() / (g.dE_dft ** 2).sum()
        ok = ~g.metal
        a.scatter(g.dE_dft[ok], g.eVh_B[ok], s=15, color=col, lw=0, alpha=0.8, label=f"{c.replace('-', '–')} ({S:.2f})")
        if (~ok).any(): a.scatter(g.dE_dft[~ok].clip(-1.5, 1.5), g.eVh_B[~ok].clip(-1.5, 1.5), marker="x", s=22, color=col, lw=1.1)
        a.plot(x, (1 - S) * x, color=col, lw=1)
    a.axhline(0, color="0.6", lw=0.5); a.axvline(0, color="0.6", lw=0.5); a.set_xlim(-1.6, 1.6); a.set_ylim(-1.6, 1.6)
    a.set_title(f, fontsize=9.5, fontweight="bold"); a.set_xlabel(r"$\Delta E_\mathrm{m}$ (eV)")
    a.legend(loc="upper left", fontsize=7.2, handletextpad=0.1, title=r"contact ($S$, all)", title_fontsize=7.2)
    a.text(0.92, 0.06, "abc"[k], transform=a.transAxes, fontweight="bold", fontsize=11)
ax[0].set_ylabel(r"$eV_\mathrm{h}^{(B)}$ (eV)")
a = ax[3]; g = D[D.eVh_B.notna() & D.eVh_A.notna()]
cmap = {"Ch-Ch": "#1f5fa8", "Ch-Ha": "#3fa34d", "Ha-Ha": "#8e5bb5", "Ch-Pn": "#e08a1e", "Pn-Pn": "#c8302a"}
for c, col in cmap.items():
    h = g[g.contact == c]; a.scatter(h.w_min, (h.eVh_A - h.eVh_B).abs(), s=14, color=col, lw=0, alpha=0.8, label=c.replace("-", "–"))
a.set_yscale("log"); a.set_ylim(1e-3, 3); a.set_xlim(0, 1); a.set_xlabel(r"$w_\mathrm{min}$"); a.set_ylabel(r"$|eV_\mathrm{h}^{(A)}-eV_\mathrm{h}^{(B)}|$ (eV)")
a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7, ncol=3, handletextpad=0.1, columnspacing=0.5); a.text(0.9, 0.88, "d", transform=a.transAxes, fontweight="bold", fontsize=11)
plt.tight_layout(); plt.savefig("figS_transfer.pdf"); plt.savefig("figS_transfer.png", dpi=170)
