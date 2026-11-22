import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "ytick.minor.visible": True, "savefig.dpi": 300, "legend.frameon": False,
    "axes.labelsize": 10, "axes.labelweight": "bold", "mathtext.fontset": "cm", "mathtext.default": "it"})
R = pd.read_csv("gen_S_final.csv"); D = pd.read_csv("gen_table.csv"); D["metal"] = D[["Eg1m", "Eg2m"]].min(axis=1) <= 0
order = ["BiTeI/Ch-Ch", "BiTeI/Ch-Ha", "BiTeI/Ha-Ha", "MXY/Ch-Ch", "L4/Ch-Ch", "L4/Ch-Pn", "L4/Pn-Pn"]
lab = ["BiTeI\nCh–Ch", "BiTeI\nCh–Ha", "BiTeI\nHa–Ha", "MXY\nCh–Ch", "L4\nCh–Ch", "L4\nCh–Pn", "L4\nPn–Pn"]
cno, cpn = "#1f5fa8", "#c8302a"
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1.35, 1]))
a = ax[0]; a.axhspan(0.72, 1.15, color=cno, alpha=0.07, lw=0); a.axhspan(0, 0.65, color=cpn, alpha=0.07, lw=0)
a.axhline(0.72, color=cno, lw=0.8, ls="--"); a.axhline(0.65, color=cpn, lw=0.8, ls="--")
for i, c in enumerate(order):
    for dx, st, mf in [(-0.13, "all", True), (0.13, "semiconducting", False)]:
        r = R[(R.set == st) & (R.cls == c)].iloc[0]; col = cpn if "Pn" in c else cno
        a.errorbar(i + dx, r.S, yerr=[[r.S - r.lo], [min(r.hi, 1.12) - r.S]], fmt="o" if mf else "s", ms=5.5, color=col,
                   mfc=col if mf else "white", mew=1.2, capsize=2.5, lw=1.1)
    a.text(i, 0.03, f"n={int(R[(R.set=='all')&(R.cls==c)].n.iloc[0])}", ha="center", fontsize=7.5, color="0.3")
for y, t in [(0.784, "XX"), (0.683, "XQ"), (0.530, "QQ")]:
    a.plot([6.55, 6.85], [y, y], color="0.25", lw=1.6); a.text(6.9, y, t, fontsize=7.5, va="center", color="0.25")
a.text(6.7, 0.88, "108-set\nN = 599", fontsize=7.5, ha="center", color="0.25")
a.set_xticks(range(7)); a.set_xticklabels(lab, fontsize=8); a.tick_params(axis="x", which="minor", bottom=False, top=False)
a.set_xlim(-0.5, 7.3); a.set_ylim(0, 1.15); a.set_ylabel("Screening factor $S$")
a.plot([], [], "o", color="0.3", label="all interfaces (pre-registered)"); a.plot([], [], "s", mfc="white", color="0.3", label="semiconducting layers only")
a.legend(loc="lower left", bbox_to_anchor=(0.0, 0.07), fontsize=7.5, handletextpad=0.3)
a.text(0.02, 0.97, "a", transform=a.transAxes, fontweight="bold", fontsize=12, va="top")
b = ax[1]; x = np.linspace(-1.4, 1.4, 10)
for k, col, nm in [(False, cno, "no Pn face"), (True, cpn, "Pn face")]:
    g = D[(D.has_Pn_face == k) & ~D.metal & D.eVh_B.notna()]; S = 1 - (g.dE_dft * g.eVh_B).sum() / (g.dE_dft ** 2).sum()
    b.scatter(g.dE_dft, g.eVh_B, s=16, color=col, alpha=0.75, lw=0, label=f"{nm} (S = {S:.2f})"); b.plot(x, (1 - S) * x, color=col, lw=1.2)
g = D[D.metal & D.eVh_B.notna()]
b.scatter(g.dE_dft.clip(-1.4, 1.4), g.eVh_B.clip(-1.4, 1.4), marker="x", s=22, color="0.35", lw=1.0, label="metallic layer (clipped)")
b.axhline(0, color="0.6", lw=0.6); b.axvline(0, color="0.6", lw=0.6)
b.set_xlim(-1.5, 1.5); b.set_ylim(-1.5, 1.5); b.set_xlabel(r"$\Delta E_\mathrm{m}$ (eV)"); b.set_ylabel(r"$eV_\mathrm{h}$ (eV)")
b.legend(loc="upper left", fontsize=7.3, handletextpad=0.2)
b.text(0.93, 0.05, "b", transform=b.transAxes, fontweight="bold", fontsize=12)
plt.tight_layout(); plt.savefig("fig_transfer.pdf"); plt.savefig("fig_transfer.png", dpi=200)
