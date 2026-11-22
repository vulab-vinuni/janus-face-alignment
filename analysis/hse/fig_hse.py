import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.linewidth": .9, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                     "xtick.minor.visible": True, "ytick.minor.visible": True, "mathtext.default": "regular", "axes.labelweight": "bold", "legend.frameon": False})
H = pd.read_csv("data/hse/hse108_vasp.csv").set_index("name"); M = pd.read_csv("data/monolayers/monolayers_strain_grid.csv"); M0 = M[M.strain == 0].set_index("name")
g = H[(M0.Eg.reindex(H.index) > .05)]; Mel = g.index.str.extract(r"(Si|Ge|Sn|Pb)")[0].values
cols = {"Si": "#1f5fa8", "Ge": "#3fa34d", "Sn": "#e08a1e", "Pb": "#c8302a"}
R = pd.read_csv("data/hse/hse_dft_set_summary.csv")
fig, ax = plt.subplots(1, 4, figsize=(10, 2.7))
a = ax[0]
for m, c in cols.items():
    k = Mel == m; a.scatter(M0.Eg.reindex(g.index)[k], (M0.Eg.reindex(g.index) + g.dCBM_hse - g.dVBM_hse)[k], s=14, color=c, lw=0, label=f"M = {m}")
a.plot([0, 3.2], [0, 3.2], "k--", lw=.7); a.set_xlim(0, 2.2); a.set_ylim(0, 3.2); a.set_xlabel("$E_g$ PBE (eV)"); a.set_ylabel("$E_g$ HSE06 (eV)")
a.legend(loc="lower right", fontsize=7, handletextpad=.1)
b = ax[1]
for m, c in cols.items():
    k = Mel == m; b.scatter(g.dVBM_hse[k], g.dCBM_hse[k], s=14, color=c, lw=0)
b.set_xlabel(r"$\Delta E_\mathrm{VBM}$ (eV)"); b.set_ylabel(r"$\Delta E_\mathrm{CBM}$ (eV)"); b.set_xlim(-.7, -.25); b.set_ylim(0, .35)
c = ax[2]; lab = ["PBE", "HSE", "HSE+SOC"]; rows = [R.iloc[0], R.iloc[1], R.iloc[3]]
x = np.arange(3); bot = np.zeros(3)
for k, col, nm in [("frac_I", "#8e5bb5", "I"), ("frac_II", "#1f5fa8", "II"), ("frac_III", "#c8302a", "III")]:
    v = np.array([r[k] for r in rows]); c.bar(x, v, .6, bottom=bot, color=col, label=f"type {nm}", edgecolor="white", lw=.5); bot += v
c.set_xticks(x); c.set_xticklabels(lab); c.tick_params(axis="x", which="minor", bottom=False, top=False); c.set_ylabel("DFT interfaces (%)"); c.set_ylim(0, 128)
c.legend(loc="upper center", ncol=3, fontsize=7, handletextpad=.2, columnspacing=.6)
d = ax[3]; sw = [int(r["switch"].split("(")[1].rstrip("%)")) for r in rows]
d.bar(x - .17, sw, .32, color="#3fa34d", label="pairs switching type")
d.bar(x + .17, [0, 95.8, 95.1], .32, color="#e08a1e", label="confident types kept")
for i, v in enumerate(sw): d.text(i - .17, v + 2, f"{v}", ha="center", fontsize=7)
for i, v in [(1, 95.8), (2, 95.1)]: d.text(i + .17, v + 2, f"{v:.0f}", ha="center", fontsize=7)
d.set_xticks(x); d.set_xticklabels(lab); d.tick_params(axis="x", which="minor", bottom=False, top=False); d.set_ylim(0, 128); d.set_ylabel("Fraction (%)")
d.legend(loc="upper left", fontsize=7, handletextpad=.3)
for i, a in enumerate(ax): a.text(-.22, 1.03, "abcd"[i], transform=a.transAxes, fontweight="bold", fontsize=11)
plt.tight_layout(w_pad=1.2); plt.savefig("/tmp/claude-0/-home-claude/1cc01a84-5ce5-5f61-b3d7-0b32c6854667/scratchpad/hseres/figS_hse.pdf"); plt.savefig("/tmp/claude-0/-home-claude/1cc01a84-5ce5-5f61-b3d7-0b32c6854667/scratchpad/hseres/figS_hse.png", dpi=170)
