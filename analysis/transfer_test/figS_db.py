import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.linewidth": .9, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                     "mathtext.default": "regular", "axes.labelweight": "bold", "legend.frameon": False})
D = pd.read_csv("data/janus1328/Supplementary_Data_2_Janus1328.csv")
fam = ["XAMQ", "L4", "MXY", "MA2Z4", "BiTeI"]; lab = ["X–A–M–Q", "L4", "MXY", "MA$_2$Z$_4$", "BiTeI"]
col = {"XAMQ": "#1f5fa8", "L4": "#e08a1e", "MXY": "#3fa34d", "MA2Z4": "#8e5bb5", "BiTeI": "#c8302a"}
fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.0)); ax = axs.ravel()
a = ax[0]; x = np.arange(5)
cats = [("intact, gapped", lambda g: g.intact & (g.Eg_PBE_eV > .05) & (g.magnetic_moment_muB.abs() <= .1), "#1f5fa8"),
        ("intact, gapless", lambda g: g.intact & (g.Eg_PBE_eV <= .05) & (g.magnetic_moment_muB.abs() <= .1), "#9bb7dd"),
        ("intact, magnetic", lambda g: g.intact & (g.magnetic_moment_muB.abs() > .1), "#8e5bb5"),
        ("not intact", lambda g: ~g.intact, "0.6")]
bot = np.zeros(5)
for nm, fn, c in cats:
    v = np.array([fn(D[D.family == f]).sum() for f in fam]); a.bar(x, v, .62, bottom=bot, color=c, label=nm, edgecolor="white", lw=.4); bot += v
for i, f in enumerate(fam): a.text(i, bot[i] + 8, str(int(bot[i])), ha="center", fontsize=7.5)
a.set_xticks(x); a.set_xticklabels(lab); a.tick_params(axis="x", which="minor", bottom=False, top=False); a.set_ylabel("Compounds"); a.set_ylim(0, 620)
a.legend(loc="upper right", fontsize=7, handlelength=1)
b = ax[1]; rng = np.random.default_rng(1)
for i, f in enumerate(fam):
    g = D[(D.family == f) & D.intact]; xs = i + rng.uniform(-.25, .25, len(g))
    b.scatter(xs[~g.used_in_transfer_test.values], g.Eg_PBE_eV[~g.used_in_transfer_test.values].clip(lower=0), s=5, color=col[f], alpha=.45, lw=0)
    b.scatter(xs[g.used_in_transfer_test.values], g.Eg_PBE_eV[g.used_in_transfer_test.values], s=14, facecolor="none", edgecolor="k", lw=.7)
b.scatter([], [], s=14, facecolor="none", edgecolor="k", lw=.7, label="used in transfer test")
b.set_xticks(x); b.set_xticklabels(lab); b.tick_params(axis="x", which="minor", bottom=False, top=False); b.set_ylabel("PBE gap (eV)"); b.set_ylim(-.1, 3.6)
b.legend(loc="upper left", fontsize=7)
c = ax[2]; bins = np.linspace(2.3, 4.6, 24); bt = np.zeros(len(bins) - 1)
for f, l in zip(fam, lab):
    h, _ = np.histogram(D[(D.family == f) & D.intact].a_A, bins); c.bar(bins[:-1], h, np.diff(bins), bottom=bt, align="edge", color=col[f], label=l, edgecolor="white", lw=.3); bt += h
c.set_xlabel("Lattice constant (Å)"); c.set_ylabel("Compounds"); c.legend(loc="upper left", fontsize=7, handlelength=1)
d = ax[3]
for f, l in zip(fam, lab):
    g = D[(D.family == f) & D.intact]; d.scatter(g.a_A, g.Eg_PBE_eV.clip(lower=0), s=5, color=col[f], alpha=.5, lw=0, label=l)
d.set_xlabel("Lattice constant (Å)"); d.set_ylabel("PBE gap (eV)"); d.set_xlim(2.3, 4.6); d.set_ylim(-.1, 3.6)
for i, a_ in enumerate(ax): a_.text(-.17, 1.03, "abcd"[i], transform=a_.transAxes, fontweight="bold", fontsize=11)
plt.tight_layout(w_pad=1.5, h_pad=1.5); plt.savefig("figS_db.pdf"); plt.savefig("figS_db.png", dpi=160)
