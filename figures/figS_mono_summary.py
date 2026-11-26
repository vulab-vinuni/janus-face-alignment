import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/tmp/claude-0/-home-claude/1cc01a84-5ce5-5f61-b3d7-0b32c6854667/scratchpad/bsoc"; sys.path.insert(0, B)
import os; cwd = os.getcwd(); os.chdir(B)
src = open("plot_soc_si.py").read().split("ds = []")[0]; ns = {}; exec(src, ns); load = ns["load"]
os.chdir(cwd)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": .9, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                     "mathtext.default": "regular", "axes.labelweight": "bold", "legend.frameon": False})
R = "/home/claude/janus-face-alignment/data/"
REP = ["SeAlSnAs", "SeInPbP", "SAlSiAs", "TeAlPbSb"]
RED, BLUE = "#C0392B", "#2E5BB8"
fig = plt.figure(figsize=(7.2, 7.6)); gs = fig.add_gridspec(2, 4, hspace=.32, wspace=.12, top=.965, bottom=.40, left=.08, right=.99); g2 = fig.add_gridspec(1, 4, wspace=.55, top=.31, bottom=.065, left=.08, right=.99)
for k, nm in enumerate(REP):
    a = fig.add_subplot(gs[0, k]); X, Y, n, Eg, tk, flat = load(f"{B}/{nm}_M")
    if Eg > 0: a.axhspan(0, Eg, color="0.92", zorder=0)
    for j in range(Y.shape[1]): a.plot(X, Y[:, j], "-", color=RED if j >= n else BLUE, lw=.8)
    if flat is not None: a.axhline(flat, color="0.55", ls="--", lw=.7)
    for v in tk[1:-1]: a.axvline(v, color="0.75", lw=.5)
    a.set_xticks(tk); a.set_xticklabels(["M", "Γ", "K", "M"]); a.set_xlim(X[0], X[-1]); a.set_ylim(-3, 3)
    a.set_title(f"{nm} ($E_g$ = {Eg:.2f} eV)" if Eg >= .005 else f"{nm} (semimetal)", fontsize=7.5, pad=3)
    if k == 0: a.set_ylabel(r"$E-E_\mathrm{VBM}$ (eV)")
    else: a.set_yticklabels([])
    a._plab = "abcd"[k]
    p = np.load(f"{R}phonons/bands/{nm}.npz"); x, w, z = p["x"], p["w"], p["zfrac"]
    b = fig.add_subplot(gs[1, k]); b.axhline(0, c="k", lw=.5, ls=":")
    for j in range(w.shape[1]): b.plot(x, w[:, j], color="#1b4f8a" if j else "#c8302a", lw=.8)
    xt = [x[0], x[80], x[160], x[-1]]
    for v in xt[1:-1]: b.axvline(v, color="0.75", lw=.5)
    b.set_xticks(xt); b.set_xticklabels(["Γ", "M", "K", "Γ"]); b.set_xlim(x[0], x[-1]); b.set_ylim(-25, 350)
    if k == 0: b.set_ylabel("Frequency (cm$^{-1}$)")
    else: b.set_yticklabels([])
    b._plab = "efgh"[k]
# summaries
S = pd.read_csv(R + "monolayers/soc_gap_reduction.csv"); S["M"] = S.name.str.extract(r"(Si|Ge|Sn|Pb)")
c = fig.add_subplot(g2[0, 0]); cols = {"Si": "#1f5fa8", "Ge": "#3fa34d", "Sn": "#e08a1e", "Pb": "#c8302a"}
bins = np.linspace(0, .7, 15)
c.hist([-S[S.M == m].dEg_SOC_eV for m in cols], bins=bins, stacked=True, color=list(cols.values()), label=list(cols), edgecolor="white", lw=.4)
c.set_xlabel(r"$-\Delta E_g^\mathrm{SOC}$ (eV)"); c.set_ylabel("Compounds"); c.legend(title="M", fontsize=6.8, title_fontsize=6.8, loc="upper right", handlelength=1)
P = pd.read_csv(R + "phonons/phon108_summary.csv")
d = fig.add_subplot(g2[0, 1]); v = P.wmin_ZA_nearG.clip(upper=0)
d.hist(v[v < 0], bins=np.linspace(-7, 0, 15), color="#1b4f8a", edgecolor="white", lw=.4)
d.set_xlabel("Lowest ZA freq. (cm$^{-1}$)"); d.set_ylabel("Compounds"); d.text(.04, .93, f"{(v>=0).sum()} free of\nimag. modes", transform=d.transAxes, va="top", fontsize=6.8)
M = pd.read_csv(R + "md300K/md300K_summary.csv")
e = fig.add_subplot(g2[0, 2]); e.hist(M.dz_max_A, bins=np.linspace(0, .6, 13), color="#3fa34d", edgecolor="white", lw=.4)
e.set_xlabel(r"MD max. $|\Delta z|$ (Å)"); e.set_ylabel("Compounds")
f = fig.add_subplot(g2[0, 3]); f.scatter(M.T_mean, M.E_drift_meV_at, s=8, color="#8e5bb5", lw=0)
f.set_xlabel(r"MD $\langle T\rangle$ (K)"); f.set_ylabel("Energy drift (meV/atom)")
from matplotlib.ticker import MaxNLocator
for a, l in zip([c, d, e, f], "ijkl"):
    a.text(-0.3, 1.04, l, transform=a.transAxes, fontweight="bold", fontsize=10, va="bottom"); a.yaxis.set_major_locator(MaxNLocator(integer=(l != "l")))
_ax = [a_ for a_ in fig.axes if hasattr(a_, "_plab")]; _w0 = min(a_.get_position().x0 for a_ in fig.axes)
_il = [a_ for a_ in fig.axes if a_.get_position().x0 == _w0 and not hasattr(a_, "_plab")][0]; _xi = _w0 - 0.3 * _il.get_position().width
for a_ in _ax:
    p_ = a_.get_position(); x_ = _xi if abs(p_.x0 - _w0) < 1e-3 else p_.x0 - 0.018
    fig.text(x_, p_.y1 + 0.006, a_._plab, fontweight="bold", fontsize=10, va="bottom", ha="left")
fig.savefig("figS_mono_summary.pdf"); fig.savefig("figS_mono_summary.png", dpi=160)
print(M.broken_bonds.sum(), M.T_mean.min(), M.T_mean.max(), (v>=0).sum())
