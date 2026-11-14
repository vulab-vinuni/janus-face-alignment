import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": .9, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "ytick.minor.visible": True, "mathtext.default": "regular"})
D = pd.read_csv("phon108/phon108_summary.csv").set_index("name"); names = sorted(D.index)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("ip_oop", ["#1F4E9C", "#7B3FA0", "#C0392B"])
with PdfPages("phonon108_F2_24.pdf") as pdf:
    for p in range(0, len(names), 24):
        nrow = int(np.ceil(min(24, len(names) - p) / 4)); fig, axs = plt.subplots(nrow, 4, figsize=(8.0, 11.0 * nrow / 6), gridspec_kw=dict(wspace=.12, hspace=.55)); axs = np.atleast_1d(axs).ravel()
        for k, ax in enumerate(axs):
            if p + k >= len(names): ax.axis("off"); continue
            n = names[p + k]; d = np.load(f"phon108/bands/{n}.npz"); x, w, z = d["x"], d["w"], d["zfrac"]
            for b in range(w.shape[1]): ax.plot(x, w[:, b], c="#1F4E9C", lw=1.0)
            xt = [x[0], x[79], x[159], x[-1]]
            for v in xt[1:-1]: ax.axvline(v, c="#999", lw=.5)
            ax.axhline(0, c="k", lw=.5, ls=":"); ax.set_xlim(x[0], x[-1]); top = max(350, w.max() * 1.05); ax.set_ylim(min(-25, w.min() * 1.3), top)
            ax.set_xticks(xt); ax.set_xticklabels(["Γ", "M", "K", "Γ"]); ax.tick_params(axis="x", which="minor", bottom=False, top=False)
            st = D.loc[n, "status"]; col = {"stable": "#2E7D32", "ZA_noise": "#B26A00", "CHECK": "#C0392B", "UNSTABLE": "#C0392B"}[st]
            ax.set_title(f"{n}", fontsize=8, pad=9); ax.text(.5, 1.015, {"stable":"stable","ZA_noise":"ZA pocket near Γ"}[st], transform=ax.transAxes, ha="center", va="bottom", fontsize=6.2, color=col)
            if k % 4 == 0: ax.set_ylabel("Frequency (cm$^{-1}$)")
            else: ax.set_yticklabels([])
        fig.suptitle(f"Phonon dispersions, MACE fine-tuned (F2) screening — {p + 1}–{min(p + 24, len(names))} of {len(names)}", fontsize=8.5, y=.925 if nrow == 6 else .96)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        if p == 0: fig.savefig("phon_p1.png", dpi=110, bbox_inches="tight") if False else None
print("ok")
