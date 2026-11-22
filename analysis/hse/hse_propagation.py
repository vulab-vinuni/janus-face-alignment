"""Propagation of monolayer HSE06 (VASP, collinear) corrections, optionally with PBE+SOC gap corrections, to the
DFT heterostructure set and to the confident alignment map.  Run from the repository root:  python analysis/hse/hse_propagation.py
dVBM/dCBM: HSE-PBE shifts of the band edges relative to the mean vacuum level (data/hse/hse108_vasp.csv);
they are applied rigidly to all strains and to the layer-resolved band edges.  The interface dipole is kept at its
PBE value (ground-state density)."""
import numpy as np, pandas as pd
H = pd.read_csv("data/hse/hse108_vasp.csv").set_index("name")
T = pd.read_csv("data/hse/hse108_time.txt", sep=r"\s+", header=None, names=["name", "st", "pbe", "hse", "nelm"]).set_index("name")
H["nelm"] = T.nelm.str.split("=").str[1].astype(int); H["conv"] = H.nelm < 140   # NELM = 150: 149 steps means not converged
M = pd.read_csv("data/monolayers/monolayers_strain_grid.csv"); M0 = M[M.strain == 0].set_index("name")
S = pd.read_csv("data/monolayers/soc_gap_reduction.csv").set_index("name")
gapped = M0.Eg > .05
print("HSE not converged:", list(H.index[~H.conv]), "| gapped in PBE:", list(H.index[~H.conv & gapped.reindex(H.index)]))
g = H[H.conv & gapped.reindex(H.index)]
print(f"98 gapped compounds: Eg PBE {M0.Eg[gapped].median():.2f} -> HSE {(M0.Eg + H.dCBM_hse - H.dVBM_hse)[gapped].median():.2f} eV (median); "
      f"dVBM {g.dVBM_hse.median():+.2f} ({g.dVBM_hse.min():+.2f} to {g.dVBM_hse.max():+.2f}), dCBM {g.dCBM_hse.median():+.2f} ({g.dCBM_hse.min():+.2f} to {g.dCBM_hse.max():+.2f}); "
      f"VBM share of gap opening {(-g.dVBM_hse / (g.dCBM_hse - g.dVBM_hse)).median():.2f}")
print(f"VASP PBE vs QE PBE (0%): |dEg| median {np.median(abs(H.Eg_pbe - M0.Eg.reindex(H.index))[gapped.reindex(H.index)]):.3f} eV, "
      f"|dIP_X| {np.median(abs(H.IP_X_pbe - M0.IP_X.reindex(H.index))[gapped.reindex(H.index)]):.3f}, |d dEvac| {np.median(abs(H.dEvac_pbe - M0.dEvac.reindex(H.index))):.3f}; "
      f"HSE-PBE change of dEvac median {np.median((H.dEvac_hse - H.dEvac_pbe)[H.conv]):+.3f} eV")
dV, dC = H.dVBM_hse, H.dCBM_hse

def typ(v1, c1, v2, c2):
    t = np.where(np.minimum(c1, c2) < np.maximum(v1, v2), "III", np.where(((v2 >= v1) & (c2 <= c1)) | ((v1 >= v2) & (c1 <= c2)), "I", "II"))
    return t, np.minimum.reduce([np.abs(v1 - v2), np.abs(c1 - c2), np.abs(np.minimum(c1, c2) - np.maximum(v1, v2))])
def shifts(L, f, soc):  # per-layer VBM and CBM shifts; f = VBM share of the (negative) SOC gap change
    sv, sc = L.map(dV).values, L.map(dC).values
    if soc:
        d = L.map(S.dEg_SOC_eV).values; sv = sv - f * d; sc = sc + (1 - f) * d
    return sv, sc

# --- DFT heterostructures (599): layer-resolved edges
D = pd.read_csv("data/heterostructures/het599_training_set.csv")
D["pair"] = [" ".join(sorted([a, b])) for a, b in zip(D.L1, D.L2)]
okH = D.L1.map(H.conv) & D.L2.map(H.conv); okS = D.L1.map(S.Eg_SOC_eV > .05) & D.L2.map(S.Eg_SOC_eV > .05)
t0, m0 = typ(D.VBM1, D.CBM1, D.VBM2, D.CBM2)
def switch_frac(D, t, sel):
    G = pd.DataFrame(dict(pair=D.pair[sel], t=t[sel])); G = G.groupby("pair").t.agg(["nunique", "size"]); G = G[G["size"] >= 2]
    return (G["nunique"] > 1).sum(), len(G)
def model_types(D, sv1, sc1, sv2, sc2):  # monolayer-based prediction with the out-of-fold contact-resolved dipole
    eVh = D["eVh_gLR-contact"].values; dEp = D.dE.values + ((sv2 + sc2) - (sv1 + sc1)) / 2 - eVh
    E1, E2 = D.Eg1.values + sc1 - sv1, D.Eg2.values + sc2 - sv2
    return typ(-E1 / 2, E1 / 2, dEp - E2 / 2, dEp + E2 / 2)[0]
loc = D.w_min >= .6 if "p_local" not in D else D.p_local >= .3
print(f"\nDFT set: {okH.sum()} interfaces with converged HSE for both layers")
rows = []
for lab, soc, fs in [("PBE", False, [None]), ("HSE", False, [None]), ("HSE+SOC", True, [1.0, .5, 0.0])]:
    for f in fs:
        sel = (okH & okS).values if soc else okH.values
        if lab == "PBE": sv1 = sc1 = sv2 = sc2 = np.zeros(len(D))
        else: (sv1, sc1), (sv2, sc2) = shifts(D.L1, f, soc), shifts(D.L2, f, soc)
        t, _ = typ(D.VBM1 + sv1, D.CBM1 + sc1, D.VBM2 + sv2, D.CBM2 + sc2)
        tm = model_types(D, sv1, sc1, sv2, sc2)
        ns, nt = switch_frac(D, t, sel)
        r = dict(case=lab + ("" if f is None else f" (VBM share {f:.1f})"), N=int(sel.sum()), changed=100 * (t[sel] != t0[sel]).mean(),
                 II_III=int(((t0 == "II") & (t == "III") & sel).sum()), III_II=int(((t0 == "III") & (t == "II") & sel).sum()),
                 I_out=int(((t0 == "I") & (t != "I") & sel).sum()), I_in=int(((t0 != "I") & (t == "I") & sel).sum()),
                 switch=f"{ns}/{nt} ({100 * ns / nt:.0f}%)", acc_all=(tm[sel] == t[sel]).mean(), acc_loc=(tm[sel & loc.values] == t[sel & loc.values]).mean(),
                 frac_I=100 * (t[sel] == "I").mean(), frac_II=100 * (t[sel] == "II").mean(), frac_III=100 * (t[sel] == "III").mean())
        rows.append(r)
R = pd.DataFrame(rows); pd.set_option("display.width", 250); print(R.round(2).to_string(index=False)); R.to_csv("data/hse/hse_dft_set_summary.csv", index=False)
# same-subset PBE reference for the SOC subset
sel = (okH & okS).values; ns, nt = switch_frac(D, t0, sel)
print(f"PBE on HSE+SOC subset ({sel.sum()}): switching {ns}/{nt} ({100*ns/nt:.0f}%), types I/II/III % = "
      f"{100*(t0[sel]=='I').mean():.0f}/{100*(t0[sel]=='II').mean():.0f}/{100*(t0[sel]=='III').mean():.0f}")

# --- confident alignment map
A = pd.read_csv("data/predictions/alignment_map7796.csv"); A = A[(A.Eg1 > .05) & (A.Eg2 > .05)].copy()
dEp = (A.dE - A.eVh_hyb).values; ta, ma = typ(-A.Eg1 / 2, A.Eg1 / 2, dEp - A.Eg2 / 2, dEp + A.Eg2 / 2)
conf = (A.p_local >= .3).values & (ma > .1) & (A.L1.map(H.conv) & A.L2.map(H.conv)).values
print(f"\nConfident map: {conf.sum()} confident interfaces")
for lab, soc, f in [("HSE", False, None), ("HSE+SOC VBM", True, 1.0), ("HSE+SOC 50/50", True, .5), ("HSE+SOC CBM", True, 0.0)]:
    (sv1, sc1), (sv2, sc2) = shifts(A.L1, f, soc), shifts(A.L2, f, soc)
    c = conf & ((A.L1.map(S.Eg_SOC_eV > .05) & A.L2.map(S.Eg_SOC_eV > .05)).values if soc else True)
    E1, E2 = A.Eg1.values + sc1 - sv1, A.Eg2.values + sc2 - sv2; d = dEp + ((sv2 + sc2) - (sv1 + sc1)) / 2
    t, m = typ(-E1 / 2, E1 / 2, d - E2 / 2, d + E2 / 2)
    print(f"{lab}: {c.sum()} interfaces, type unchanged {100 * (t[c] == ta[c]).mean():.1f}%, still margin>0.1 eV {100 * ((t[c] == ta[c]) & (m[c] > .1)).mean():.1f}%; "
          f"I/II/III before {np.mean(ta[c]=='I')*100:.0f}/{np.mean(ta[c]=='II')*100:.0f}/{np.mean(ta[c]=='III')*100:.0f}% after {np.mean(t[c]=='I')*100:.0f}/{np.mean(t[c]=='II')*100:.0f}/{np.mean(t[c]=='III')*100:.0f}%")
