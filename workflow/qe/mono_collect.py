#!/usr/bin/env python3
"""Collect MONO/<name>/e<ε>/ results -> mono_strain.csv and provide interpolation in the lattice constant.
   Columns: name, strain, a, Etot, VBM, CBM, Eg (eV, absolute), Evac_X (bottom), Evac_Q (top), IP_X, IP_Q, EA_X, EA_Q, dEvac=Evac_Q-Evac_X,
        m_v, m_c, g_v, g_c, flat_X, flat_Q
   python3 mono_collect.py MONO   ->  mono_strain.csv
   from mono_collect import Mono; M = Mono('mono_strain.csv'); M.at('SAlGeAs', a_bar) -> interpolated dict (quadratic for Etot, linear/quadratic for the rest)"""
import sys, os, glob, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qeio import read_xml, planar_avg, vac_levels


def collect(root):
    import pandas as pd
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*", "e*"))):
        try:
            meta = dict(l.split() for l in open(f"{d}/meta.txt")); info = json.load(open(f"{d}/edges.json"))
            X = read_xml(f"{d}/tmp/m.save/data-file-schema.xml"); z, V = planar_avg(f"{d}/avg.dat")
            zat = [p[2] for _, p in X["atoms"]]; vl = vac_levels(z, V, zat, X["cell"][2, 2])
            vbm, cbm = info["vbm_grid"], info["cbm_grid"]
            rows.append(dict(name=meta["name"], strain=float(meta["strain"]), a=float(meta["a"]), Etot=info["etot"], VBM=vbm, CBM=cbm,
                             Eg=cbm - vbm, Evac_X=vl["bot"], Evac_Q=vl["top"], IP_X=vl["bot"] - vbm, IP_Q=vl["top"] - vbm,
                             EA_X=vl["bot"] - cbm, EA_Q=vl["top"] - cbm, dEvac=vl["top"] - vl["bot"], m_v=info.get("m_v"),
                             m_c=info.get("m_c"), g_v=info["g_v"], g_c=info["g_c"], flat_X=vl["flat_bot"], flat_Q=vl["flat_top"]))
        except Exception as ex:
            print("skipped", d, ex)
    D = pd.DataFrame(rows)
    gm = {}
    for f in glob.glob(os.path.join(root, "*", "e+0.0", "dos_mass.json")):
        j = json.load(open(f)); gm[f.split(os.sep)[-3]] = (j["gm_v_0.10"], j["gm_c_0.10"])
    if len(D):
        D["gm_v"] = D.name.map(lambda n: gm.get(n, (np.nan, np.nan))[0]); D["gm_c"] = D.name.map(lambda n: gm.get(n, (np.nan, np.nan))[1])
    return D


class Mono:
    def __init__(self, csv):
        import pandas as pd
        self.D = pd.read_csv(csv)

    def at(self, name, a):
        g = self.D[self.D.name == name].sort_values("a"); out = dict(name=name, a=a)
        if len(g) == 0: raise KeyError(name)
        if len(g) == 1: return dict(g.iloc[0])
        x = g.a.values
        if not (x.min() - 1e-6 <= a <= x.max() + 1e-6): out["extrap"] = True
        for c in ("Etot",):
            out[c] = np.polyval(np.polyfit(x, g[c], min(2, len(g) - 1)), a)
        for c in ("gm_v", "gm_c"):
            if c in g: out[c] = float(g[c].iloc[0])
        for c in ("VBM", "CBM", "Eg", "IP_X", "IP_Q", "EA_X", "EA_Q", "dEvac", "m_v", "m_c"):
            y = g[c].values.astype(float); ok = np.isfinite(y)
            out[c] = np.interp(a, x[ok], y[ok]) if ok.sum() > 1 else (y[ok][0] if ok.any() else np.nan)
        for c in ("g_v", "g_c"):
            out[c] = int(g.iloc[np.abs(x - a).argmin()][c])
        return out


if __name__ == "__main__":
    D = collect(sys.argv[1] if len(sys.argv) > 1 else "MONO"); D.to_csv("mono_strain.csv", index=False)
    print(D.round(4).to_string() if len(D) < 30 else D.describe().round(3).to_string())
