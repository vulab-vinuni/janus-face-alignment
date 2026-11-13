#!/usr/bin/env python3
"""Anderson / midgap / linear-response (sLR) / generalized linear-response (gLR) alignment models extended to Janus faces.

Conventions (L1 bottom, face f1 up; L2 top, face f2 down; energies relative to the vacuum level in the vdW GAP):
    E_m,i  = -(IP_i,f + EA_i,f)/2           face-resolved midgap (CNL) of the strained monolayer at the common lattice constant
    ΔE     = E_m2 - E_m1 ;  ΔE' = ΔE - eVh  (eq. 1)
    sLR    : eVh = (1-S) ΔE                 (eq. 5)          α_eff = (1-S)/S
    gLR    : eVh = α_eff ΔE' + sgn(ΔE) γ (|ΔE'| - Eg_sum/2) Θ(E_broken)   (eq. 1 of the paper), solved self-consistently; Θ ~ ½tanh(E/η)+½
             γ = e² d_int/(ε0 π ħ²) · 2|m_v||m_c|/(|m_v|+|m_c|)   (m_v of the layer with higher E_m, m_c of the layer with lower E_m)
             option: multiply the DOS mass by the valley degeneracy g
Two DFT measures of eVh (relaxed heterostructure):
    (A) band-resolved : eVh_A = ΔE - (E'_m2 - E'_m1)_bilayer
    (B) electrostatic: eVh_B = (s1 + s2) - ΔΦ_bi ; s_i = intrinsic vacuum step of layer i oriented as in the bilayer
    The A–B difference measures hybridization / non-rigid behaviour.
"""
import numpy as np
from scipy.optimize import brentq, least_squares
GAMMA0 = 1.602176634e-19 ** 2 * 9.1093837015e-31 * 1e-10 / (8.8541878128e-12 * np.pi * 1.054571817e-34 ** 2)  # ≈7.56 /(Å·m_e)


def gamma(d_int, mv, mc, gv=1, gc=1):
    Dv, Dc = gv * abs(mv), gc * abs(mc)
    return GAMMA0 * d_int * 2 * Dv * Dc / (Dv + Dc)


def theta(x, eta):
    return 0.5 * np.tanh(x / eta) + 0.5 if eta > 0 else float(x > 0)


def eVh_glr(dE, egsum, S, gam, eta=0.01):
    """Self-consistent solution of the gLR equation. dE = E_m2 - E_m1 (eV)."""
    if abs(dE) < 1e-9: return 0.0
    a = (1 - S) / S; sg = np.sign(dE)
    f = lambda x: x - (a * (dE - x) + sg * gam * (abs(dE - x) - egsum / 2) * theta(abs(dE - x) - egsum / 2, eta))
    lo, hi = sorted((0.0, dE))
    try:
        return brentq(f, lo, hi, xtol=1e-7)
    except ValueError:                      # numerical edge case (smooth θ, very small Eg) -> widen the bracket, then closed form
        try:
            return brentq(f, lo - abs(dE), hi + abs(dE), xtol=1e-7)
        except ValueError:
            return eVh_closed(dE, egsum, S, gam)


def eVh_closed(dE, egsum, S, gam):
    """Closed form, for checking."""
    a = (1 - S) / S; x = a / (1 + a) * dE
    if abs(dE - x) - egsum / 2 > 0:
        x = (a + gam) / (a + gam + 1) * dE - np.sign(dE) * gam / (2 * (a + gam + 1)) * egsum
    return x


def predict(dE, eg1, eg2, model, S=1.0, gam=None, eta=0.01):
    if model == "anderson": v = np.zeros_like(dE)
    elif model == "midgap": v = dE.copy()
    elif model == "slr": v = (1 - S) * dE
    elif model == "glr": v = np.array([eVh_glr(d, g1 + g2, S, gm, eta) for d, g1, g2, gm in zip(dE, eg1, eg2, gam)])
    dEp = dE - v                                   # E'_m2 - E'_m1
    v1, c1, v2, c2 = -eg1 / 2, eg1 / 2, dEp - eg2 / 2, dEp + eg2 / 2
    eg = np.minimum(c1, c2) - np.maximum(v1, v2)   # heterostructure gap (negative = broken gap)
    return v, eg, kind(v1, c1, v2, c2)


def kind(v1, c1, v2, c2):
    t = np.where((v1 <= v2) & (c1 >= c2) | (v2 <= v1) & (c2 >= c1), "I", "II")
    return np.where((np.maximum(v1, v2) > np.minimum(c1, c2)), "III", t)


def r2(y, f):
    y, f = np.asarray(y), np.asarray(f); return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def build_inputs(D, valley=False):
    """D needs: dE, Eg1, Eg2, mv1, mc1, mv2, mc2 (+ gv1,gc1,gv2,gc2). Returns γ(d_int) oriented by ΔE."""
    up = D.dE.values > 0
    mv = np.where(up, D.mv2, D.mv1); mc = np.where(up, D.mc1, D.mc2)
    gv = np.where(up, D.gv2, D.gv1) if valley else 1; gc = np.where(up, D.gc1, D.gc2) if valley else 1
    return lambda d: gamma(d, mv, mc, gv, gc)


def fit(D, target="eVh_A", model="glr", groups=None, valley=False, eta=0.01):
    """Fit S (and d_int for gLR) globally; groups (e.g. contact types 'XX','XQ','QQ') -> separate S per group (Janus extension)."""
    y = D[target].values; dE = D.dE.values; g1, g2 = D.Eg1.values, D.Eg2.values; G = build_inputs(D, valley)
    labs = np.zeros(len(D), int) if groups is None else D[groups].astype("category").cat.codes.values
    cats = [None] if groups is None else list(D[groups].astype("category").cat.categories); nS = len(cats)
    def res(p):
        S = p[:nS][labs]
        if model == "slr": f = (1 - S) * dE
        else:
            gm = G(p[nS]); f = np.array([eVh_glr(d, a + b, s, m, eta) for d, a, b, s, m in zip(dE, g1, g2, S, gm)])
        return f - y
    p0 = [0.85] * nS + ([1.2] if model == "glr" else []); lb = [0.05] * nS + ([0.05] if model == "glr" else [])
    ub = [1.0] * nS + ([10.0] if model == "glr" else [])
    o = least_squares(res, p0, bounds=(lb, ub)); f = res(o.x) + y
    out = dict(S=dict(zip([str(c) for c in cats], o.x[:nS])), r2=r2(y, f), mae=np.abs(f - y).mean(), pred=f)
    if model == "glr": out["d_int"] = o.x[nS]
    return out


if __name__ == "__main__":
    # self-test: closed form = self-consistent solution (η→0), and the fit recovers parameters from synthetic data
    rng = np.random.default_rng(0); n = 600
    D = __import__("pandas").DataFrame(dict(dE=rng.normal(0, 1.2, n), Eg1=rng.uniform(.1, 2, n), Eg2=rng.uniform(.1, 2, n),
                                            mv1=rng.uniform(.2, 1.5, n), mc1=rng.uniform(.2, 1, n), mv2=rng.uniform(.2, 1.5, n),
                                            mc2=rng.uniform(.2, 1, n), gv1=1, gc1=1, gv2=1, gc2=1))
    G = build_inputs(D)(1.18)
    e1 = np.array([eVh_glr(d, a + b, .88, g, eta=1e-5) for d, a, b, g in zip(D.dE, D.Eg1, D.Eg2, G)])
    e2 = np.array([eVh_closed(d, a + b, .88, g) for d, a, b, g in zip(D.dE, D.Eg1, D.Eg2, G)])
    print("max|self-consistent - closed form| = %.2e eV" % np.abs(e1 - e2).max(), "| type III (gLR):", (predict(D.dE.values, D.Eg1.values, D.Eg2.values, 'glr', .88, G)[2] == 'III').mean().round(3))
    D["eVh_A"] = np.array([eVh_glr(d, a + b, .88, g) for d, a, b, g in zip(D.dE, D.Eg1, D.Eg2, G)]) + rng.normal(0, .03, n)
    o = fit(D); print("gLR fit: S=%.3f d_int=%.3f Å r2=%.3f (true: 0.88, 1.18)" % (o["S"]["None"], o["d_int"], o["r2"]))
    o = fit(D, model="slr"); print("sLR fit: S=%.3f r2=%.3f" % (o["S"]["None"], o["r2"]))
