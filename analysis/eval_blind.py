"""Score the blind test: compare the FROZEN predictions (blind60_predictions_FROZEN.csv) with DFT (glr_table_blind.csv from assemble.py).
   python3 eval_blind.py blind60_predictions_FROZEN.csv glr_table_blind.csv"""
import sys, numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
P = pd.read_csv(sys.argv[1]); D = pd.read_csv(sys.argv[2])
M = P.merge(D[["id", "VBM1", "CBM1", "VBM2", "CBM2", "Eg_bi", "eVh_B", "eVh_A", "type_dft", "w_min", "d_int", "Eb_meV_A2"]], on="id", how="left")
print(f"with DFT: {M.type_dft.notna().sum()}/{len(M)}"); M = M[M.type_dft.notna()].copy()
v1 = -M.Eg1 / 2; v2 = M.dE - M.eVh_glr - M.Eg2 / 2
M["vbo_pred"] = v1 - v2; M["vbo_dft"] = M.VBM1 - M.VBM2
if "eVh_hyb" in M: print(f"MAE eVh: gLR {np.abs(M.eVh_B - M.eVh_glr).mean():.3f} | hybrid (gLR+GBR) {np.abs(M.eVh_B - M.eVh_hyb).mean():.3f} | type accuracy: gLR {(M.type_glr == M.type_dft).mean():.2f} hybrid {(M.type_hyb == M.type_dft).mean():.2f}")
M["ok_type"] = M.type_glr == M.type_dft; M["sbo_dft"] = ((M.type_dft != "I") & (M.Eg_bi.abs() < .15)).astype(int)
def rep(s, name):
    if len(s) == 0: return
    print(f"{name:22s} n={len(s):3d} | type accuracy {s.ok_type.mean():.2f} | MAE eVh {np.abs(s.eVh_B - s.eVh_glr).mean():.3f} | MAE VBO {np.abs(s.vbo_dft - s.vbo_pred).mean():.3f} | mean VBO bias {np.mean(s.vbo_pred - s.vbo_dft):+.3f} eV")
rep(M, "ALL"); rep(M[M.regime == "localized"], "predicted localized"); rep(M[M.regime == "hybridized"], "predicted hybridized")
for p, g in M.groupby("part"): rep(g, p)
for c, g in M.groupby("contact"): rep(g, "contact " + c)
F = M[M.part.str.startswith("F_")]; F["pair"] = ["|".join(sorted([a, b])) for a, b in zip(F.L1, F.L2)]
fl = F.groupby("pair").agg(n=("id", "size"), pred=("type_glr", "nunique"), dft=("type_dft", "nunique"))
fl = fl[fl.n == 2]; print(f"face-switch pairs: DFT switches type {int((fl.dft > 1).sum())}/{len(fl)}; model correct (switch / no switch) {int(((fl.pred > 1) == (fl.dft > 1)).sum())}/{len(fl)}")
if M.sbo_dft.nunique() > 1: print(f"SVM SBO: NA {balanced_accuracy_score(M.sbo_dft, M.SBO_svm):.2f} (SBO DFT: {M.sbo_dft.sum()})")
print(f"p_local calibration: fraction with w_min≥0.6 in predicted localized {(M[M.regime=='localized'].w_min >= .6).mean():.2f}, predicted hybridized {(M[M.regime=='hybridized'].w_min >= .6).mean():.2f}")
print("Preset criteria: localized type accuracy ≥0.85; MAE eVh ≤ ~0.11 eV (1.3×CV 0.083); face-switch pairs ≥8/10; mean VBO bias ≈ 0")
M.to_csv("blind_eval.csv", index=False)
