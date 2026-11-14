# Contact-face-resolved band alignment in Janus van der Waals heterostructures

Data, models and scripts for the manuscript

> V. K. An, V. K. Dien, V. M. Canh, *The contact face controls band alignment and its predictability in Janus van der Waals heterostructures* (submitted).

The study covers 108 Janus X–A–M–Q monolayers (X = S, Se, Te; A = Al, Ga, In; M = Si, Ge, Sn, Pb; Q = P, As, Sb). It includes 832 explicit DFT heterostructure calculations, a contact-resolved linear-response model of the interface dipole with a machine-learned correction, a blind test whose predictions were frozen before the DFT calculations, and a confidence-resolved alignment map of 7 796 lattice-matched interfaces.

## Contents

| Path | Description |
|---|---|
| `data/monolayers/monolayers_strain_grid.csv` | 108 monolayers × 9 biaxial strains (−2 … +2 %): lattice constant, total energy, band edges, vacuum levels on the X and Q sides, face-resolved IP/EA, vacuum-level step `dEvac`, DOS effective masses |
| `data/monolayers/formation_energy.csv` | Formation energy per atom relative to the elemental ground states (PBE+D3(BJ)) and PBE gap |
| `data/heterostructures/het832_dft.csv` | All 832 DFT heterostructures: monolayer descriptors at the common lattice constant, stacking, layer-resolved band edges (`VBM1`, `CBM1`, `VBM2`, `CBM2`), vacuum step across the bilayer `dPhi_bi`, band-edge layer weight `w_min`, binding energy `Eb_meV_A2`, interface dipoles `eVh_A` (band-resolved) and `eVh_B` (electrostatic), DFT alignment type |
| `data/heterostructures/het832_relaxed.extxyz` | Relaxed geometries of the 832 heterostructures in their lowest-energy stacking (PBE+D3(BJ), fixed in-plane cell, c = 35 Å); `info` holds id, stack, layers, faces and contact type |
| `data/heterostructures/het_all_stackings_relaxed.extxyz` | All 1 099 relaxed configurations, including the six stackings of the pilot set and the blind-test interfaces (`lowest_stack`, `in_dft832` flags) |
| `data/heterostructures/het599_training_set.csv` | The 599 gapped heterostructures used for fitting, with out-of-fold predictions of each model |
| `data/predictions/interfaces7796_descriptors.csv` | Monolayer descriptors of all 7 796 lattice-matched face-resolved interfaces |
| `data/predictions/alignment_map7796.csv` | Predicted dipole, alignment type, band offsets, localization probability `p_local`, SBO class and confidence regime |
| `data/blind_test/blind60_predictions_FROZEN.csv` | Blind-test predictions recorded **before** the DFT calculations (not modified afterwards) |
| `data/blind_test/blind_eval.csv` | DFT results of the 59 completed blind interfaces and their evaluation |
| `data/phonons/` | Phonon summary (`phon108_summary.csv`; `status_raw` is the automatic label before reassigning near-Γ pockets with mixed polarisation) and dispersions (`bands/*.npz`, Γ–M–K–Γ) of the 108 monolayers from `xamq_F2.model` |
| `models/mace/` | Fine-tuned MACE potential used in the paper (`xamq_F2.model`; two-stage fine-tuning from MACE-MP-0 medium via `xamq_F.model`) and its data set (`train_F.xyz`, 9 453; `valid_F.xyz`, 1 093 configurations; extended XYZ with `REF_energy`, `REF_forces`, `REF_stress` from QE PBE). All configurations of 15 compounds, including SeInPbP and TeAlSnSb, are excluded. `archive_B330/` holds an earlier model for reference |
| `data/monolayers/monolayers108_relaxed.extxyz` | Relaxed unit cells of the 108 monolayers (PBE+D3(BJ), QE) |
| `models/model_frozen.json` | Frozen parameters of the contact-resolved model (S_XX, S_XQ, S_QQ, d_int) and classifier thresholds used in the blind test |
| `workflow/qe/` | Quantum ESPRESSO workflow: monolayer strain grid, effective masses, bilayer relaxation and post-processing (layer projection, electrostatic potential), result assembly |
| `analysis/` | Linear-response models, ML residual, localization and SBO classifiers, blind-test selection and evaluation, alignment-map prediction, MACE phonon and MD scripts |
| `figures/` | Scripts that generate the main and supplementary figures |

Notation: `f1`, `f2` are the faces in contact (X = chalcogen, Q = pnictogen); `contact` ∈ {XX, XQ, QQ}; energies in eV, lengths in Å.

## Requirements

Python ≥ 3.10 with numpy, pandas, scipy, scikit-learn and matplotlib; ASE, phonopy and MACE for the phonon scripts; Quantum ESPRESSO ≥ 7.2 for the workflow. Cluster paths in the shell scripts are given as `$WORK`, `$SCRATCH` and `$HOME` and must be adapted.

## Quick start

```bash
pip install numpy pandas scipy scikit-learn matplotlib
python analysis/predict_all.py      # alignment map from the frozen model
python analysis/eval_blind.py data/blind_test/blind60_predictions_FROZEN.csv data/blind_test/blind_eval.csv
```

## Licence

Code: MIT licence (`LICENSE`). Data: Creative Commons Attribution 4.0 (CC BY 4.0).

## Citation

Please cite the article above (details will be added on publication).

## Contact

Vuong Khiet An — an.kv@vinuni.edu.vn
HE³ Lab, Center for Materials Innovation and Technology (CMIT), VinUniversity, Hanoi, Vietnam
