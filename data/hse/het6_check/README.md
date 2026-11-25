# Direct HSE06 (+SOC) check on six heterostructures (Supplementary Note 8)
Tests whether the monolayer HSE06 (and SOC) corrections, applied to the layer-resolved PBE band edges of the
heterostructures with the PBE interface dipole, reproduce a direct hybrid-functional calculation of the bilayer.
Six DFT interfaces (lowest stacking, 1x1 cells, 8 atoms): XX localized H03752, XX hybridized H10072, XQ localized H00877,
XQ hybridized H03913, QQ H00855 (w_min 0.59) and QQ hybridized H23023; HSE06+SOC for H10072, H00877, H23023 (soc3.txt).
Settings as hse108_vasp (VASP 6.5, ENCUT 400, 12x12, dipole correction) plus LORBIT = 11 for layer projections.
1. python data/hse/het6_check/mk_het6.py   (already done: calc/<id>/POSCAR, layers.txt)
2. copy het6_check/ to Taiwania; MONO=<path to hse108_vasp> sbatch --array=0-5 run_het6.sh
3. python ana_het6.py  -> het6_direct_vs_monolayer.csv
