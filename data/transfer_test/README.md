# Transfer test (Supplementary Note 8)
- `gen_selection_FROZEN.csv`, `gen_registration.json`: selection and pass criteria recorded before any DFT (sha256 0bcdd3babcfbbcd8...).
- `gen_mace_final.csv`: MACE-MP-0 stacking scan (24 interfaces rescanned on 1.6–4.4 Å; `rescanned` column).
- `mono_strain_gen.csv`: 116 monolayers at −1.5, 0, +1.5 % strain (PBE+D3(BJ), QE, SG15; band edges from the SCF 12×12 mesh).
- `gen_table.csv`: one row per interface (face-resolved ΔE_m, s1, s2, ΔΦ_bi, eVh_A, eVh_B, w_min, ...). G070: eVh_B from the planar-averaged potential.
- `gen_S_final.csv`: screening factors per family/contact class (all interfaces and semiconducting only), pair-bootstrap 95 % CI.
Scripts: `analysis/transfer_test/`.
