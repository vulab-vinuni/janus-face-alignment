# Direct HSE06+SOC check (Supplementary Note 8, Table 17)
VASP 6.5 vasp_ncl, settings of the HSE06 runs + LSORBIT, NBANDS doubled, preceded by PBE+SOC on the same 12x12 mesh.
Energies in eV; shifts relative to the mean vacuum level; err = additive (HSE06 + PBE+SOC - PBE) minus direct HSE06+SOC.
Run run_soc6.sh as a Slurm array from hse108_vasp/, then ana_soc6.py there.
