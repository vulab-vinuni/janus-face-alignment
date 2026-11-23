#!/bin/bash
#SBATCH -J hsesoc6
#SBATCH -A acd115225
#SBATCH -p ct56
#SBATCH -N 1
#SBATCH --ntasks-per-node=56
#SBATCH -t 06:00:00
#SBATCH -o log/%x-%A_%a.out
#SBATCH -e log/%x-%A_%a.err
# run from hse108_vasp/ ; array task i = compound i of soc6.txt ; uses calc/<name>/POSCAR, potcars.txt, nbands.txt
module purge; module load intel/2024 intelmpi/2021.11
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; ulimit -s unlimited
VASP=/work/u3830097/apps/vasp.6.5.0/bin/vasp_ncl
POT=/home/u3830097/potpaw_PBE.54
cd $SLURM_SUBMIT_DIR; R=$PWD
mapfile -t L < soc6.txt; n=${L[$SLURM_ARRAY_TASK_ID]}; d=$R/calc/$n; cd $d
[ -f DONE_SOC ] && exit 0
NB=$(( $(cat nbands.txt) * 2 ))
for e in $(cat potcars.txt); do cat $(ls $POT/$e/POTCAR* | head -1); done > POTCAR
mkdir -p 3_pbesoc 4_hsesoc
cp POSCAR POTCAR $R/KPOINTS 3_pbesoc/; sed "s/__NB__/$NB/" $R/INCAR.pbesoc > 3_pbesoc/INCAR
cd 3_pbesoc; t0=$(date +%s); mpiexec.hydra -n $SLURM_NTASKS $VASP > stdout.txt 2>&1; t1=$(date +%s); cd ..
grep -q "F=" 3_pbesoc/OSZICAR || { echo "$n PBESOC_FAIL $((t1-t0))s" >> $R/soc6_time.txt; exit 1; }
cp POSCAR POTCAR $R/KPOINTS 4_hsesoc/; sed "s/__NB__/$NB/" $R/INCAR.hsesoc > 4_hsesoc/INCAR; mv 3_pbesoc/WAVECAR 4_hsesoc/
cd 4_hsesoc; t2=$(date +%s); mpiexec.hydra -n $SLURM_NTASKS $VASP > stdout.txt 2>&1; t3=$(date +%s); rm -f WAVECAR CHG CHGCAR; cd ..
n_el=$(grep -c 'DMP:' 4_hsesoc/OSZICAR)
if grep -q "F=" 4_hsesoc/OSZICAR && [ $n_el -lt 140 ]; then touch DONE_SOC; s=OK; else s=HSESOC_FAIL; fi
echo "$n $s pbesoc=$((t1-t0))s hsesoc=$((t3-t2))s nelm=$n_el mem=$(grep 'Maximum memory used' 4_hsesoc/OUTCAR | awk '{print $NF}')kb" >> $R/soc6_time.txt
