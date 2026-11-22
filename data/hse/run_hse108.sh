#!/bin/bash
#SBATCH -J hse108
#SBATCH -A acd115225
#SBATCH -p ct56
#SBATCH -N 1
#SBATCH --ntasks-per-node=56
#SBATCH -t 04:00:00
#SBATCH -o log/%x-%A_%a.out
#SBATCH -e log/%x-%A_%a.err
# one compound at a time on a full node; array task i takes compounds i, i+N, i+2N, ...
module purge; module load intel/2024 intelmpi/2021.11
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; ulimit -s unlimited
VASP=/work/u3830097/apps/vasp.6.5.0/bin/vasp_std
POT=/home/u3830097/potpaw_PBE.54
cd $SLURM_SUBMIT_DIR; R=$PWD
N=${SLURM_ARRAY_TASK_COUNT:-1}; I=${SLURM_ARRAY_TASK_ID:-0}
mapfile -t L < ${LIST:-list.txt}
for ((j=I; j<${#L[@]}; j+=N)); do
  n=${L[$j]}; d=$R/calc/$n; cd $d
  [ -f DONE ] && continue
  NB=$(cat nbands.txt)
  for e in $(cat potcars.txt); do cat $(ls $POT/$e/POTCAR* | head -1); done > POTCAR
  mkdir -p 1_pbe 2_hse
  cp POSCAR POTCAR $R/KPOINTS 1_pbe/; sed "s/__NB__/$NB/" $R/INCAR.pbe > 1_pbe/INCAR
  cd 1_pbe; t0=$(date +%s); mpiexec.hydra -n $SLURM_NTASKS $VASP > stdout.txt 2>&1; t1=$(date +%s); cd ..
  if ! grep -q "F=" 1_pbe/OSZICAR; then echo "$n PBE_FAIL $((t1-t0))" >> $R/hse108_time.txt; cd $R; continue; fi
  cp POSCAR POTCAR $R/KPOINTS 2_hse/; sed "s/__NB__/$NB/" $R/INCAR.hse > 2_hse/INCAR; mv 1_pbe/WAVECAR 2_hse/
  cd 2_hse; t2=$(date +%s); mpiexec.hydra -n $SLURM_NTASKS $VASP > stdout.txt 2>&1; t3=$(date +%s); rm -f WAVECAR CHG CHGCAR; cd ..
  if grep -q "F=" 2_hse/OSZICAR; then touch DONE; s=OK; else s=HSE_FAIL; fi
  echo "$n $s pbe=$((t1-t0))s hse=$((t3-t2))s nelm=$(grep -c 'DMP:' 2_hse/OSZICAR)" >> $R/hse108_time.txt
  cd $R
done
