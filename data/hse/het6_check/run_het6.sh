#!/bin/bash
#SBATCH -J het6hse
#SBATCH -A acd115225
#SBATCH -p ct56
#SBATCH -N 1
#SBATCH --ntasks-per-node=56
#SBATCH -t 12:00:00
#SBATCH -o log/%x-%A_%a.out
#SBATCH -e log/%x-%A_%a.err
# Direct HSE06 (all six) and HSE06+SOC (soc3.txt) of DFT heterostructures. Array task i = line i of het6.txt.
# Same VASP, POTCARs and settings as hse108_vasp; POTCAR labels are taken from hse108_vasp/calc/<monolayer>/potcars.txt.
# Usage (from het6_check/):  MONO=/work/u3830097/3.Janus108/H2_het/hse_vasp/hse108_vasp sbatch --array=0-5 run_het6.sh
module purge; module load intel/2024 intelmpi/2021.11
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; ulimit -s unlimited
VS=/work/u3830097/apps/vasp.6.5.0/bin/vasp_std; VN=/work/u3830097/apps/vasp.6.5.0/bin/vasp_ncl
POT=/home/u3830097/potpaw_PBE.54; MONO=${MONO:?set MONO to the hse108_vasp directory}
cd $SLURM_SUBMIT_DIR; R=$PWD; mkdir -p log
mapfile -t L < het6.txt; n=${L[$SLURM_ARRAY_TASK_ID]}; d=$R/calc/$n; cd $d
# POTCAR: each species line of POSCAR (8 entries, layer 1 then layer 2) mapped to the label used for that element in the monolayers
read l1 l2 < layers.txt; : > potcars.txt
lab() { local m=$1 el=$2; paste <(sed -n 6p $MONO/calc/$m/POSCAR | tr -s ' ' '\n' | grep -v '^$') <(tr -s ' ' '\n' < $MONO/calc/$m/potcars.txt | grep -v '^$') | awk -v e=$el '$1==e{print $2; exit}'; }
i=0; for el in $(sed -n 6p POSCAR); do m=$l1; [ $i -ge 4 ] && m=$l2; p=$(lab $m $el); [ -z "$p" ] && { echo "no POTCAR label for $el in $m"; exit 1; }; echo $p >> potcars.txt; i=$((i+1)); done
for e in $(cat potcars.txt); do cat $(ls $POT/$e/POTCAR* | head -1); done > POTCAR
NEL=$(awk '/ZVAL/{z[++k]=$6} END{s=0; for(i=1;i<=k;i++) s+=z[i]; print s}' POTCAR)   # one atom per species block
NB=$(( (${NEL%.*} + 1) / 2 + 24 )); echo $NEL $NB > nbands.txt
run() { # $1 dir  $2 INCAR template  $3 nbands  $4 binary  $5 wavecar-from
  mkdir -p $1; cp POSCAR POTCAR $R/KPOINTS $1/; sed "s/__NB__/$3/" $R/$2 > $1/INCAR; [ -n "$5" ] && mv $5/WAVECAR $1/
  cd $1; t0=$(date +%s); mpiexec.hydra -n $SLURM_NTASKS $4 > stdout.txt 2>&1; t1=$(date +%s); cd ..
  ne=$(grep -c ':' $1/OSZICAR); nl=$(grep -o 'NELM *= *[0-9]*' $1/INCAR | grep -o '[0-9]*$'); grep -q "F=" $1/OSZICAR && [ $ne -lt $nl ] && s=OK || s=FAIL
  echo "$n $1 $s $((t1-t0))s nscf=$ne" >> $R/het6_time.txt; [ $s = OK ]; }
if [ ! -f DONE ]; then
  run 1_pbe INCAR.pbe$SUF $NB $VS "" && run 2_hse INCAR.hse$SUF $NB $VS 1_pbe && { rm -f 2_hse/WAVECAR 2_hse/CHG*; touch DONE; }
fi
if grep -qx $n $R/soc3.txt && [ ! -f DONE_SOC ]; then
  run 3_pbesoc INCAR.pbesoc$SUF $((2*NB)) $VN "" && run 4_hsesoc INCAR.hsesoc$SUF $((2*NB)) $VN 3_pbesoc && { rm -f 4_hsesoc/WAVECAR 4_hsesoc/CHG*; touch DONE_SOC; }
fi
rm -f */WAVECAR
