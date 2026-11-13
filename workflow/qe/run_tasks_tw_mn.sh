#!/bin/bash
#SBATCH --job-name=glr
#SBATCH --account=acd115225
#SBATCH --partition=ct224
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=56
#SBATCH --time=12:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
# MULTI-NODE VERSION (partitions requiring ≥57 cores/job): 2 nodes per job (or -N k), slots spread over all nodes.
# Taiwania 3: ONE node per job (original version), runs 56/CPT small tasks in parallel (default CPT=14 -> 4 tasks), pinning cores per slot.
# LIST is split over the job array: element i takes lines i, i+N, i+2N... (N = array size).
#   Test (ctest, 2 h):  sbatch -p ctest -t 02:00:00 --array=0 --export=ALL,LIST=test_list.txt glr_janus/run_tasks_tw.sh
#   Production:        sbatch --array=0-19 --export=ALL,LIST=all_list.txt glr_janus/run_tasks_tw.sh   (20 nodes in parallel)
# If time runs out, resubmit unchanged: directories with DONE (or post/DONE) are skipped. Early stop: touch STOP_glr
module purge
module load intel/2024 intelmpi/2021.11
module use ~/modulefiles; module load qe/7.5
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; ulimit -s unlimited
# Python with numpy + scipy: submit from a shell with a suitable environment (--export=ALL carries PATH), or set PYACT=<.../bin/activate>
[ -n "$PYACT" ] && source "$PYACT"
module load python/3.12.2 2>/dev/null
[ -n "$VIRTUAL_ENV" ] && export PATH=$VIRTUAL_ENV/bin:$PATH
python3 -c "import numpy, scipy" || { echo "numpy/scipy missing in python3"; exit 1; }
for x in pw.x pp.x average.x; do which $x >/dev/null || { echo "$x missing (build QE with 'make pw pp')"; exit 1; }; done
ROOT=${SLURM_SUBMIT_DIR:-$PWD}; cd $ROOT
export G=${G:-$ROOT/glr_janus} PSEUDO=${PSEUDO:-$ROOT/pseudo_xamq}
TASK=${TASK:-glr_janus/task_auto.sh}; CPT=${CPT:-14}; NPN=${SLURM_NTASKS_PER_NODE:-56}; SPN=$((NPN / CPT))
HOSTS=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); NSLOT=$((SPN * ${#HOSTS[@]}))
echo "node: ${HOSTS[*]} | khe: $NSLOT"
export NK=${NK:-2}
I=${SLURM_ARRAY_TASK_ID:-0}; N=${SLURM_ARRAY_TASK_COUNT:-1}
declare -A BUSY
run_one() {   # $1 directory, $2 slot
  local d=$1 s=$2 t0=$(date +%s)
  local h=${HOSTS[$((s / SPN))]} ls=$((s % SPN))
  local pin="-genv I_MPI_PIN_PROCESSOR_LIST=$((ls * CPT))-$((ls * CPT + CPT - 1))"; [ -n "$NOPIN" ] && pin=""
  export MPIRUN="${LAUNCH:-mpiexec.hydra} -n $CPT -hosts $h $pin"
  ( cd $ROOT/$d && bash $ROOT/$TASK < /dev/null > task.log 2>&1 ); rc=$?
  echo "$d rc=$rc $(( $(date +%s) - t0 ))s cores=$CPT host=$h" >> $ROOT/glr_time_tw.txt
}
while read -r d; do
  [ -f $ROOT/STOP_glr ] && break
  if [[ "$TASK" == *task_dos* ]]; then [ -f $ROOT/$d/DOS_DONE ] && continue
  elif [[ "$TASK" == *redo* ]]; then [ -f $ROOT/$d/post/REDO_DONE ] && continue
  else { [ -f $ROOT/$d/DONE ] || [ -f $ROOT/$d/post/DONE ]; } && continue; fi
  while :; do
    free=""; for s in $(seq 0 $((NSLOT - 1))); do p=${BUSY[$s]}; if [ -z "$p" ] || ! kill -0 $p 2>/dev/null; then free=$s; break; fi; done
    [ -n "$free" ] && break; wait -n
  done
  run_one "$d" "$free" & BUSY[$free]=$!
done < <(awk -v i=$I -v n=$N 'NF && (NR-1)%n==i' $LIST)
wait
