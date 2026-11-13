#!/bin/bash
#SBATCH --job-name=glr
#SBATCH --partition=terai
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=48
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# Generic dispatcher: runs small tasks in parallel, CPT cores each; several tasks per node, pinned cores per slot.
#   sbatch --export=ALL,LIST=mono_list.txt,TASK=task_mono.sh,CPT=12 run_tasks_rudra.sh        (monolayers: 4 slots/node)
#   sbatch --export=ALL,LIST=post_list.txt,TASK=task_bilayer.sh,CPT=24 run_tasks_rudra.sh     (bilayers: 2 slots/node)
# LIST: one directory per line (relative to the submit directory). Directories with DONE are skipped. Stop: touch STOP_glr
module use $HOME/modulefiles
module --ignore_cache load mpi/2021.13 tbb/2021.13 compiler-rt/2024.2.1 mkl/2024.2 intel_oneapi/2026.1 ifort/2024.2.1
module --ignore_cache load quantum-espresso/cpu/7.5
ulimit -s unlimited; export OMP_NUM_THREADS=1
ROOT=${SLURM_SUBMIT_DIR:-$PWD}; cd $ROOT
export G=${G:-$ROOT/glr_janus} PSEUDO=${PSEUDO:-$ROOT/pseudo_xamq}
CPT=${CPT:-12}; NPN=${SLURM_NTASKS_PER_NODE:-48}; NSLOT=$((NPN / CPT))
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
declare -A BUSY; SLOTS=()
for n in "${NODES[@]}"; do for s in $(seq 0 $((NSLOT - 1))); do SLOTS+=("$n:$s"); done; done
run_one() {   # $1 directory, $2 node:slot
  local d=$1 node=${2%:*} s=${2#*:} t0=$(date +%s)
  export MPIRUN="mpirun -np $CPT -ppn $CPT -hosts $node -genv I_MPI_PIN_PROCESSOR_LIST=$((s * CPT))-$((s * CPT + CPT - 1))"
  ( cd $ROOT/$d && bash $ROOT/$TASK < /dev/null > task.log 2>&1 ); rc=$?
  echo "$d rc=$rc $(( $(date +%s) - t0 ))s $2" >> $ROOT/glr_time_${TASK%.sh}.txt
}
while read -r d; do
  [ -z "$d" ] && continue; [ -f $ROOT/STOP_glr ] && break
  [ -f $ROOT/$d/DONE ] || [ -f $ROOT/$d/post/DONE ] && continue
  while :; do
    free=""; for sl in "${SLOTS[@]}"; do p=${BUSY[$sl]}; if [ -z "$p" ] || ! kill -0 $p 2>/dev/null; then free=$sl; break; fi; done
    [ -n "$free" ] && break; wait -n
  done
  run_one "$d" "$free" & BUSY[$free]=$!
done < $LIST
wait; echo "xong: $(grep -c 'rc=0' glr_time_${TASK%.sh}.txt) ok / $(wc -l < glr_time_${TASK%.sh}.txt)"
