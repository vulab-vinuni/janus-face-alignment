#!/bin/bash
# One monolayer task (run in MONO/<name>/e<ε>): relax -> pp.x -> average.x -> effective-mass NSCF
# Environment variables: MPIRUN (full MPI command, e.g. "mpirun -np 12 -hosts cn001"), G (glr_janus directory), PSEUDO
set -o pipefail
sed -i "s#pseudo_dir = .*#pseudo_dir = '$PSEUDO'#" relax.in
grep -q "JOB DONE" relax.out 2>/dev/null || $MPIRUN pw.x -nk ${NK:-2} -inp relax.in > relax.out 2> relax.err || exit 1
grep -q "bfgs converged\|End of BFGS" relax.out || echo "WARNING: relaxation not converged" >&2
pp.x < pp.in > pp.out 2>&1 && average.x < avg.in > avg.out 2>&1 || exit 2
python3 $G/mono_mass.py write || exit 3
$MPIRUN pw.x -nk ${NK:-2} -inp nscf_mass.in > nscf_mass.out 2> nscf_mass.err && python3 $G/mono_mass.py parse > mass.txt || exit 4
rm -f tmp/m.save/wfc*.dat tmp/m.wfc* tmp/*.mix* vtot.dat      # keep data-file-schema.xml + charge density for checks
touch DONE
