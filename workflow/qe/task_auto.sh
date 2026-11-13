#!/bin/bash
# Combined task for a single job: detects the directory type
#   - MONO/<name>/e<ε>/ (contains meta.txt)      -> task_mono.sh (monolayer relaxation + potential + effective masses)
#   - het108/qe/<id>__<stack>/ (contains relax.in) -> bilayer relaxation followed by task_bilayer.sh (45 Å SCF + parse)
set -o pipefail
if [ -f meta.txt ]; then exec bash $G/task_mono.sh; fi
sed -i "s#pseudo_dir = .*#pseudo_dir = '$PSEUDO'#" relax.in
if ! grep -q "bfgs converged" relax.out 2>/dev/null; then
  $MPIRUN pw.x -nk ${NK:-4} -inp relax.in > relax.out 2> relax.err || exit 11
  grep -q "bfgs converged" relax.out || exit 12
  rm -rf tmp
fi
exec bash $G/task_bilayer.sh
