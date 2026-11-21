#!/bin/bash
# A1 round b: like glr_janus/task_bilayer.sh, but post cell CPOST=60 A (2D cutoff valid for separated layers),
# softer mixing and more SCF steps (one 45 A point did not converge in 100 steps).
set -o pipefail
export CPOST=${CPOST:-60}
python3 $G/bilayer_post.py write || exit 1
cd post && sed -i "s#pseudo_dir = .*#pseudo_dir = '$PSEUDO'#; s#mixing_beta = 0.3#mixing_beta = 0.2\n    electron_maxstep = 300#" scf.in
grep -q "JOB DONE" scf.out 2>/dev/null || $MPIRUN pw.x -nk ${NK:-4} -inp scf.in > scf.out 2> scf.err || exit 2
grep -q "convergence NOT achieved" scf.out && exit 3
pp.x < pp.in > pp.out 2>&1 && average.x < avg.in > avg.out 2>&1 || exit 4
cd .. && python3 $G/bilayer_post.py parse > post/parse.txt || exit 5
rm -rf post/tmp/b.save/wfc*.dat post/tmp/b.wfc* post/b.pdos* post/vtot.dat
touch post/DONE
