#!/bin/bash
# One bilayer post-processing task (run in het108/qe/<id>__<stack> after relaxation): SCF (c = 45 Å cell) -> pp/average -> parse (layer weights from wfcN.dat)
set -o pipefail
python3 $G/bilayer_post.py write ${GLR_TEST:+--test} || exit 1
cd post && sed -i "s#pseudo_dir = .*#pseudo_dir = '$PSEUDO'#" scf.in
grep -q "JOB DONE" scf.out 2>/dev/null || $MPIRUN pw.x -nk ${NK:-4} -inp scf.in > scf.out 2> scf.err || exit 2
# projwfc not used: SG15 has no PP_PSWFC -> layer weights are computed from wfcN.dat in bilayer_post.py parse
pp.x < pp.in > pp.out 2>&1 && average.x < avg.in > avg.out 2>&1 || exit 4
cd .. && python3 $G/bilayer_post.py parse > post/parse.txt || exit 5
rm -rf post/tmp/b.save/wfc*.dat post/tmp/b.wfc* post/b.pdos* post/vtot.dat
touch post/DONE
