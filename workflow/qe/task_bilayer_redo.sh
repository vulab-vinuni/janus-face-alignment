#!/bin/bash
# Recompute layer weights for bilayers with an existing post/ (SCF done, wavefunctions deleted): NSCF on the same k-mesh from the saved density -> new parse
set -o pipefail
cd post || exit 1
[ -f layerw.npz ] || {
  sed -e "s/calculation = 'scf'/calculation = 'nscf'/" -e "s#pseudo_dir = .*#pseudo_dir = '$PSEUDO'#" scf.in > nscf_w.in
  grep -q "diagonalization" nscf_w.in || sed -i "s/    mixing_mode = 'local-TF'/&\n    diagonalization = 'cg'/" nscf_w.in
  $MPIRUN pw.x -nk ${NK:-2} -inp nscf_w.in > nscf_w.out 2> nscf_w.err || exit 2
}
cd .. && python3 $G/bilayer_post.py parse > post/parse.txt || exit 3
rm -f post/tmp/b.save/wfc*.dat post/tmp/b.wfc*
touch post/REDO_DONE
