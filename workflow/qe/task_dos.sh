#!/bin/bash
# DOS masses (run in a relaxed MONO/<name>/e+0.0): dense-mesh NSCF (cg) -> dos_mass.json
set -o pipefail
python3 $G/mono_dos.py write --k ${KDOS:-48} || exit 1
sed -i "s#pseudo_dir = .*#pseudo_dir = '$PSEUDO'#" nscf_dos.in
$MPIRUN pw.x -nk ${NK:-2} -inp nscf_dos.in > nscf_dos.out 2> nscf_dos.err || exit 2
python3 $G/mono_dos.py parse > dos.txt || exit 3
rm -f tmp/m.save/wfc*.dat tmp/m.wfc*
touch DOS_DONE
