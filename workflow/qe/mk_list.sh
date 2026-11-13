#!/bin/bash
# Build the task list: pilot bilayers first (longer), monolayers afterwards.  bash glr_janus/mk_list.sh > all_list.txt
ls -d het108/qe/*__* 2>/dev/null
find MONO -name relax.in -printf '%h\n' 2>/dev/null | sort
