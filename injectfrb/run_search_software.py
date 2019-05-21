import os
import sys

import numpy as np

fnfil = sys.argv[1]
dm_min = 10.
dm_max = 3000.
outdir = './'
fnout = 'output'

heim_args = (fnfil, dm_min, dm_max)
heimdall_str = 'heimdall -v -f %s -dm %f %f -rfi_no_narrow -rfi_no_broad -output_dir /tmp/' % heim_args
heimdall_post_str = 'cat /tmp/*.cand > %s/%s.cand' % (outdir, fnout)

amber_str = 'python run_amber_args.py %s' % fnfil

fredda_str = 'cudafdmt -t 512 -d 16384 -x 6 -o %s/%s.fredda %s' % (outdir, fnout, fnfil)

print("Starting Heimdall")
os.system(heimdall_str)
os.system(heimdall_post_str)
print("Starting Amber")
os.system(amber_str)
print("Starting Fredda")
os.system(fredda_str)