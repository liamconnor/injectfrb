import os
import sys

import numpy as np

fnfil = sys.argv[1]
dm_min = 10.
dm_max = 3000.
outdir = './'
fnout = 'output'

heim_args = (fnfil, dm_min, dm_max, outdir)
heimdall_str = 'heimdall -f %s -dm %f %f -rfi_no_narrow -rfi_no_broad -output_dir %s' % heim_args

amber_str = 'python run_amber_args.py %s' % fnfil

fredda_str = 'cudafdmt -t 512 -d 16384 -x 6 -o %s/%s.fredda %s' % (outdir, fnout, fnfil)

os.system(heimdall_str)
os.system(amber_str)
os.system(fredda_str)