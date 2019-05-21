import os
import sys

import numpy as np

fnfil = sys.argv[1]

try: 
	fntruth = sys.argv[2]
except:
	fntruth = fnfil.strip('.fil') + '.txt'

dm_min = 10.
dm_max = 3000.
outdir = './'
fnout = 'output'

heim_args = (fnfil, dm_min, dm_max)
heimdall_str = 'heimdall -v -f %s -dm %f %f -rfi_no_narrow -rfi_no_broad -output_dir /tmp/' % heim_args
heimdall_post_str = 'cat /tmp/*.cand > %s/%s.cand' % (outdir, fnout)

amber_str = 'python run_amber_args.py %s' % fnfil

fredda_str = 'cudafdmt -t 512 -d 16384 -x 6 -o %s/%s.fredda %s' % (outdir, fnout, fnfil)

print("\n==========Starting Heimdall==========")
os.system(heimdall_str)
os.system(heimdall_post_str)
print("\n==========Starting Amber==========")
os.system(amber_str)
print("\n==========Starting Fredda==========")
os.system(fredda_str)

print('\nStarting thing\n')
blind_detection_args = (fntruth, fnout, fnout, fnout, fnout)
blind_detection_str = 'python blind_detection.py %s --fn_cand_files %s.cand,%s.fredda,%s.fredda --mk_plot --fnout %s.results' % blind_detection_args
print(blind_detection_str)
os.system(blind_detection_str)
