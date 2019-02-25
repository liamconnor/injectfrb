#!/usr/bin/env python

import os
import time

import glob

N_FRB = 25
SNR_MIN = 7
backend = 'PRESTO'

infile = '/data1/output/snr_tests_liam/backgrounddata_CB00.fil'
outdir = '/data1/output/snr_tests_liam/'

if not os.path.isdir(outdir):
    os.mkdir(outdir)

timestr = time.strftime("%Y%m%d-%H%M")
os.system('python inject_frb.py %s %s --nfrb %d --dm_list 10.0 --calc_snr True --gaussian_noise' \
          % (infile, outdir, N_FRB))
