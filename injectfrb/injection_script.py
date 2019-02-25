import os
import time

import glob

N_FRB = 500
SNR_MIN = 7
backend = 'PRESTO'

infile = '/data1/output/snr_tests_liam/backgrounddata_CB00.fil'

#infile = '/data2/output/20190121/2019-01-21-10:14:45.B1933+16/filterbank/CB24.fil'

if not os.path.isdir(outdir):
    os.mkdir(outdir)

timestr = time.strftime("%Y%m%d-%H%M")
os.system('python inject_frb.py %s %s --nfrb %d --dm_list 10.0 --calc_snr True --gaussian' % (infile, outdir, N_FRB))
