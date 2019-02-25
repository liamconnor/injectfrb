import os
import time

import glob

N_FRB = 500
SNR_MIN = 7
backend = 'PRESTO'
AMBER_PATH = '~/test/amber_arg.sh'

outdir = '/data/03/Triggers/injection/%s' % time.strftime("%Y%m%d")
outdir = '/data2/output/snr_tests_liam/%s' % time.strftime("%Y%m%d")
infile = '/data/03/Triggers/injection/sky_data_nofrb.fil'
infile = '/data2/output/snr_tests_liam/CB21.fil'
infile = '/data2/output/snr_tests_liam/backgrounddata_CB00.fil'
infile = '/data2/output/dm_time_generation/CB21.fil'

#infile = '/data2/output/20190121/2019-01-21-10:14:45.B1933+16/filterbank/CB24.fil'

if not os.path.isdir(outdir):
    os.mkdir(outdir)

timestr = time.strftime("%Y%m%d-%H%M")
os.system('python inject_frb.py %s %s --nfrb %d --dm_list 10.0 --calc_snr True --gaussian' % (infile, outdir, N_FRB))
