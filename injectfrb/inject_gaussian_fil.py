#!/usr/bin/env python

import sys
import os
import time

import sigproc

fnfil = './test_new.fil'

filhdr = {'telescope_id': 10,
          'az_start': 0.0,
          'nbits': 8,
          'source_name': 'J1813-1749',
          'data_type': 1,
          'nchans': 1536,
          'machine_id': 15,
          'tsamp': 8.192e-05,
          'foff': -0.1953125,
          'src_raj': 181335.2,
          'src_dej': -174958.1,
          'tstart': 58523.3437492,
          'nbeams': 1,
          'fch1': 1549.8046875,
          'za_start': 0.0,
          'rawdatafile': fnfil,
          'nifs': 1,
          'nsamples': 7204148}

newhdr = ""
newhdr += sigproc.addto_hdr("HEADER_START", None)
for k,v in filhdr.items():
    newhdr += sigproc.addto_hdr(k, v)
newhdr += sigproc.addto_hdr("HEADER_END", None)
print "Writing new header to '%s'" % fnfil
outfile = open(fnfil, 'wb')
outfile.write(newhdr)

N_FRB = 25
SNR_MIN = 7

outdir = './data/'

if not os.path.isdir(outdir):
    os.mkdir(outdir)

timestr = time.strftime("%Y%m%d-%H%M")
os.system('python inject_frb.py %s %s --nfrb %d --dm_list 10.0 --calc_snr True --gaussian_noise' \
          % (fnfil, outdir, N_FRB))

