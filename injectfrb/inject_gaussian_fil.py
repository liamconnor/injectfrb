#!/usr/bin/env python

import sys
import os
import time

import numpy as np
import optparse

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
      'rawdatafile': '',
      'nifs': 1,
      'nsamples': 7204148}

if __name__=='__main__':

  parser = optparse.OptionParser(prog="inject_frb.py", \
                      version="", \
                      usage="%prog FN_FILTERBANK OUTDIR [OPTIONS]", \
                      description="Inject FRBs into filterbank data")

  parser.add_option('--fnfil', dest='fnfil', default=None,\
                      help="input filterbank file", 
                    type='float')

  parser.add_option('--nfrb', dest='nfrb', type='int', \
                      help="Number of FRBs to inject(Default: 50).", \
                      default=1)

  parser.add_option('--outdir', dest='outdir', default='data/',\
                      help="directory to output .fil")

  options, args = parser.parse_args()

  fnfil = options.fnfil

  filhdr['rawdatafile'] = fnfil

  try:
      import sigproc

      newhdr = ""
      newhdr += sigproc.addto_hdr("HEADER_START", None)
      for k,v in filhdr.items():
          newhdr += sigproc.addto_hdr(k, v)
      newhdr += sigproc.addto_hdr("HEADER_END", None)
      print "Writing new header to '%s'" % fnfil
      outfile = open(fnfil, 'wb')
      outfile.write(newhdr)
      spectrum = np.zeros([filhdr['nchans']], dtype=np.uint8)
      outfile.write(spectrum)
      outfile.close()
  except:
      print("Could not load sigproc")

  if not os.path.isfile(fnfil):
    print("Need either a test .fil file or sigproc")
    exit()

  outdir = 'data/'

  if not os.path.isdir(outdir):
      os.mkdir(outdir)

  timestr = time.strftime("%Y%m%d-%H%M")
  os.system('python inject_frb.py %s %s --nfrb %d --dm_list 10.0 --calc_snr True --gaussian_noise' \
            % (fnfil, options.outdir, options.nfrb))

