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
                      description="Inject FRBs into gaussian filterbank data")

  parser.add_option('--fnfil', dest='fnfil', default='./data/test.fil',\
                      help="input filterbank file", 
                    type=str)

  parser.add_option('--nfrb', dest='nfrb', type='int', \
                      help="Number of FRBs to inject.", \
                      default=1)

  parser.add_option('--outdir', dest='outdir', default='data/',\
                      help="directory to output .fil")

  parser.add_option('--upchan_factor', dest='upchan_factor', type='int', \
                      help="Upchannelize data by this factor before injecting. Rebin after.", \
                      default=2)

  parser.add_option('--upsamp_factor', dest='upsamp_factor', type='int', \
                      help="Upsample data by this factor before injecting. Downsample after.", \
                      default=2)

  parser.add_option('--simulator', dest='simulator', type='str', \
                      help="Either Liam Connor's inject_frb or Kendrick Smith's simpulse", \
                      default="injectfrb")


  options, args = parser.parse_args()

  fnfil = options.fnfil

  filhdr['rawdatafile'] = fnfil

--upchan_factor 1 --upsamp_factor 1 --simulator simpulse

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
  os.system('python inject_frb.py %s %s --nfrb %d --dm_list 10.0 \
            --calc_snr True --gaussian_noise --upchan_factor %d --upsamp_factor %d --simulator %s' \
            % (fnfil, options.outdir, options.nfrb, \
              options.upsamp_factor, options.upchan_factor, \
              options.simulator))





