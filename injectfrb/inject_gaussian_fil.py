#!/usr/bin/env python

import sys
import os
import time

import numpy as np
import optparse

from injectfrb import simulate_frb
from injectfrb import simulation_config

spec_ind_min = -4
spec_ind_max = 4
width_mean = 0.001

if __name__=='__main__':

  parser = optparse.OptionParser(prog="inject_frb.py", \
                      version="", \
                      usage="%prog FN_FILTERBANK OUTDIR [OPTIONS]", \
                      description="Inject FRBs into gaussian filterbank data")

  parser.add_option('--fnfil', dest='fnfil', default='injectfrb/data/test.fil',\
                      help="input filterbank file", 
                    type=str)

  parser.add_option('--nfrb', dest='nfrb', type='int', \
                      help="Number of FRBs to inject.", \
                      default=10)

  parser.add_option('--outdir', dest='outdir', default='data/',\
                      help="directory to output .fil")

  parser.add_option('--upchan_factor', dest='upchan_factor', type='int', \
                      help="Upchannelize data by this factor before injecting. Rebin after.", \
                      default=1)

  parser.add_option('--upsamp_factor', dest='upsamp_factor', type='int', \
                      help="Upsample data by this factor before injecting. Downsample after.", \
                      default=1)

  parser.add_option('--simulator', dest='simulator', type='str', \
                      help="Either Liam Connor's inject_frb or Kendrick Smith's simpulse", \
                      default="injectfrb")

  parser.add_option('--dm_min', dest='dm_min', default=10.,\
                      help="min dm to use, either float or tuple", 
                    type='float')

  parser.add_option('--dm_max', dest='dm_max', default=2000.,\
                      help="min dm to use, either float or tuple", 
                    type='float')

  parser.add_option('--fluence_min', dest='fluence_min', default=1.,\
                    help="fluence_min=1 is calibrated to S/N_min=10", 
                    type='float')

  parser.add_option('--calc_snr', dest='calc_snr', action='store_true',
                    help="write only Gaussian data to fil files", default=False)

  parser.add_option('--paramsfile', dest='paramsfile', default=None,\
                    help="txt file with parameters to simulate", 
                    type="str")

  options, args = parser.parse_args()

  fnfil = options.fnfil
  
  simulation_config.create_new_filterbank(fnfil)

  if not os.path.isfile(fnfil):
    print("Need either a test .fil file or sigproc")
    exit()

  if not os.path.isdir(options.outdir):
      os.mkdir(options.outdir)

  timestr = time.strftime("%Y%m%d-%H%M")
  if options.paramsfile is None:
    paramsfile = options.outdir + '/params%s.txt' % timestr
    
    ES = simulate_frb.EventSimulator()
    ES.draw_event_parameters_array(fluence_min=options.fluence_min, dm_min=options.dm_min, dm_max=options.dm_max, 
                                 nfrb=options.nfrb, spec_ind_min=spec_ind_min, spec_ind_max=spec_ind_max, width_mean=width_mean, 
                                 width_sig=1, fnout=paramsfile)
  else:
    paramsfile = options.paramsfile

  os.system('python injectfrb/inject_frb.py %s %s --nfrb %d --dm_list 10.0 \
            --gaussian_noise --upchan_factor %d \
            --upsamp_factor %d --simulator %s\
            --dm_low %f --dm_high %f --paramslist %s --calc_snr %s' \
            % (fnfil, options.outdir, options.nfrb, \
              options.upsamp_factor, options.upchan_factor, \
               options.simulator, options.dm_min, options.dm_max, paramsfile, options.calc_snr))





