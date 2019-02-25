import os

import inject_frb

fn_fil = 'data/test.fil'
fn_fil_out = 'data/'

print("Reading in test.fil to verify filterbank reader/writer works")

inject_frb.inject_in_filterbank(fn_fil, fn_fil_out, N_FRB=10,
                                NTIME=2**15, rfi_clean=False,
                                calc_snr=False, start=0,
                                dm=0, gaussian=True, 
                                gaussian_noise=False)