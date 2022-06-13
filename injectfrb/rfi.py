import sys

import numpy as np 
import pandas as pd
from joblib import Parallel, delayed

import reader

class RFI:

    def __init__(self, data, dumb_mask=[]):
        self.data = data
        self.nfreq, self.ntime = data.shape 
        self.dumb_mask = dumb_mask

    def apply_dumb_mask(self):
        if len(self.dumb_mask):
            self.data[dumb_mask] = 0.0

    def remove_bandpass_Tsys(self):
        """ Remove bandpass based on system temperature
        """
        T_sys = np.mean(self.data, axis=1)
        bad_chans = T_sys < 0.001 * np.median(T_sys)
        T_sys[bad_chans] = 1
        self.data /= T_sys[:,None]
        self.data[bad_chans,:] = 0

    def per_channel_sigmacut(self, frebin=1, sigma_thresh=3):
        if frebin!=1:
            data_rb = self.data.reshape(self.nfreq//frebin, frebin, self.ntime)
            data_rb = data_rb.mean(1)
        else:
            data_rb = self.data

        for ii in range(data_rb.shape[0]):
            sig_ii = np.std(data_rb[ii])
            dmed_ii = np.median(data_rb[ii])
            bad_samp = np.where(data_rb[ii]>dmed_ii + sigma_thresh*sig_ii)[0]
            self.data[frebin*ii:frebin*(ii+1), bad_samp] = dmed_ii

    def per_channel_sigmacut_mproc(self, data, frebin=1, sigma_thresh=3):
        data_rb = data

        for ii in range(data_rb.shape[0]):
            sig_ii = np.std(data_rb[ii])
            dmed_ii = np.median(data_rb[ii])
            bad_samp = np.where(data_rb[ii]>dmed_ii + sigma_thresh*sig_ii)[0]
            data_rb[frebin*ii:frebin*(ii+1), bad_samp] = dmed_ii

        return data_rb

    def per_sample_sigmacut(self, sigma_thresh=3):
        pass 

    def mask_bad_channels(self):
        pass 

    def dm_zero_filter(self, sigma_thresh=7.0):
        dmzero = np.mean(self.data,0)
        dmzero = dmzero - np.median(dmzero)
        stdev = 1.4826 * pd.Series(dmzero).mad()

        # Find DM=0 outliers 
        bad_samp = np.where(np.abs(dmzero) > sigma_thresh*stdev)[0]
        data_replace = np.mean(self.data, 1).repeat(len(bad_samp))

        # Replace bad samples with mean spectrum 
        self.data[:, bad_samp] = data_replace.reshape(self.nfreq, 
                                                      len(bad_samp))

def apply_rfi_filters(data, sigma_thresh_chan=3.0,
                    sigma_thresh_dm0=7., ncpu=20):
    R = RFI(data)
    R.remove_bandpass_Tsys()
#    R.per_channel_sigmacut(1, sigma_thresh_chan)
    R.dm_zero_filter(sigma_thresh_dm0)

    R.data = Parallel(n_jobs=20)(delayed(R.per_channel_sigmacut_mproc)(R.data[9*ii:9*(ii+1)],) for ii in range(ncpu))

    return R.data

if __name__=='__main__':
    fn_fil = sys.argv[1]
    fn_out_fil = sys.argv[2]
    chunksize = 2**15
    
    for ii in range(int(1e8)):
        data_fil_obj, freq_arr, dt, header = reader.read_fil_data(fn_fil, 
                                            start=ii*chunksize, stop=chunksize)
        if data_fil_obj.data.shape[1]==0:
            break

        data = apply_rfi_filters(data_fil_obj.data)
        if ii==0:
            reader.write_to_fil(np.zeros([header['nchans'], 0]), header, fn_out_fil)

        fil_obj = reader.filterbank.FilterbankFile(fn_out_fil, mode='readwrite')
        fil_obj.append_spectra(data.transpose())

    if data_fil_obj.data.shape[1]!=0:
        print("Did not reach end of file, maybe crank up loop range")













