"""
Liam Connor 26 April 2019

FRB-benchmarking tools 
Code to check if a given FRB guess matches the true 
DM and arrival times. 
"""

import numpy as np
import matplotlib.pylab as plt
from scipy import interpolate 

import simulate_frb
#import simpulse 
import tools 

class CompareInjectors:
    """ Class to generate pulses with both simpulse 
    and injectfrb packages. 

    Computes correlation coefficient between two 
    pulse profiles or spectra.
    """


    def __init__(self, nfreq=1024, fluence=1, width=0.001,
                 dm=100., dt=0.001, freq=(1550, 1250), spec_ind=0.,
                 scat_tau_ref=0., freq_ref=np.inf):
        self.nfreq = nfreq
        self.fluence = fluence 
        self.dm = dm
        self.dt = dt
        self.width = width 
        self.freq_hi_MHz, self.freq_lo_MHz = freq
        self.scat_tau_ref = scat_tau_ref
        self.spec_ind = spec_ind
        self.freq_ref = freq_ref
        self.freq_arr = np.linspace(freq[0], freq[1], nfreq)

    def gen_injfrb_pulse(self, upchan_factor=1, upsamp_factor=1, conv_dmsmear=False):
        """ Generate pulse dynamic spectrum 
        with injectfrb.simulate_frb
        """
        data_bg = np.zeros([upchan_factor*self.nfreq, upsamp_factor*self.ntime])
        data_injfrb, p = simulate_frb.gen_simulated_frb(NFREQ=upchan_factor*self.nfreq, 
                                     NTIME=upsamp_factor*self.ntime,
                                     sim=True, fluence=self.fluence, 
                                     spec_ind=self.spec_ind, width=self.width, dm=self.dm, 
                                     background_noise=data_bg,
                                     delta_t=self.dt/upsamp_factor, plot_burst=False, 
                                     freq=(self.freq_hi_MHz, self.freq_lo_MHz), 
                                     FREQ_REF=self.freq_ref, scintillate=False, 
                                     scat_tau_ref=self.scat_tau_ref, 
                                     disp_ind=2.0, conv_dmsmear=conv_dmsmear)

        data_injfrb = data_injfrb.reshape(self.nfreq, upchan_factor, self.ntime, upsamp_factor)
        data_injfrb = data_injfrb.mean(1).mean(-1)

        return data_injfrb

    def gen_simpulse(self):
        """ Generate pulse dynamic spectrum 
        with simpulse 
        """
        undispersed_arrival_time = 0.5*self.ntime*self.dt 
#        undispersed_arrival_time -= 4148*self.dm*(self.freq_hi_MHz**-2)
        sm = self.scat_tau_ref
        sp = simpulse.single_pulse(self.ntime, self.nfreq, self.freq_lo_MHz, self.freq_hi_MHz,
                           self.dm, sm, self.width, self.fluence,
                           self.spec_ind, undispersed_arrival_time)

        data_simpulse = np.zeros([self.nfreq, self.ntime])
        sp.add_to_timestream(data_simpulse, 0.0, self.ntime*self.dt)
        data_simpulse = data_simpulse[::-1]

        return data_simpulse


class DetectionDecision():
    """ Class to decide if an FRB has been 
    detected or not. Each method 
    compares the true pulse parameters 
    with the 'guess' and makes a decision. 
    """

    def __init__(self, dm, t0, width_i=0.001, 
                 spec_ind=0., freq_ref=None, scat_tau_ref=0.0,
                 freq=(1550, 1250), nfreq=1536, fluence=1, dt=8.192e-5):

        self._dm = dm
        self._t0 = t0
        self._width_i = width_i
        self._spec_ind = spec_ind
        self._freq_ref = freq_ref

        if freq_ref is None:
            self._freq_ref = 0.5*(freq[0]+freq[-1])

        self._scat_tau_ref = scat_tau_ref
        self._freqs = np.linspace(freq[0], freq[-1], nfreq)
        self._bw = self._freqs.max() - self._freqs.min()
        self._delta_freq = self._bw / nfreq
        self._nfreq = nfreq
        self._freq_hi_MHz, self._freq_lo_MHz = freq
        self._fluence = fluence 
        self._dt = dt 

    def gen_simpulse(self, ntime=10000):
        """ Generate pulse dynamic spectrum 
        with simpulse 
        """

        undispersed_arrival_time = 0.5*ntime*self.dt 
#        undispersed_arrival_time -= 4148*self.dm*(self.freq_hi_MHz**-2)
        sp = simpulse.single_pulse(ntime, self._nfreq, self._freq_lo_MHz, self._freq_hi_MHz,
                           self._dm, self._scat_tau_ref, self._width_i, self._fluence,
                           self.spec_ind, undispersed_arrival_time)

        data_simpulse = np.zeros([self._nfreq, ntime])
        sp.add_to_timestream(data_simpulse, 0.0, ntime*self._dt)
        data_simpulse = data_simpulse[::-1]

        return data_simpulse

    def gen_injfrb_pulse(self, ntime=10000, upchan_factor=1, upsamp_factor=1, conv_dmsmear=False):
        """ Generate pulse dynamic spectrum 
        with injectfrb.simulate_frb
        """
        data_bg = np.zeros([upchan_factor*self._nfreq, upsamp_factor*ntime])
        data_injfrb, p = simulate_frb.gen_simulated_frb(NFREQ=upchan_factor*self._nfreq, 
                                     NTIME=upsamp_factor*ntime,
                                     sim=True, fluence=self._fluence, 
                                     spec_ind=self._spec_ind, width=self._width_i, dm=self._dm, 
                                     background_noise=data_bg,
                                     delta_t=self._dt/upsamp_factor, plot_burst=False, 
                                     freq=(self._freq_hi_MHz, self._freq_lo_MHz), 
                                     FREQ_REF=self._freq_ref, scintillate=False, 
                                     scat_tau_ref=self._scat_tau_ref, 
                                     disp_ind=2.0, conv_dmsmear=conv_dmsmear)

        data_injfrb = data_injfrb.reshape(self._nfreq, upchan_factor, ntime, upsamp_factor)
        data_injfrb = data_injfrb.mean(1).mean(-1)

        return data_injfrb

    def gen_dm_time_gaussian(self):
        sigdm = 0.1*self._dm
        sigt = 0.1*(1 + self._width_i/0.001) # scale for pulse width, min 0.1 sec 
        ntime = 1000
        ndm = 1000

        times = np.linspace(0, 100*self._dt*ntime, ntime)
        times -= 0.5*ntime*self._dt*100
        times += self._t0
        dms = np.linspace(0.*self._dm, 2*self._dm, ndm)

        dmtarr = np.exp(-0.5*(self._t0-times[None])**2/sigt**2-0.5*(self._dm-dms[:,None])**2/sigdm**2)
        dmtarr_function = interpolate.interp2d(times, dms, dmtarr)

        return dmtarr, dms, times, dmtarr_function

    def gen_dm_time_bowtie(self, simulator='simpulse'):
        if simulator is 'injectfrb':
            data = self.gen_injfrb_pulse(ntime=10000, upchan_factor=1, 
                                         upsamp_factor=1, conv_dmsmear=False)
        elif simulator is 'simpulse':
            data = self.gen_simpulse(ntime=10000)
        else:
            print("Expected either (injectfrb, simpulse)")
            return 

        freq_ref = 0.5*(self._freqs[0]+self._freqs[-1])

        data_event_copy = data.copy()

        data_event_copy = tools.dedisperse(data_event_copy, self._dm, self._dt, 
                                           freq=(self._freqs[0],self._freqs[-1]), 
                                           freq_ref=freq_ref)

        mm = np.argmax(data_event_copy.mean(0))             
        data_event_copy = data_event_copy[:, mm-500:mm+500]                                                                        
        dmtarr, dms, times = tools.dm_transform(data_event_copy, self._freqs, dt=self._dt,                                                                           
                                    freq_ref=freq_ref,                                                                       
                                    dm_min=-50, dm_max=50, ndm=50)                                                                 
        times += self._t0
        dms += self._dm

        dmtarr /= dmtarr.max()
        dmtarr_function = interpolate.interp2d(times, dms, dmtarr)

        return dmtarr, dms, times, dmtarr_function

    def plot_dm_time_boxes(self, dm_arr, t_arr, t_err=0.5, width=None):
        ntrig = len(dm_arr)
        dm_err = 0.1*np.ones([ntrig])

        if width is not None:
            t_err += 0.1*width/0.001
            dm_err = 0.1*dm_arr #30.18*1e3*width*(self._freqs.mean()/1400.)**-3
            dm_low = dm_arr - dm_err
            dm_high = dm_arr + dm_err
        else:
            t_err = np.ones([ntrig])*t_err
            dm_low = ((1-dm_err)*dm_arr)
            dm_high = ((1+dm_err)*dm_arr)            

        fig = plt.figure()
        plt.scatter(t_arr, dm_arr, s=5, facecolor='none', edgecolors='k')

        for ii in range(ntrig):
            times = np.linspace(t_arr[ii]-t_err[ii], t_arr[ii]+t_err[ii], 10)

            plt.fill_between(times, dm_low[ii].repeat(10), dm_high[ii].repeat(10), alpha=0.4)

        plt.xlabel('Time [s]', fontsize=15)
        plt.ylabel('DM [pc cm**-3]', fontsize=15)
        plt.show()


    def find_parameter_guess(self, dm_arr, t_arr, snr_arr, 
                             dm_err=0.5, t_err=1.):

        dm_stat = np.abs(1.-dm_arr/self._dm)
        t_stat = np.abs(self._t0 - t_arr)

        ind = np.where((dm_stat<dm_err) & (t_stat<t_err))[0]

        if len(ind)==0:
            return [],[],[]

        snr_max_ind = np.argmax(snr_arr[ind])

        return dm_arr[ind[snr_max_ind]], t_arr[ind[snr_max_ind]], snr_arr[ind[snr_max_ind]]


    def dm_time_contour_decision(self, dm_guess, t0_guess, thresh=0.1, 
                                 simulator='simpulse', dmtarr_function='box'):

        if type(dmtarr_function) is str:
            if dmtarr_function=='bowtie':
                dmtarr, dms, times, dmtarr_function = self.gen_dm_time_bowtie(simulator=simulator)
            elif dmtarr_function=='gaussian':
                dmtarr, dms, times, dmtarr_function = self.gen_dm_time_gaussian()
            elif dmtarr_function=='box':
                decision = self.dm_time_box_decision(dm_guess, t0_guess)
                return decision

        val = dmtarr_function(t0_guess, dm_guess)
        decision = val > thresh

        return decision[0]

    def dm_time_box_decision(self, dm_guess, t0_guess, 
                    dm_err=0.1, t_err=0.1):
        """ method to test if parameter 
        guess is within the allowed DM_time_box

        Parameters
        ----------
        dm_guess : 
            guess of FRB's dispersion measure 
        t0_guess : 
            guess of FRB's arrival time 
        dm_err : 
            allowed fractional error on DM 
        t_err : 
            allowed arrival time error in seconds 
        """
        t_err += self._width_i / 0.001

        dm_stat = np.abs(1.-np.float(dm_guess)/self._dm)
        t_stat = np.abs(self._t0 - t0_guess)

        decision = (dm_stat < dm_err) & (t_stat < t_err)

        return decision

def get_decision_array(fn_truth, fn_cand, dmtarr_function='box'):
    dm_truth, sig_truth, t_truth, w_truth = tools.read_singlepulse(fn_truth)
    dm_cand, sig_cand, t_cand, w_cand = tools.read_singlepulse(fn_cand)

    decision_arr = []

    for ii in range(len(dm_truth)):
        D = DetectionDecision(dm_truth[ii], t_truth[ii])
        dm_guess, t_guess, sig_guess = D.find_parameter_guess(dm_cand, 
                                        t_cand, sig_cand, dm_err=0.5, t_err=1.)

        if dm_guess==[]:
            decision_arr.append(0)
            continue

        decision = 1+int(D.dm_time_contour_decision(dm_guess, t_guess, 
                                                  simulator='injectfrb', 
                                                  dmtarr_function=dmtarr_function))
        decision_arr.append(decision)
    
    decision_arr  = np.array(decision_arr)

    n_no_guess = np.sum(decision_arr==0)
    n_wrong_guess = np.sum(decision_arr==1)
    n_correct_guess = np.sum(decision_arr==2)
    ntrig = len(decision_arr)

    print("No guess: %d/%d" % (n_no_guess, ntrig))
    print("Wrong guess: %d/%d" % (n_wrong_guess, ntrig))
    print("Correct guess: %d/%d" % (n_correct_guess, ntrig))

    return decision_arr













































