"""
Liam Connor 26 April 2019

FRB-benchmarking tools 
Code to check if a given FRB guess matches the true 
DM and arrival times. 
"""
import sys

import numpy as np
import matplotlib.pylab as plt
from scipy import interpolate 
import optparse

import simulate_frb
#import simpulse 
import tools 

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
        sigdm = 5. + 0.025*self._dm
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

        return plt

    def find_parameter_guess(self, dm_arr, t_arr, snr_arr, 
                             dm_err=0.5, t_err=1.):
        """ The dm/time guess is generated by finding the 
        highest S/N event in a box around the true candidate 
        with dimensions dm_err by t_err
        """

        dm_stat = np.abs(1.-dm_arr/self._dm)
        t_stat = np.abs(self._t0 - t_arr)

        ind = np.where((dm_stat<dm_err) & (t_stat<t_err))[0]

        if len(ind)==0:
            return [],[],[]

        snr_max_ind = np.argmax(snr_arr[ind])

        return dm_arr[ind[snr_max_ind]], t_arr[ind[snr_max_ind]], snr_arr[ind[snr_max_ind]]


    def dm_time_contour_decision(self, dm_guess, t0_guess, thresh=0.1, 
                                 simulator='simpulse', dmtarr_function='box', t_err=0.1, dm_err=0.1):

        """ Submit DM/time guess for true FRB parameters using 
        one of three dmtarr_function contours (box, gaussian, bowtie). t_err and 
        dm_err apply for the box guess. 

        Method returns: boolean decision, dmtarr, dm/time extent list 
        """

        if type(dmtarr_function) is str:
            if dmtarr_function=='bowtie':
                dmtarr, dms, times, dmtarr_function = self.gen_dm_time_bowtie(simulator=simulator)
                extent = [times[0], times[-1], dms[-1], dms[0]]
            elif dmtarr_function=='gaussian':
                dmtarr, dms, times, dmtarr_function = self.gen_dm_time_gaussian()
                extent = [times[0], times[-1], dms[-1], dms[0]]
            elif dmtarr_function=='box':
                decision = self.dm_time_box_decision(dm_guess, t0_guess, dm_err=dm_err, t_err=t_err)
                extent = [t0_guess-t_err, t0_guess+t_err, dm_guess*(1+dm_err), dm_guess*(1-dm_err)]
                return decision, [], []

        val = dmtarr_function(t0_guess, dm_guess)
        decision = val > thresh

        return decision[0], dmtarr, extent

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
        t_err = t_err * (1 + self._width_i / 0.01)

        dm_stat = np.abs(1.-np.float(dm_guess)/self._dm)
        t_stat = np.abs(self._t0 - t0_guess)

        decision = (dm_stat < dm_err) & (t_stat < t_err)

        return decision

def get_decision_array(fn_truth, fn_cand, dmtarr_function='box', 
                       freq_ref_truth=1400., freq_ref_cand=1400., mk_plot=False):
    """ Step through each candidate in truth file and find set of triggers 
    in fn_cand within a box around true candidate. The highest S/N in that 
    box is the codes guess 
    """
    dm_truth, sig_truth, t_truth, w_truth = tools.read_singlepulse(fn_truth)
    dm_cand, sig_cand, t_cand, w_cand = tools.read_singlepulse(fn_cand)

    t_truth += 4148*dm_truth*(freq_ref_cand**-2 - freq_ref_truth**-2)

    decision_arr = []

    if dmtarr_function is 'box':
        # hack until i can plot both boxes
        mk_plot = False 

    for ii in range(len(dm_truth)):
        D = DetectionDecision(dm_truth[ii], t_truth[ii])
        dm_guess, t_guess, sig_guess = D.find_parameter_guess(dm_cand, 
                                                              t_cand, sig_cand, dm_err=0.75, t_err=2.0)

        if dm_guess==[]:
            decision_arr.append(0)
            continue

        dec_bool, dmtarr, extent = D.dm_time_contour_decision(dm_guess, t_guess,
                                                  simulator='injectfrb',
                                                  dmtarr_function=dmtarr_function)
        decision = 1+int(dec_bool)
        decision_arr.append(decision)
        
        if mk_plot:
            fig = plt.figure()
            ind = np.where( (np.abs(1-dm_cand/dm_truth[ii])<0.5) & (np.abs(t_cand-t_truth[ii])<5) )[0]
            times = np.linspace(t_truth[ii]-1.0, t_truth[ii]+1.0, 10)
            plt.fill_between(times, np.ones([10])*dm_truth[ii]*(1-0.2), np.ones([10])*dm_truth[ii]*1.2, alpha=0.15, color='C1')
            plt.scatter(t_truth[ii], dm_truth[ii], s=10, marker='*', color='red')
            plt.scatter(t_cand[ind], dm_cand[ind], sig_cand[ind], color='k', alpha=0.25)
            plt.scatter(t_guess, dm_guess, 20, color='C6', alpha=0.85, marker='s', edgecolor='k')
            plt.contour(dmtarr, 0.1, color='C0', extent=extent)
            plt.xlim(t_truth[ii]-5, t_truth[ii]+5)
            plt.ylim(dm_truth[ii]*0.5, dm_truth[ii]*1.5)
            plt.grid('on', alpha=0.5)
            plt.xlabel('Time [s]', fontsize=15)
            plt.ylabel('DM', fontsize=15)
            plt.title('Truth: DM=%0.2f  t=%0.2fs  S/N:%0.2f' % (dm_truth[ii], t_truth[ii], sig_truth[ii]))
            plt.savefig('DM:%0.2f_t0:%0.2f.pdf' % (dm_truth[ii], t_truth[ii]))
    
    decision_arr  = np.array(decision_arr)

    n_no_guess = np.sum(decision_arr==0)
    n_wrong_guess = np.sum(decision_arr==1)
    n_correct_guess = np.sum(decision_arr==2)
    ntrig = len(decision_arr)

    print("No guess: %d/%d" % (n_no_guess, ntrig))
    print("Wrong guess: %d/%d" % (n_wrong_guess, ntrig))
    print("Correct guess: %d/%d" % (n_correct_guess, ntrig))

    return decision_arr

if __name__=='__main__':

    def foo_callback(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))

    parser = optparse.OptionParser(prog="inject_frb.py", \
                        version="", \
                        usage="%prog FN_FILTERBANK OUTDIR [OPTIONS]", \
                        description="Inject FRBs into filterbank data")

    parser.add_option('--fn_cand_files', type='string', action='callback', callback=foo_callback)

    parser.add_option('--freq_ref_cand_files', dest='freq_ref_cand_files', type='str', \
                        help='Comma-separated list of reference frequencies for candidate files [MHz]', \
                        default='1400', callback=foo_callback, action='callback')

    parser.add_option('--freq_ref_truth', dest='freq_ref_truth', type='float', \
                        help='reference frequency of arrival times in truth file [MHz]', \
                        default=1400.)

    parser.add_option('--mk_plot', action='store_true',
                        help="Plot each candidate guess", default=False)

    parser.add_option('--dmtarr_function', dest='dmtarr_function', 
                        help="Function to determine DM/time boundary (box, gaussian, bowtie)", default='gaussian')

    parser.add_option('--fnout', dest='fnout', type='str', \
                        help="output text file", \
                        default='output.txt')

    options, args = parser.parse_args()
    fn_truth = args[0]

    fn_truth_arr = np.genfromtxt(fn_truth)
    ntrig = len(fn_truth_arr)

    header = 'DM     Sigma     Time (s)   Sample  Downfact  Width_intrins  With_obs  Spec_ind  Scat_tau_ref '
    fmt = '%5.3f    %3.2f    %5.5f    %9d    %d    %5f    %5f    %2f    %1.5f    '

    dec_arr_full = []    

    if len(options.fn_cand_files)!=len(options.freq_ref_cand_files):
        print("File/freq mismatch: Assuming all candidate reference frequencies are 1400 MHz")
        freq_ref_cand = 1400.*np.ones([len(options.fn_cand_files)])
    else:
        freq_ref_cand = np.array(options.freq_ref_cand_files).astype(float)


    for ii, fn_cand in enumerate(options.fn_cand_files):
        print("\nProcessing %s" % fn_cand)
        dec_arr = get_decision_array(fn_truth, fn_cand, dmtarr_function=options.dmtarr_function, 
                                    freq_ref_truth=options.freq_ref_truth,
                                    freq_ref_cand=freq_ref_cand[ii], mk_plot=options.mk_plot)

        dec_arr_full.append(dec_arr)

        header += 'code%d ' % ii
        fmt += '%5d    '

    dec_arr_full = np.concatenate(dec_arr_full).reshape(-1, ntrig).transpose()
    # Add the new result columns to truth txt file
    results_arr = np.concatenate([fn_truth_arr, dec_arr_full], axis=1)
    np.savetxt(options.fnout, results_arr, fmt=fmt, header=header)












