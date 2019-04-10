import numpy as np

import matplotlib 
#matplotlib.use('Agg', warn=False)
import matplotlib.pyplot as plt

import simpulse
import simulate_frb

class CompareInjectors:
    """ Class to generate pulses with both simpulse 
    and injectfrb packages. 

    Computes correlation coefficient between two 
    pulse profiles or spectra.
    """


    def __init__(self, nfreq=1024, ntime=1024, fluence=1, width=0.001,
                 dm=100., dt=0.001, freq=(2000, 1000), spec_ind=0.,
                 scat_tau_ref=0., freq_ref=np.inf):
        self.nfreq = nfreq
        self.ntime = ntime 
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

    def corr_coeff(self, data1, data2):
        """ 
        Both data arrays should be 1D of equal length
        """
                
        r = np.dot(data1, data2) / np.sqrt(np.dot(data1, data1)*np.dot(data2, data2))

        return r

    def plot_comparison(self, data1, data2, title1='', title2=''):
        """ Plot dynamic spectra and pulse profiles 
        for two simulated pulses 
        """
        fig = plt.figure(figsize=(8,8))

        plt.subplot(221)
        plt.title(title1)
        plt.plot(data1[-50])
        plt.subplot(222)
        plt.title(title2)
        plt.plot(data2[-50])
        plt.subplot(223)
        plt.imshow(data1, aspect='auto', cmap='Greys')
        plt.subplot(224)
        plt.imshow(data2, aspect='auto', cmap='Greys')
        plt.show()

    def corr_coeff_arr(self, data_arr1, data_arr2):
        """ Compute correlation coefficient of pulse profiles 
        for each frequency in array
        data.shape = (nfreq, ntime) 
        """ 

        r_arr = []

        for ii, ff in enumerate(self.freq_arr):
            data1 = np.roll(data_arr1[ii], self.ntime//2-np.argmax(data_arr1[ii]))
            data2 = np.roll(data_arr2[ii], self.ntime//2-np.argmax(data_arr2[ii]))
            r = self.corr_coeff(data1, data2)
            if ii % 10 == 0:
                print("Correlation coefficient: %f at %0.1f MHz" % (r,ff))
            r_arr.append(r)

        return np.array(r_arr)

def gen_corrcoef_grid_dm_width(ndm=20, nwidth=20):
    DMs = np.linspace(10., 2000, ndm)
    widths = np.logspace(-4, -1.5, nwidth)
    dt = 0.001

    r_arr = np.empty([ndm, nwidth])

    for ii, dm in enumerate(DMs):
        for jj, width in enumerate(widths):
            nt = max(2500, int(3*4183*dm*(1000**-2)/dt))
            C = CompareInjectors(ntime=nt, dm=dm, width=width, dt=dt)
            data_injfrb = C.gen_injfrb_pulse(upchan_factor=1, upsamp_factor=10, conv_dmsmear=True)
            data_simpulse = C.gen_simpulse()
            r = C.corr_coeff(data_injfrb[512], data_simpulse[512])
            r_arr[ii, jj] = r
            print("r=%.2f nt=%d DM=%.4f w=%.4f" % (r, nt, dm, width))

    r_arr = np.array(r_arr)
    fnout = 'corr_arr_DM=%d-%d_w=%.2f-%.2f' % (DMs.min(), DMs.max(), 1e3*widths.min(), 1e3*widths.max())
    np.save(fnout, r_arr)
    extent = [np.log10(1e3*widths[0]), np.log10(1e3*widths[-1]), DMs[-1], DMs[0]]
    plt.imshow(np.log10(1-r_arr), aspect='auto', extent=extent)
    plt.xlabel('log10(width [ms])')
    plt.ylabel('DM')
    plt.title('Log of deviation \nfrom perfect correlation: log10(1-r)')
    plt.colorbar()
    plt.savefig(fnout+'.pdf')
    plt.show()

    return r_arr

def gen_corrcoef_grid_spec_scat(nscat=5, nspecind=5):
    scat_tau_refs = np.logspace(-2, -1, nscat)
    spec_inds = np.linspace(-3, 3, nspecind)
    dm = 0.
    width = 0.001
    dt = 0.0005

    r_arr = np.empty([nscat, nspecind])

    for ii, scat_tau_ref in enumerate(scat_tau_refs):
        for jj, spec_ind in enumerate(spec_inds):
            nt = max(2500, int(3*4183*dm*(1000**-2)/dt))
            C = CompareInjectors(ntime=nt, dm=dm, width=width, 
                                 dt=dt, scat_tau_ref=scat_tau_ref, 
                                 spec_ind=spec_ind)

            data_injfrb = C.gen_injfrb_pulse(upchan_factor=1, upsamp_factor=10, conv_dmsmear=True)
            data_simpulse = C.gen_simpulse()
            C.plot_comparison(data_injfrb, data_simpulse, title1='', title2='')
            r = C.corr_coeff(data_injfrb[512], data_simpulse[512])
            r_arr[ii, jj] = r
            print("r=%.2f nt=%d SM=%d specind=%.4f" % (r, nt, scat_tau_ref, spec_ind))

    r_arr = np.array(r_arr)
    fnout = 'corr_arr_SM=%d-%d_specind=%.2f-%.2f' % (-5, -1, -5, 5)
    np.save(fnout, r_arr)
    #extent = [np.log10(widths[0]), np.log10(1e3*widths[-1]), DMs[-1], DMs[0]]
    plt.imshow(np.log10(1-r_arr), aspect='auto')#, extent=extent)
    plt.xlabel('specind')
    plt.ylabel('scat_tau_ref')
    plt.title('Log of deviation \nfrom perfect correlation: log10(1-r)')
    plt.colorbar()
    plt.savefig(fnout+'.pdf')
    plt.show()

    return r_arr

def test_gen_corrcoef_grid():
    r_arr = gen_corrcoef_grid(ndm=3, nwidth=3)
    assert (r_arr==1).all(), "this"

def test_gen_injfrb():
    C = CompareInjectors()
    data = C.gen_injfrb_pulse()

def test_gen_simpulse():
    C = CompareInjectors()
    data = C.gen_simpulse()

def test_imitate_simpulse():
    C = CompareInjectors(ntime=25000, dm=100., width=0.005, dt=0.001)
    data_injfrb = C.gen_injfrb_pulse(upchan_factor=10, conv_dmsmear=True)
    data_simpulse = C.gen_simpulse()
    np.save('injarr', data_injfrb)
    np.save('simparr', data_simpulse)
    data_injfrb_prof = np.mean(data_injfrb, axis=0)
    data_simpulse_prof = np.mean(data_simpulse, axis=0)

    r_arr = C.corr_coeff_arr(data_injfrb, data_simpulse)

    fig = plt.figure(figsize=(9,9))
    plt.subplot(131)
    plt.plot(C.freq_arr[::2], r_arr[::2], '.', color='k')
    plt.ylabel('Correlation Coeff <injfrb vs. simpulse>', fontsize=14)
    plt.xlabel('Freq [MHz]', fontsize=14)
    plt.ylim(0.5, 1.05)
    title = '\nDM=%d\npulse_width=%0.1f ms\ndt=%0.1f ms' % (100, 5, 1)
    plt.title('Resolved ' + title, fontsize=12)

    C = CompareInjectors(ntime=15000, nfreq=256, dm=500., width=0.0005, dt=0.0005)
    data_injfrb = C.gen_injfrb_pulse(upchan_factor=10, conv_dmsmear=True)
    data_simpulse = C.gen_simpulse()
    data_injfrb_prof = np.mean(data_injfrb, axis=0)
    data_simpulse_prof = np.mean(data_simpulse, axis=0)

    r_arr = C.corr_coeff_arr(data_injfrb, data_simpulse)
    plt.subplot(132)
    plt.plot(C.freq_arr[:], r_arr[:], '.', color='k')
    plt.xlabel('Freq [MHz]', fontsize=14)
    plt.ylim(0.5, 1.05)
    title = '\nDM=%d\npulse_width=%0.1f ms\ndt=%0.1f ms' % (500, 0.5, 0.5)
    plt.title('DM-smeared ' + title, fontsize=12)

    C = CompareInjectors(ntime=25000, dm=0., width=0.0001, dt=0.005)
    data_injfrb = C.gen_injfrb_pulse(upchan_factor=10, conv_dmsmear=True)
    data_simpulse = C.gen_simpulse()
    np.save('injarr', data_injfrb)
    np.save('simparr', data_simpulse)
    data_injfrb_prof = np.mean(data_injfrb, axis=0)
    data_simpulse_prof = np.mean(data_simpulse, axis=0)

    r_arr = C.corr_coeff_arr(data_injfrb, data_simpulse)
    plt.subplot(133)
    plt.plot(C.freq_arr[::2], r_arr[::2], '.', color='k')
    plt.xlabel('Freq [MHz]', fontsize=14)
    plt.ylim(0.5, 1.05)
    title = '\nDM=%d\npulse_width=%0.1f ms\ndt=%0.1f ms' % (0, 0.1, 5)
    plt.title('Sample-smeared ' + title, fontsize=12)
    plt.tight_layout()
    plt.show()

def test_plot_comparison():
    C = CompareInjectors(ntime=15000, nfreq=256, dm=1000., width=0.0001, dt=0.0005)
    data_injfrb = C.gen_injfrb_pulse(upsamp_factor=4, conv_dmsmear=True)
    data_simpulse = C.gen_simpulse()
        
    data_injfrb_prof = np.mean(data_injfrb, axis=0)
    data_simpulse_prof = np.mean(data_simpulse, axis=0)

    data_simpulse_prof = np.roll(data_simpulse_prof, C.ntime//2-np.argmax(data_simpulse_prof))

    C.plot_comparison(data_injfrb, data_simpulse, title1='INJFRB', title2='SIMPULSE')


if __name__=='__main__':
#    test_gen_injfrb()
#    test_gen_simpulse()
#    test_corr_coeff()
#    test_plot_comparison()
    test_imitate_simpulse()
    gen_corrcoef_grid_dm_width(ndm=4, nwidth=4)
    print("Note gen_corrcoef_grid_spec_scat still fails")





