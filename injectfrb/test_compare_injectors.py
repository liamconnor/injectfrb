import numpy as np
import matplotlib.pylab as plt

import simpulse
import simulate_frb

class CompareInjectors:


    def __init__(self, nfreq=1024, ntime=1024, fluence=1, width=0.001,
                 dm=100., dt=0.001, freq=(2000, 1000), spec_ind=0.,
                 scat_tau_ref=0., freq_ref=1500.):
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

    def gen_injfrb_pulse(self):
        data_bg = np.zeros([self.nfreq, self.ntime])
        data_injfrb, p = simulate_frb.gen_simulated_frb(NFREQ=self.nfreq, NTIME=self.ntime,
                                     sim=True, fluence=self.fluence, 
                                     spec_ind=self.spec_ind, width=self.width, dm=self.dm, 
                                     background_noise=data_bg,
                                     delta_t=self.dt, plot_burst=False, 
                                     freq=(self.freq_hi_MHz, self.freq_lo_MHz), 
                                     FREQ_REF=self.freq_ref, scintillate=False, 
                                     scat_tau_ref=self.scat_tau_ref, 
                                     disp_ind=2.0)
        return data_injfrb

    def gen_simpulse(self):
        undispersed_arrival_time = 0.1
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

    def plot_comparison(self, data1, data2):
        fig = plt.figure(figsize=(8,8))

        plt.subplot(121)
        plt.plot(data1)
        plt.subplot(122)
        plt.plot(data2)
        plt.show()

    def corr_coeff_arr(self, data_arr1, data_arr2):
        """ (nfreq, ntime) array
        """ 

        for ii, ff in enumerate(self.freq_arr):
            data1 = np.roll(data_arr1[ii], self.ntime//2-np.argmax(data_arr1[ii]))
            data2 = np.roll(data_arr2[ii], self.ntime//2-np.argmax(data_arr2[ii]))
            r = self.corr_coeff(data1, data2)
            print("Correlation coefficient: %f at %0.1f MHz" % (r,ff))
            r_arr.append(r)

        return np.array(r_arr)

def test_gen_injfrb():
    C = CompareInjectors()
    data = C.gen_injfrb_pulse()

def test_gen_simpulse():
    C = CompareInjectors()
    data = C.gen_simpulse()

def test_corr_coeff():
    C = CompareInjectors(ntime=5000, dm=100., width=0.0001, dt=0.005)
    data_injfrb = C.gen_injfrb_pulse()
    data_simpulse = C.gen_simpulse()
        
    data_injfrb_prof = np.mean(data_injfrb, axis=0)
    data_simpulse_prof = np.mean(data_simpulse, axis=0)

    r_arr = C.corr_coeff_arr(data_injfrb, data_simpulse)

    plt.plot(C.freq_arr, r_arr)
    plt.show()

def test_plot_comparison():
    C = CompareInjectors(ntime=5000, dm=100., width=0.0001, dt=0.005)
    data_injfrb = C.gen_injfrb_pulse()
    data_simpulse = C.gen_simpulse()
        
    data_injfrb_prof = np.mean(data_injfrb, axis=0)
    data_simpulse_prof = np.mean(data_simpulse, axis=0)

    data_simpulse_prof = np.roll(data_simpulse_prof, C.ntime//2-np.argmax(data_simpulse_prof))

    C.plot_comparison(data_injfrb_prof, data_simpulse_prof)



if __name__=='__main__':
    test_gen_injfrb()
    test_gen_simpulse()
    test_corr_coefficient()
    test_plot_comparison()






