import numpy as np

import simpulse
import simulate_frb

class CompareInjectors:


	def __init__(self, nfreq=1024, ntime=1024, fluence=1, width=0.001,
				 dm=100., dt=0.001, freq=(1000, 2000), spec_ind=0.,
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
                           self.dm, sm, self.intrinsic_width, self.fluence,
                           self.spec_ind, undispersed_arrival_time)

		data_simpulse = np.zeros([self.nfreq, self.ntime])
		sp.add_to_timestream(data_simpulse, 0.0, self.ntime*self.dt)
		data_simpulse = data_simpulse[::-1]

		return data_simpulse

	def corr_coefficient(self, data1, data2):
		""" 
		Both data arrays should be 1D of equal length
		"""

		r = np.dot(data1, data2) / np.sqrt(np.dot(data1), np.dot(data1))

		return r


def test_gen_injfrb():
	C = CompareInjectors()
	data = C.gen_injfrb_pulse()

def test_gen_simpulse():
	C = CompareInjectors()
	data = C.gen_simpulse()

def test_corr_coefficient():
	C = CompareInjectors()
	data_injfrb = C.gen_injfrb_pulse()
	data_simpulse = C.gen_simpulse()

	r = C.corr_coefficient(data_injfrb, data_simpulse)
	print("Correlation coefficient: %f" % r)


if __name__=='__main__':
	test_gen_injfrb()
	test_gen_simpulse()
	test_corr_coefficient()






