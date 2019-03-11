""" simpulse.single_pulse and injectfrb.simulate_frb deal with 
	temporal smearing slightly differently. simulate_frb calculates 
	final width from dispersion smearing and sampling and then 
	generates a Gaussian pulse with that width. single_pulse 
	generates a Gaussian pulse with some intrinsic width and 
	injects into upchannelized/upsampled data. 

	this script is meant to test the difference between those 
	two methods
"""

import numpy as np
import matplotlib.pylab as plt

import simpulse
import simulate_frb

dt = 0.001
freq_lo_MHz = 1000.0
freq_hi_MHz = 2000.0
dm = 1000.0
sm = 0.0
intrinsic_width = 0.0005
fluence = 1.0
spectral_index = 0.
undispersed_arrival_time = 0.10
dedisp_delay = 4148*dm*(freq_lo_MHz**-2.-freq_hi_MHz**-2.)
pulse_nt = int(2*dedisp_delay/dt + 1000)
nfreq = 1024
freq_ref = 1500.

sp = simpulse.single_pulse(pulse_nt, nfreq, freq_lo_MHz, freq_hi_MHz,
                           dm, sm, intrinsic_width, fluence,
                           spectral_index, undispersed_arrival_time)

data_simpulse = np.zeros([nfreq, pulse_nt])
sp.add_to_timestream(data_simpulse, 0.0, pulse_nt*dt)
data_simpulse = data_simpulse[::-1]

data_injfrb, p = simulate_frb.gen_simulated_frb(NFREQ=nfreq, NTIME=pulse_nt, sim=True, fluence=1.0, 
    								 spec_ind=0.0, width=intrinsic_width, dm=dm, 
    								 background_noise=np.zeros([nfreq, pulse_nt]),
    								 delta_t=dt, plot_burst=False, 
    								 freq=(freq_hi_MHz, freq_lo_MHz), 
    								 FREQ_REF=freq_ref, 
     	    						 scintillate=False, scat_tau_ref=0.0, 
     	    						 disp_ind=2.0)

data_simpulse /= np.max(data_simpulse)
data_injfrb /= np.max(data_injfrb)

fig = plt.figure()

plt.subplot(121)
plt.plot(np.roll(data_simpulse[0], pulse_nt//2-np.argmax(data_simpulse[0])), color='k')
plt.plot(np.roll(data_simpulse[-100], pulse_nt//2-np.argmax(data_simpulse[-100])), color='C1')

maxind = pulse_nt//2-np.argmax(data_simpulse[0])
plt.xlim(maxind-100, maxind+100)

plt.subplot(122)
plt.plot(np.roll(data_injfrb[0], pulse_nt//2-np.argmax(data_injfrb[0])), color='k')
plt.plot(np.roll(data_injfrb[-100], pulse_nt//2-np.argmax(data_injfrb[-100])), color='C1')

maxind = pulse_nt//2-np.argmax(data_injfrb[0])
plt.xlim(maxind-100, maxind+100)

plt.show()
