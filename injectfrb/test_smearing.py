import numpy as np

import simpulse
import simulate_frb

dt = 0.001
freq_lo_MHz = 1000.0
freq_hi_MHz = 2000.0
dm = 250.0
sm = 0.0
intrinsic_width = 0.005
fluence = 1.0
spectral_index = 0.
undispersed_arrival_time = 0.10
dedisp_delay = 4148*dm*(freq_lo_MHz**-2.-freq_hi_MHz**-2.)
pulse_nt = int(2*dedisp_delay/dt)
nfreq = 1024
freq_ref = 1500.


data_simpulse = simpulse.single_pulse(pulse_nt, nfreq, freq_lo_MHz, freq_hi_MHz,
                           dm, sm, intrinsic_width, fluence,
                           spectral_index, undispersed_arrival_time)

data = np.zeros([nfreq, pulse_nt])
data_simpulse.add_to_timestream(data, 0.0, pulse_nt*dt)

data_injfrb, p = s.gen_simulated_frb(NFREQ=nfreq, NTIME=pulse_nt, sim=True, fluence=1.0, 
    								 spec_ind=0.0, width=0.0001, dm=1000, 
    								 background_noise=np.zeros([nfreq, pulse_nt])
    								 delta_t=dt, plot_burst=False, 
    								 freq=(freq_hi_MHz, freq_lo_MHz), 
    								 FREQ_REF=freq_ref, 
     	    						 scintillate=False, scat_tau_ref=0.0, 
     	    						 disp_ind=2.0)

fig = plt.figure()

plt.plot(data_simpulse[0], '--', color='k')
plt.plot(data_simpulse[-100], '--', color='k')
plt.plot(data_simpulse[0], '.', color='C1')
plt.plot(data_simpulse[-100], '--', color='C1')

plt.show()
