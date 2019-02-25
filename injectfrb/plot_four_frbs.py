import numpy as np
import matplotlib.pylab as plt

import simulate_frb

def plot_four_frbs():
    fig = plt.figure(figsize=(8,6))
    cmap = 'Greys'

    print("Generating Apertif-like FRB with blue spectrum")
    NTIME = 2**13
    NFREQ = 1536
    dt = 0.000081
    upchan_factor = 1
    freq = np.linspace(1520., 1220., upchan_factor*NFREQ)
    freq_ref = 1400.
    dm = 500.

    data_apertif, p = simulate_frb.gen_simulated_frb(NFREQ=NFREQ, NTIME=NTIME, sim=True, fluence=1., 
                            spec_ind=4., width=dt, dm=dm, 
                            background_noise=np.zeros([NFREQ, NTIME]), 
                            delta_t=dt, plot_burst=False, 
                            freq=(freq[0],freq[-1]), FREQ_REF=freq_ref,
                            scintillate=False, scat_tau_ref=0.0)

    data_apertif = data_apertif.reshape(-1, upchan_factor, NTIME).mean(1)

    ext = [0, NTIME*dt, freq[-1], freq[0]]
    plt.subplot(221)
    plt.imshow(data_apertif, aspect='auto', cmap=cmap, extent=ext)
    plt.ylabel('Frequency [MHz]')
    plt.text(NTIME*dt*.6, freq[NFREQ//2], 'Apertif\nNo noise \nBlue spectrum\
                                      \nDM=%d' % (dm), alpha=1.0)# bbox=dict(facecolor='white', alpha=1.0))

    print("Generating CHIME-like FRB with scattering")
    NTIME = 2**12
    NFREQ = 16384
    dt = 0.001
    upchan_factor = 1
    freq = np.linspace(800., 400., upchan_factor*NFREQ)
    freq_ref = 600.
    dm = 100.

    data_chime, p = simulate_frb.gen_simulated_frb(NFREQ=NFREQ, NTIME=NTIME, sim=True, fluence=1., 
                            spec_ind=-3., width=2*dt, dm=dm, 
                            background_noise=np.zeros([NFREQ, NTIME]), 
                            delta_t=dt, plot_burst=False, 
                            freq=(freq[0],freq[-1]), FREQ_REF=freq_ref, 
                            scintillate=False, scat_tau_ref=0.025)

    data_chime = data_chime.reshape(-1, upchan_factor, NTIME).mean(1)

    ext = [0, NTIME*dt, freq[-1], freq[0]]
    plt.subplot(222)
    plt.imshow(data_chime, aspect='auto', cmap=cmap, extent=ext)
    plt.text(NTIME*dt*.6, freq[NFREQ//2], 'CHIME\nNo noise \nScattered\
                                      \nDM=%d' % (dm), alpha=1.0)#
                                      #                                     bbox=dict(facecolor='white', alpha=1.0))
    print("Generating Breakthrough-listen-like FRB with Gaussian noise")
    NTIME = 2**8
    NFREQ = 2048
    dt = 0.001
    upchan_factor = 1
    freq = np.linspace(8000., 4000., upchan_factor*NFREQ)
    freq_ref = 6000.
    dm = 1000.

    data_noise_breakthrough, p = simulate_frb.gen_simulated_frb(NFREQ=NFREQ, NTIME=NTIME, 
                            sim=True, fluence=100000., 
                            spec_ind=4., width=dt, dm=500., 
                            background_noise=None,
                            delta_t=dt, plot_burst=False, 
                            freq=(freq[0],freq[-1]), FREQ_REF=freq_ref,
                            scintillate=True, scat_tau_ref=0.0)

    data_noise_breakthrough = data_noise_breakthrough.reshape(-1, upchan_factor, NTIME).mean(1)

    plt.subplot(223)
    ext = [0, NTIME*dt, freq[-1], freq[0]]
    plt.ylabel('Frequency [MHz]')
    plt.imshow(data_noise_breakthrough, aspect='auto', cmap=cmap, extent=ext)
    plt.text(NTIME*dt*.6, freq[NFREQ//2], 'Breakthrough Listen\nGaussian noise \nScintillated\
                                      \nDM=%d' % (dm), alpha=1.0)#, bbox=dict(facecolor='white', alpha=1.0))
    plt.xlabel('Time [sec]')

    print("Generating Parkes-like FRB with Gaussian noise")
    NTIME = 2**15
    NFREQ = 1024
    dt = 0.000064
    upchan_factor = 1
    freq = np.linspace(1520., 1220., upchan_factor*NFREQ)
    freq_ref = 1400.
    dm = 2000.

    data_noise_dm2000, p = simulate_frb.gen_simulated_frb(NFREQ=NFREQ, NTIME=NTIME, 
                            sim=True, fluence=5000000., 
                            spec_ind=4., width=50*dt, dm=dm, 
                            background_noise=None,
                            delta_t=dt, plot_burst=False, 
                            freq=(freq[0],freq[-1]), FREQ_REF=freq_ref,
                            scintillate=False, scat_tau_ref=0.0)

    data_noise_dm2000 = data_noise_dm2000.reshape(-1, upchan_factor, NTIME).mean(1)

    plt.subplot(224)
    ext = [0, NTIME*dt, freq[-1], freq[0]]
    plt.imshow(data_noise_dm2000, aspect='auto', cmap=cmap, extent=ext)
    plt.xlabel('Time [sec]')
    plt.text(NTIME*dt*.6, freq[NFREQ//2], 'Parkes\nGaussian noise\nDM=%d' % dm, 
             alpha=1.0)#, bbox=dict(facecolor='white', alpha=1.0))

    fnout = './test_fig_4FRBs.pdf'
    print("Saving figure to %s" % fnout)
    plt.savefig(fnout)
    plt.show()

if __name__=='__main__':
    plot_four_frbs()




