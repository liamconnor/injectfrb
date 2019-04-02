#!/usr/bin/env python
#
# Plot triggers output by the ML classifier

import sys

import numpy as np
import h5py
import matplotlib
matplotlib.use('pdf', warn=False)
import matplotlib.pyplot as plt

def plot_two_panel(data_freq_time, params, times=None, cb=None, prob=None, 
                   freq_low=1250.09765625, freq_up=1549.90234375, 
                   cand_no=1, fnout='out.pdf', suptitle=''):
    """ Plot data in two panels
    """
    snr, dm, bin_width, t0, delta_t = params
    nfreq, ntime = data_freq_time.shape

    if times is None:
        times = np.arange(ntime)*delta_t*bin_width*1e3  # ms

    times *= 1e3 #convert to ms
    freqs = np.linspace(freq_low, freq_up, nfreq)

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, 
                                gridspec_kw=dict(height_ratios=[1, 2]))

    # timeseries
    ax1.plot(times, np.sum(data_freq_time, axis=0)/
            np.sqrt(data_freq_time.shape[0]), c='k')
    ax1.set_ylabel('S/N', labelpad=10)
    # add what a DM=0 signal would look like for ms tres
    DM0_delays = dm * 4.15E6 * (freq_low**-2 - freqs**-2)#/(delta_t*bin_width*1.e3)
    ax2.plot(DM0_delays, freqs, c='r', lw='2', alpha=0.5)
    # scaling: std = 1, median=0
    extent = [times[0], times[-1], freq_low, freq_up]

    ax2.imshow(data_freq_time, cmap='viridis', vmin=-3, vmax=3, 
               interpolation='nearest', aspect='auto', 
               origin='upper', extent=extent)

    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Freq (MHz)', labelpad=10)

    if cb is None:
        cb = -1
    if prob is None:
        prob = -1

    suptitle = 'p:%.2f %s' % (prob, suptitle)

#    try:
#        fig.suptitle("p: {:.2f}, S/N: {:.0f}, DM: {:.2f}, \
#                  T0: {:.2f}, CB: {:02d}".format(prob, snr, dm, t0, cb))
#        figname = "plots/cand_{:04d}_snr{:.0f}_dm{:.0f}.pdf".format(cand_no, snr, dm)
#    except:
#        fig.suptitle("p: %.2f, S/N: %.0f, DM: %.2f, T0: %.2f, CB: %02d" \
#                     % (prob, snr, dm, t0, cb))
#        figname = "plots/cand_%04d_snr%.0f_dm%.0f.pdf" % (cand_no, snr, dm)

    fig.suptitle(suptitle)
    plt.savefig(fnout)
    plt.close(fig)

def plot_three_panel(data_freq_time, data_dm_time, params, dms, times=None, 
                     freq_up=1549.90234375, freq_low=1250.09765625,
                     cmap="RdBu", suptitle="", fnout="out.pdf", cb=None,
                     cand_no=1, prob=None):
    """ Plot freq/time, time, and dm/time data in 
    three panels. 
    """
    snr, dm, bin_width, t0, delta_t = params
    nfreq, ntime = data_freq_time.shape
    freqs = np.linspace(freq_low, freq_up, nfreq)

    if times is None:
        times = np.arange(ntime)*delta_t*bin_width*1e3  # ms

    figure = plt.figure()
    ax1 = plt.subplot(311)

    times *= 1e3 # convert to ms from s
    plt.imshow(data_freq_time, aspect='auto', vmax=4, vmin=-4, 
               extent=[0, times[-1], freq_low, freq_up], 
               interpolation='nearest', cmap=cmap)
    plt.ylabel('Freq [MHz]', labelpad=10)

    plt.subplot(312, sharex=ax1)
    plt.plot(times, data_freq_time.mean(0), color='k')
    plt.ylabel('Flux', labelpad=10)

    plt.subplot(313, sharex=ax1)
    plt.imshow(data_dm_time, aspect='auto', 
               extent=[0, times[-1], dms[-1], dms[0]], 
               interpolation='nearest', cmap=cmap, origin='upper')
    plt.xlabel('Time [ms]')
    plt.ylabel('DM', labelpad=10)

    DM0_delays = dm * 4.15E6 * (freq_low**-2 - freqs**-2)#/(delta_t*bin_width*1.e3)
    ax1.plot(DM0_delays, freqs, c='r', lw='2', alpha=0.5)
    plt.xlim(0, times[-1])

    if cb is None:
        cb = -1
    if prob is None:
        prob = -1

    suptitle = 'p:%.2f %s' % (prob, suptitle)

    plt.suptitle(suptitle, fontsize=14)
#    plt.tight_layout()
    plt.savefig(fnout)

def plot_from_h5(fn, cb, freq_low=1250.09765625, freq_up=1549.90234375, 
                 ):
    # read dataset 
    with h5py.File(fn, 'r') as f:
        data_frb_candidate = f['data_frb_candidate'][:]
        probability = f['probability'][:]
        params = f['params'][:]  # snr, DM, boxcar width, arrival time

    for i, cand in enumerate(data_frb_candidate):
        data_freq_time = cand[:, :, 0]

        plot_two_panel(data_freq_time, params[i], cb=cb, freq_low=freq_low, 
                    freq_up=freq_up, prob=probability[i], cand_no=i)


def mk_histograms(params, fnout='summary_hist.pdf', 
                  suptitle='Trigger summary', alpha=0.25):
    """ Take parameter array (ntrig, 4) and make 
    histograms of each param 
    """
    assert(len(params)==4)

    # if int(mpl.__version__[0])<2:
    #     alpha=0.5
    # else:
    #     alpha=1.

    figure = plt.figure(figsize=(8,8))
    ax1 = plt.subplot(221)
    plt.hist(params[0], log=True, color='C0', alpha=alpha, bins=30)
    plt.xlabel('DM [pc cm**-3]', fontsize=12)

    plt.subplot(222)
    plt.hist(params[1], color='C1', alpha=alpha, log=True, bins=30)
    plt.xlabel('S/N', fontsize=12)

    plt.subplot(223)
    plt.hist(params[2], color='C2', alpha=alpha, log=True, bins=30)
    plt.xlabel('Time [sec]', fontsize=12)

    plt.subplot(224)
    plt.hist(np.log2(params[3]), color='C3', alpha=alpha, bins=8, log=True)
    plt.xlabel('log2(Width) [samples]', fontsize=12)

    suptitle = "%s %d events" % (suptitle, len(params))
    plt.suptitle(suptitle)

    plt.show()
    plt.savefig(fnout)

def plot_against_truth(par_match_arr1, par_match_arr2, algo1='algo1', algo2='algo2'):

    fig = plt.figure(figsize=(5,5))

    suptitle = "%s vs. %s" % (algo1, algo2)

    snr_1 = par_match_arr1[0,:,0]
    snr_1_truth = par_match_arr1[0,:,1]

    snr_2 = par_match_arr2[0,:,0]
    snr_2_truth = par_match_arr2[0,:,1]

#    snr_1, snr_2, snr_1t, snr_2t = par_1[0], par_2[0], par_1_truth[0], par_2_truth[0]
#    dm_1, dm_2, dm_1t, dm_2t = par_1[1], par_2[1], par_1_truth[1], par_2_truth[1]
#    t_1, t_2 = par_1[2], par_2[2]
#    width_1, width_2 = par_1[3], par_2[3]

    plt.plot(snr_1_truth, snr_1, '.', color='k')
    plt.plot(snr_2_truth, snr_2, '.', color='grey')
    plt.plot(snr_1_truth, snr_1_truth, '--', color='k')
    plt.legend([algo1, algo2])
    plt.loglog()
    plt.show()

def plot_comparison(par_1, par_2, par_match_arr, 
                    ind_missed, figname='./test.pdf', 
                    algo1='algo1', algo2='algo2'):
    fig = plt.figure(figsize=(12,14))

    suptitle = "%s vs. %s" % (algo1, algo2)

    snr_1, snr_2 = par_1[0], par_2[0]
    dm_1, dm_2 = par_1[1], par_2[1]
    t_1, t_2 = par_1[2], par_2[2]
    width_1, width_2 = par_1[3], par_2[3]

    snr_1_match = par_match_arr[0,:,0]
    snr_2_match = par_match_arr[0,:,1]

    dm_1_match = par_match_arr[1,:,0]
    dm_2_match = par_match_arr[1,:,1]

    width_1_match = par_match_arr[3,:,0]
    width_2_match = par_match_arr[3,:,1]

    fig.add_subplot(331)
    plt.plot(snr_1[ind_missed], 5+np.zeros([len(ind_missed)]), '.', color='orange')
    plt.plot(snr_1_match, snr_2_match, '.')
    plt.plot(snr_1, snr_1, color='k')
    plt.loglog()
    plt.xlabel('%s S/N' % algo1, fontsize=10)
    plt.ylabel('%s S/N' % algo2, fontsize=10)        
    plt.legend(['Missed', 'Matched', 'Equal S/N'], fontsize=10)

    fig.add_subplot(334)
    plt.plot(dm_1[ind_missed], np.zeros([len(ind_missed)]), '.', color='orange')
    plt.plot(dm_1_match, snr_1_match/snr_2_match, '.')
    plt.plot(dm_1, np.ones_like(dm_1), '--', color='k')
    plt.xlabel('DM', fontsize=10)
    plt.ylabel('S/N$_1$ : S/N$_2$', fontsize=10)        
    plt.legend(['Missed','Detected','Equal S/N'], fontsize=10)

    plt.subplot(332)
    plt.hist(dm_1, log=True, alpha=0.8, bins=30, color='k', histtype='step', linestyle='--')
    plt.hist(dm_2, log=True, alpha=1., bins=30, color='k', histtype='step')
    plt.hist(dm_1[ind_missed], log=True, alpha=0.3, bins=30, color='C1')
    plt.xlabel('DM [pc cm**-3]', fontsize=10)
    plt.legend([algo1, algo2, 'Missed'], fontsize=10)

    plt.subplot(333)
    plt.hist(np.log10(snr_1), alpha=0.8, log=True, bins=30, color='k', histtype='step', linestyle='--')
    plt.hist(np.log10(snr_2), alpha=1, log=True, bins=30, color='k', histtype='step')
    plt.hist(np.log10(snr_1[ind_missed]), alpha=0.3, log=True, bins=30, color='C1')
    plt.xlabel('S/N', fontsize=10)
    plt.legend([algo1, algo2, 'Missed'], fontsize=10)

    plt.subplot(335)
#    plt.hist(t_1, alpha=0.5, log=True, bins=30)
#    plt.hist(t_2, alpha=0.5, log=True, bins=30)
    plt.plot(t_1, np.log10(.1+dm_1), 'o', alpha=0.3, color='k')
    plt.plot(t_2, np.log10(.1+dm_2), '.', alpha=0.8, color='k')    
    plt.xlabel('Time [sec]', fontsize=10)
    plt.ylabel('log10(DM)', fontsize=10)

    plt.subplot(336)
    plt.hist(np.log10(width_1), alpha=0.8, bins=8, log=True, color='k', histtype='step', linestyle='--')
    plt.hist(np.log10(width_2), alpha=1., bins=8, log=True, color='k', histtype='step')
    plt.hist(np.log10(width_1[ind_missed]), alpha=0.3, bins=8, log=True, color='C1')
    plt.xlabel('log10(Width) [samples]', fontsize=10)
    plt.legend([algo1, algo2, 'Missed'], fontsize=10)

    # fig.add_subplot(337)
    # plt.hist(width_1[ind_missed], bins=50, alpha=0.3, normed=True, color='C3')
    # plt.hist(width_1, bins=50, alpha=0.3, normed=True)
    # plt.hist(width_2, bins=50, alpha=0.3, normed=True)
    # plt.xlabel('Width [samples]', fontsize=12)

    fig.add_subplot(325)
    plt.plot(np.log10(width_1), np.log10(snr_1), 'o', color='k', alpha=0.3)
    plt.plot(np.log10(width_1_match), np.log10(snr_2_match),'.', alpha=0.8, color='k')
#    plt.plot(width_1_match, snr_2_match,'.')
    plt.xlabel('log10(Width) [samples]', fontsize=10)
    plt.ylabel('log10(S/N)', fontsize=10)
    plt.legend(['%s all' % algo1, 'Matched'], fontsize=10)
    plt.grid()

    fig.add_subplot(326)
    plt.plot(np.log10(width_1), np.log10(0.1+dm_1),'o',color='k', alpha=0.3)
    plt.plot(np.log10(width_1_match), np.log10(0.1+dm_2_match),'.', alpha=0.8, color='k')
    #plt.plot(np.log2(width_1_match), np.log10(0.1+dm_2_match),'.')
    plt.xlabel('log10(Width) [samples]', fontsize=10)
    plt.ylabel('log10(DM)', fontsize=10)
    plt.legend(['%s all' % algo1, 'Matched'], fontsize=10)
    plt.grid()

    snr_ratio = np.mean(snr_1_match / snr_2_match)
    frac_missed = np.float(len(ind_missed))/len(snr_1)

    suptitle += ('   avg S/N$_1$/S/N$_2$: ' + np.str(np.round(snr_ratio,2)))
    suptitle += '\n      frac$_{missed}$=%0.2f' % frac_missed
    suptitle += '\n N$_{%s}=%d$' % (algo1, len(width_1))
    suptitle += '    N$_{%s}=%d$' % (algo2, len(width_2))
    suptitle += '    N$_{%s}=%d$' % ('missed', len(ind_missed))

    plt.suptitle(suptitle, fontsize=15)
#    plt.tight_layout()
    plt.show()
    plt.savefig(figname)

if __name__ == '__main__':
#     # input hdf5 file
    print('\nExpecting: data_file CB <freq_low> <freq_up>\n')
    fn = sys.argv[1]
    cb = int(sys.argv[2])

    try:
        freq_low = np.float(sys.argv[3])
    except:
        freq_low = 1250.09765625

    try:
        freq_up = np.float(sys.argv[4])
    except:
        freq_up = 1549.90234375
        
    plot_from_h5(fn, cb, freq_low=freq_low, freq_up=freq_up)


