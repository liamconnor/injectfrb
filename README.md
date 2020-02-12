injectfrb
==========

python software to simulate fast radio bursts and inject them into filterbank data

to test that the FRB generator works, run the script "test_plot_four_frbs.py"
to test that the filterbank reader, writer, injector works, run "test_filterbank_injector.py "
to test that the Gaussian-noise filterbank writer works:

python inject_gaussian_fil.py --nfrb 50 --simulator injectfrb --dm_min 100. --dm_max 100. --outdir outputdir --upchan_factor 1 --upsamp_factor 1

this should create a filterbank file with 50 FRBs all with DM=100 injected into Gaussian noise called: 

outputdir/injectfrb_nfrb50_*.fil 

along with a text file with the relevant trigger information in:

outputdir/injectfrb_nfrb50_*.txt

Requires:
========== 

simpulse https://github.com/kmsmith137/simpulse
