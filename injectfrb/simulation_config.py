filhdr = {'telescope_id': 10,
      'az_start': 0.0,
      'nbits': 8,
      'source_name': 'J1813-1749',
      'data_type': 1,
      'nchans': 336,
      'machine_id': 15,
#      'tsamp': 8.192e-5,
      'tsamp' : 0.0013,
#      'foff': -0.1953125,
      'foff': -1.,
      'src_raj': 181335.2,
      'src_dej': -174958.1,
      'tstart': 58523.3437492,
      'nbeams': 1,
#      'fch1' : 2000.0,
#      'fch1': 1549.700927734375,
      'fch1' : 1549.700927734375,
      'za_start': 0.0,
      'rawdatafile': '',
      'nifs': 1,
      'nsamples': 7204148}

def create_new_filterbank(fnfil):
  try:
      import sigproc
      filhdr['rawdatafile'] = fnfil

      newhdr = ""
      newhdr += sigproc.addto_hdr("HEADER_START", None)
      for k,v in filhdr.items():
          newhdr += sigproc.addto_hdr(k, v)
      newhdr += sigproc.addto_hdr("HEADER_END", None)
      print("Writing new header to '%s'" % fnfil)
      outfile = open(fnfil, 'wb')
      outfile.write(newhdr)
      spectrum = np.zeros([filhdr['nchans']], dtype=np.uint8)
      outfile.write(spectrum)
      outfile.close()
  except:
      print("Either could not load sigproc or create filterbank")

