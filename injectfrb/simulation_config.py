import numpy as np

filhdr_ASKAP = {'telescope_id': 1,
      'az_start': 0.0,
      'nbits': 8,
      'source_name': 'J1813-1749',
      'data_type': 1,
      'nchans': 336,
      'machine_id': 15,
      'tsamp' : 0.0012656,
      'foff': -1.,
      'src_raj': 181335.2,
      'src_dej': -174958.1,
      'tstart': 58523.3437492,
      'nbeams': 1,
      'fch1' : 1500.,
      'za_start': 0.0,
      'rawdatafile': '',
      'nifs': 1,
      #'nsamples': 12500
      }


filhdr_Apertif = {'telescope_id': 2,
      'az_start': 0.0,
      'nbits': 8,
      'source_name': 'J1813-1749',
      'data_type': 1,
      'nchans': 1536,
      'machine_id': 15,
      'tsamp': 8.192e-5,
      'foff': -0.1953125,
      'src_raj': 181335.2,
      'src_dej': -174958.1,
      'tstart': 58523.3437492,
      'nbeams': 1,
      'fch1': 1519.50561523,
      'za_start': 0.0,
      'rawdatafile': '',
      'nifs': 1,
      #'nsamples': 12500
      }

filhdr_CHIME = {'telescope_id': 3,
      'az_start': 0.0,
      'nbits': 8,
      'source_name': 'J1813-1749',
      'data_type': 1,
      'nchans': 16384,
      'machine_id': 15,
      'tsamp': 0.00983,
      'foff': -0.0244,
      'src_raj': 181335.2,
      'src_dej': -174958.1,
      'tstart': 58523.3437492,
      'nbeams': 1,
      'fch1' : 800.,
      'za_start': 0.0,
      'rawdatafile': '',
      'nifs': 1,
      #'nsamples': 12500
      }


def create_new_filterbank(fnfil, telescope='ASKAP'):
   if telescope in ('ASKAP', 'Askap', 'askap'):
      filhdr = filhdr_ASKAP
   elif telescope in ('Apertif', 'APERTIF', 'apertif'):
      filhdr = filhdr_Apertif
   elif telescope in ('CHIME', 'Chime', 'chime'):
      filhdr = filhdr_CHIME
   else:
      raise ValueError("Could not find telescope name")

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
