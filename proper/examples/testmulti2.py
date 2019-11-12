#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np


def testmulti2():
    wavelength = 0.6
    gridsize = 256

    # create different DM ripple patterns (50 nm amplitude)
    npatterns = 3
    optval = np.repeat({'use_dm': True, 'dm': np.zeros([48,48], dtype=np.float64)}, 3)
    x = np.dot((np.arange(48.)/47 * (2*np.pi)).reshape(48,1), np.ones([1,48], dtype = np.float64))

    for i in range(npatterns):
        optval[i]['dm'] = 5.e-8 * np.cos(4*x*(i+1))
    
    # generate monochromatic field in parallel
    (fields, sampling) = proper.prop_run_multi('multi_example', wavelength, gridsize, PASSVALUE = optval)

    return
    
if __name__ == '__main__':
    testmulti2()
