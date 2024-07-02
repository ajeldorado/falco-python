#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np


def testmulti1():
    lambda_min = 0.5
    lambda_max = 0.7
    nlambda = 9
    gridsize = 256
    npsf = 256
    final_sampling = 1.5e-6

    # generate array of wavelengths
    wavelength = np.arange(nlambda) / (nlambda - 1.) * (lambda_max - lambda_min) + lambda_min

    # Create DM pattern (a couple of 0.1 micron pokes)
    optval = {'use_dm': True, 'dm': np.zeros([48,48], dtype = np.float64)}
    optval['dm'][20,20] = 0.2e-6
    optval['dm'][15,25] = 0.2e-6

    # generate monchromatic fields in parallel
    (fields, sampling) = proper.prop_run_multi('multi_example', wavelength, gridsize, PASSVALUE = optval) 

    # resample fields to same scale, convert to PSFs
    psfs = np.zeros([nlambda, npsf, npsf], dtype = np.float64)
    for i in range(nlambda):
        mag = sampling[i] / final_sampling
        field = proper.prop_magnify(fields[i,:,:], mag, npsf, CONSERVE = True)
        psfs[i,:,:] = np.abs(field)**2
    
    # add PSFs together
    psf = np.sum(psfs, axis = 0) / nlambda
    
    return
    

if __name__ == '__main__':
    testmulti1()
