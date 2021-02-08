#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#   Modified by J. Krist - 19 April 2019 - got rid of the nthreads option
#   and switched it to compute for the pre-defined number of threads for
#   non-multiprocessing (1<=nthreads<=4) and multiprocessing (nthreads=1)

import os
import proper
import _pickle as pickle
import multiprocessing
import numpy as np

def prop_fftw_wisdom(gridsize, direction = 'FFTW_FORWARD'):
    '''Determine FFTW wisdom for a given array size and number of threads.

    Write the results out to a "wisdom file" that can be used by prop_fftw.


    Parameters
    ----------
    gridsize : int
        Wavefront grid size

    direction : {'FFTW_FORWARD', 'FFTW_BACKWARD'}
        Fourier transform direction


    '''
    try:
        import pyfftw
    except ImportError:
        raise ImportError("pyfftw is not installed. Stopping.")

    home_dir = os.path.expanduser( '~' )

    data = np.ones( (gridsize,gridsize), dtype=np.complex128 )

    # threads for single PROPER process

    print( "Computing wisdom for " + str(proper.fftw_single_nthreads) + " threads" )
    pyfftw.forget_wisdom()
    pyfftw.FFTW(data, data, direction='FFTW_FORWARD', axes=(0,1), threads=proper.fftw_single_nthreads, flags=['FFTW_MEASURE','FFTW_UNALIGNED'] )
    pyfftw.FFTW(data, data, direction='FFTW_BACKWARD', axes=(0,1), threads=proper.fftw_single_nthreads, flags=['FFTW_MEASURE','FFTW_UNALIGNED'])

    wisdom = pyfftw.export_wisdom() # Export the wisdom file
    with open( os.path.join( home_dir, '.proper_{}pix'.format(str(gridsize)) + '{}threads'.format(str(proper.fftw_single_nthreads)) + '_wisdomfile' ), 'wb') as outfile:
        pickle.dump(wisdom, outfile, 2)

    # threads for multiprocessing PROPER 

    if proper.fftw_single_nthreads != proper.fftw_multi_nthreads:
        print( "Computing wisdom for " + str(proper.fftw_multi_nthreads) + " threads" )
        pyfftw.forget_wisdom()
        pyfftw.FFTW(data, data, direction='FFTW_FORWARD', axes=(0,1), threads=proper.fftw_multi_nthreads, flags=['FFTW_MEASURE','FFTW_UNALIGNED'] )
        pyfftw.FFTW(data, data, direction='FFTW_BACKWARD', axes=(0,1), threads=proper.fftw_multi_nthreads, flags=['FFTW_MEASURE','FFTW_UNALIGNED'])

        wisdom = pyfftw.export_wisdom() # Export the wisdom file
        with open( os.path.join( home_dir, '.proper_{}pix'.format(str(gridsize)) + '{}threads'.format(str(proper.fftw_multi_nthreads)) + '_wisdomfile' ), 'wb') as outfile:
            pickle.dump(wisdom, outfile, 2)

    data = 0

    return

