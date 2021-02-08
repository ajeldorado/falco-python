#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
# modified 18 April 2019 by J. Krist - Removed user-specified NTHREADS option, 
# instead set to experimentally-derived optimums.

import os
import proper
import numpy as np
import _pickle as pickle
import multiprocessing as mp


def prop_fftw( a, directionFFTW = 'FFTW_FORWARD' ):
    """Compute FFT of wavefront array using FFTW or MKL Intel FFT library routines

    Parameters
    ----------
    a : numpy ndarray
        Input wavefront

    directionFFTW : str
        Direction for the Fourier transform

    Returns
    ----------
    out : numpy ndarray
        Fourier transform of input complex array

    Raises
    ------
    ValueError
        Input array is not 2D.

    ValueError
        Data type is not double complex.
    """
    # Check array size and type
    if len(a.shape) != 2:
        raise ValueError('PROP_FFTW: Input array is not 2D. Stopping.')

    # check if the data type is double complex
    if a.dtype != np.complex128:
        raise ValueError('PROP_FFTW: Data type is not double complex. Stopping.')

    if proper.use_ffti:
        if directionFFTW == 'FFTW_FORWARD':
            proper.prop_ffti.fft2(a)
        else:
            proper.prop_ffti.ifft2(a)
    else:
        try:
            import pyfftw
        except ImportError:
            raise ImportError("pyfftw not installed. Stopping.")

        if proper.fftw_use_wisdom:
            flags = ['FFTW_UNALIGNED']
        else:
            flags = ['FFTW_UNALIGNED','FFTW_ESTIMATE']

        if directionFFTW == 'FFTW_FORWARD':
            fftw_obj = pyfftw.FFTW( a, a, direction='FFTW_FORWARD', axes=(0,1), flags=flags, threads=proper.fft_nthreads )
        else:
            fftw_obj = pyfftw.FFTW( a, a, direction='FFTW_BACKWARD', axes=(0,1), flags=flags, threads=proper.fft_nthreads )

        a = fftw_obj()

    return 
