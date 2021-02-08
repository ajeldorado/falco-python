#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#  Modified 18 April 2019 by J. Krist - Changed to in-place transform to
#  fix memory leak; defined global variable fft_nthreads which sets the number
#  of threads the Intel FFT can use.  From experimentation, for multiprocessing
#  it should be 1, otherwise <=4.

import os
import proper
import numpy as np
import ctypes as _ctypes
from proper.prop_dftidefs import *


def mkl_fft2(a, norm = None, direction = 'forward', mkl_dir = None):
    """Wrapper for the MKL FFT routines. 
    
    This implements very fast FFT on Intel processors, much faster than the 
    stock fftpack routines in numpy/scipy. Forward/backward 2D single- or 
    double-precision FFT.

        
    Parameters
    ----------
    a : numpy ndarray
        Wavefront array
        
    norm : {None, 'ortho'}
        Normalization
        
    direction : str
        Direction for fourier transform
        
    
    Other Parameters
    ----------------
    mkl_dir : str
        Intel MKL library path if not installed in default location

    Returns
    ----------
    a : numpy ndarray 
        Fourier transform 2D array 
    """
    if os.name == 'posix':
        if  proper.system == 'Linux':
            ## for Linux system search for libmkl_rt.so
            if mkl_dir:
                mkl_lib = os.path.join(mkl_dir, 'libmkl_rt.so')
            else:
                mkl_lib = os.path.join('/opt/intel/mkl/lib/intel64', 'libmkl_rt.so')
                
            try:           
                mkl = _ctypes.cdll.LoadLibrary(mkl_lib)
            except:
                raise SystemExit('Intel MKL Library not found. Stopping.')
        elif proper.system == 'Darwin':
            ## for mac osx search for libmkl_rt.dylib
            if mkl_dir:
                mkl_lib = os.path.join(mkl_dir, 'libmkl_rt.dylib')
            else:
                mkl_lib = os.path.join('/opt/intel/mkl/lib/intel64', 'libmkl_rt.dylib')
                
            try:
                mkl = _ctypes.cdll.LoadLibrary(mkl_lib)
            except:
                raise SystemExit('Intel MKL Library not found. Stopping.')
    elif os.name == 'nt':
        ## for windows search for mk2_rt.dll
        if mkl_dir:
            mkl_lib = os.path.join(mkl_dir, 'mkl_rt.lib')
        else:
            mkl_lib = os.path.join('C:/Program Files(x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64', 'mkl_rt.lib')
        
        try:
            mkl = _ctypes.cdll.LoadLibrary(mkl_lib)
        except:
            raise SystemExit('Intel MKL Library not found. Stopping.')
    else:
        raise ValueError('Unsupported operating system %s. Stopping.' %(os.name))
    
    
    if a.dtype != np.complex128 and a.dtype != np.complex64:
        raise ValueError('prop_fftw: Unsupported data type.  Must be complex64 or complex128.')

    # Create the description handle
    Desc_Handle = _ctypes.c_void_p(0)
    dims = (_ctypes.c_int64*2)(*a.shape)
   
    if a.dtype == np.complex64:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_SINGLE, DFTI_COMPLEX, _ctypes.c_int(2), dims)
    elif a.dtype == np.complex128:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_DOUBLE, DFTI_COMPLEX, _ctypes.c_int(2), dims)

    # Set normalization factor
    if norm == 'ortho':
        if a.dtype == np.complex64:
            scale = _ctypes.c_float(1.0 / np.sqrt(np.prod(a.shape)))
        else:
            scale = _ctypes.c_double(1.0 / np.sqrt(np.prod(a.shape)))
        mkl.DftiSetValue(Desc_Handle, DFTI_FORWARD_SCALE, scale)
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)
    elif norm is None:
        if a.dtype == np.complex64:
            scale = _ctypes.c_float(1.0 / np.prod(a.shape))
        else:
            scale = _ctypes.c_double(1.0 / np.prod(a.shape))
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)

    # Set input strides if necessary
    if not a.flags['C_CONTIGUOUS']:
        in_strides = (_ctypes.c_int*3)(0, a.strides[0] // a.itemsize, a.strides[1] // a.itemsize)
        mkl.DftiSetValue(Desc_Handle, DFTI_INPUT_STRIDES, _ctypes.byref(in_strides))

    if direction == 'forward':
        fft_func = mkl.DftiComputeForward
    elif direction == 'backward':
        fft_func = mkl.DftiComputeBackward
    else:
        assert False

    mkl.DftiSetValue( Desc_Handle, DFTI_THREAD_LIMIT, _ctypes.c_int(proper.fft_nthreads) )

    # In-place FFT
    mkl.DftiCommitDescriptor( Desc_Handle )
    fft_func( Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p) )
    mkl.DftiFreeDescriptor( _ctypes.byref(Desc_Handle) )

    return 


def fft2(a, norm=None):
    """Computes the forward 2D FFT using Intel's MKL routines.

    Parameters
    ----------
    a : numpy ndarray
        Input array to transform.
        
    norm : {None, 'ortho'}
        Normalization of the transform. None (default) is same as numpy;
        'ortho' gives an orthogonal (norm-preserving) transform.
        
    Returns
    -------
    numpy ndarray
        The transformed output array.

    """
    mkl_fft2(a, norm = norm, direction = 'forward')
    return 

def ifft2(a, norm=None):
    """Computes the inverse 2D FFT using Intel's MKL routines.

    Parameters
    ----------
    a : ndarray
        Input array to transform.
        
    norm : {None, 'ortho'}
        Normalization of the transform. None (default) is same as numpy;
        'ortho' gives an orthogonal (norm-preserving) transform.
        
    Returns
    -------
    numpy ndarray
        The transformed output array.
    """
    mkl_fft2(a, norm = norm, direction = 'backward')
    return 
