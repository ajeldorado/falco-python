""" 
Wrapper for the MKL FFT routines. This implements very fast FFT on Intel
processors, much faster than the stock fftpack routines in numpy/scipy.

"""
from __future__ import division, print_function

import numpy as np
import ctypes as _ctypes
import os

from dftidefs import *

def load_libmkl():
    r"""Loads the MKL library if it can be found in the library load path.

    Raises
    ------
    ValueError
        If the MKL library cannot be found.

    """

    if os.name == 'posix':
        try:
            lib_mkl = os.getenv('LIBMKL')
            if lib_mkl is None:
                raise ValueError('LIBMKL environment variable not found')
            return _ctypes.cdll.LoadLibrary(lib_mkl)
        except:
            pass
        try:
            return _ctypes.cdll.LoadLibrary("libmkl_rt.dylib")
        except:
            raise ValueError('MKL Library not found')

    else:
        try:
            return _ctypes.cdll.LoadLibrary("mkl_rt.dll")
        except:
            raise ValueError('MKL Library not found')

mkl = load_libmkl()

def mkl_rfft(a, n=None, axis=-1, norm=None, direction='forward', out=None, scrambled=False):
    r"""Forward/backward 1D double-precision real-complex FFT.

    Uses the Intel MKL libraries distributed with Anaconda Python.
    Normalisation is different from Numpy!
    By default, allocates new memory like 'a' for output data.
    Returns the array containing output data.

    See Also
    --------
    rfft, irfft

    """

    if axis == -1:
        axis = a.ndim-1

    # This code only works for 1D and 2D arrays
    assert a.ndim < 3
    assert (axis < a.ndim and axis >= -1)
    assert (direction == 'forward' or direction == 'backward')

    # Convert input to complex data type if real (also memory copy)
    if direction == 'forward' and a.dtype != np.float32 and a.dtype != np.float64:
        if a.dtype == np.int64 or a.dtype == np.uint64:
            a = np.array(a, dtype=np.float64)
        else:
            a = np.array(a, dtype=np.float32)

    elif direction == 'backward' and a.dtype != np.complex128 and a.dtype != np.complex64:
        if a.dtype == np.int64 or a.dtype == np.uint64 or a.dtype == np.float64:
            a = np.array(a, dtype=np.complex128)
        else:
            a = np.array(a, dtype=np.complex64)


    order = 'C'
    if a.flags['F_CONTIGUOUS'] and not a.flags['C_CONTIGUOUS']:
        order = 'F'

    # Add zero padding or truncate if needed (incurs memory copy)
    if n is not None:
        m = n if direction == 'forward' else (n // 2 + 1)
        if a.shape[axis] < m:
            # pad axis with zeros
            pad_width = np.zeros((a.ndim, 2), dtype=np.int)
            pad_width[axis,1] = m - a.shape[axis]
            a = np.pad(a, pad_width, mode='constant')
        elif a.shape[axis] > m:
            # truncate along axis
            b = np.swapaxes(a, axis, 0)[:m,]
            a = np.swapaxes(b, 0, axis).copy()

    elif direction == 'forward':
        n = a.shape[axis]

    elif direction == 'backward':
        n = 2*(a.shape[axis]-1)


    # determine output type
    if direction == 'backward':
        out_type = np.float64
        if a.dtype == np.complex64:
            out_type = np.float32
    elif direction == 'forward':
        out_type = np.complex128
        if a.dtype == np.float32:
            out_type = np.complex64

    # Configure output array
    assert a is not out
    if out is not None:
        assert out.dtype == out_type
        for i in range(a.ndim):
            if i != axis:
                assert a.shape[i] == out.shape[i]
        if direction == 'forward':
            assert (n // 2 + 1) == out.shape[axis]
        else:
            assert out.shape[axis] == n
        assert not np.may_share_memory(a, out)
    else:
        size = list(a.shape)
        size[axis] = n // 2 + 1 if direction == 'forward' else n
        out = np.empty(size, dtype=out_type, order=order)

    # Define length, number of transforms strides
    length = _ctypes.c_int(n)
    n_transforms = _ctypes.c_int(np.prod(a.shape) // a.shape[axis])

    # For strides, the C type used *must* be long
    strides = (_ctypes.c_long*2)(0, a.strides[axis] // a.itemsize)
    if a.ndim == 2:
        if axis == 0:
            distance = _ctypes.c_int(a.strides[1] // a.itemsize)
            out_distance = _ctypes.c_int(out.strides[1] // out.itemsize)
        else:
            distance = _ctypes.c_int(a.strides[0] // a.itemsize)
            out_distance = _ctypes.c_int(out.strides[0] // out.itemsize)

    double_precision = True
    if (direction == 'forward' and a.dtype == np.float32) or (direction == 'backward' and a.dtype == np.complex64):
        double_precision = False

    # Create the description handle
    Desc_Handle = _ctypes.c_void_p(0)
    if not double_precision:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_SINGLE, DFTI_REAL, _ctypes.c_int(1), length)
    else:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_DOUBLE, DFTI_REAL, _ctypes.c_int(1), length)

    # set the storage type
    mkl.DftiSetValue(Desc_Handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX)

    # set normalization factor
    if norm == 'ortho':
        scale = _ctypes.c_double(1 / np.sqrt(n))
        mkl.DftiSetValue(Desc_Handle, DFTI_FORWARD_SCALE, scale)
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)
    elif norm is None:
        scale = _ctypes.c_double(1. / n)
        s = mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)

    # set all values if necessary
    if a.ndim != 1:
        mkl.DftiSetValue(Desc_Handle, DFTI_NUMBER_OF_TRANSFORMS, n_transforms)
        mkl.DftiSetValue(Desc_Handle, DFTI_INPUT_DISTANCE, distance)
        mkl.DftiSetValue(Desc_Handle, DFTI_OUTPUT_DISTANCE, out_distance)
        mkl.DftiSetValue(Desc_Handle, DFTI_INPUT_STRIDES, _ctypes.byref(strides))
        mkl.DftiSetValue(Desc_Handle, DFTI_OUTPUT_STRIDES, _ctypes.byref(strides))

    if scrambled:
        s = mkl.DftiSetValue(Desc_Handle, DFTI_ORDERING, DFTI_BACKWARD_SCRAMBLED)

    if direction == 'forward':
        fft_func = mkl.DftiComputeForward
    elif direction == 'backward':
        fft_func = mkl.DftiComputeBackward
    else:
        assert False

    # Not-in-place FFT
    mkl.DftiSetValue(Desc_Handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE)

    mkl.DftiCommitDescriptor(Desc_Handle)
    fft_func(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p), out.ctypes.data_as(_ctypes.c_void_p) )

    mkl.DftiFreeDescriptor(_ctypes.byref(Desc_Handle))

    return out


def mkl_fft(a, n=None, axis=-1, norm=None, direction='forward', out=None, scrambled=False):
    r"""Forward/backward 1D single- or double-precision FFT.

    Uses the Intel MKL libraries distributed with Anaconda Python.
    Normalisation is different from Numpy!
    By default, allocates new memory like 'a' for output data.
    Returns the array containing output data.

    See Also
    --------
    fft, ifft

    """

    # This code only works for 1D and 2D arrays
    assert a.ndim < 3
    assert axis < a.ndim and axis >= -1

    # Add zero padding if needed (incurs memory copy)
    '''
    if n is not None and n != a.shape[axis]:
        pad_width = np.zeros((a.ndim, 2), dtype=np.int)
        pad_width[axis,1] = n - a.shape[axis]
        a = np.pad(a, pad_width, mode='constant')
    '''

    if n is not None:
        if a.shape[axis] < n:
            # pad axis with zeros
            pad_width = np.zeros((a.ndim, 2), dtype=np.int)
            pad_width[axis,1] = n - a.shape[axis]
            a = np.pad(a, pad_width, mode='constant')
        elif a.shape[axis] > n:
            # truncate along axis
            b = np.swapaxes(a, axis, -1)[...,:n]
            a = np.swapaxes(b, -1, axis).copy()

    # Convert input to complex data type if real (also memory copy)
    if a.dtype != np.complex128 and a.dtype != np.complex64:
        if a.dtype == np.int64 or a.dtype == np.uint64 or a.dtype == np.float64:
            a = np.array(a, dtype=np.complex128)
        else:
            a = np.array(a, dtype=np.complex64)

    # Configure in-place vs out-of-place
    inplace = False
    if out is a:
        inplace = True
    elif out is not None:
        assert out.dtype == a.dtype
        assert a.shape == out.shape
        assert not np.may_share_memory(a, out)
    else:
        out = np.empty_like(a)

    # Define length, number of transforms strides
    length = _ctypes.c_int(a.shape[axis])
    n_transforms = _ctypes.c_int(np.prod(a.shape) // a.shape[axis])
    
    # For strides, the C type used *must* be long
    strides = (_ctypes.c_long*2)(0, a.strides[axis] // a.itemsize)
    if a.ndim == 2:
        if axis == 0:
            distance = _ctypes.c_int(a.strides[1] // a.itemsize)
        else:
            distance = _ctypes.c_int(a.strides[0] // a.itemsize)

    # Create the description handle
    Desc_Handle = _ctypes.c_void_p(0)
    if a.dtype == np.complex64:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_SINGLE, DFTI_COMPLEX, _ctypes.c_int(1), length)
    elif a.dtype == np.complex128:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_DOUBLE, DFTI_COMPLEX, _ctypes.c_int(1), length)

    # Set normalization factor
    if norm == 'ortho':
        if a.dtype == np.complex64:
            scale = _ctypes.c_float(1 / np.sqrt(a.shape[axis]))
        else:
            scale = _ctypes.c_double(1 / np.sqrt(a.shape[axis]))
        mkl.DftiSetValue(Desc_Handle, DFTI_FORWARD_SCALE, scale)
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)
    elif norm is None:
        if a.dtype == np.complex64:
            scale = _ctypes.c_float(1. / a.shape[axis])
        else:
            scale = _ctypes.c_double(1. / a.shape[axis])
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)

    # set all values if necessary
    if a.ndim != 1:
        mkl.DftiSetValue(Desc_Handle, DFTI_NUMBER_OF_TRANSFORMS, n_transforms)
        mkl.DftiSetValue(Desc_Handle, DFTI_INPUT_DISTANCE, distance)
        mkl.DftiSetValue(Desc_Handle, DFTI_OUTPUT_DISTANCE, distance)
        mkl.DftiSetValue(Desc_Handle, DFTI_INPUT_STRIDES, _ctypes.byref(strides))
        mkl.DftiSetValue(Desc_Handle, DFTI_OUTPUT_STRIDES, _ctypes.byref(strides))

    if scrambled:
        s = mkl.DftiSetValue(Desc_Handle, DFTI_ORDERING, DFTI_BACKWARD_SCRAMBLED)
        DftiErrorMessage(s)

    if direction == 'forward':
        fft_func = mkl.DftiComputeForward
    elif direction == 'backward':
        fft_func = mkl.DftiComputeBackward
    else:
        assert False

    if inplace:
        # In-place FFT
        mkl.DftiCommitDescriptor(Desc_Handle)
        fft_func(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p) )

    else:
        # Not-in-place FFT
        mkl.DftiSetValue(Desc_Handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE)

        mkl.DftiCommitDescriptor(Desc_Handle)
        fft_func(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p), out.ctypes.data_as(_ctypes.c_void_p) )


    mkl.DftiFreeDescriptor(_ctypes.byref(Desc_Handle))

    return out

def proper_fft2(a, norm=None, direction='forward', mkl_dir=None, fft_nthreads=0):
    r"""Forward/backward 2D single- or double-precision FFT.

    Uses the Intel MKL libraries distributed with Enthought Python.
    Normalisation is different from Numpy!
    By default, allocates new memory like 'a' for output data.
    Returns the array containing output data.

    See Also
    --------
    fft2, ifft2

    """

    # input must be complex!  Not exceptions
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

    mkl.DftiSetValue( Desc_Handle, DFTI_THREAD_LIMIT, _ctypes.c_int(fft_nthreads) )

    # In-place FFT
    mkl.DftiCommitDescriptor( Desc_Handle )
    fft_func( Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p) )
    mkl.DftiFreeDescriptor( _ctypes.byref(Desc_Handle) )

    return 

def mkl_fft2(a, norm=None, direction='forward', out=None):
    r"""Forward/backward 2D single- or double-precision FFT.

    Uses the Intel MKL libraries distributed with Enthought Python.
    Normalisation is different from Numpy!
    By default, allocates new memory like 'a' for output data.
    Returns the array containing output data.

    See Also
    --------
    fft2, ifft2

    """

    # convert input to complex data type if real (also memory copy)
    if a.dtype != np.complex128 and a.dtype != np.complex64:
        if a.dtype == np.int64 or a.dtype == np.uint64 or a.dtype == np.float64:
            a = np.array(a, dtype=np.complex128)
        else:
            a = np.array(a, dtype=np.complex64)

    # Configure in-place vs out-of-place
    inplace = False
    if out is a:
        inplace = True
    elif out is not None:
        assert out.dtype == a.dtype
        assert a.shape == out.shape
        assert not np.may_share_memory(a, out)
    else:
        out = np.empty_like(a)

    # Create the description handle
    Desc_Handle = _ctypes.c_void_p(0)
    dims = (_ctypes.c_long*2)(*a.shape)
   
    if a.dtype == np.complex64:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_SINGLE, DFTI_COMPLEX, _ctypes.c_int(2), dims)
    elif a.dtype == np.complex128:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_DOUBLE, DFTI_COMPLEX, _ctypes.c_int(2), dims)


    # Set normalization factor
    if norm == 'ortho':
        scale = _ctypes.c_double(1.0 / np.sqrt(np.prod(a.shape)))
        mkl.DftiSetValue(Desc_Handle, DFTI_FORWARD_SCALE, scale)
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)

    elif norm is None:
        scale = _ctypes.c_double(1.0 / np.prod(a.shape))
        mkl.DftiSetValue(Desc_Handle, DFTI_FORWARD_SCALE, _ctypes.c_double(1.0))
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)

        scale = _ctypes.c_float(0.)
        mkl.DftiGetValue(Desc_Handle, DFTI_BACKWARD_SCALE, _ctypes.byref(scale))


    # Set input strides if necessary
    if not a.flags['C_CONTIGUOUS']:
        in_strides = (_ctypes.c_long*3)(0, a.strides[0] // a.itemsize, a.strides[1] // a.itemsize)
        mkl.DftiSetValue(Desc_Handle, DFTI_INPUT_STRIDES, _ctypes.byref(in_strides))

    if direction == 'forward':
        fft_func = mkl.DftiComputeForward
    elif direction == 'backward':
        fft_func = mkl.DftiComputeBackward
    else:
        assert False

    if inplace:
        # In-place FFT
        mkl.DftiCommitDescriptor(Desc_Handle)
        fft_func(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p) )

    else:
        # Not-in-place FFT
        mkl.DftiSetValue(Desc_Handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE)

        # Set output strides if necessary
        if not out.flags['C_CONTIGUOUS']:
            out_strides = (_ctypes.c_long*3)(0, out.strides[0] // out.itemsize, out.strides[1] // out.itemsize)
            mkl.DftiSetValue(Desc_Handle, DFTI_OUTPUT_STRIDES, _ctypes.byref(out_strides))

        mkl.DftiCommitDescriptor(Desc_Handle)
        fft_func(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p), out.ctypes.data_as(_ctypes.c_void_p) )

    mkl.DftiFreeDescriptor(_ctypes.byref(Desc_Handle))

    return out

def cce2full(A):

    # Assume all square for now

    N = A.shape
    N_half = N[0]//2 + 1
    out = np.empty((A.shape[0], A.shape[0]), dtype=A.dtype)
    out[:, :N_half] = A

    out[1:, N_half:] = np.rot90(A[1:, 1:-1], 2).conj()

    # Complete the first row
    out[0, N_half:] = A[0, -2:0:-1].conj()

    return out

def mkl_rfft2(a, norm=None, direction='forward', out=None):
    r"""Forward/backward single- or double-precision real-complex 2D FFT.

    For more details:

    See Also
    --------
    rfft2, irfft2

    """

    assert (a.dtype == np.float32) or (a.dtype == np.float64)

    out_type = np.complex128
    if a.dtype == np.float32:
        out_type = np.complex64

    n = a.shape[1]

    # Allocate memory if needed
    if out is not None:
        assert out.dtype == out_type
        assert out.shape[1] == n // 2 + 1
        assert not np.may_share_memory(a, out)
    else:
        size = list(a.shape)
        size[1] = n // 2 + 1
        out = np.empty(size, dtype=out_type)

    # Create the description handle
    Desc_Handle = _ctypes.c_void_p(0)
    dims = (_ctypes.c_long*2)(*a.shape)
   
    if a.dtype == np.float32:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_SINGLE, DFTI_REAL, _ctypes.c_int(2), dims)
    elif a.dtype == np.float64:
        mkl.DftiCreateDescriptor(_ctypes.byref(Desc_Handle), DFTI_DOUBLE, DFTI_REAL, _ctypes.c_int(2), dims)

    # Set the storage type
    mkl.DftiSetValue(Desc_Handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX)

    # Set normalization factor
    if norm == 'ortho':
        if a.dtype == np.float32:
            scale = _ctypes.c_float(1.0 / np.sqrt(np.prod(a.shape)))
        else:
            scale = _ctypes.c_double(1.0 / np.sqrt(np.prod(a.shape)))
        
        mkl.DftiSetValue(Desc_Handle, DFTI_FORWARD_SCALE, scale)
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)
    elif norm is None:
        if a.dtype == np.float64:
            scale = _ctypes.c_float(1.0 / np.prod(a.shape))
        else:
            scale = _ctypes.c_double(1.0 / np.prod(a.shape))
        
        mkl.DftiSetValue(Desc_Handle, DFTI_BACKWARD_SCALE, scale)
    
    # For strides, the C type used *must* be long
    in_strides = (_ctypes.c_long*3)(0, a.strides[0] // a.itemsize, a.strides[1] // a.itemsize)
    out_strides = (_ctypes.c_long*3)(0, out.strides[0] // out.itemsize, out.strides[1] // out.itemsize)
    
    # mkl.DftiSetValue(Desc_Handle, DFTI_INPUT_STRIDES, _ctypes.byref(in_strides))
    mkl.DftiSetValue(Desc_Handle, DFTI_OUTPUT_STRIDES, _ctypes.byref(out_strides))

    if direction == 'forward':
        fft_func = mkl.DftiComputeForward
    elif direction == 'backward':
        fft_func = mkl.DftiComputeBackward
    else:
        assert False

    # Not-in-place FFT
    mkl.DftiSetValue(Desc_Handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE)

    # Set output strides if necessary
    if not out.flags['C_CONTIGUOUS']:
        out_strides = (_ctypes.c_int*3)(0, out.strides[0] // out.itemsize, out.strides[1] // out.itemsize)
        mkl.DftiSetValue(Desc_Handle, DFTI_OUTPUT_STRIDES, _ctypes.byref(out_strides))

    mkl.DftiCommitDescriptor(Desc_Handle)
    fft_func(Desc_Handle, a.ctypes.data_as(_ctypes.c_void_p), out.ctypes.data_as(_ctypes.c_void_p) )

    mkl.DftiFreeDescriptor(_ctypes.byref(Desc_Handle))

    return out


def rfft(a, n=None, axis=-1, norm=None, out=None, scrambled=False):
    r"""Computes the forward real-complex FFT using Intel's MKL routines.

    Faster than mkl_fft.fft for real arrays. 

    Parameters 
    ----------
    a : ndarray
        Input array to transform. It must be real.
    n : int
        Size of the transform.
    axis : int
        Axis along which the transform is computed (default is -1, summation
        over the last axis).
    norm : {None, 'ortho'}
        Normalization of the transform. None (default) is same as numpy;
        'ortho' gives an orthogonal (norm-preserving) transform.
    out : ndarray
        Points to the output array. Used when the array is preallocated or an
        in-place transform is desired. Default is None, meaning that the
        memory is allocated for the output array of the same shape as a.
    scrambled: bool, optional (default False)
        Allows the output of the FFT to be out of order if set to true. This
        can sometimes lead to better performance.

    Returns
    -------
        The transformed output array.

    """
    return mkl_rfft(a, n=n, axis=axis, norm=norm, direction='forward', out=out, scrambled=scrambled)

def irfft(a, n=None, axis=-1, norm=None, out=None, scrambled=False):
    r"""Computes the inverse complex-real FFT using Intel's MKL routines.

    Faster than mkl_fft.ifft for conjugate-even arrays. 

    Parameters
    ----------
    a : ndarray
        Input array to transform. It should be stored in the conjugate-even
        format (i.e. like output of rfft).
    n : int
        Size of the transform.
    axis : int
        Axis along which the transform is computed (default is -1, summation
        over the last axis).
    norm : {None, 'ortho'}
        Normalization of the transform. None (default) is same as numpy;
        'ortho' gives an orthogonal (norm-preserving) transform.
    out : ndarray
        Points to the output array. Used when the array is preallocated or an
        in-place transform is desired. Default is None, meaning that the
        memory is allocated for the output array of the same shape as a.
    scrambled: bool, optional (default False)
        Allows the input of the iFFT to be out of order if set to true. This
        can sometimes lead to better performance.

    Returns
    -------
        The transformed output array.

    """
    return mkl_rfft(a, n=n, axis=axis, norm=norm, direction='backward', out=out, scrambled=scrambled)

def fft(a, n=None, axis=-1, norm=None, out=None, scrambled=False):
    r"""Computes the forward FFT using Intel's MKL routines.

    Parameters
    ----------
    a : ndarray
        Input array to transform.
    n : int
        Size of the transform.
    axis : int
        Axis along which the transform is computed (default is -1, summation
        over the last axis).
    norm : {None, 'ortho'}
        Normalization of the transform. None (default) is same as numpy;
        'ortho' gives an orthogonal (norm-preserving) transform.
    out : ndarray
        Points to the output array. Used when the array is preallocated or an
        in-place transform is desired. Default is None, meaning that the
        memory is allocated for the output array of the same shape as a.
    scrambled: bool, optional (default False)
        Allows the output of the FFT to be out of order if set to true. This
        can sometimes lead to better performance.

    Returns
    -------
        The transformed output array.

    """
    return mkl_fft(a, n=n, axis=axis, norm=norm, direction='forward', out=out, scrambled=scrambled)

def ifft(a, n=None, axis=-1, norm=None, out=None, scrambled=False):
    r"""Computes the inverse FFT using Intel's MKL routines.

    Parameters
    ----------
    a : ndarray
        Input array to transform.
    n : int
        Size of the transform.
    axis : int
        Axis along which the transform is computed (default is -1, summation
        over the last axis).
    norm : {None, 'ortho'}
        Normalization of the transform. None (default) is same as numpy;
        'ortho' gives an orthogonal (norm-preserving) transform.
    out : ndarray
        Points to the output array. Used when the array is preallocated or an
        in-place transform is desired. Default is None, meaning that the
        memory is allocated for the output array of the same shape as a.
    scrambled: bool, optional (default False)
        Allows the input of the iFFT to be out of order if set to true. This
        can sometimes lead to better performance.

    Returns
    -------
        The transformed output array.

    """
    return mkl_fft(a, n=n, axis=axis, norm=norm, direction='backward', out=out, scrambled=scrambled)

def fft2(a, norm=None, out=None):
    r"""Computes the forward 2D FFT using Intel's MKL routines.

    Parameters
    ----------
    a : ndarray
        Input array to transform.
    norm : {None, 'ortho'}
        Normalization of the transform. None (default) is same as numpy;
        'ortho' gives an orthogonal (norm-preserving) transform.
    out : ndarray
        Points to the output array. Used when the array is preallocated or an
        in-place transform is desired. Default is None, meaning that the
        memory is allocated for the output array of the same shape as a.

    Returns
    -------
        The transformed output array.

    """
    proper_fft2(a, norm=norm, direction='forward')
    return fftshift(a)
    #return mkl_fft2(a, norm=norm, direction='forward', out=out)

def ifft2(a, norm=None, out=None):
    r"""Computes the inverse 2D FFT using Intel's MKL routines.

    Parameters
    ----------
    a : ndarray
        Input array to transform.
    norm : {None, 'ortho'}
        Normalization of the transform. None (default) is same as numpy;
        'ortho' gives an orthogonal (norm-preserving) transform.
    out : ndarray
        Points to the output array. Used when the array is preallocated or an
        in-place transform is desired. Default is None, meaning that the
        memory is allocated for the output array of the same shape as a.

    Returns
    -------
        The transformed output array.

    """

    proper_fft2(a, norm=norm, direction='backward')
    return fftshift(a)
    #return mkl_fft2(a, norm=norm, direction='backward', out=out)


def rfft2(a, norm=None, out=None):
    r"""Computes the forward real -> complex conjugate-even 2D FFT using
    Intel's MKL routines.

    Faster than mkl_fft.fft2 for real arrays.

    Parameters
    ----------
    a : ndarray
        Input array to transform. It must be real.
    norm : {None, 'ortho'}
        Normalization of the transform. None (default) is same as numpy;
        'ortho' gives an orthogonal (norm-preserving) transform.
    out : ndarray
        Points to the output array. Used when the array is preallocated or an
        in-place transform is desired. Default is None, meaning that the
        memory is allocated for the output array of the same shape as a.

    Returns
    -------
        The transformed output array.

    """

    return mkl_rfft2(a, norm=None, direction='forward', out=None)


def irfft2(a, norm=None, out=None):
    r"""Computes the forward conjugate-even -> real 2D FFT using Intel's MKL
    routines.

    Faster than mkl_fft.ifft2 for conjugate-even arrays.

    Parameters
    ----------
    a : ndarray
        Input array to transform. It should be stored in the conjugate-even
        format (i.e. like output of rfft2).
    norm : {None, 'ortho'}
        Normalization of the transform. None (default) is same as numpy;
        'ortho' gives an orthogonal (norm-preserving) transform.
    out : ndarray
        Points to the output array. Used when the array is preallocated or an
        in-place transform is desired. Default is None, meaning that the
        memory is allocated for the output array of the same shape as a.

    Returns
    -------
        The transformed output array.

    """

    return mkl_rfft2(a, norm=None, direction='backward', out=None)

def fftshift(x, additional_shift=None, axes=None):
    """
    Shift the zero-frequency component to the center of the spectrum, or with
    some additional offset from the center.

    This is a more generic fork of `~numpy.fft.fftshift`, which doesn't support
    additional shifts.


    Parameters
    ----------
    x : array_like
        Input array.
    additional_shift : list of length ``M``
        Desired additional shifts in ``x`` and ``y`` directions respectively
    axes : int or shape tuple, optional
        Axes over which to shift.  Default is None, which shifts all axes.

    Returns
    -------
    y : `~numpy.ndarray`
        The shifted array.
    """
    tmp = np.asarray(x)
    ndim = len(tmp.shape)
    if axes is None:
        axes = list(range(ndim))
    elif isinstance(axes, integer_types):
        axes = (axes,)

    # If no additional shift is supplied, reproduce `numpy.fft.fftshift` result
    if additional_shift is None:
        additional_shift = [0, 0]

    y = tmp
    for k, extra_shift in zip(axes, additional_shift):
        n = tmp.shape[k]
        if (n+1)//2 - extra_shift < n:
            p2 = (n+1)//2 - extra_shift
        else:
            p2 = abs(extra_shift) - (n+1)//2
        mylist = np.concatenate((np.arange(p2, n), np.arange(0, p2)))
        y = np.take(y, mylist, k)
    return y


if __name__ == "__main__":

    import time

    n_iter = 200
    N = 256

    np.seterr(all='raise')

    algos = {
            'Numpy fft2 complex128': {'transform': np.fft.fft2, 'dtype': np.complex128},
            'MKL fft2 complex128': {'transform': fft2, 'dtype': np.complex128},
            'Numpy fft2 complex64': {'transform': np.fft.fft2, 'dtype': np.complex64},
            'MKL fft2 complex64': {'transform': fft2, 'dtype': np.complex64},
            'Numpy fft complex128': {'transform': np.fft.fft, 'dtype': np.complex128},
            'MKL fft complex128': {'transform': fft, 'dtype': np.complex128},
            'Numpy fft complex64': {'transform': np.fft.fft, 'dtype': np.complex64},
            'MKL fft complex64': {'transform': fft, 'dtype': np.complex64},
            'Numpy rfft float64': {'transform': np.fft.rfft, 'dtype': np.float64},
            'MKL rfft float64': {'transform': rfft, 'dtype': np.float64},
            'Numpy rfft float32': {'transform': np.fft.rfft, 'dtype': np.float32},
            'MKL rfft float32': {'transform': rfft, 'dtype': np.float32},
            }

    for algo in algos.keys():

        A = algos[algo]['dtype'](np.random.randn(N, N))
        #C = np.zeros((N, N), dtype='complex128')
        start_time = time.time()
        for i in range(n_iter):
            algos[algo]['transform'](A)
        total = time.time() - start_time
        print(algo,":")
        print("--- %s seconds ---" % total)


