#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#   Revised 5 March 2018 - John Krist - Changed GRID option to operate
#   as it does for IDL's interpolate function.


import os
import proper
import numpy as np
import ctypes as ct
import multiprocessing as mp


def prop_cubic_conv(image_in, xval, yval, THREADED = True, GRID = True):
    """ Cubic convolution interpolation.

    Python routines calls the cubic convolution interpolation C routine.

    Parameters
    ----------
    image_in : numpy ndarray
        2-D image (real or complex)

    x : numpy ndarray
        Vector of X coordinates in input image (size_out_x elements)

    y : numpy ndarray
        Vector of Y coordinates in input image (size_out_y elements)

    THREADED : bool
        Run cubic convolution interpolation with multiple threads? Default is
        True.

    GRID : bool
        If True, x and y are vectors equal to the X and Y axes dimensions of
        the output image. If false, x and y are arrays with the same number
        of points as the output image.

    Returns
    -------
    b : numpy ndarray
        Output interpolated array

    Notes
    -----
    If compling manually, use following command to compile the C routine

        a) on Linux:
            gcc -shared -Wall -fPIC -o libcconv.so cubic_conv_c.c
            gcc -shared -Wall -fPIC -o libcconvthread.so cubic_conv_threaded_c.c

        b) on Mac
            gcc -shared -o libcconv.dylib  cubic_conv_c.c -framework Python
            gcc -shared -o libcconvthread.dylib  cubic_conv_threaded_c.c -framework Python
    """
    # Check data type of input array
    image_in_dtype = image_in.dtype
    if image_in_dtype == np.dtype("complex128") or image_in_dtype == np.dtype("complex64"):
        is_complex = 1
    else:
        is_complex = 0

    # Number of threads to use?
    if THREADED:
        nthreads = mp.cpu_count()
    else:
        nthreads = 1

    # Load cubic convolution interpolation shared library
    if proper.use_cubic_conv:
        if THREADED:
            libcconv = ct.cdll.LoadLibrary(os.path.join(proper.lib_dir, proper.cubic_conv_threaded_lib))
        else:
            libcconv = ct.cdll.LoadLibrary(os.path.join(proper.lib_dir, proper.cubic_conv_lib))
    else:
        raise ValueError('Cubic convolution interpolation shared library does not exist. Stopping.')

    # Cast input arrays as double
    xval = np.asarray(xval).astype(np.double)
    yval = np.asarray(yval).astype(np.double)

    if GRID:
        x = np.tile( xval, yval.shape[0] )
        y = np.repeat( yval, xval.shape[0] )
        xdim = xval.shape[0]
        ydim = yval.shape[0]
    else:
        x = xval.flatten()
        y = yval.flatten()
        if xval.ndim == 1:
            xdim = xval.shape[0]
            ydim = 1
        else:
            xdim = xval.shape[1]
            ydim = xval.shape[0]

    if is_complex:
        image_out = np.empty((ydim, xdim), dtype = np.complex128)

        # real and imaginary part of the complex array are contiguous in memory.
        # need to copy real and imaginary parts to double arrays to process
        image_in_real = np.copy(image_in.real.astype(np.double))
        image_in_imag = np.copy(image_in.imag.astype(np.double))

        image_out_real = np.empty((ydim,xdim), dtype = np.double)
        image_out_imag = np.empty((ydim,xdim), dtype = np.double)

        # Interplolate the real input array
        libcconv.cubic_conv_c(image_in_real.ctypes.data_as(ct.c_void_p), ct.c_int(image_in_real.shape[0]),
        ct.c_int(image_in_real.shape[1]), image_out_real.ctypes.data_as(ct.c_void_p), ct.c_int(image_out_real.shape[0]),
        ct.c_int(image_out_real.shape[1]), x.ctypes.data_as(ct.c_void_p), y.ctypes.data_as(ct.c_void_p), ct.c_int(nthreads))

        # Interplolate the imag input array
        libcconv.cubic_conv_c(image_in_imag.ctypes.data_as(ct.c_void_p), ct.c_int(image_in_imag.shape[0]),
        ct.c_int(image_in_imag.shape[1]), image_out_imag.ctypes.data_as(ct.c_void_p), ct.c_int(image_out_imag.shape[0]),
        ct.c_int(image_out_imag.shape[1]), x.ctypes.data_as(ct.c_void_p), y.ctypes.data_as(ct.c_void_p), ct.c_int(nthreads))

        image_out.real = image_out_real
        image_out.imag = image_out_imag

        del(image_in_real, image_in_imag)
    else:
        # Cast input arrays as double
        image_in = np.asarray(image_in).astype(np.double)
        image_out = np.zeros((ydim,xdim), dtype = np.double)

        # Interplolate the input array
        libcconv.cubic_conv_c(image_in.ctypes.data_as(ct.c_void_p), ct.c_int(image_in.shape[0]),
        ct.c_int(image_in.shape[1]), image_out.ctypes.data_as(ct.c_void_p), ct.c_int(image_out.shape[0]),
        ct.c_int(image_out.shape[1]), x.ctypes.data_as(ct.c_void_p), y.ctypes.data_as(ct.c_void_p), ct.c_int(nthreads))

    if image_in_dtype == np.int:
        image_out = image_out.astype(np.int)

    if ydim == 1:
        image_out = np.reshape(image_out, xdim)

    return image_out
