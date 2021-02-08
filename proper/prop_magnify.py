#   Copyright 2016, 2017, 2018 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#   Revised 5 March 2018 - John Krist - bug fixes; added GRID=True switch
#   to call to prop_cubic_conv
#   Modified by J. Krist on 19 April 2019 - reduced need for an extra
#   intermediate array

import proper
import numpy as np

if not proper.use_cubic_conv:
    from scipy.ndimage.interpolation import map_coordinates


def prop_magnify(image_in, mag0, size_out0 = 0, **kwargs):
    """Resample input image using damped sinc interpolation.

    Parameters
    ----------
    image_in : numpy ndarray
        2D real or complex image (square only, nx = ny)

    mag0 : float
        Magnification (e.g. 0.5 = shrink image by a factor of 2)

    size_out0 : float
        Dimension of new image (size_out by size_out). If not specified, the
        input image dimension is multiplied by the magnification.


    Returns
    -------
    image_out : numpy ndarray
        Resampled image


    Other Parameteres
    -----------------
    AMP_CONSERVE : bool
        If set, the real-valued image is field amplitude rather than intensity,
        and to conserve the resulting intensity, the interpolated image is
        divided by the magnification.

    CONSERVE : bool
        If set, the intensity in the image will be conserved. If the image is
        complex, then it is assumed that the input is an electric field, so the
        interpolated result will be divided by the magnification. If the image
        is not complex, then it is assumed that it is intensity (modulus-square
        of the amplitude), so the interpolated result will be divided by the
        square of the magnification. If the non-complex image is amplitude and
        not intensity, specify AMP_CONSERVE instead.

    QUICK : bool
        If set, python's "interpolate" function is used (with the CUBIC
        interpolation) instead of the more exact, but much slower, sinc
        interpolator. Most of the time this will work as well.
    """
    if np.ndim(image_in) != 2:
        raise Exception( "PROP_MAGNIFY: Input image is not 2-D." )

    size_in = image_in.shape[0]
    mag = float(mag0)

    if size_out0 == 0:
        size_out = int(size_in * mag)
    else:
        size_out = int(size_out0)

    if image_in.dtype == np.dtype("complex128") or image_in.dtype == np.dtype("complex64"):
        is_complex = 1
    else:
        is_complex = 0

    if proper.switch_set("QUICK",**kwargs):
        x = (np.arange(size_out, dtype = np.float64) - np.floor(size_out/2))/mag + np.floor(size_in/2)
        image_out = np.zeros([size_out, size_out], dtype = np.complex128)

        if proper.use_cubic_conv:
            image_out = proper.prop_cubic_conv(image_in, x, x, GRID=True)
        else:
            xxgrid = np.meshgrid(x, x)
            if is_complex:
                image_out.real = map_coordinates(image_in.real.T, xxgrid, order = 3, mode = "nearest")
                image_out.imag = map_coordinates(image_in.imag.T, xxgrid, order = 3, mode = "nearest")
            else:
                image_out = map_coordinates(image_in.T, xxgrid, order = 3, mode = "nearest")
    else:
        if proper.use_szoom_c:
            import ctypes as ct
            libszoom_c = ct.cdll.LoadLibrary(proper.szoom_c_lib)

            if is_complex:
                image_out = np.empty([size_out, size_out], dtype = np.complex128)

                # real and imaginary part of the complex array are contiguous in memory.
                # need to copy real and imaginary parts to double arrays to process

                image_in_array = np.copy(image_in.real.astype(np.double))
                image_out_array = np.empty((size_out,size_out), dtype = np.double)
                libszoom_c.prop_szoom_c(image_in_array.T.ctypes.data_as(ct.c_void_p),
                            ct.c_int(image_in_array.T.shape[0]),
                            image_out_array.ctypes.data_as(ct.c_void_p),
                            ct.c_int(size_out), ct.c_double(mag)
                           )
                image_out.real = image_out_array

                image_in_array[:,:] = np.copy(image_in.imag.astype(np.double))
                libszoom_c.prop_szoom_c(image_in_array.T.ctypes.data_as(ct.c_void_p),
                            ct.c_int(image_in_array.T.shape[0]),
                            image_out_array.ctypes.data_as(ct.c_void_p),
                            ct.c_int(size_out), ct.c_double(mag)
                           )
                image_out.imag = image_out_array

                del( image_in_array, image_out_array )
            else:
                image_in = image_in.astype(np.double)
                image_out = np.zeros([size_out, size_out], dtype = np.double)
                libszoom_c.prop_szoom_c(image_in.T.ctypes.data_as(ct.c_void_p),
                            ct.c_int(image_in.T.shape[0]),
                            image_out.ctypes.data_as(ct.c_void_p),
                            ct.c_int(size_out), ct.c_double(mag)
                           )
        else:
            image_out = proper.prop_szoom(image_in, mag, size_out)

    # Conserve intensity
    if proper.switch_set("CONSERVE",**kwargs):
        if is_complex:                    # image is electric field (amplitude, phase)
            image_out /= mag
        else:                             # image is intensity
            image_out /= mag**2
    elif proper.switch_set("AMP_CONSERVE",**kwargs):  #  image is amplitude
        image_out /= mag                  # image is amplitude, conserve intensity

    return image_out
