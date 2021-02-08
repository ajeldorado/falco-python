#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#   Modified 3 Feb 2020 - JEK - fixed missing ".real", which was a bug


import numpy as np
import proper

def round(val):
    """
    Round the float value to nearest integer.

    Parameters
    ----------
    val : float
        Floating point value

    Returns
    -------
        : int
        Nearest integer value
    """
    if val < 0:
        return np.floor(val)
    else:
        return np.ceil(val)


def prop_szoom(image_in, mag, n_out = 0):
    """Perform image magnification about the center using damped sinc (Lancoz)
    interpolation.

    This is a slower version; typically, the external C routine prop_szoom_c
    would be used, if it was compiled. This is intended to be called from
    prop_magnify.

    Parameters
    ----------
    image_in : numpy ndarray
        Image to be magnified (must be square); may be real or complex

    mag : float
        Magnification (>1 magnifies image, <1 shrinks)

    n_out : int
        Output dimension of magnified image (n_out by n_out)

    Returns
    -------
    image_out : numpy ndarray
        Returns magnified image
    """
    dk = 6
    k = 13

    s = image_in.shape
    n_in = s[0]
    is_complex = np.iscomplexobj(image_in)
    xc_in = n_in // 2

    if n_out == 0:
        n_out = int(np.fix(n_in * mag) - k)

    u = np.zeros(k, dtype = np.float64)
    table = np.zeros([n_out, k], dtype = np.float64)

    kk = np.arange(k, dtype = np.float64) - k//2

    # precompute damped sinc kernel table
    for i in range(n_out):
        u[:] = 1.
        x_in = (i - n_out//2) / float(mag)
        xphase = x_in - round(x_in)
        x = kk - xphase
        mask = np.abs(x) <= dk
        x *= np.pi
        x_ne_0 = np.where(x != 0)
        xx = x[x_ne_0]
        u[x_ne_0] = np.sin(xx)/xx * np.sin(xx/dk)/(xx/dk)

        table[i,:] = u * mask

    if is_complex:
        image_out = np.zeros([n_out, n_out], dtype = np.complex128)
    else:
        image_out = np.zeros([n_out, n_out], dtype = np.float64)

    # Apply kernels
    for j in range(n_out):
        y_in = (j - n_out//2) / float(mag)
        y_pix = round(y_in) + n_in//2
        y1 = int(y_pix) - k//2
        y2 = int(y_pix) + k//2 + 1
        if y1 < 0 or y2 > n_in:
            continue
        strip = image_in[y1:y2,:]
        strip = np.multiply(strip, np.tile(table[j,:], (n_in,1)).T)
        if is_complex:
            for i in range(n_out):
                x_in = (i - n_out//2) / float(mag)
                x_pix = round(x_in) + n_in//2
                x1 = int(x_pix) - k//2
                x2 = int(x_pix) + k//2 + 1
                if (x1 < 0 or x2 > n_in):
                    continue
                t = table[i,:]
                ur = np.sum(strip[:,x1:x2].real, axis = 0) * t
                ui = np.sum(strip[:,x1:x2].imag, axis = 0) * t
                image_out[j,i] = np.sum(ur) + np.sum(ui)*1j
        else:
            for i in range(n_out):
                x_in = (i - n_out//2) / float(mag)
                x_pix = round(x_in) + n_in//2
                x1 = int(x_pix) - k//2
                x2 = int(x_pix) + k//2 + 1
                if (x1 < 0 or x2 > n_in):
                    continue
                u = np.sum(strip[:,x1:x2], axis = 0) * table[i,:]
                image_out[j,i] = np.sum(u)

    return image_out
