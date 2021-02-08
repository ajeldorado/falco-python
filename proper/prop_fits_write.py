#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#   Modified by J. Krist - 19 April 2019: switched to astropy.io.fits and
#   allow file overwrite

import os
import astropy.io.fits as pyfits
import proper

def prop_fits_write(fname, img, **kwargs):
    """Function to write FITS image with optional header keywords.

    Parameters
    ----------
    fname : str
        FITS image name

    img : numpy ndarray
        2D image array

    Returns
    -------
        None

    Other Parameters
    ----------------
    HEADER : dict
        Dictionary of FITS image header keywords and value
    """
    hdu = pyfits.PrimaryHDU(img)

    if "HEADER" in kwargs:
        for key,value in kwargs["HEADER"].items():
            hdu.header.set(key, value[0], value[1])

    if os.path.isfile(fname):
        os.remove(fname)

    try:
        hdu.writeto(fname, overwrite=True)
    except IOError:
        raise IOError('Unable to write FITS image %s. Stopping.' %fname)

    return
