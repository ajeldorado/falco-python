#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#   Modified by J. Krist - 19 April 2019 - switched to astropy.io.fits

import astropy.io.fits as pyfits
import proper
import numpy as np

def prop_fits_read(fname, header = False):
    """Function to read an input FITS image.

    Parameters
    ----------
    fname : str
        FITS image name

    header : bool
        Get FITS image header? Default is False.


    Returns
    -------
    fitsarr : numpy ndarray
        2D array of input image
    """
    try:
        imgarr, imgheader = pyfits.getdata(fname, header=True, ignore_missing_end=True)
    except IOError:
        raise IOError("Unable to read FITS image %s. Stopping" %(fname))

    imgarr = imgarr.astype(np.float64)

    if header:
        return (imgarr, imgheader)
    else:
        return imgarr
