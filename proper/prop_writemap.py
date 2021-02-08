#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#   Modified by J. Krist - 19 April 2019 - switched to astropy.io.fits and
#   allow file overwrite

import astropy.io.fits as pyfits
import proper

def prop_writemap(dmap, filename, **kwargs):
    """Write an aberration (phase or amplitude) map to a FITS file.

    Parameters
    ----------
    dmap : numpy ndarray
        2-D error map

    filename : str
        Name of file, including extension


    Returns
    -------
        None


    Other Parameters
    ----------------
    AMPLITUDE, MIRROR, WAVEFRONT : bool
        Indicates type of map (only one may be specified)
        AMPLITUDE : amplitude map
        WAVEFRONT : wavefront map in METERS
        MIRROR : mirror surface (not wavefront) map in METERS

        Default is WAVEFRONT

    RADIUS_PIX : float
        Specifies the beam radius in the map in pixels.  If specified, the
        value of SAMPLING (if provided) is ignored.  When this file is read
        by prop_errormap, the map will be resampled as necessary to match the
        sampling of the beam.

    SAMPLING : float
        Map sampling in METERS; ignored if RADIUS_PIX is specified.
    """
    if not "RADIUS_PIX" in kwargs and not "SAMPLING" in kwargs:
        raise ValueError('PROP_WRITEMAP: Either SAMPLING or RADIUS_PIX keyword \
                    is required. Stopping.')

    nx, ny = dmap.shape

    if proper.switch_set("AMPLITUDE",**kwargs):
        maptype = "amplitude"
    elif proper.switch_set("MIRROR",**kwargs):
        maptype = "mirror"
    else:
        maptype = "wavefront"

    # assume wavefront map if maptype is not specified
    hdu = pyfits.PrimaryHDU()
    hdu.header.set("MAPTYPE", maptype, " error map type")
    hdu.header.set("X_UNIT", "meters", " X & Y units")
    if not maptype == "amplitude":
        hdu.header.set("Z_UNIT", "meters", " Error units")

    if "RADIUS_PIX" in kwargs:
        hdu.header.set("RADPIX", kwargs["RADIUS_PIX"], " beam radius in pixels")
    elif "SAMPLING" in kwargs:
        hdu.header.set("PIXSIZE", kwargs["SAMPLING"], " spacing in meters")

    hdu.header.set("XC_PIX", nx//2, " Center X pixel coordinate")
    hdu.header.set("YC_PIX", ny//2, " Center Y pixel coordinate")
    hdu.data = dmap
    hdu.writeto( filename, overwrite=True )

    return
