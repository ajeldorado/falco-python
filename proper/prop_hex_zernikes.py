#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import numpy as np
import proper

def prop_hex_zernikes(zindex, zcoeff, n, dx, hexrad, xhex = 0.0, yhex = 0.0, **kwargs):
    """Return a 2D array that is the sum of specified hexagonal Zernike
    polynomials.

    Parameters
    ----------
    zindex : numpy ndarray
        Array of zernike polynomial indices (1 to 22).

    zcoeff : numpy ndarray
        Array of coefficiencts giving the RMS amount of aberration for the
        corresponding Zernike polynomial specified by the indices in "zindex".

    n : int
        Size of array grid ( n by n pixels)

    dx : float
        Spacing of array grid in meters.

    hexred : float
        Distance from center of hexagon to a vertec in meters.

    xhex, yhex : float
        Offset of the center of the hexagon in meters from the center of the
        wavefront. If not specified, the hex is assumed to be centered in the
        wavefront at (xhex, yhex) = (0.0, 0.0).


    Returns
    -------
    abmap : numpy ndarray
        Returns the 2D array of summed zernike polynmials in meters.


    Other Parameters
    ----------------
    ROTATION : float
        If given, the aberration array will be rotated by angle degrees about
        (xhex, yhex) in the clockwise direction (first pixel in lower left).


    Notes
    -----
    The aberration map extends beyond the hexagonal region inside of
    which it is properly normalized. If adding together a number of aberration
    maps generated with this routine, to create a wavefront map for a hexagonally
    segmented aperture for example, then each segment's map must multiplied by
    its corresponding aperture prior to addition.  PROP_HEX_WAVEFRONT will do
    all this.

    The normalization error (difference between the requested and actual RMS of
    the aberration) is about 0.5%, except for Z18 (1.5%) and Z22 (1%).

    The user provides an array of Zernike polynomial indicies (1 to 22) and an
    array of corresponding polynomial coefficients that specify the RMS amount
    of each aberration. The polynomials are circular but normalized over a
    hexagonal aperture with the zero-angle axis passing through a vertex of the
    hexagon.  The order of the zernikes is the same as the first 22 circular
    zernikes used in PROPER (see prop_print_zernikes or prop_zernikes).
    Derivation of these polynomials is provided by Mahajan & Dai, J. Opt. Soc.
    Am. A, 24, 2994 (2007).
    """
    if not "ROTATION" in kwargs:
        angle = 0.0
    else:
        angle = kwargs["ROTATION"]

    zindex = np.asarray(zindex)
    zcoeff = np.asarray(zcoeff)

    hex_xc_pix = int(round(xhex / dx))
    hex_yc_pix = int(round(yhex / dx))
    hex_rad_pix = int(np.fix(hexrad / dx))

    rpix = hex_rad_pix + 2
    dpix = 2 * rpix + 1

    x1 = hex_xc_pix - rpix
    x2 = hex_xc_pix + rpix
    y1 = hex_yc_pix - rpix
    y2 = hex_yc_pix + rpix


    x = (np.arange(dpix, dtype = np.float64) + x1) * dx - xhex
    x = np.tile(x, (dpix,1))

    y = (np.arange(dpix, dtype = np.float64) + y1) * dx - yhex
    y = np.tile(y, (dpix,1)).T

    r = np.sqrt(x**2 + y**2) / hexrad
    t = np.arctan2(y, x) - angle * np.pi/180.
    zer = 0.

    for i in range(len(zcoeff)):
        iz = zindex[i]
        z = zcoeff[i]

        if iz == 1:
            zer = zer + z
        elif iz == 2:
            zer += z * 2 * np.sqrt(6/5.) * r * np.cos(t)
        elif iz == 3:
            zer += z * 2 * np.sqrt(6/5.) * r * np.sin(t)
        elif iz == 4:
            zer += z * np.sqrt(5/43.) * (12*r**2 - 5)
        elif iz == 5:
            zer += z * 2 * np.sqrt(15/7.) * r**2 * np.sin(2*t)
        elif iz == 6:
            zer += z * 2 * np.sqrt(15/7.) * r**2 * np.cos(2*t)
        elif iz == 7:
            zer += z * 4 * np.sqrt(42/3685.) * (25*r**3 - 14*r) * np.sin(t)
        elif iz == 8:
            zer += z * 4 * np.sqrt(42/3685.) * (25*r**3 - 14*r) * np.cos(t)
        elif iz == 9:
            zer += z * (4/3.) * np.sqrt(10.) * r**3 * np.sin(3*t)
        elif iz == 10:
            zer += z * 4 * np.sqrt(70/103.) * r**3 * np.cos(3*t)
        elif iz == 11:
            zer += z * (3 / np.sqrt(1072205.)) * (6020*r**4 - 5140*r**2 + 737)
        elif iz == 12:
            zer += z * (30 / np.sqrt(492583.)) * (392*r**4 - 249*r**2) * np.cos(2*t)
        elif iz == 13:
            zer += z * (30 / np.sqrt(492583.)) * (392*r**4 - 249*r**2) * np.sin(2*t)
        elif iz == 14:
            zer += z * (10/3.) * np.sqrt(7/99258181.) * ( 10. * ((297. - 598*r**2) * r**2 * np.cos(2*t)) + 5413 * r**4 * np.cos(4*t))
        elif iz == 15:
            zer += z * (10/3.) * np.sqrt(7/99258181.) * (-10. * ((297. - 598*r**2) * r**2 * np.sin(2*t)) + 5413 * r**4 * np.sin(4*t))
        elif iz == 16:
            zer += z * 2 * np.sqrt(6/1089382547.) * (70369*r - 322280*r**3 + 309540*r**5) * np.cos(t)
        elif iz == 17:
            zer += z * 2 * np.sqrt(6/1089382547.) * (70369*r - 322280*r**3 + 309540*r**5) * np.sin(t)
        elif iz == 18:
            zer += z * 4 * np.sqrt(385/295894589.) * (4365*r**5 - 3322*r**3) * np.cos(3*t)
        elif iz == 19:
            zer += z * 4 * np.sqrt(5/97.) * (35*r**5 - 22*r**3) * np.sin(3*t)
        elif iz == 20:
            zer += z * ((-2.17600248 * r + 13.23551876 *r**3 - 16.15533716 *r**5)* np.cos(t) + 5.95928883 *(r**5)*np.cos(5*t))
        elif iz == 21:
            zer += z * ((2.17600248 *r - 13.23551876 *r**3 + 16.15533716 * r**5)*np.sin(t) + 5.95928883 *(r**5)*np.sin(5*t))
        elif iz == 22:
            zer += z * (70.01749250 *r**6 - 93.07966445 *r**4 + 33.14780774 * r**2 - 2.47059083)

    zery, zerx = zer.shape
    abmap = np.zeros([n, n], dtype = np.float64)

    xleft = int(x1 + n/2)
    xright = int(xleft + dpix) - 1
    xa = 0
    xb = dpix - 1
    if (xleft < 0):
        xa = -xleft
        xleft = 0

    if (xright >= n):
        xb = dpix - 2 - xright + n

    ybottom = int(y1 + n/2)
    ytop = int(ybottom + dpix ) - 1
    ya = 0
    yb = dpix - 1
    if (ybottom < 0):
        ya = -ybottom
        ybottom = 0

    if (ytop >= n):
        yb = dpix - 2 - ytop + n

    if (xleft < n and ybottom < n and xright >= 0 and ytop >= 0):
        abmap[ybottom:ytop+1,xleft:xright+1] = zer[ya:yb+1,xa:xb+1]

    return abmap
