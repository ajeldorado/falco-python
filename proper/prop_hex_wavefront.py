#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#   Revised 28 Oct 2019 - JEK - Fixed NO_APPLY and DARKCENTER logic to allow
#   False setting


import proper
import numpy as np


def prop_hex_wavefront(wf, nrings, hexrad, hexsep, zernike_val = 0, **kwargs):
    """Compute transmission and wavefront phase errors for a segmented hexagonal
    aperture and apply them to the current wavefront.

    The hexagons making up the aperture have antialiased edges. Each segment
    has user-defined hexagonal Zernike phase errors (up to Z22).

    Parameters
    ----------
    wf : obj
        WaveFront class object

    nrings : int
        Number of rings of hexagons in aperture (e.g. 1 = a central hexagon
        surrounded by a ring of hexagons)

    hexrad : float
        The distance in meters from the center of a hexagonal segment to a
        vertex.

    hexsep : float
        The distance between the centers of adjacent hexagonal segments.

    zernike_val : numpy ndarray
        Array of dimensions (22, nhex) where nhex is the number of hexagonal
        segments [nhex = nrings*(nrings+1)*3+1]. Each line in this array
        specifies the hexagonal Zernike polynomial coefficients (Z1 to Z22) for
        a segment. The Zernikes are Noll-ordered (see prop_zernikes in the
        manual for a list of them or look at the prop_hex_zernikes.py routine in
        the PROPER library). The values are in meters of RMS wavefront error.
        Even if the center segment is made dark by /DARKCENTER, there must be an
        entry for it.


    Returns
    -------
    aperture : numpy ndarray
        Transmission map for a segmented hexagonal aperture

    phase : numpy ndarray, optional
        Wavefront phase errors for a segmented hexagonal aperture


    Other Parameters
    ----------------
    DARKCENTER : bool
        If set, the central hexagonal segment transmission will be set to 0.0.

    NO_APPLY : bool
        If set, the aperture and phase maps are created but not applied to the
        wavefront.

    ROTATION : float
        The counterclockwise rotation in degrees of the aperture about its center.

    XCENTER, YCENTER : float
        The offset in meters of the aperture from the center of the wavefront.
        By default, the aperture is centered within the wavefront.
    """

    if not "XCENTER" in kwargs:
        xc = 0.

    if not "YCENTER" in kwargs:
        yc = 0.

    n = proper.prop_get_gridsize(wf)
    dx = proper.prop_get_sampling(wf)

    aperture = np.zeros([n,n], dtype = np.float64)

    if "ROTATION" in kwargs:
        angle = kwargs["ROTATION"]
        anglerad = angle * np.pi / 180
    else:
        angle = 0.
        anglerad =0.

    ## 1 May 2018 - Navtej
    ## Taking absolute value of zernike_val before summing up
    include_phase_errors = np.sum(np.abs(zernike_val)) != 0.

    if include_phase_errors:
        zernike_index = np.arange(22, dtype = np.int) + 1
        phase = np.zeros([n,n], dtype = np.float64)
    else:
        phase = 0

    segi = 0

    for iring in range(0, nrings+1):
        x = hexsep * np.cos(30*np.pi/180) * iring
        y = -nrings * hexsep + iring * hexsep * 0.5

        for iseg in range(0, 2*nrings-iring+1):
            # Create hex segment on one side
            xhex = x * np.cos(anglerad) - y * np.sin(anglerad) + xc
            yhex = x * np.sin(anglerad) + y * np.cos(anglerad) + yc

            segment = proper.prop_polygon(wf, 6, hexrad, xhex, yhex, ROTATION = angle)
      
            if (iring == 0 and iseg == nrings and proper.switch_set("DARKCENTER",**kwargs)) == False:
                aperture += segment

            segment = segment != 0

            if include_phase_errors:
                phase += segment * proper.prop_hex_zernikes(zernike_index, zernike_val[:,segi], n, dx, hexrad, xhex, yhex, ROTATION = angle)

            segi += 1

            # create hex segment on opposite side
            if iring != 0:
                xhex = -x * np.cos(anglerad) - y * np.sin(anglerad) + xc
                yhex = -x * np.sin(anglerad) + y * np.cos(anglerad) + yc

                segment = proper.prop_polygon(wf, 6, hexrad, xhex, yhex, ROTATION = angle)
                aperture += segment
                segment = segment != 0

                if include_phase_errors:
                    phase += segment * proper.prop_hex_zernikes(zernike_index, zernike_val[:,segi], n, dx, hexrad, xhex, yhex, ROTATION = angle)

                segi += 1

            y += hexsep

    if not proper.switch_set("NO_APPLY",**kwargs):
        proper.prop_multiply(wf, aperture)
        if include_phase_errors:
            proper.prop_add_phase(wf, phase)

    if include_phase_errors:
        return (aperture, phase)
    else:
        return aperture
