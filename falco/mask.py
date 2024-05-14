import numpy as np
import math
from numpy import cos, sin
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import rotate

import falco.proper as proper
import falco
from . import check
from falco.util import ceil_even, cosd, sind


def _init_proper(Dmask, dx, centering):
    """Initialize PROPER for generating mask representations."""
    check.centering(centering)

    # number of points across output array:
    if centering == "pixel":
        # Sometimes requires two more pixels when pixel centered.
        # Same size as width when interpixel centered.
        Narray = ceil_even(Dmask / dx + 0.5)
    else:
        Narray = ceil_even(Dmask / dx + 0.0)

    wl_dummy = 1e-6  # Dummy value--no propagation for just masks

    return proper.prop_begin(Narray * dx, wl_dummy, Narray, 1.0)


def falco_gen_DM_stop(dx, diamMask, centering):
    """
    Make a circular aperture to place centered on the beam at a DM.

    Parameters
    ----------
    dx : float
        spatial resolution for a pixel. Any units as long as they match that of
        diamMask
    diamMask : float
        diameter of the aperture mask. Any units as long as they match that of
        dx
    centering :
        centering of beam in array. Either 'pixel' or 'interpixel'

    Returns
    -------
    numpy ndarray
        2-D square array of a circular stop at a DM. Cropped down to the
        smallest even-sized array with no extra zero padding.

    """
    check.real_positive_scalar(dx, 'dx', TypeError)
    check.real_positive_scalar(diamMask, 'Dmask', TypeError)
    check.centering(centering)

    wf = _init_proper(diamMask, dx, centering)

    # 0 shift for pixel-centered pupil, or -dx shift for interpixel centering
    cshift = -dx / 2 * (centering == "interpixel")

    # Outer diameter of aperture
    proper.prop_circular_aperture(wf, diamMask/2., cshift, cshift)

    return np.fft.ifftshift(np.abs(wf.wfarr))


def falco_gen_pupil_Roman_CGI_20200513(Nbeam, centering, changes={}):
    """
    Generate WFIRST pupil CGI-20200513.

    Generate WFIRST pupil CGI-20200513. Options to change
    the x- or y-shear, clocking, or magnification via keys in the dict
    called changes.

    Parameters
    ----------
    Nbeam : float, int
        Number of pixels across the diameter of the aperture.
    centering : string
        String specifying the centering of the output array
    changes : dict
        Optional dictionary of specifications for altering the nominal pupil

    Returns
    -------
    pupil : numpy ndarray
        2-D amplitude map of the WFIRST pupil CGI-20200513
    """
    check.real_positive_scalar(Nbeam, 'Nbeam', TypeError)
    check.centering(centering)
    check.dictionary(changes, 'changes', TypeError)

    # Define the best-fit values for ellipses and rectangles (DO NOT CHANGE)
    primaryRadiusYpixels = 4027.25
    ODpixels = 2*primaryRadiusYpixels

    primaryRadiusX = 3990.0/ODpixels
    primaryRadiusY = primaryRadiusYpixels/ODpixels
    primaryCenterX = 0.0
    primaryCenterY = 0.0

    secondaryRadiusX = 1209.65/ODpixels
    secondaryRadiusY = 1220.0/ODpixels
    secondaryCenterX = 0.0/ODpixels
    secondaryCenterY = -2.95/ODpixels

    strutEndVecX1 = np.array([843.9, 728.0, 47.5, -192.0, -676.0, -816.65])/ODpixels
    strutEndVecY1 = np.array([550.85, 580.35, -970.65, -1097.15, 605.55, 458.85])/ODpixels

    strutEndVecX2 = np.array([1579.9, 3988.0, 2429.7, -2484.0, -3988.0, -1572.65])/ODpixels
    strutEndVecY2 = np.array([3866.85, -511.65, -3214.65, -3256.15, -504.45, 3866.85])/ODpixels

    strutCenterVecX = (strutEndVecX1 + strutEndVecX2)/2.0
    strutCenterVecY = (strutEndVecY1 + strutEndVecY2)/2.0

    strutWidthVec = np.array([257.0, 259.0, 258.0, 258.0, 259.0, 257.0])/ODpixels

    strutAngleVec = np.arctan2(strutEndVecY2-strutEndVecY1,
                               strutEndVecX2-strutEndVecX1)*(180/np.pi)

    tabRadiusVecX = np.array([1343.0, 1343.0, 1364.0])/ODpixels
    tabRadiusVecY = np.array([1353.0, 1353.0, 1374.0])/ODpixels
    tabCenterVecX = np.array([0.0, 0.0, 0.0])/ODpixels
    tabCenterVecY = np.array([53.85, 53.85, 67.6])/ODpixels

    lStrut = 0.55

    deltaAngle = 2.5*np.pi/16
    angTabStart = np.array([0.616 - deltaAngle/2.0,
                            2.54 - deltaAngle/2.0,
                            -1.57 - deltaAngle/2.0])
    angTabEnd = np.array([0.616 + deltaAngle/2.0,
                          2.54 + deltaAngle/2.0,
                          -1.57 + deltaAngle/2.0])

    xcStrutVec = strutCenterVecX
    ycStrutVec = strutCenterVecY
    angStrutVec = strutAngleVec
    wStrutVec = strutWidthVec

    # Changes to the pupil

    # (Optional) Lyot stop mode (concentric, circular ID and OD)
    if 'flagLyot' not in changes:
        flagLyot = False
    else:
        flagLyot = changes['flagLyot']
    if flagLyot:
        if 'ID' in changes:
            ID = changes['ID']
        else:
            raise ValueError('ID must be defined in Lyot stop mode.')

        if 'OD' in changes:
            OD = changes['OD']
        else:
            raise ValueError('OD must be defined in Lyot stop mode.')

    # Oversized strut features: overwrite defaults if values specified
    if 'wStrut' in changes:
        wStrutVec = np.ones(6)*changes["wStrut"]
    if 'wStrutVec' in changes:
        wStrutVec = changes["wStrutVec"]

    # Padding values for obscuration
    # Defaults of Bulk Changes:
    # (All length units are pupil diameters. All angles are in degrees.)
    xShear = changes.get("xShear", 0)
    yShear = changes.get("yShear", 0)
    magFac = changes.get("magFac", 1)
    flagRot180 = changes.get("flagRot180", False)
    clock_deg = changes.get("clock_deg", 0)
    clockRad = np.radians(clock_deg)

    # Values to use for bulk clocking, magnification, and translation
    pad_all = changes.get("pad_all", 0)
    pad_strut = changes.get("pad_strut", 0) + pad_all
    pad_COBS = changes.get("pad_COBS", 0) + pad_all
    pad_COBStabs = changes.get("pad_COBStabs", 0) + pad_all
    pad_OD = changes.get("pad_OD", 0) + pad_all

    # Rotation matrix used on center coordinates.
    rotMat = np.array([[math.cos(clockRad), -math.sin(clockRad)],
                       [math.sin(clockRad), math.cos(clockRad)]])

    # Nominal Mask Coordinates
    if centering.lower() == 'pixel':
        Narray = ceil_even(Nbeam*np.max(2*np.abs((xShear, yShear))) +
                           Nbeam*np.max([magFac, 1.0]) + 1)
    elif centering.lower() == 'interpixel':
        Narray = ceil_even(Nbeam*np.max(2*np.abs((xShear, yShear))) +
                           Nbeam*np.max([magFac, 1.0]))

    if centering.lower() == 'interpixel':
        xs = np.linspace(-(Narray-1)/2, (Narray-1)/2, Narray)/Nbeam
    else:
        xs = np.linspace(-(Narray/2), Narray/2-1, Narray)/Nbeam

    [XS, YS] = np.meshgrid(xs, xs)

    # Proper Setup Values
    Dbeam = 1  # Diameter of aperture, normalized to itself
    wl = 1e-6  # wavelength (m); Dummy value--no propagation here, so not used.
    bdf = Nbeam/Narray  # beam diameter factor in output array
    dx = Dbeam/Nbeam
    nSubsamples = 101

    if centering.lower() in ('interpixel', 'even'):
        cshift = -dx/2.0
    elif centering.lower() in ('pixel', 'odd'):
        cshift = 0
        if flagRot180:
            cshift = -dx

    # INITIALIZE PROPER
    bm = proper.prop_begin(Dbeam, wl, Narray, bdf)
    proper.prop_set_antialiasing(nSubsamples)

    # Struts
    for iStrut in range(6):
        angDeg = angStrutVec[iStrut] + clock_deg  # degrees
        wStrut = magFac*(wStrutVec[iStrut] + 2*pad_strut)
        lStrutIn = magFac*lStrut
        xc = magFac*(xcStrutVec[iStrut])
        yc = magFac*(ycStrutVec[iStrut])
        cxy = rotMat @ np.array([xc, yc]).reshape((2, 1))
        xc = cxy[0].item() + xShear
        yc = cxy[1].item() + yShear
        proper.prop_rectangular_obscuration(bm, lStrutIn, wStrut, xc+cshift,
                                            yc+cshift, ROTATION=angDeg)

    if not flagLyot:

        # PRIMARY MIRROR (OUTER DIAMETER)
        ra_OD_x = magFac*(primaryRadiusX-pad_OD)
        ra_OD_y = magFac*(primaryRadiusY-pad_OD)
        cx_OD = magFac*primaryCenterX
        cy_OD = magFac*primaryCenterY
        cxy = rotMat @ np.array([cx_OD, cy_OD]).reshape((2, 1))
        cx_OD = cxy[0].item() + xShear + cshift
        cy_OD = cxy[1].item() + yShear + cshift
        proper.prop_elliptical_aperture(bm, ra_OD_x, ra_OD_y, cx_OD, cy_OD,
                                        ROTATION=-clock_deg)

        # SECONDARY MIRROR (INNER DIAMETER)
        ra_ID_x = magFac*(secondaryRadiusX + pad_COBS)
        ra_ID_y = magFac*(secondaryRadiusY + pad_COBS)
        cx_ID = magFac*secondaryCenterX
        cy_ID = magFac*secondaryCenterY
        cxy = rotMat @ np.array([cx_ID, cy_ID]).reshape((2, 1))
        cx_ID = cxy[0].item() + xShear + cshift
        cy_ID = cxy[1].item() + yShear + cshift
        proper.prop_elliptical_obscuration(bm, ra_ID_x, ra_ID_y, cx_ID, cy_ID,
                                           ROTATION=-clock_deg)

        # Tabs where Struts Meet Secondary Mirror
        nTabs = 3
        tabCube = np.ones((Narray, Narray, nTabs))

        for iTab in range(nTabs):
            cobsTabsMask = np.zeros((Narray, Narray))

            xyShear = rotMat @ np.array([tabCenterVecX[iTab],
                                         tabCenterVecY[iTab]]).reshape((2, 1))

            XSnew = (XS + magFac*xyShear[0]) - xShear
            YSnew = (YS + magFac*xyShear[1]) - yShear
            THETAS = np.arctan2(YSnew, XSnew)

            ang1 = angTabStart[iTab] + clockRad
            ang2 = angTabEnd[iTab] + clockRad
            while (ang2 > 2*np.pi) and (ang1 > 2*np.pi):
                ang1 = ang1 - 2*np.pi
                ang2 = ang2 - 2*np.pi
            while (ang2 < -2*np.pi) and (ang1 < -2*np.pi):
                ang1 = ang1 + 2*np.pi
                ang2 = ang2 + 2*np.pi
            if (((ang2 < 2*np.pi) and (ang2 > np.pi)) and
                ((ang1 < np.pi) or ang1 > 0)):
                THETAS[THETAS < 0] = THETAS[THETAS < 0] + 2*np.pi
            elif (ang2 > 2*np.pi) and ((ang1 > np.pi) and (ang1 < 2*np.pi)):
                THETAS = THETAS + 2*np.pi
            elif (ang1 < -np.pi) and (ang2 > -np.pi):
                THETAS[THETAS > 0] = THETAS[THETAS > 0] - 2*np.pi
            elif (ang1 < -np.pi) and (ang2 < -np.pi):
                THETAS = THETAS - 2*np.pi

            cobsTabsMask[(THETAS <= ang2) & (THETAS >= ang1)] = 1.0

            # Full ellipse to be multiplied by the mask to get just tabs
            cx_tab = magFac*tabCenterVecX[iTab]
            cy_tab = magFac*tabCenterVecY[iTab]
            cxy = rotMat @ np.array([cx_tab, cy_tab]).reshape((2, 1))
            cx_tab = cxy[0].item() + xShear
            cy_tab = cxy[1].item() + yShear
            tabRadiusX = magFac*(tabRadiusVecX[iTab] + pad_COBStabs)
            tabRadiusY = magFac*(tabRadiusVecY[iTab] + pad_COBStabs)
            bm2 = proper.prop_begin(Dbeam, wl, Narray, bdf)
            proper.prop_set_antialiasing(nSubsamples)
            proper.prop_elliptical_obscuration(bm2, tabRadiusX, tabRadiusY,
                                               cx_tab+cshift, cy_tab+cshift,
                                               ROTATION=-clock_deg)
            tabEllipse = 1 - np.fft.ifftshift(np.abs(bm2.wfarr))

            tabSector = cobsTabsMask*tabEllipse

            tabCube[:, :, iTab] = 1 - tabSector

        pupil = tabCube[:, :, 0] * tabCube[:, :, 1] * tabCube[:, :, 2] *\
            np.fft.ifftshift(np.abs(bm.wfarr))

    else:  # (Lyot stop mode)

        # OUTER DIAMETER
        ra_OD = magFac*(OD/2.)
        cx_OD = xShear
        cy_OD = yShear
        proper.prop_circular_aperture(bm, ra_OD, cx_OD+cshift, cy_OD+cshift)

        # INNER DIAMETER
        ra_ID = magFac*(ID/2.)
        cx_ID = xShear
        cy_ID = yShear
        proper.prop_circular_obscuration(bm, ra_ID, cx_ID+cshift, cy_ID+cshift)

        pupil = np.fft.ifftshift(np.abs(bm.wfarr))

    if(flagRot180):
        pupil = np.rot90(pupil, 2)

    return pupil


def falco_gen_pupil_WFIRST_CGI_180718(Nbeam, centering, changes={}):
    """
    Generate WFIRST pupil CGI-180718.

    Generate WFIRST pupil CGI-180718. Options to change the x- or y-shear,
    clocking, or magnification via the keyword argument changes.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    centering : string
        String specifying the centering of the output array
    Returns
    -------
    pupil : numpy ndarray
        2-D amplitude map of the WFIRST pupil CGI-180718
    """
    check.real_positive_scalar(Nbeam, 'Nbeam', TypeError)
    check.centering(centering)
    check.dictionary(changes, 'changes', TypeError)

    OD = 1.000130208333333
    xcOD = 8.680555555555557e-06
    ycOD = 8.680555555555557e-06
    ID = 3.030133333333332e-01
    xcCOBS = -1.155555555555556e-04
    ycCOBS = -6.133333333333334e-04
    IDtabs = 3.144078947368421e-01
    xcCOBStabs = -1.973684210526340e-04
    ycCOBStabs = -6.250000000000000e-03

    wStrutVec = np.array([
        3.219259259259259e-02,
        3.219259259259259e-02,
        3.219259259259259e-02,
        3.219259259259258e-02,
        3.219259259259259e-02,
        3.219259259259259e-02,
    ])
    lStrut = 5.500000000000000e-01
    angStrutVec = np.array([
        4.308638741879215e+01,
        1.828091850580443e+01,
        -7.736372240624411e+01,
        7.746228722667239e+01,
        -1.833049685311381e+01,
        -4.310697246349373e+01,
    ])
    xcStrutVec = np.array([
        1.637164789600492e-01,
        3.311169704392094e-01,
        1.542050924925356e-01,
        -1.556442459316893e-01,
        -3.075636241385107e-01,
        -1.712399202747162e-01,
    ])
    ycStrutVec = np.array([
        2.695837795868052e-01,
        7.744558909460633e-03,
        -2.885875977555251e-01,
        -2.874651682155463e-01,
        -7.319997758726773e-04,
        2.748434070552074e-01])
    angTabStart = np.array([
        1.815774989921760e+00,
        -3.487710035839058e-01,
        -2.416523875732038e+00])
    angTabEnd = np.array([
        1.344727938801013e+00,
        -7.527300509955320e-01,
        -2.822938064533701e+00])

    # Oversized strut features: overwrite defaults if values specified
    if 'OD' in changes: OD = changes["OD"]
    if 'ID' in changes: ID = changes["ID"]
    if 'wStrut' in changes: wStrutVec = np.ones(6)*changes["wStrut"]
    if 'wStrutVec' in changes: wStrutVec = changes["wStrutVec"]

    # Padding values for obscuration
    # Defaults of Bulk Changes: (All length units are pupil diameters. 
    # All angles are in degrees.)
    if not 'xShear' in changes: changes["xShear"] = 0
    if not 'yShear' in changes: changes["yShear"] = 0
    if not 'magFac' in changes: changes["magFac"] = 1.0
    if not 'clock_deg' in changes: changes["clock_deg"] = 0.0
    if not 'flagRot180' in changes: changes["flagRot180"] = False

    # Defaults for obscuration padding: (All length units are pupil diameters.)
    if not 'pad_all' in changes: changes["pad_all"] = 0.
    if not 'pad_strut' in changes: changes["pad_strut"] = 0.
    if not 'pad_COBS' in changes: changes["pad_COBS"] = 0.
    if not 'pad_COBStabs' in changes: changes["pad_COBStabs"] = 0.
    if not 'pad_OD' in changes: changes["pad_OD"] = 0.

    # Values to use for bulk clocking, magnification, and translation
    xShear = changes["xShear"]  # - xcOD;
    yShear = changes["yShear"]  # - ycOD;
    magFac = changes["magFac"]
    clock_deg = changes["clock_deg"]
    clockRad = clock_deg*(np.pi/180.)
    flagRot180 = changes["flagRot180"]

    # Padding values. (pad_all is added to all the rest)
    pad_all = changes["pad_all"]     #0.2/100; # Uniform padding on all features
    pad_strut = changes["pad_strut"] + pad_all
    pad_COBS = changes["pad_COBS"] + pad_all
    pad_COBStabs = changes["pad_COBStabs"] + pad_all
    pad_OD = changes["pad_OD"] + pad_all  # Radial padding at the edge

    # Rotation matrix used on center coordinates.
    rotMat = np.array([[math.cos(clockRad), -math.sin(clockRad)],
                       [math.sin(clockRad), math.cos(clockRad)]])

    # Coordinates
    if centering.lower() in ('pixel', 'odd'):
        Narray = ceil_even(Nbeam*np.max(2*np.abs((xShear, yShear))) + magFac*(Nbeam+1))
    else:
        Narray = ceil_even(Nbeam*np.max(2*np.abs((xShear, yShear))) + magFac*Nbeam)

    if centering.lower() == 'interpixel':
        xs = np.linspace(-(Nbeam-1)/2, (Nbeam-1)/2, Nbeam)/Nbeam
    else:
        xs = np.linspace(-(Narray/2), Narray/2-1, Narray)/Nbeam

    [XS, YS] = np.meshgrid(xs, xs)

    # Proper Setup Values
    Dbeam = 1  # Diameter of aperture, normalized to itself
    wl = 1e-6  # Dummy value--no propagation here, so not used.
    bdf = Nbeam/Narray  # beam diameter factor in output array
    dx = Dbeam/Nbeam

    if centering.lower() in ('interpixel', 'even'):
        cshift = -dx/2
    elif centering.lower() in ('pixel', 'odd'):
        cshift = 0
        if flagRot180:
            cshift = -dx

    # INITIALIZE PROPER
    bm = proper.prop_begin(Dbeam, wl, Narray, bdf)

    # PRIMARY MIRROR (OUTER DIAMETER)
    ra_OD = magFac*(OD/2 - pad_OD)
    cx_OD = magFac*xcOD
    cy_OD = magFac*ycOD
    cxy = np.matmul(rotMat, np.array([[cx_OD], [cy_OD]]))
    cx_OD = cxy[0]+xShear
    cy_OD = cxy[1]+yShear
    proper.prop_circular_aperture(bm, ra_OD, cx_OD+cshift, cy_OD+cshift)
    # bm = proper.prop_circular_aperture(bm, ra_OD, cx_OD+cshift, cy_OD+cshift)

    # SECONDARY MIRROR (INNER DIAMETER)
    ra_ID = magFac*(ID/2 + pad_COBS)
    cx_ID = magFac*xcCOBS
    cy_ID = magFac*ycCOBS
    cxy = np.matmul(rotMat, np.array([[cx_ID], [cy_ID]]))
    cx_ID = cxy[0]+xShear
    cy_ID = cxy[1]+yShear
    proper.prop_circular_obscuration(bm, ra_ID, cx_ID+cshift, cy_ID+cshift)

    # Struts
    for istrut in range(6):
        angDeg = angStrutVec[istrut] + clock_deg  # degrees
        wStrut = magFac*(wStrutVec[istrut] + 2*pad_strut)
        lStrutIn = magFac*lStrut
        xc = magFac*(xcStrutVec[istrut])
        yc = magFac*(ycStrutVec[istrut])
        cxy = np.matmul(rotMat, np.array([[xc], [yc]]))
        xc = cxy[0]+xShear
        yc = cxy[1]+yShear
        proper.prop_rectangular_obscuration(bm, lStrutIn, wStrut, xc+cshift,\
                                            yc+cshift, ROTATION=angDeg)

    wf1 = bm.wfarr  # assign for later use

    # TABS ON SECONDARY MIRROR
    # Compute as new shape, and then multiply the obscuration with the rest of
    # the pupil.

    # SOFTWARE MASK:
    XSnew = (1/1*XS+xcCOBStabs)-xShear
    YSnew = (1/1*YS+ycCOBStabs)-yShear

    overSizeFac = 1.3
    cobsTabsMask = np.zeros([Narray, Narray])
    THETAS = np.arctan2(YSnew, XSnew)
    clockRad = np.deg2rad(clock_deg)

    if angTabStart[0] > angTabEnd[0]:
        msk1 = (XSnew**2 + YSnew**2) <= (overSizeFac*magFac*IDtabs/2)**2
        msk2 = np.logical_and(THETAS >= angTabEnd[0]+clockRad, THETAS <= angTabStart[0]+clockRad)
        msk3 = np.logical_and(THETAS >= angTabEnd[1]+clockRad, THETAS <= angTabStart[1]+clockRad)
        msk4 = np.logical_and(THETAS >= angTabEnd[2]+clockRad, THETAS <= angTabStart[2]+clockRad)
        cobsTabsMask[np.logical_and(msk1,np.logical_or(msk2,np.logical_or(msk3, msk4)))] = 1
    else:
        msk1 = (XSnew**2 + YSnew**2) <= (overSizeFac*magFac*IDtabs/2)**2
        msk2 = np.logical_and(THETAS <= angTabEnd[0]+clockRad, THETAS >= angTabStart[0]+clockRad)
        msk3 = np.logical_and(THETAS <= angTabEnd[1]+clockRad, THETAS >= angTabStart[1]+clockRad)
        msk4 = np.logical_and(THETAS <= angTabEnd[2]+clockRad, THETAS >= angTabStart[2]+clockRad)
        cobsTabsMask[np.logical_and(msk1, np.logical_or(msk2, np.logical_or(msk3, msk4)))] = 1


    # CIRCLE:
    # Initialize PROPER
    bm = proper.prop_begin(Dbeam, wl, Narray, bdf)

    # Full circle of COBS tabs--to be multiplied by the mask to get just tabs
    ra_tabs = magFac*(IDtabs/2 + pad_COBStabs)
    cx_tabs = magFac*(xcCOBStabs)
    cy_tabs = magFac*(ycCOBStabs)
    cxy = np.matmul(rotMat,np.array([[cx_tabs], [cy_tabs]]))
    cx_tabs = cxy[0]+xShear
    cy_tabs = cxy[1]+yShear

    # bm2 = prop_circular_obscuration(bm, ra_tabs,'XC',cx_tabs+cshift,'YC',cy_tabs+cshift)
    proper.prop_circular_obscuration(bm, ra_tabs, cx_tabs+cshift, cy_tabs+cshift)

    temp = 1-np.fft.ifftshift(np.abs(bm.wfarr))
    temp = cobsTabsMask*temp

    cobsTabs = 1-temp

    # Output
    pupil = cobsTabs*np.fft.ifftshift(np.abs(wf1))
    if(flagRot180):
        pupil = np.rot90(pupil, 2)

    return pupil


def falco_gen_SW_mask(inputs):
    """
    Generate binary (0 or 1) software masks for the focal plane.

    Used for making the scoring and correction regions in the focal plane.

    Detailed description here

    Parameters
    ----------
    inputs: structure with several fields:
        -pixresFP: pixels per lambda_c/D
        -rhoInner: radius of inner FPM amplitude spot (in lambda_c/D)
        -rhoOuter: radius of outer opaque FPM ring (in lambda_c/D)
        -angDeg: angular opening (degrees) on the left/right/both sides.
        -whichSide: which sides to have open.
        -centering: centering of the coordinates. 'pixel' or 'interpixel'
        -FOV: minimum desired field of view (in lambda_c/D)
        -shape: 'square' makes a square. Omitting makes a circle.
        -clockAngDeg: Dark hole rotation about the z-axis (deg)

    Returns
    -------
    maskSW: rectangular, even-sized, binary-valued software mask
    xis: vector of coordinates along the horizontal axis (in lambda_c/D)
    etas: : vector of coordinates along the vertical axis (in lambda_c/D)
    """
    check.dictionary(inputs, 'inputs', TypeError)

    # Required inputs
    pixresFP = inputs["pixresFP"]  # pixels per lambda_c/D
    rhoInner = inputs["rhoInner"]  # inner radius (in lambda_c/D)
    rhoOuter = inputs["rhoOuter"]  # outer radius (in lambda_c/D)
    angDeg = inputs["angDeg"]  # angular opening (input in degrees)
    angRad = np.radians(angDeg)
    whichSide = inputs["whichSide"]

    # Optional inputs
    centering = inputs.get("centering", "pixel")
    check.centering(centering)
    darkHoleShape = inputs.get("shape", "circle").lower()
    clockAngDeg = inputs.get("clockAngDeg", 0)
    FOV = inputs.get("FOV", rhoOuter)
    xiOffset = inputs.get("xiOffset", 0.)
    etaOffset = inputs.get("etaOffset", 0.)
    if darkHoleShape in {'square', 'rect', 'rectangle'}:
        maxExtent = np.max((1, 2*np.abs(np.cos(np.radians(clockAngDeg)))))
    else:
        maxExtent = 1
    minFOVxi = inputs.get("xiFOV", maxExtent*FOV + np.abs(xiOffset))
    minFOVeta = inputs.get("etaFOV", maxExtent*FOV + np.abs(etaOffset))

    # Output array dimensions
    if centering == "pixel":
        Nxi0 = ceil_even(2*(minFOVxi*pixresFP + 1/2))
        Neta0 = ceil_even(2*(minFOVeta*pixresFP + 1/2))
    elif centering == "interpixel":
        Nxi0 = ceil_even(2*minFOVxi*pixresFP)
        Neta0 = ceil_even(2*minFOVeta*pixresFP)
    Nxi = inputs.get("Nxi", Nxi0)
    Neta = inputs.get("Neta", Neta0)

    # Focal Plane Coordinates
    deta = dxi = 1/pixresFP
    if centering == "interpixel":
        xis = np.arange(-(Nxi - 1)/2, (Nxi + 1)/2)*dxi
        etas = np.arange(-(Neta - 1)/2, (Neta + 1)/2)*deta
    elif centering == "pixel":
        xis = np.arange(-Nxi/2, Nxi/2) * dxi
        etas = np.arange(-Neta/2, Neta/2) * deta

    [XIS, ETAS] = np.meshgrid(xis, etas)
    XIS = XIS - xiOffset
    ETAS = ETAS - etaOffset
    RHOS = np.sqrt(XIS ** 2 + ETAS ** 2)
    THETAS = np.arctan2(ETAS, XIS)

    if whichSide in {'r', 'right', 'lr', 'rl', 'leftright', 'rightleft',
                     'both'}:
        clockAngRad = 0
    elif whichSide in {'l', 'left'}:
        clockAngRad = np.pi
    elif whichSide in {'t', 'top', 'u', 'up', 'tb', 'bt', 'ud', 'du',
                       'topbottom', 'bottomtop', 'updown', 'downup'}:
        clockAngRad = np.pi/2
    elif whichSide in {'b', 'bottom', 'd', 'down'}:
        clockAngRad = 3/2*np.pi
    else:
        raise ValueError('Invalid value given for inputs["whichSide"]')

    clockAngRad = clockAngRad + np.radians(clockAngDeg)

    # Generate the Outer Mask
    # Avoidy a ratty line from the higher numerical noise floor
    # introduced by RHOS*cos().
    eps = np.finfo(float).eps
    rhoInner = rhoInner - 13*eps
    rhoOuter = rhoOuter + 13*eps
    if darkHoleShape in {'circle', 'annulus'}:
        softwareMask0 = np.logical_and(RHOS >= rhoInner,
                                       RHOS <= rhoOuter)
    elif darkHoleShape in {'square'}:
        softwareMask0 = np.logical_and(
            np.logical_or(
            np.logical_and(np.logical_and(np.logical_and(RHOS*cos(THETAS-clockAngRad)<=rhoOuter,
                                                         RHOS*cos(THETAS-clockAngRad)>=-rhoOuter),
                                                         RHOS*sin(THETAS-clockAngRad)<=rhoOuter),
                                                         RHOS*sin(THETAS-clockAngRad)>=-rhoOuter),
            np.logical_and(np.logical_and(np.logical_and(RHOS*cos(THETAS-clockAngRad)>=-rhoOuter,
                                                         RHOS*cos(THETAS-clockAngRad)<=rhoOuter),
                                                         RHOS*sin(THETAS-clockAngRad)<=rhoOuter),
                                                         RHOS*sin(THETAS-clockAngRad)>=-rhoOuter)
            ),
            RHOS >= rhoInner
        )
    elif darkHoleShape in {'rect', 'rectangle'}:
        softwareMask0 = np.logical_or(
            np.logical_and(np.logical_and(np.logical_and(RHOS*cos(THETAS-clockAngRad)>=rhoInner,
                                                         RHOS*cos(THETAS-clockAngRad)<=rhoOuter),
                                                         RHOS*sin(THETAS-clockAngRad)<=rhoOuter),
                                                         RHOS*sin(THETAS-clockAngRad)>=-rhoOuter),
            np.logical_and(np.logical_and(np.logical_and(RHOS*cos(THETAS-clockAngRad)<=-rhoInner,
                                                         RHOS*cos(THETAS-clockAngRad)>=-rhoOuter),
                                                         RHOS*sin(THETAS-clockAngRad)<=rhoOuter),
                                                         RHOS*sin(THETAS-clockAngRad)>=-rhoOuter)
            )
    elif darkHoleShape in {'d'}:
        softwareMask0 = np.logical_and(np.logical_or(
            RHOS*cos(THETAS-clockAngRad) >= rhoInner,
            RHOS*cos(THETAS-clockAngRad) <= -rhoInner),
            RHOS <= rhoOuter)
    else:
        raise ValueError('Invalid value given for inputs["shape"].')

    softwareMask = np.logical_and(softwareMask0, np.abs(np.angle(np.exp(1j*(THETAS-clockAngRad)))) <= angRad/2)

    if whichSide in {'both', 'lr', 'rl', 'leftright', 'rightleft', 'tb', 'bt',
                     'ud', 'du', 'topbottom', 'bottomtop', 'updown', 'downup'}:
        softwareMask2 = np.logical_and(softwareMask0, np.abs(np.angle(np.exp(1j *
                                (THETAS-(clockAngRad+np.pi))))) <= angRad/2)
        softwareMask = np.logical_or(softwareMask, softwareMask2)

    return softwareMask, xis, etas


def falco_gen_bowtie_FPM(inputs):
    return gen_bowtie_fpm(inputs)


def gen_bowtie_fpm(inputs):
    """
    Generate a bowtie FPM using PROPER.

    Parameters
    ----------
    inputs : dict
        dictionary of input values

    Returns
    -------
    mask : array_like
        2-D FPM representation
    """
    check.dictionary(inputs, 'inputs', TypeError)

    # Required keys
    ang = inputs["ang"]  # Opening angle on each side of the bowtie
    pixresFPM = inputs["pixresFPM"]
    rhoInner = inputs["rhoInner"]
    rhoOuter = inputs["rhoOuter"]

    # Optional keys
    xOffset = inputs.get("xOffset", 0)  # [lambda0/D]
    yOffset = inputs.get("yOffset", 0)  # [lambda0/D]
    centering = inputs.get("centering", "pixel")
    clocking = inputs.get("clocking", 0)  # [degrees]

    dx = 1.0 / pixresFPM  # lambda_c/D per pixel.
    maxAbsOffset = np.max(np.array([np.abs(xOffset), np.abs(yOffset)]))

    if centering == "interpixel":
        Narray = ceil_even(2*rhoOuter/dx + 2*maxAbsOffset/dx)
    elif centering == "pixel":
        Narray = ceil_even(2*rhoOuter/dx + 2*maxAbsOffset/dx + 1)

    Dmask = 2 * pixresFPM * rhoOuter  # Diameter of the mask

    if "Narray" in inputs:
        Narray = inputs["Narray"]

    Darray = Narray * dx  # width of array in lambda_c/D
    bdf = Dmask / Darray
    wl_dummy = 1e-6  # Dummy value--no propagation here, so not used.

    if centering == "interpixel":
        cshift = -Darray / 2 / Narray
    else:
        cshift = 0

    # INITIALIZE PROPER. Note that:  bm.dx = diam / bdf / np;
    wf = proper.prop_begin(Dmask, wl_dummy, Narray, bdf)

    # Outer opaque ring of FPM
    cx_OD = 0 + cshift + xOffset
    cy_OD = 0 + cshift + yOffset
    proper.prop_circular_aperture(wf, rhoOuter, cx_OD, cy_OD)

    # Inner spot of FPM (Amplitude transmission can be nonzero)
    cx_ID = 0 + cshift + xOffset
    cy_ID = 0 + cshift + yOffset
    innerSpot = proper.prop_ellipse(wf, rhoInner, rhoInner, cx_ID, cy_ID,
                                    DARK=True)

    # Create the bowtie region
    if ang < 180:
        # Top part
        Lside = 2*rhoOuter

        rotMat = np.array([[cosd(clocking), -sind(clocking)],
                           [sind(clocking), cosd(clocking)]])
        xTop = np.array([0, Lside*cosd(ang/2), Lside*cosd(ang/2),
                -Lside*cosd(ang/2), -Lside*cosd(ang/2)])
        yTop = np.array([0, Lside*sind(ang/2), Lside, Lside,
                         Lside*sind(ang/2)])
        for ii in range(len(xTop)):
            xy = rotMat @ np.array([xTop[ii], yTop[ii]]).reshape((2, 1))
            xTop[ii] = xy[0].item()
            yTop[ii] = xy[1].item()
        xvert = cshift + xOffset + xTop
        yvert = cshift + yOffset + yTop
        bowtieTop = proper.prop_irregular_polygon(wf, xvert, yvert, DARK=True)

        # Bottom part
        xBottom = np.array([0, Lside*cosd(ang/2), Lside*cosd(ang/2),
                            -Lside*cosd(ang/2), -Lside*cosd(ang/2)])
        yBottom = -1*np.array([0, Lside*sind(ang/2), Lside, Lside,
                               Lside*sind(ang/2)])
        for ii in range(len(xBottom)):
            xy = rotMat @ np.array([xBottom[ii], yBottom[ii]]).reshape((2, 1))
            xBottom[ii] = xy[0].item()
            yBottom[ii] = xy[1].item()
        xvert = cshift + xOffset + xBottom
        yvert = cshift + yOffset + yBottom
        bowtieBottom = proper.prop_irregular_polygon(wf, xvert, yvert,
                                                     DARK=True)

    else:
        bowtieTop = 1
        bowtieBottom = 1

    mask = np.fft.ifftshift(np.abs(wf.wfarr))  # undo PROPER's fftshift
    return mask * innerSpot * bowtieTop * bowtieBottom


def falco_gen_annular_FPM(inputs):
    return gen_annular_fpm(inputs)


def gen_annular_fpm(inputs):
    """
    Generate an annular FPM using PROPER.

    Outside the outer ring is opaque.If rhoOuter = infinity, then the outer
    ring is omitted and the mask is cropped down to the size of the inner spot.
    The inner spot has a specifyable amplitude value. The output array is the
    smallest size that fully contains the mask.

    Parameters
    ----------
    inputs : dict
        dictionary of input values

    Returns
    -------
    mask : array_like
        2-D FPM representation
    """
    check.dictionary(inputs, 'inputs', TypeError)

    # Required keys
    pixresFPM = inputs["pixresFPM"]
    rhoInner = inputs["rhoInner"]
    rhoOuter = inputs["rhoOuter"]

    # Optional keys
    xOffset = inputs.get("xOffset", 0)
    yOffset = inputs.get("yOffset", 0)
    centering = inputs.get("centering", "pixel")
    FPMampFac = inputs.get("FPMampFac", 0)

    dx = 1.0 / pixresFPM  # lambda_c/D per pixel.
    maxAbsOffset = np.max(np.array([np.abs(xOffset), np.abs(yOffset)]))

    if np.isinf(rhoOuter):
        if centering == "interpixel":
            Narray = ceil_even(2*rhoInner/dx + 2*maxAbsOffset/dx)
        elif centering == "pixel":
            Narray = ceil_even(2*rhoInner/dx + 2*maxAbsOffset/dx + 1)

        Dmask = 2 * pixresFPM * rhoInner  # Diameter of the mask

    else:

        if centering == "interpixel":
            Narray = ceil_even(2*rhoOuter/dx + 2*maxAbsOffset/dx)
        elif centering == "pixel":
            Narray = ceil_even(2*rhoOuter/dx + 2*maxAbsOffset/dx + 1)

        Dmask = 2 * pixresFPM * rhoOuter  # Diameter of the mask

    if "Narray" in inputs:
        Narray = inputs["Narray"]

    Darray = Narray * dx  # width of array in lambda_c/D
    bdf = Dmask / Darray
    wl_dummy = 1e-6  # wavelength (m); Dummy value

    if centering == "interpixel":
        cshift = -Darray / 2 / Narray
    else:
        cshift = 0

    # INITIALIZE PROPER. Note that:  bm.dx = Darray / bdf / np;
    wf = proper.prop_begin(Dmask, wl_dummy, Narray, bdf)
    proper.prop_set_antialiasing(101)

    if not np.isinf(rhoOuter):
        # Outer opaque ring of FPM
        cx_OD = 0 + cshift + xOffset
        cy_OD = 0 + cshift + yOffset
        proper.prop_circular_aperture(wf, rhoOuter, cx_OD, cy_OD)

    # Inner spot of FPM (Amplitude transmission can be nonzero)
    cx_ID = 0 + cshift + xOffset
    cy_ID = 0 + cshift + yOffset
    innerSpot = proper.prop_ellipse(wf, rhoInner, rhoInner, cx_ID, cy_ID,
                                    DARK=True) * (1 - FPMampFac) + FPMampFac

    mask = np.fft.ifftshift(np.abs(wf.wfarr))  # undo PROPER's fftshift

    return mask * innerSpot  # Include the inner FPM spot


def falco_gen_bowtie_LS(inputs):
    return gen_bowtie_lyot_stop(inputs)


def gen_bowtie_lyot_stop(inputs):
    """
    Generate a sideways-bowtie-shaped Lyot stop using PROPER.

    Parameters
    ----------
    inputs : dict
        dictionary of input values.

    Returns
    -------
    mask : numpy ndarray
        2-D output mask.

    """
    check.dictionary(inputs, 'inputs', TypeError)

    Nbeam = inputs["Nbeam"]  # number of points across the incoming beam
    ID = inputs["ID"]  # inner diameter of mask (in pupil diameters)
    OD = inputs["OD"]  # outer diameter of mask (in pupil diameters)
    ang = inputs["ang"]  # opening angle of the upper and lower bowtie wedges [degrees]

    Dbeam = 1.  # inputs.Dbeam; # diameter of the beam at the mask [pupil diameters]
    dx = Dbeam/Nbeam
    Dmask = Dbeam  # width of the beam (so can have zero padding if LS is undersized) [meters]

    # Optional inputs and their defaults
    centering = inputs['centering'] if 'centering' in inputs.keys() else 'pixel'  # Default to pixel centering
    xShear = inputs['xShear'] if 'xShear' in inputs.keys() else  0.  # x-axis shear of mask [pupil diameters]
    yShear = inputs['yShear'] if 'yShear' in inputs.keys() else  0.  # y-axis shear of mask [pupil diameters]
    clocking = -inputs['clocking'] if 'clocking' in inputs.keys() else 0.  # Clocking of the mask [degrees]
    magfac = inputs['magfac'] if 'magfac' in inputs.keys() else 1.  # magnification factor of the pupil diameter

    if(centering == 'pixel'):
        Narray = ceil_even(magfac*Nbeam + 1 + 2*Nbeam*np.max(np.abs(np.array([xShear, yShear]))))
    elif(centering == 'interpixel'):
        Narray = ceil_even(magfac*Nbeam + 2*Nbeam*np.max(np.abs(np.array([xShear, yShear]))))

    Darray = Narray*dx  # width of the output array [meters]
    bdf = Dmask/Darray  # beam diameter factor in output array
    wl_dummy = 1e-6  # wavelength (m); Dummy value--no propagation here, so not used.

    # No shift for pixel-centered pupil, or -Dbeam/Narray/2 shift for inter-pixel centering
    cshift = -dx/2 if 'interpixel' in centering else 0.

    # % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

    # INITIALIZE PROPER
    bm = proper.prop_begin(Dmask, wl_dummy, Narray, bdf)

    # PRIMARY MIRROR (OUTER DIAMETER)
    ra_OD = (Dbeam*OD/2.)*magfac
    cx_OD = cshift + xShear
    cy_OD = cshift + yShear
    proper.prop_circular_aperture(bm, ra_OD, cx_OD, cy_OD)

    # SECONDARY MIRROR (INNER DIAMETER)
    ra_ID = (Dbeam*ID/2.)*magfac
    cx_ID = cshift + xShear
    cy_ID = cshift + yShear
    proper.prop_circular_obscuration(bm, ra_ID, cx_ID, cy_ID)

    mask = np.fft.ifftshift(np.abs(bm.wfarr))

    # Create the bowtie region
    if(ang < 180):
        ang2 = 90. - ang/2.
        bm2 = bm
        Lside = 1.1*ra_OD  # Have the triangle go a little past the edge of the circle

        yvert0 = np.array([0., Lside*sind(ang2), Lside*sind(ang2),
                           -Lside*sind(ang2), -Lside*sind(ang2), 0.])

        # Right triangular obscuration
        xvert0 = np.array([0., Lside*cosd(ang2), Lside, Lside,
                           Lside*cosd(ang2), 0.])
        xvert = xvert0.copy()
        yvert = yvert0.copy()
        for ii in range(len(xvert0)):
            xy = np.array([[cosd(clocking), sind(clocking)],
                           [-sind(clocking), cosd(clocking)]]) @ \
                np.array([xvert0[ii], yvert0[ii]]).reshape(2, 1)
            xvert[ii] = xy[0]
            yvert[ii] = xy[1]
            pass
        bowtieRight = proper.prop_irregular_polygon(bm, cshift+xShear+xvert,
                                                    cshift+yShear+yvert, DARK=True)

        # Left triangular obscuration
        xvert0 = -np.array([0., Lside*cosd(ang2), Lside, Lside,
                            Lside*cosd(ang2), 0.])
        xvert = xvert0.copy()
        yvert = yvert0.copy()
        for ii in range(len(xvert0)):
            xy = np.array([[cosd(clocking), sind(clocking)],
                           [-sind(clocking), cosd(clocking)]]) @ \
                np.array([xvert0[ii], yvert0[ii]]).reshape(2, 1)
            xvert[ii] = xy[0]
            yvert[ii] = xy[1]
            pass
        bowtieLeft = proper.prop_irregular_polygon(bm2, cshift+xShear+xvert,
                                                   cshift+yShear+yvert, DARK=True)

        mask = mask*bowtieRight*bowtieLeft
        pass

    return mask


def falco_gen_pupil_LUVOIR_A_final(inputs):
    """
    Generate the LUVOIR Design A (Final) telescope pupil.

    Parameters
    ----------
    inputs : dict
        dictionary of input values.

    Returns
    -------
    mask : numpy ndarray
        2-D output mask.

    """
    check.dictionary(inputs, 'inputs', TypeError)

    # Optional inputs and their defaults
    centering = inputs.get("centering", "pixel")
    check.centering(centering)
    xShear = inputs.get("xShear", 0)
    yShear = inputs.get("yShear", 0)
    clockDeg = inputs.get("clock_deg", 0)
    magFac = inputs.get("magFac", 1)
    wGap_m = inputs.get("wGap_m", 6.0e-3)
    flagRot180deg = inputs.get("flagRot180deg", False)
    shearMax = np.max(np.abs(np.array([xShear, yShear])))  # [pupil diameters]

    # (Optional) Lyot stop mode (concentric, circular ID and OD)
    flagLyot = inputs.get("flagLyot", False)
    if flagLyot:
        if "ID" not in inputs or "OD" not in inputs:
            raise ValueError('ID and OD must be defined in Lyot stop mode.')
        else:
            ID = inputs['ID']
            OD = inputs['OD']

    # Rotation matrix used on center coordinates.
    rotMat = np.array([[cosd(clockDeg), -sind(clockDeg)],
                       [sind(clockDeg), cosd(clockDeg)]])

    # % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

    # USER INPUTS
    Nbeam = inputs["Nbeam"]  # number of points across the incoming beam
    Nap = Nbeam  # number of points across FULL usable pupil
    width_hex0 = 1.2225  # flat-to-flat (m)
    Dap = (12*width_hex0 + 11*wGap_m)  # (12*width_hex0 + 12*wGap_m)
    dx = Dap/Nap
    xShearM = Dap*xShear  # meters
    yShearM = Dap*yShear  # meters

    if "pixel" == centering:
        Narray = 2+ceil_even((1+2*shearMax)*Nbeam*np.max(np.array([magFac, 1])) + 1)
        cShift = -dx if flagRot180deg else 0.
    elif "interpixel" == centering:
        Narray = 2+ceil_even((1+2*shearMax)*Nbeam*np.max(np.array([magFac, 1])))
        cShift = -dx/2.

    # For PROPER
    wl_dummy = 1e-6  # wavelength (m)
    bdf = Nbeam/Narray  # beam diameter factor in output array

    width_hex = magFac*width_hex0  # flat-to-flat (m)
    nrings = 6
    hexrad = 2./np.sqrt(3.)*width_hex/2.
    hexgap = magFac*wGap_m  # (m)
    hexsep = width_hex + hexgap  # center to center of neighboring segments
    if "wStrut" in inputs:
        wStrutM = inputs["wStrut"] * Dap  # meters
    else:
        wStrutM = 0.15*magFac  # meters

    # Generate the input pupil for LUVOIR
    bm = proper.prop_begin(Dap, wl_dummy, Narray, bdf)
    proper.prop_set_antialiasing(33)

    if not flagLyot:
        ap = falco_hex_aperture_LUVOIR_A(bm, nrings, hexrad, hexsep,
                                         cShift+xShearM, cShift+yShearM,
                                         DARK=True, ROTATION=clockDeg)
    else:
        ap = 1

        # OUTER DIAMETER
        ra_OD = magFac * (OD/2) * Dap
        proper.prop_circular_aperture(bm, ra_OD, cShift+xShearM,
                                      cShift+yShearM)

        # INNER DIAMETER
        ra_ID = magFac * (ID/2) * Dap
        proper.prop_circular_obscuration(bm, ra_ID, cShift+xShearM,
                                         cShift+yShearM)

    # Add the struts
    if wStrutM > 0:
        xc = 0
        yc = magFac*Dap/4
        xyc = rotMat @ np.array([xc, yc]).reshape((2, 1))
        xc = xyc[0]
        yc = xyc[1]
        proper.prop_rectangular_obscuration(bm, wStrutM, 7*width_hex,
                                            xc+xShearM+cShift,
                                            yc+yShearM+cShift,
                                            ROTATION=clockDeg)
        len_1b = (np.sqrt(93)+0.5)*hexrad
        xc = 1.5*hexrad
        yc = -11*np.sqrt(3)/4*hexrad
        xyc = rotMat @ np.array([xc, yc]).reshape((2, 1))
        xc = xyc[0]
        yc = xyc[1]
        proper.prop_rectangular_obscuration(bm, wStrutM, len_1b,
                                            xc+xShearM+cShift,
                                            yc+yShearM+cShift,
                                            ROTATION=12.7+clockDeg)

        xc = -1.5*hexrad
        yc = -11*np.sqrt(3)/4*hexrad
        xyc = rotMat @ np.array([xc, yc]).reshape((2, 1))
        xc = xyc[0]
        yc = xyc[1]
        proper.prop_rectangular_obscuration(bm, wStrutM, len_1b,
                                          xc+xShearM+cShift,
                                          yc+yShearM+cShift,
                                          ROTATION=-12.7+clockDeg)

    mask = np.fft.ifftshift(np.abs(bm.wfarr)) * ap
    # Get rid of overlapping segment edges at low res if gap size is 0.
    mask[mask > 1] = 1.
    if flagRot180deg:
        mask = np.rot90(mask, 2)

    return mask


def falco_gen_pupil_LUVOIR_B(inputs, **kwargs):
    """
    Generate the LUVOIR B telescope pupil.

    Parameters
    ----------
    inputs : dict
        dictionary of input values.

    Returns
    -------
    mask : numpy ndarray
        2-D output mask.

    Notes
    -----
    Coordinates of hex segments to skip:
    1 13 114 115 126 127
    1 12 113 114 125 126
    """
    check.dictionary(inputs, 'inputs', TypeError)

    # Optional inputs and their defaults
    centering = inputs.get("centering", "pixel")
    check.centering(centering)
    xShear = inputs.get("xShear", 0)
    yShear = inputs.get("yShear", 0)
    clockDeg = inputs.get("clock_deg", 0)
    magFac = inputs.get("magFac", 1)
    wGap_m = inputs.get("wGap_m", 6.0e-3)
    flagRot180deg = inputs.get("flagRot180deg", False)
    shearMax = np.max(np.abs(np.array([xShear, yShear])))  # [pupil diameters]

    Nbeam0 = inputs["Nbeam"]  # points across the circumscribed circle
    scaleFac = 0.96075
    Nbeam = scaleFac*Nbeam0  # Change Nbeam to be flat-to-flat. makes the beam size match the hypergaussian approach
    nrings = 4
    width_hex0 = 0.955  # flat-to-flat (m)
    Dap = ((2*nrings)*width_hex0 + (2*nrings-1)*wGap_m)  # (12*width_hex0 + 12*hexgap0)
    dx = Dap/Nbeam
    xShearM = 1/scaleFac*Dap*xShear  # meters
    yShearM = 1/scaleFac*Dap*yShear  # meters

    if "pixel" == centering:
        Narray = ceil_even((1+2*shearMax) * 1.02 * Nbeam *
                           np.max(np.array([magFac, 1])) + 1)
        if flagRot180deg:
            cshift = -dx
        else:
            cshift = 0
    elif "interpixel" == centering:
        Narray = ceil_even((1+2*shearMax) * 1.02 * Nbeam *
                           np.max(np.array([magFac, 1])))
        cshift = -dx/2

    # For PROPER
    wl_dummy = 1e-6  # wavelength (m)
    bdf = Nbeam/Narray  # beam diameter factor in output array

    width_hex = magFac*width_hex0  # flat-to-flat (m)

    hexrad = 2./np.sqrt(3.)*width_hex/2.
    hexgap = magFac*wGap_m  # (m)
    hexsep = width_hex + hexgap  # segments' center-to-center distance

    # Generate the input pupil for LUVOIR
    bm = proper.prop_begin(Dap, wl_dummy, Narray, bdf)
    proper.prop_set_antialiasing(33)
    ap = falco_hex_aperture_LUVOIR_B(bm, nrings, hexrad, hexsep,
                                     cshift+xShearM, cshift+yShearM,
                                     ROTATION=clockDeg)

    mask = np.fft.ifftshift(np.abs(bm.wfarr)) * ap

    return mask


def falco_hex_aperture_LUVOIR_A(wf, nrings, hexrad, hexsep, xc=0.0, yc=0.0,
                                **kwargs):
    """
    Return a mask consisting of multiple hexagons for the LUVOIR A pupil.

    This is used for generating the primary mirror for the LUVOIR telescope.
    The hexagons have antialiased edges. This routine does not modify the
    wavefront.

    Parameters
    ----------
    wf : object
        WaveFront class object
    nrings : int
        Number of rings of hexagons in aperture (e.g. 1 = a central hexagon
        surrounded by a ring of hexagons)
    hexrad : float
        Distance in meters from center of a hexagon segment to a vertex.
    hexsep : float
        The distance between the centers of adjacent hexagons.
    xc, yc : float
        The offset in meters of the aperture from the center of the wavefront.
        By default, the aperture is centered within the wavefront.

    Optional Keywords
    -----------------
    DARK : boolean
        If set, the central hexagonal segment will be set to 0.0.
    ROTATION : float
        The counterclockwise rotation in degrees of the aperture about its
        center.

    Returns
    -------
       ap : numpy ndarray
        2-D mask made of hexagons
    """
    ngrid = wf.ngrid

    ap = np.zeros([ngrid, ngrid], dtype=np.float64)

    isDark = True if "DARK" in kwargs else False
    angle = kwargs.get("ROTATION", 0)

    counter = 0
    for iring in range(nrings+1):
        x = hexsep * iring * cosd(30)
        y = hexsep * (iring * cosd(60) - nrings)

        for iseg in range(2*nrings-iring+1):
            xhex = xc + x*cosd(angle) - y*sind(angle)
            yhex = yc + x*sind(angle) + y*cosd(angle)

            if iring != 0 or not (iseg == nrings and isDark):
                counter += 1
                if not any(counter == np.array([1, 12, 113, 114, 125, 126])):
                    ap += proper.prop_polygon(wf, 6, hexrad, xhex, yhex,
                                              ROTATION=angle)

            if iring != 0:
                xhex = -x*cosd(angle) - y*sind(angle) + xc
                yhex = -x*sind(angle) + y*cosd(angle) + yc
                counter += 1
                if not any(counter == np.array([1, 12, 113, 114, 125, 126])):
                    ap = ap + proper.prop_polygon(wf, 6, hexrad, xhex, yhex,
                                                  ROTATION=angle)

            y += hexsep

    return ap


def falco_hex_aperture_LUVOIR_B(wf, nrings, hexrad, hexsep, xc=0., yc=0.,
                                **kwargs):
    """
    Return a mask consisting of multiple hexagons for the LUVOIR B pupil.

    This is used for generating the primary mirror for the LUVOIR telescope.
    The hexagons have antialiased edges. This routine does not modify the
    wavefront.

    Parameters
    ----------
    wf : object
        WaveFront class object
    nrings : int
        Number of rings of hexagons in aperture (e.g. 1 = a central hexagon
        surrounded by a ring of hexagons)
    hexrad : float
        The distance in meters from the center of a hexagon segment to a
        vertex.
    hexsep : float
        The distance between the centers of adjacent hexagons.
    xc, yc : float
        The offset in meters of the aperture from the center of the wavefront.
        By default, the aperture is centered within the wavefront.

    Optional Keywords
    -----------------
    DARK : boolean
        If set, the central hexagonal segment will be set to 0.0.
    ROTATION : float
        The counterclockwise rotation in degrees of the aperture about its
        center.

    Returns
    -------
        numpy ndarray
        A hexagonal mask
    """
    ngrid = wf.ngrid

    ap = np.zeros([ngrid, ngrid], dtype=np.float64)

    if "ROTATION" in kwargs:
        angle = kwargs["ROTATION"]
        angle_rad = angle * np.pi/180.
    else:
        angle = 0.0
        angle_rad = 0.0

    counter = 0
    for iring in range(0, nrings+1):
        x = hexsep * np.cos(30 * np.pi/180.) * iring
        y = -nrings * hexsep + iring * hexsep * 0.5
        for iseg in range(0, 2*nrings-iring+1):
            xhex = x * np.cos(angle_rad) - y * np.sin(angle_rad) + xc
            yhex = x * np.sin(angle_rad) + y * np.cos(angle_rad) + yc
            if (iring != 0 or not (iseg == nrings and "DARK" in kwargs)):
                counter += 1
                if not any(counter == np.array([1, 9, 52, 60])):
                    ap = ap + proper.prop_polygon(wf, 6, hexrad, xhex, yhex,
                                                  ROTATION=angle)

            if (iring != 0):
                xhex = -x * np.cos(angle_rad) - y * np.sin(angle_rad) + xc
                yhex = -x * np.sin(angle_rad) + y * np.cos(angle_rad) + yc
                counter += 1
                if not any(counter == np.array([1, 9, 53, 61])):
                    ap = ap + proper.prop_polygon(wf, 6, hexrad, xhex, yhex,
                                                  ROTATION=angle)

            y += hexsep

    return ap


def falco_gen_pupil_Simple(inputs):
    """
    Generate a custom simple circular pupil with an ID, OD, and struts.

    Parameters
    ----------
    inputs : dict
        Dictionary of input parameters

    Returns
    -------
    pupil : numpy ndarray
        2-D pupil mask

    """
    check.dictionary(inputs, 'inputs', TypeError)

    # Required dictionary keys
    Nbeam = inputs["Nbeam"]  # Aperture diameter in pixel widths
    Narray = inputs["Npad"]  # Number of points across 2-D, NxN output array
    OD = inputs["OD"]  # pupil outer diameter, can be < 1

    # Optional dictionary keys
    wStrut = inputs.get("wStrut", 0.)  # width of each strut [pupil diameters]
    angStrut = inputs.get("angStrut", [])
    angStrutVec = np.atleast_1d(angStrut)

    # if 'angStrut' in inputs:
    #     angStrutVec = inputs["angStrut"]  # Azimuthal locations
    #     angStrutVec = np.array(angStrutVec)
    # else:
    #     angStrutVec = np.array([])
    #     wStrut = 0
    ID = inputs.get("ID", 0.)  # central obscuration diam [pupil diameters]
    centering = inputs.get("centering", "pixel")
    xStretch = inputs.get("xStretch", 1.)
    clocking = inputs.get("clocking", 0.)  # [degrees]
    xShear = inputs.get("xShear", 0.)  # [pupil diameters]
    yShear = inputs.get("yShear", 0.)  # [pupil diameters]
    flagHG = inputs.get("flagHG", False)

    # Checks on dict keys
    check.real_nonnegative_scalar(wStrut, 'wStrut', TypeError)
    check.centering(centering)
    if not isinstance(flagHG, bool):
        raise TypeError("inputs['flagHG'] must be a bool")
    if ID > OD:
        raise ValueError("Inner diameter is larger than outer diameter.")

    # By default, don't use hyger-gaussians for anti-aliasing the edges.
    if not flagHG:

        # Create outer aperture
        inpOuter = {}
        inpOuter["Nbeam"] = Nbeam
        inpOuter["Narray"] = Narray
        inpOuter["radiusX"] = xStretch*0.5*OD
        inpOuter["radiusY"] = 0.5*OD
        inpOuter["centering"] = centering
        inpOuter["clockingDegrees"] = clocking
        inpOuter["xShear"] = xShear
        inpOuter["yShear"] = yShear
        apOuter = gen_ellipse(inpOuter)

        # Create inner obscuration
        if ID > 0:
            inpInner = {}
            inpInner["Nbeam"] = Nbeam
            inpInner["Narray"] = Narray
            inpInner["radiusX"] = xStretch*0.5*ID
            inpInner["radiusY"] = 0.5*ID
            inpInner["centering"] = centering
            inpInner["clockingDegrees"] = clocking
            inpInner["xShear"] = xShear
            inpInner["yShear"] = yShear
            apInner = 1.0 - gen_ellipse(inpInner)
        else:
            apInner = 1

        # Create strut obscurations
        if angStrutVec.size == 0:
            apStruts = 1
        else:
            # INITIALIZE PROPER
            Dbeam = 1.0  # diameter of beam (normalized to itself)
            dx = Dbeam/Nbeam
            Darray = Narray*dx
            wl_dummy = 1e-6  # dummy value
            bdf = Dbeam/Darray  # beam diameter fraction
            if centering == 'pixel':
                cshift = 0
            elif centering == 'interpixel':
                cshift = -dx/2.
            bm = proper.prop_begin(Dbeam, wl_dummy, Narray, bdf)
            proper.prop_set_antialiasing(101)

            # STRUTS
            lStrut = 0.6  # [pupil diameters]
            rcStrut0 = lStrut / 2.0
            for iStrut in range(angStrutVec.size):
                ang = angStrutVec[iStrut] + clocking
                proper.prop_rectangular_obscuration(
                    bm, lStrut, wStrut, rcStrut0*cosd(ang)+cshift+xShear,
                    rcStrut0*sind(ang)+cshift+yShear, ROTATION=ang)
            apStruts = np.fft.ifftshift(np.abs(bm.wfarr))

        # Combine all features
        pupil = apOuter*apInner*apStruts

    else:
        hg_expon = 1000  # hyper-gaussian exponent for anti-aliasing
        hg_expon_spider = 100  # hyper-gaussian exponent for anti-aliasing
        apRad = Nbeam/2.  # aperture radius in samples

        # Create coordinates
        if centering == 'pixel':
            x = np.arange(-Narray/2, Narray/2)
        elif centering == 'interpixel':
            x = np.arange(-(Narray-1)/2, (Narray-1)/2+1)
        RHO = falco.util.radial_grid(x, xStretch=xStretch)
        THETA = falco.util.azimuthal_grid(x, xStretch=xStretch)

        if ID > 0:
            pupil = np.exp(-(RHO/(apRad*OD))**hg_expon) - \
                np.exp(-(RHO/(apRad*ID))**hg_expon)
        else:
            pupil = np.exp(-(RHO/(apRad*OD))**hg_expon)

        # Create spiders
        if wStrut > 0:
            try:
                halfwidth = wStrut*apRad
                for ang in angStrutVec:
                    pupil *= (1 - np.exp(-(RHO*np.sin(THETA-ang*np.pi/180)
                                           / halfwidth)**hg_expon_spider) *
                              (RHO*np.cos(THETA-ang*np.pi/180) > 0))
            except:
                raise TypeError("inputs['angStrut'] must be an iterable")

    return pupil


def falco_gen_pupil_customHex(inputs):
    """
    Generate a custom segmented pupil comprised of hexagonal segments.

    Parameters
    ----------
    inputs : dict
        Dictionary of input parameters

    Returns
    -------
    pupil : numpy ndarray
        2-D pupil mask

    """
    check.dictionary(inputs, 'inputs', TypeError)

    hg_expon = 1000  # hyper-gaussian exponent for anti-aliasing
    hg_expon_spider = 100  # hyper-gaussian exponent for anti-aliasing

    N = inputs["Npad"]  # Number of samples in NxN grid
    OD = inputs["OD"]  # pupil outer diameter, can be < 1
    ID = inputs["ID"]  # central obscuration radius
    apRad = inputs["Nbeam"]/2.  # aperture radius in samples

    if 'wStrut' in inputs:
        wStrut = inputs["wStrut"]  # width of all struts [pupil diameters], float
        check.real_nonnegative_scalar(wStrut, 'wStrut', TypeError)

        angStrutVec = inputs["angStrut"]  # Azimuthal locations of the radial struts, array_like
    else:
        wStrut = 0

    # Create coordinates
    [X, Y] = np.meshgrid(np.arange(-N/2, N/2), np.arange(-N/2, N/2))
    [THETA, RHO] = falco.util.cart2pol(X, Y)

    inputs["apDia"] = inputs["Nbeam"]
    if('pistons' in inputs.keys()):
        pupil0 = falco.hexsegmirror.get_field(inputs)
    else:
        pupil0 = falco.hexsegmirror.get_support(inputs)

    # Create inner and outer circles
    if(ID > 0):
        pupil = np.exp(-(RHO/(apRad*OD))**hg_expon) - np.exp(-(RHO/(apRad*ID))**hg_expon)
    else:
        pupil = np.exp(-(RHO/(apRad*OD))**hg_expon)

    pupil = pupil*pupil0

    # Create spiders
    if(wStrut > 0):
        halfwidth = wStrut*2.*apRad
        try:
            for ang in angStrutVec:
                pupil = pupil*(1.-np.exp(-(RHO*np.sin(THETA-ang*np.pi/180.)/halfwidth)**hg_expon_spider) *
                                (RHO*np.cos(THETA-ang*np.pi/180.)>0))
        except:  # not iterable
            raise TypeError("inputs['angStrut'] must be an iterable")

    return pupil


# def falco_gen_pupil_LUVOIR_B(Nbeam):
#     """
#     Generate the LUVOIR B pupil.

#     Parameters
#     ----------
#     Nbeam : float, int
#         Number of points across the pupil diameter.

#     Returns
#     -------
#     pupil : numpy ndarray
#         2-D pupil amplitude for LUVOIR B

#     """
#     D = 7.989  # meters, circumscribed. Segments 0.955m flat-to-flat. 6mm gaps.
#     wGap = 6e-3/D  # Fractional width of segment gaps

#     inputs = {}
#     inputs["Nbeam"] = Nbeam/0.925  # number of points across the pupil diameter
#     inputs["wGap"] = wGap*Nbeam  # number of samples across segment gaps
#     inputs["numRings"] = 4  # Number of rings in hexagonally segmented mirror
#     inputs["Npad"] = int(2**(falco.util.nextpow2(Nbeam)))
#     inputs["ID"] = 0  # central obscuration radius
#     inputs["OD"] = 1  # pupil outer diameter, can be < 1
#     inputs["angStrut"] = np.array([])  # Angles of the struts (deg)
#     inputs["wStrut"] = 0  # np.array([]) # Width of the struts (fraction of pupil diam.)

#     missingSegments = np.ones(falco.hexsegmirror.count_segments(inputs["numRings"]),)
#     for index in range(6):
#         missingSegments[38+index*4 - 1] = 0

#     inputs["missingSegments"] = missingSegments

#     pupil = falco_gen_pupil_customHex(inputs)

#     return pupil


def falco_gen_vortex_mask(charge, N):
    """
    Generate a vortex phase mask.

    Parameters
    ----------
    charge : int, float
        Charge of the vortex mask.
    N : int
        Number of points across output array.

    Returns
    -------
    vortex : numpy ndarray
        2-D vortex phase mask

    """
    check.real_scalar(charge, 'charge', TypeError)
    check.positive_scalar_integer(N, 'N', TypeError)
    return np.exp(1j*charge*falco.util.azimuthal_grid(np.arange(-N/2., N/2.)))


def gen_ellipse(inputs):

    pupil = falco_gen_ellipse(inputs)

    return pupil


def falco_gen_ellipse(inputs):
    """
    Generate a rotated ellipse with antialiased edges.

    Parameters
    ----------
    inputs : dict
        dictionary of input values.

    Returns
    -------
    pupil : numpy ndarray
        2-D output mask.

    """
    check.dictionary(inputs, 'inputs', TypeError)

    Nbeam = inputs['Nbeam']
    Narray = inputs['Narray']
    radiusX = inputs['radiusX']
    radiusY = inputs['radiusY']

    # Optional dictionary keys
    centering = inputs["centering"] if('centering' in inputs) else 'pixel'
    clockingDegrees = inputs["clockingDegrees"] if('clockingDegrees' in inputs) else 0.
    clockingRadians = (np.pi/180.)*clockingDegrees
    xShear = inputs["xShear"] if('xShear' in inputs) else 0.
    yShear = inputs["yShear"] if('yShear' in inputs) else 0.
    magFac = inputs["magFac"] if('magFac' in inputs) else 1

    if centering == 'pixel':
        x = np.linspace(-Narray/2., Narray/2. - 1, Narray)/float(Nbeam)
    elif centering == 'interpixel':
        x = np.linspace(-(Narray-1)/2., (Narray-1)/2., Narray)/float(Nbeam)

    y = x
    x = x - xShear
    y = y - yShear
    [X, Y] = np.meshgrid(x,y)
    dx = x[1] - x[0]
    radius = 0.5

    RHO = 1/magFac*0.5*np.sqrt(
        1/(radiusX)**2*(np.cos(clockingRadians)*X + np.sin(clockingRadians)*Y)**2
        + 1/(radiusY)**2*(np.sin(clockingRadians)*X - np.cos(clockingRadians)*Y)**2
        )

    halfWindowWidth = np.max(np.abs((RHO[1, 0]-RHO[0, 0], RHO[0, 1] - RHO[0, 0])))
    pupil = -1*np.ones(RHO.shape)
    pupil[np.abs(RHO) < radius - halfWindowWidth] = 1
    pupil[np.abs(RHO) > radius + halfWindowWidth] = 0
    grayInds = np.array(np.nonzero(pupil == -1))
    # print('Number of grayscale points = %d' % grayInds.shape[1])

    upsampleFactor = 101
    dxUp = dx/float(upsampleFactor)
    xUp = np.linspace(-(upsampleFactor-1)/2., (upsampleFactor-1)/2., upsampleFactor)*dxUp
    # xUp = (-(upsampleFactor-1)/2:(upsampleFactor-1)/2)*dxUp
    [Xup, Yup] = np.meshgrid(xUp, xUp)

    subpixel = np.zeros((upsampleFactor, upsampleFactor))

    for iInterior in range(grayInds.shape[1]):

        subpixel = 0*subpixel

        xCenter = X[grayInds[0, iInterior], grayInds[1, iInterior]]
        yCenter = Y[grayInds[0, iInterior], grayInds[1, iInterior]]
        RHOup = 0.5*np.sqrt(
        1/(radiusX)**2*(np.cos(clockingRadians)*(Xup+xCenter) +
                        np.sin(clockingRadians)*(Yup+yCenter))**2
        + 1/(radiusY)**2*(np.sin(clockingRadians)*(Xup+xCenter) -
                          np.cos(clockingRadians)*(Yup+yCenter))**2)

        subpixel[RHOup <= radius] = 1
        pixelValue = np.sum(subpixel)/float(upsampleFactor**2)
        pupil[grayInds[0, iInterior], grayInds[1, iInterior]] = pixelValue

    return pupil


def rotate_shift_downsample_pupil_mask(arrayIn, nBeamIn, nBeamOut, xOffset,
                                       yOffset, rotDeg):
    """
    Translate, rotate, and downsample a pixel-centered mask.

    Parameters
    ----------
    arrayIn : np.ndarray
        2-D, pixel-centered array containing the pupil mask.
    nBeamIn : float
        Number of points across beam at starting resolution.
    nBeamOut : float
        Number of points across beam at final resolution.
    xOffset, yOffset : float
        x- and y-offsets of the mask in output-sized pixels
    rotDeg : float
        Amount to rotate the array about the center pixel in degrees.

    Returns
    -------
    arrayOut : np.ndarray
        2-D, even-sized, square array containing the resized mask.
    """
    check.twoD_array(arrayIn, 'arrayIn', TypeError)
    check.real_positive_scalar(nBeamIn, 'nBeamIn', TypeError)
    check.real_positive_scalar(nBeamOut, 'nBeamOut', TypeError)
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)
    check.real_scalar(rotDeg, 'rotDeg', TypeError)

    if nBeamIn <= nBeamOut:
        raise ValueError('This function is for downsampling only.')
    else:
        if arrayIn.shape[0] % 2 == 0:  # if in an even-sized array
            # Crop assuming mask is pixel centered with row 0 and column 0 empty.
            arrayIn = arrayIn[1::, 1::]

        # Array sizes
        dxIn = 1./nBeamIn
        dxOut = 1./nBeamOut
        nArrayIn = arrayIn.shape[0]
        nArrayOut = falco.util.ceil_odd(nArrayIn*dxIn/dxOut + 2
                             + 2.*np.max((np.abs(xOffset), np.abs(yOffset))))
        # 2 pixels added to guarantee the offset mask is fully contained in
        # the output array.

        # array-centered coordinates of input matrix [pupil diameters]
        x0 = np.arange(-(nArrayIn-1.)/2., (nArrayIn)/2., 1)*dxIn
        [X0, Y0] = np.meshgrid(x0, x0)
        R0 = np.sqrt(X0**2 + Y0**2)
        Window = 0*R0
        Window[R0 <= dxOut/2.] = 1
        Window = Window/np.sum(Window)

        # To get good grayscale edges, convolve with the correct window
        # before downsampling.
        f_window = np.fft.ifft2(np.fft.ifftshift(Window))*nArrayIn
        f_arrayin = np.fft.ifft2(np.fft.ifftshift(arrayIn))*nArrayIn
        A = np.fft.fftshift(np.fft.fft2(f_window*f_arrayin))
        A = np.real(A)

        if not rotDeg == 0:
            A = rotate(A, -rotDeg, reshape=False)

        x1 = (np.arange(-(nArrayOut-1.)/2., nArrayOut/2., 1) - xOffset)*dxOut
        y1 = (np.arange(-(nArrayOut-1.)/2., nArrayOut/2., 1) - yOffset)*dxOut

        # RectBivariateSpline is faster in 2-D than interp2d
        interp_spline = RectBivariateSpline(x0, x0, A)
        Atemp = interp_spline(y1, x1)

        arrayOut = np.zeros((nArrayOut+1, nArrayOut+1))
        arrayOut[1::, 1::] = np.real(Atemp)

        return arrayOut
