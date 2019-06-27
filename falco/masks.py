import numpy as np
import os
import poppy
import proper
import scipy.interpolate
import scipy.ndimage

from . import utils


def annular_fpm(pixres_fpm, rho_inner, rho_outer, fpm_amp_factor=0.0,
                rot180=False, centering='pixel', **kwargs):
    """
    Generate an annular FPM using POPPY.

    Outside the outer ring is opaque. If rho_outer = infinity, the outer
    mask is omitted and the mask is cropped down to the size of the inner spot.
    The inner spot has a specifyable amplitude value.
    The output array is the smallest size that fully contains the mask.

    Parameters
    ----------
    pixres_fpm : float
        resolution in pixels per lambda_c/D
    rho_inner : float
        radius of inner FPM amplitude spot (in lambda_c/D)
    rho_outer : float
        radius of outer opaque FPM ring (in lambda_c/D). Set to
        infinity for an occulting-spot-only FPM.
    fpm_amp_factor : float
        amplitude transmission of inner FPM spot. Default is 0.0.
    rot180 : bool
        Optional, flag to rotate
    centering : string
        Either 'pixel' or 'interpixel'

    Returns
    -------
    mask: ndarray
        cropped-down, 2-D FPM representation. amplitude only
    """

    dxi_ul = 1 / pixres_fpm  # lambda_c/D per pixel. "UL" for unitless

    offset = 1/2 if centering == 'interpixel' else 0

    if not np.isfinite(rho_outer):
        # number of points across the inner diameter of the FPM.
        narray = utils.ceil_even(2 * (rho_inner / dxi_ul + offset))
    else:
        # number of points across the outer diameter of the FPM.
        narray = utils.ceil_even(2 * (rho_outer / dxi_ul + offset))

    xshift = 0
    yshift = 0
    darray = narray * dxi_ul  # width of array in lambda_c/D

    # 0 for pixel-centered FPM, or -diam/Narray for inter-pixel centering
    if centering == 'interpixel':
        cshift = -dxi_ul / 2
    elif centering == 'pixel':
        cshift = -dxi_ul if rot180 else 0

    else:
        raise ValueError("Invalid value for centering parameter")

    # Method note: The algorithm in falco-matlab works in units of lambda/D.
    # Everything in POPPY works natively in arcseconds or meters. We can
    # make a shortcut here and just substitute coordinates in arcsec for lambda/D.
    # That's fine to do for the present purposes of just drawing a circle.

    fpm = poppy.AnnularFieldStop(radius_inner=rho_inner,
                                 radius_outer=rho_outer,
                                 shift_x=cshift + xshift,
                                 shift_y=cshift + yshift)
    mask = fpm.sample(npix=narray, grid_size=darray)

    if fpm_amp_factor != 0:
        # poppy doesn't support gray circular occulting masks, but we can
        # simulate that by just adding back some of the intensity.
        fpm.radius_inner = 0
        mask_no_inner = fpm.sample(npix=narray, grid_size=darray)
        mask = mask * fpm_amp_factor + mask_no_inner * (1 - fpm_amp_factor)

    return mask


def _init_proper(Dmask, dx, centering):
    assert(centering in ("pixel", "interpixel"))

    # number of points across output array:
    if centering == "pixel":
        # Sometimes requires two more pixels when pixel centered. Same size as width when interpixel centered.
        Narray = 2 * np.ceil(0.5 * (Dmask / dx + 0.5))
    else:
        Narray = 2 * np.ceil(0.5 * (Dmask / dx + 0.0))  # Same size as width when interpixel centered.

    wl_dummy = 1e-6  # % wavelength (m); Dummy value--no propagation here, so not used.
    return proper.prop_begin(Narray * dx, wl_dummy, Narray, 1.0)


def falco_gen_DM_stop(dx, Dmask, centering):
    diam = Dmask  # diameter of the mask (meters)
    # minimum even number of points across to fully contain the actual aperture (if interpixel centered)
    NapAcross = Dmask / dx

    wf = _init_proper(Dmask, dx, centering)

    # 0 shift for pixel-centered pupil, or -dx shift for inter-pixel centering
    cshift = -dx / 2 * (centering == "interpixel")

    # Outer diameter of aperture
    proper.prop_circular_aperture(wf, diam / 2, cshift, cshift)

    return np.fft.ifftshift(np.abs(wf.wfarr))

def falco_gen_pupil_WFIRST_CGI_180718(Nbeam, centering, **kwargs):
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
     2.748434070552074e-01,
     ])
    angTabStart = np.array([
     1.815774989921760e+00,
     -3.487710035839058e-01,
     -2.416523875732038e+00,
    ])
    angTabEnd =np.array([
     1.344727938801013e+00,
     -7.527300509955320e-01,
     -2.822938064533701e+00,
    ])

    ### Changes to the pupil

def falco_gen_pupil_WFIRST_20180103(Nbeam, centering, rot180deg=False):
    pupil_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "pupil_WFIRST_CGI_20180103.png")
    pupil0 = scipy.misc.imread(pupil_file)
    pupil0 = np.rot90(pupil0, 2 + 2 * rot180deg)

    pupil1 = np.sum(pupil0, axis=2)
    pupil1 = pupil1/np.max(pupil1)

    # Temporarily using 0th order interpolation to ensure the result is identical to MATLAB's.
    # In MATLAB code, this is equivalent to floor(interp2(Xs0,Xs0.',pupil1,Xs1,Xs1.','nearest',0));
    if centering in ("interpixel", "even"):
        xs = np.arange(0, Nbeam + 1) * len(pupil1) / float(Nbeam)
        Xs = np.meshgrid(xs, xs, indexing="ij")
        return np.floor(scipy.ndimage.map_coordinates(pupil1, Xs, order=0, prefilter=False))
    else:
        xs = np.arange(0, Nbeam + 1) * len(pupil1) / float(Nbeam) - 0.5
        Xs = np.meshgrid(xs, xs, indexing="ij")
        temp = np.floor(scipy.ndimage.map_coordinates(pupil1, Xs, order=0, prefilter=False))
        return np.pad(temp, ((1, 0), (1, 0)), "constant", constant_values=(0, 0))


def falco_gen_SW_mask(pixresFP, rhoInner, rhoOuter, angDeg, whichSide, FOV=None, centering="pixel"):
    if FOV is None:
        FOV = rhoOuter

    angRad = np.radians(angDeg)

    # Number of points across each axis. Crop the vertical (eta) axis if angDeg<180 degrees.
    if centering == "interpixel":
        Nxi = utils.ceil_even(2 * FOV * pixresFP)  # Number of points across the full FPM
        Neta = utils.ceil_even(2 * np.sin(angRad / 2) * FOV * pixresFP)
    else:
        Nxi = utils.ceil_even(2 * (FOV * pixresFP + 0.5))  # Number of points across the full FPM
        Neta = utils.ceil_even(2 * (np.sin(angRad / 2) * FOV * pixresFP + 0.5))

    # Focal Plane Coordinates
    deta = dxi = 1.0 / pixresFP

    if centering == "interpixel":
        xis = np.arange(-(Nxi - 1) / 2, (Nxi + 1) / 2) * dxi
        etas = np.arange(-(Neta - 1) / 2, (Neta + 1) / 2) * deta
    else:
        xis = np.arange(-Nxi / 2, Nxi / 2) * dxi
        etas = np.arange(-Neta / 2, Neta / 2) * deta

    [XIS, ETAS] = np.meshgrid(xis, etas)
    RHOS = np.sqrt(XIS ** 2 + ETAS ** 2)
    TAN = np.arctan(ETAS / XIS)

    # Generate the Software Mask
    maskSW = 1.0 * (RHOS >= rhoInner) * (RHOS <= rhoOuter) * (TAN <= angRad/2) * (TAN >= -angRad/2)

    # Determine if it is one-sided or not
    if whichSide in ("L", "left"):
        maskSW[XIS >= 0] = 0
    elif whichSide in ("R", "right"):
        maskSW[XIS <= 0] = 0
    elif whichSide in ("T", "top"):
        maskSW[ETAS <= 0] = 0
    elif whichSide in ("B", "bottom"):
        maskSW[ETAS >= 0] = 0

    return maskSW, xis, etas


def falco_gen_pupil_WFIRSTcycle6_LS(Nbeam, Dbeam, ID, OD, strut_width, centering, rot180deg=False):
    strut_width = strut_width * Dbeam  # now in meters
    dx = Dbeam / Nbeam

    clock_deg = 0
    magfacD = 1
    xshift = 0
    yshift = 0
    pad_strut = 0
    Dmask = Dbeam  # % width of the beam (so can have zero padding if LS is undersized) (meters)
    diam = Dmask  # width of the mask (meters)
    # minimum even number of points across to fully contain the actual aperture (if interpixel centered)
    NapAcross = Dmask / dx

    wf = _init_proper(Dmask, dx, centering)

    # 0 shift for pixel-centered pupil, or -dx shift for inter-pixel centering
    if centering == "interpixel":
        cshift = -dx / 2
    elif rot180deg:
        cshift = -dx
    else:
        cshift = 0

    # DATA FROM THE VISIO FILE
    D0 = 8  # inches, pupil diameter in Visio file
    x0 = -26  # inches, pupil center in x in Visio file
    y0 = 20.25  # inches, pupil center in y in Visio file
    Dconv = diam/D0  # conversion factor from inches and Visio units to meters

    # PRIMARY MIRROR (OUTER DIAMETER)
    ra_OD = (Dbeam*OD / 2) * magfacD
    cx_OD = cshift + xshift
    cy_OD = cshift + yshift
    proper.prop_circular_aperture(wf, ra_OD, cx_OD, cy_OD)

    # SECONDARY MIRROR (INNER DIAMETER)
    ra_ID = (Dbeam * ID / 2) * magfacD
    cx_ID = cshift + xshift
    cy_ID = cshift + yshift
    proper.prop_circular_obscuration(wf, ra_ID, cx_ID, cy_ID)

    sx_s = magfacD * (3.6*(diam/D0) + pad_strut)
    sy_s = magfacD * (strut_width + pad_strut)
    clock_rot = np.array([[np.cos(np.radians(clock_deg)), -np.sin(np.radians(clock_deg))],
                          [np.sin(np.radians(clock_deg)), np.cos(np.radians(clock_deg))]])

    def _get_strut_cxy(x, y):
        cx_s = (x - x0) * Dconv
        cy_s = (y - y0) * Dconv
        cxy = magfacD*clock_rot.dot([cx_s, cy_s]) + cshift
        return cxy + [xshift, yshift]

    # STRUT 1
    rot_s1 = 77.56 + clock_deg  # degrees
    cx_s1, cy_s1 = _get_strut_cxy(-24.8566, 22.2242)
    proper.prop_rectangular_obscuration(wf, sx_s, sy_s, cx_s1, cy_s1, ROTATION=rot_s1)

    # STRUT 2
    rot_s2 = -17.56 + clock_deg  # degrees
    cx_s2, cy_s2 = _get_strut_cxy(-23.7187, 20.2742)
    proper.prop_rectangular_obscuration(wf, sx_s, sy_s, cx_s2, cy_s2, ROTATION=rot_s2)

    # STRUT 3
    rot_s3 = -42.44 + clock_deg  # degrees
    cx_s3, cy_s3 = _get_strut_cxy(-24.8566, 18.2758)
    proper.prop_rectangular_obscuration(wf, sx_s, sy_s, cx_s3, cy_s3, ROTATION=rot_s3)

    # STRUT 4
    rot_s4 = 42.44 + clock_deg  # degrees
    cx_s4, cy_s4 = _get_strut_cxy(-27.1434, 18.2758)
    proper.prop_rectangular_obscuration(wf, sx_s, sy_s, cx_s4, cy_s4, ROTATION=rot_s4)

    # STRUT 5
    rot_s5 = 17.56 + clock_deg  # degrees
    cx_s5, cy_s5 = _get_strut_cxy(-28.2813, 20.2742)
    proper.prop_rectangular_obscuration(wf, sx_s, sy_s, cx_s5, cy_s5, ROTATION=rot_s5)

    # STRUT 6
    rot_s6 = 102.44 + clock_deg  # degrees
    cx_s6, cy_s6 = _get_strut_cxy(-27.1434, 22.2242)
    proper.prop_rectangular_obscuration(wf, sx_s, sy_s, cx_s6, cy_s6, ROTATION=rot_s6)

    mask = np.fft.ifftshift(np.abs(wf.wfarr))

    if rot180deg:
        mask = np.rot90(mask, 2)

    return mask


def falco_gen_annular_FPM(pixresFPM, rhoInner, rhoOuter, FPMampFac, centering, rot180=False):
    dxiUL = 1.0 / pixresFPM  # lambda_c/D per pixel. "UL" for unitless

    if np.isinf(rhoOuter):
        if centering == "interpixel":
            # number of points across the inner diameter of the FPM.
            Narray = utils.ceil_even((2 * rhoInner / dxiUL))
        else:
            # number of points across the inner diameter of the FPM. Another half pixel added for pixel-centered masks.
            Narray = utils.ceil_even(2 * (rhoInner / dxiUL + 0.5))
    else:
        if centering == "interpixel":
            # number of points across the outer diameter of the FPM.
            Narray = utils.ceil_even(2 * rhoOuter / dxiUL)
        else:
            # number of points across the outer diameter of the FPM. Another half pixel added for pixel-centered masks.
            Narray = utils.ceil_even(2 * (rhoOuter / dxiUL + 0.5))

    xshift = 0  # translation in x of FPM (in lambda_c/D)
    yshift = 0  # translation in y of FPM (in lambda_c/D)

    Darray = Narray * dxiUL  # width of array in lambda_c/D
    diam = Darray
    wl_dummy = 1e-6  # wavelength (m); Dummy value--no propagation here, so not used.

    if centering == "interpixel":
        cshift = -diam / 2 / Narray
    elif rot180:
        cshift = -diam / Narray
    else:
        cshift = 0

    wf = proper.prop_begin(diam, wl_dummy, Narray, 1.0)

    if not np.isinf(rhoOuter):
        # Outer opaque ring of FPM
        cx_OD = 0 + cshift + xshift
        cy_OD = 0 + cshift + yshift
        proper.prop_circular_aperture(wf, rhoOuter, cx_OD, cy_OD)

    # Inner spot of FPM (Amplitude transmission can be nonzero)
    ra_ID = (rhoInner)
    cx_ID = 0 + cshift + xshift
    cy_ID = 0 + cshift + yshift
    innerSpot = proper.prop_ellipse(wf, rhoInner, rhoInner, cx_ID,
                                    cy_ID, DARK=True) * (1 - FPMampFac) + FPMampFac

    mask = np.fft.ifftshift(np.abs(wf.wfarr))  # undo PROPER's fftshift
    return mask * innerSpot  # Include the inner FPM spot
