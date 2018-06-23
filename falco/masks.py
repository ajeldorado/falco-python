import numpy as np
#import poppy
import proper

from . import utils


def annular_fpm(pixres_fpm, rho_inner, rho_outer, fpm_amp_factor=0.0,
        rot180=False, centering='pixel', **kwargs):
    """Generate an annular FPM using POPPY
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

    dxi_ul = 1/pixres_fpm  # lambda_c/D per pixel. "UL" for unitless

    offset = 1/2 if centering=='interpixel' else 0

    if not np.isfinite(rho_outer):
        # number of points across the inner diameter of the FPM.
        narray = utils.ceil_even(2*(rho_inner/dxi_ul+offset))
    else:
        # number of points across the outer diameter of the FPM.
        narray = utils.ceil_even(2*(rho_outer/dxi_ul+offset))

    xshift = 0
    yshift = 0
    darray = narray*dxi_ul  # width of array in lambda_c/D

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

    fpm = poppy.AnnularFieldStop(radius_inner = rho_inner,
                                 radius_outer = rho_outer,
                                 shift_x = cshift + xshift,
                                 shift_y = cshift + yshift)
    mask = fpm.sample(npix=narray, grid_size=darray)

    if fpm_amp_factor != 0:
        # poppy doesn't support gray circular occulting masks, but we can
        # simulate that by just adding back some of the intensity.
        fpm.radius_inner = 0
        mask_no_inner = fpm.sample(npix=narray, grid_size=darray)
        mask = mask * fpm_amp_factor + mask_no_inner * (1-fpm_amp_factor)

    return mask

def _init_proper(Dmask, dx, centering):
    #number of points across output array:
    if centering=="pixel":
    	Narray = 2*np.ceil(0.5*(Dmask/dx + 0.5)) #Sometimes requires two more pixels when pixel centered. Same size as width when interpixel centered.
    else:
        Narray = 2*np.ceil(0.5*(Dmask/dx + 0.0)) #Same size as width when interpixel centered.

    wl_dummy = 1e-6 #% wavelength (m); Dummy value--no propagation here, so not used.
    return proper.prop_begin(Narray*dx, wl_dummy, Narray, 1.0)

def falco_gen_DM_stop(dx, Dmask, centering):
    diam = Dmask #diameter of the mask (meters)
    NapAcross = Dmask/dx #minimum even number of points across to fully contain the actual aperture (if interpixel centered)

    wf = _init_proper(Dmask, dx, centering)

    cshift = -dx/2*(centering=="interpixel") #0 shift for pixel-centered pupil, or -dx shift for inter-pixel centering

    #Outer diameter of aperture
    proper.prop_circular_aperture(wf, diam/2, cshift, cshift)

    return np.fft.ifftshift(np.abs(wf.wfarr))

def falco_gen_pupil_WFIRSTcycle6_LS(Nbeam, Dbeam, ID, OD, strut_width, centering, rot180deg=False):
    strut_width = strut_width*Dbeam #now in meters
    dx = Dbeam/Nbeam

    clock_deg = 0
    magfacD = 1
    xshift = 0
    yshift = 0
    pad_strut = 0
    Dmask = Dbeam #% width of the beam (so can have zero padding if LS is undersized) (meters)
    diam = Dmask #width of the mask (meters)
    NapAcross = Dmask/dx #minimum even number of points across to fully contain the actual aperture (if interpixel centered)

    wf = _init_proper(Dmask, dx, centering)

    #0 shift for pixel-centered pupil, or -dx shift for inter-pixel centering
    if centering=="interpixel":
        cshift = -dx/2
    elif rot180deg:
        cshift = -dx
    else:
        cshift = 0

    #DATA FROM THE VISIO FILE
    D0 = 8 #inches, pupil diameter in Visio file
    x0 = -26 #inches, pupil center in x in Visio file
    y0 = 20.25 #inches, pupil center in y in Visio file
    Dconv = diam/D0 #conversion factor from inches and Visio units to meters 


    #PRIMARY MIRROR (OUTER DIAMETER)
    ra_OD = (Dbeam*OD/2)*magfacD
    cx_OD = cshift + xshift
    cy_OD = cshift + yshift
    proper.prop_circular_aperture(wf, ra_OD, cx_OD ,cy_OD)


    #SECONDARY MIRROR (INNER DIAMETER)
    ra_ID = (Dbeam*ID/2)*magfacD
    cx_ID = cshift + xshift
    cy_ID = cshift + yshift
    proper.prop_circular_obscuration(wf, ra_ID, cx_ID ,cy_ID)

    sx_s = magfacD*(3.6*(diam/D0) + pad_strut)
    sy_s = magfacD*(strut_width + pad_strut)
    clock_rot = np.array([[np.cos(np.radians(clock_deg)), -np.sin(np.radians(clock_deg))], [np.sin(np.radians(clock_deg)), np.cos(np.radians(clock_deg))]])
    def _get_strut_cxy(x,y):
        cx_s = (x - x0)*Dconv
        cy_s = (y - y0)*Dconv
        cxy = magfacD*clock_rot.dot([cx_s,cy_s]) + cshift
        return cxy + [xshift, yshift]

    #STRUT 1
    rot_s1 = 77.56 + clock_deg #degrees
    cx_s1, cy_s1 = _get_strut_cxy(-24.8566, 22.2242)
    proper.prop_rectangular_obscuration(wf, sx_s, sy_s, cx_s1, cy_s1, ROTATION=rot_s1)

    #STRUT 2
    rot_s2 = -17.56 + clock_deg #degrees
    cx_s2, cy_s2 = _get_strut_cxy(-23.7187, 20.2742)
    proper.prop_rectangular_obscuration(wf, sx_s, sy_s, cx_s2, cy_s2, ROTATION=rot_s2)

    #STRUT 3
    rot_s3 = -42.44 + clock_deg #degrees
    cx_s3, cy_s3 = _get_strut_cxy(-24.8566, 18.2758)
    proper.prop_rectangular_obscuration(wf, sx_s, sy_s, cx_s3, cy_s3, ROTATION=rot_s3)

    #STRUT 4
    rot_s4 = 42.44 + clock_deg #degrees
    cx_s4, cy_s4 = _get_strut_cxy(-27.1434, 18.2758)
    proper.prop_rectangular_obscuration(wf, sx_s, sy_s, cx_s4, cy_s4, ROTATION=rot_s4)

    #STRUT 5
    rot_s5 = 17.56 + clock_deg #degrees
    cx_s5, cy_s5 = _get_strut_cxy(-28.2813, 20.2742)
    proper.prop_rectangular_obscuration(wf, sx_s, sy_s, cx_s5, cy_s5, ROTATION=rot_s5)

    #STRUT 6
    rot_s6 = 102.44 + clock_deg #degrees
    cx_s6, cy_s6 = _get_strut_cxy(-27.1434, 22.2242)
    proper.prop_rectangular_obscuration(wf, sx_s, sy_s, cx_s6, cy_s6, ROTATION=rot_s6)

    mask = np.fft.ifftshift(np.abs(wf.wfarr));

    if rot180deg:
        mask = np.rot90(mask,2)

    return mask
