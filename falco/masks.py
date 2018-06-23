import numpy as np
import poppy

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

