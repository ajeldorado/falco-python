import numpy as np
import os
import poppy
import proper
import scipy.interpolate
import scipy.ndimage
import math
from . import utils
import falco

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
    """
    Function to generate a circular aperture to place centered on the beam at 
    a deformable mirror.

    Corrected on 2018-08-16 by A.J. Riggs to compute 'beam_diam_fraction' correctly.
    Created on 2017-11-15 by A.J. Riggs (JPL).

    Parameters
    ----------
    dx:        spatial resolution for a pixel [any units as long as they match that of Dmask]
    Dmask:     diameter of the aperture mask [any units as long as they match that of dx]
    centering: centering of beam in array. Either 'pixel' or 'interpixel'

    Returns
    -------
    mask:  2-D square array of a circular stop at a DM. Cropped down to the 
           smallest even-sized array with no extra zero padding. 
    """    
    
    diam = Dmask  # diameter of the mask (meters)
    # minimum even number of points across to fully contain the actual aperture (if interpixel centered)
    NapAcross = Dmask / dx

    wf = _init_proper(Dmask, dx, centering)

    # 0 shift for pixel-centered pupil, or -dx shift for inter-pixel centering
    cshift = -dx / 2 * (centering == "interpixel")

    # Outer diameter of aperture
    proper.prop_circular_aperture(wf, diam / 2, cshift, cshift)

    return np.fft.ifftshift(np.abs(wf.wfarr))


def falco_gen_pupil_WFIRST_CGI_180718(Nbeam, centering, changes={}):
    """
    Function to generate WFIRST pupil CGI-180718.

    Function to generate WFIRST pupil CGI-180718. Options to change the x- or y-shear, 
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

#    ### Changes to the pupil
#    class Changes(object):
#        pass
#    
#    class Struct(object):
#        def __init__(self, **entries):
#            self.__dict__.update(entries)
#
#    if len(kwargs) > 0:
#        #raise ValueError('falco_gen_pupil_WFIRST_CGI_180718.m: Too many inputs')
#        changes=Struct(**kwargs)
#    else:
#        changes = Changes()
#        changes.dummy = 1
    
    ### Oversized strut features: overwrite defaults if values specified
    if 'OD' in changes.keys(): OD = changes["OD"]
    if 'ID' in changes.keys(): ID = changes["ID"]
    if 'wStrut' in changes.keys(): wStrutVec = np.ones(6)*changes["wStrut"]
    if 'wStrutVec' in changes.keys(): wStrutVec = changes["wStrutVec"]

#    if hasattr(changes, 'OD'):
#        OD = changes.OD
#    if hasattr(changes, 'ID'):
#        ID = changes.ID
#    if hasattr(changes, 'wStrut'):
#        wStrutVec = changes.wStrut * np.ones([6])
#    if hasattr(changes, 'wStrutVec'):
#        wStrutVec = changes.wStrutVec

    ### Padding values for obscuration
    #--Defaults of Bulk Changes: (All length units are pupil diameters. All angles are in degrees.)
    if not 'xShear' in changes.keys(): changes["xShear"] = 0
    if not 'yShear' in changes.keys(): changes["yShear"] = 0
    if not 'magFac' in changes.keys(): changes["magFac"] = 1.0
    if not 'clock_deg' in changes.keys(): changes["clock_deg"] = 0.0
    if not 'flagRot180' in changes.keys(): changes["flagRot180"] = False

    #--Defaults for obscuration padding: (All length units are pupil diameters.)
    if not 'pad_all' in changes.keys(): changes["pad_all"] = 0.
    if not 'pad_strut' in changes.keys(): changes["pad_strut"] = 0.
    if not 'pad_COBS' in changes.keys(): changes["pad_COBS"] = 0.
    if not 'pad_COBStabs' in changes.keys(): changes["pad_COBStabs"] = 0.
    if not 'pad_OD' in changes.keys(): changes["pad_OD"] = 0.

    #--Values to use for bulk clocking, magnification, and translation
    xShear = changes["xShear"]  # - xcOD;
    yShear = changes["yShear"]  # - ycOD;
    magFac = changes["magFac"]
    clock_deg = changes["clock_deg"]
    flagRot180 = changes["flagRot180"]
    
    #--Padding values. (pad_all is added to all the rest)
    pad_all = changes["pad_all"]     #0.2/100; #--Uniform padding on all features
    pad_strut = changes["pad_strut"] + pad_all
    pad_COBS = changes["pad_COBS"] + pad_all
    pad_COBStabs = changes["pad_COBStabs"] + pad_all
    pad_OD = changes["pad_OD"] + pad_all  #--Radial padding at the edge

    #--Rotation matrix used on center coordinates.
    rotMat = np.array([[math.cos(math.radians(clock_deg)), -math.sin(math.radians(clock_deg))], [math.sin(math.radians(clock_deg)), math.cos(math.radians(clock_deg))]])

    ## Coordinates
    if centering.lower() in ('pixel', 'odd'):
        Narray = falco.utils.ceil_even(Nbeam + 1) #--number of points across output array. Requires two more pixels when pixel centered.
    else:
        Narray = falco.utils.ceil_even(Nbeam) #--No zero-padding needed if beam is centered between pixels

    if centering.lower() == 'interpixel':
        xs = np.linspace(-(Nbeam-1)/2, (Nbeam-1)/2,Nbeam)/Nbeam
    else:
        xs = np.linspace(-(Narray/2), Narray/2-1,Narray)/Nbeam

    [XS, YS] = np.meshgrid(xs,xs)

    ## Proper Setup Values
    Dbeam = 1                 #--Diameter of aperture, normalized to itself
    wl   = 1e-6               # wavelength (m); Dummy value--no propagation here, so not used.
    bdf = Nbeam/Narray        #--beam diameter factor in output array
    dx = Dbeam/Nbeam

    if centering.lower() in ('interpixel', 'even'):
        cshift = -dx/2
    elif centering.lower() in ('pixel', 'odd'):
        cshift = 0
        if flagRot180:
            cshift = -dx

    ## INITIALIZE PROPER
    bm = proper.prop_begin(Dbeam, wl, Narray,bdf)
    
    ## PRIMARY MIRROR (OUTER DIAMETER)
    ra_OD = magFac*(OD/2 - pad_OD)
    cx_OD = magFac*xcOD
    cy_OD = magFac*ycOD
    cxy = np.matmul(rotMat, np.array([[cx_OD],[cy_OD]]))
    cx_OD = cxy[0]+xShear
    cy_OD = cxy[1]+yShear
    proper.prop_circular_aperture(bm, ra_OD, cx_OD+cshift, cy_OD+cshift)
    #bm = proper.prop_circular_aperture(bm, ra_OD, cx_OD+cshift, cy_OD+cshift)


    ## SECONDARY MIRROR (INNER DIAMETER)
    ra_ID = magFac*(ID/2 + pad_COBS)
    cx_ID = magFac*xcCOBS
    cy_ID = magFac*ycCOBS
    cxy = np.matmul(rotMat,np.array([[cx_ID],[cy_ID]]))
    cx_ID = cxy[0]+xShear
    cy_ID = cxy[1]+yShear
    proper.prop_circular_obscuration(bm, ra_ID, cx_ID+cshift, cy_ID+cshift)
    #bm = proper.prop_circular_obscuration(bm, ra_ID, cx_ID+cshift, cy_ID+cshift)

    ## Struts
    for istrut in range(6):
        angDeg = angStrutVec[istrut] + clock_deg # degrees
        wStrut = magFac*(wStrutVec[istrut] + 2*pad_strut)
        lStrutIn = magFac*lStrut
        xc = magFac*(xcStrutVec[istrut])
        yc = magFac*(ycStrutVec[istrut])
        cxy = np.matmul(rotMat, np.array([[xc], [yc]]))
        xc = cxy[0]+xShear;
        yc = cxy[1]+yShear;
        proper.prop_rectangular_obscuration(bm, lStrutIn, wStrut, xc+cshift, yc+cshift, ROTATION=angDeg)
    
    wf1=bm.wfarr #assign for later use
    
    ## TABS ON SECONDARY MIRROR
    #--Compute as new shape, and then multiply the obscuration with the rest of
    #the pupil.
    
    #--SOFTWARE MASK:
    XSnew = (1/1*XS+xcCOBStabs)-xShear
    YSnew = (1/1*YS+ycCOBStabs)-yShear
    
    overSizeFac = 1.3
    cobsTabsMask = np.zeros([Narray,Narray])
    THETAS = np.arctan2(YSnew,XSnew)
    clock_rad = np.deg2rad(clock_deg)

    
    if angTabStart[0] > angTabEnd[0]:
        msk1=(XSnew**2 + YSnew**2) <= (overSizeFac*magFac*IDtabs/2)**2
        msk2=np.logical_and(THETAS>=angTabEnd[0]+clock_rad,THETAS<=angTabStart[0]+clock_rad)
        msk3=np.logical_and(THETAS>=angTabEnd[1]+clock_rad,THETAS<=angTabStart[1]+clock_rad)
        msk4=np.logical_and(THETAS>=angTabEnd[2]+clock_rad,THETAS<=angTabStart[2]+clock_rad)
        cobsTabsMask[np.logical_and(msk1,np.logical_or(msk2,np.logical_or(msk3,msk4)))]=1
    else:
        msk1=(XSnew**2 + YSnew**2) <= (overSizeFac*magFac*IDtabs/2)**2
        msk2=np.logical_and(THETAS<=angTabEnd[0]+clock_rad,THETAS>=angTabStart[0]+clock_rad)
        msk3=np.logical_and(THETAS<=angTabEnd[1]+clock_rad,THETAS>=angTabStart[1]+clock_rad)
        msk4=np.logical_and(THETAS<=angTabEnd[2]+clock_rad,THETAS>=angTabStart[2]+clock_rad)
        cobsTabsMask[np.logical_and(msk1,np.logical_or(msk2,np.logical_or(msk3,msk4)))]=1
    

    ##--CIRCLE:
    ##--Initialize PROPER
    bm = proper.prop_begin(Dbeam, wl, Narray,bdf)
    
    ##--Full circle of COBS tabs--to be multiplied by the mask to get just tabs
    ra_tabs = magFac*(IDtabs/2 + pad_COBStabs)
    cx_tabs = magFac*(xcCOBStabs)
    cy_tabs = magFac*(ycCOBStabs)
    cxy = np.matmul(rotMat,np.array([[cx_tabs],[cy_tabs]]))
    cx_tabs = cxy[0]+xShear
    cy_tabs = cxy[1]+yShear
    
    #bm2 = prop_circular_obscuration(bm, ra_tabs,'XC',cx_tabs+cshift,'YC',cy_tabs+cshift)
    proper.prop_circular_obscuration(bm, ra_tabs,cx_tabs+cshift,cy_tabs+cshift)

    temp = 1-np.fft.ifftshift(np.abs(bm.wfarr))
    temp = cobsTabsMask*temp

    cobsTabs = 1-temp

    ## Output
    pupil = cobsTabs*np.fft.ifftshift(np.abs(wf1))
    if(flagRot180):
        pupil = np.rot90(pupil,2)

    return pupil


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


def falco_gen_SW_mask(inputs):
    """
    Function to generate binary (0-1) software masks for the focal plane. 
    This can be used as a field stop, or for making the scoring and correction 
    regions in the focal plane.

    Detailed description here

    Parameters
    ----------
    inputs: structure with several fields:
   -pixresFP: pixels per lambda_c/D
   -rhoInner: radius of inner FPM amplitude spot (in lambda_c/D)
   -rhoOuter: radius of outer opaque FPM ring (in lambda_c/D)
   -angDeg: angular opening (degrees) on the left/right/both sides.
   -whichSide: which sides to have open. 'left','right', 'top', 'bottom', or 'both'
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
#    class Struct(object):
#        def __init__(self, **entries):
#            self.__dict__.update(entries)
#
#    if len(kwargs) > 0:
#        #raise ValueError('falco_gen_pupil_WFIRST_CGI_180718.m: Too many inputs')
#        inputs=Struct(**kwargs)
    
    #--Read in user inputs
    pixresFP = inputs["pixresFP"]    #--pixels per lambda_c/D
    rhoInner = inputs["rhoInner"]    # radius of inner FPM amplitude spot (in lambda_c/D)
    rhoOuter = inputs["rhoOuter"]    # radius of outer opaque FPM ring (in lambda_c/D)
    angDeg = inputs["angDeg"]        #--angular opening (input in degrees) on the left/right/both sides of the dark hole.
    whichSide = inputs["whichSide"]  #--which (sides) of the dark hole have open
    
    
    
    if 'FOV' in inputs.keys(): #hasattr(inputs, 'FOV'): 
        FOV = inputs["FOV"]
    else:
        inputs["FOV"] = rhoOuter
    
        
    if 'centering' in inputs.keys(): #hasattr(inputs, 'centering'):
        centering = inputs["centering"]
    else:
        centering = 'pixel'
        

    if 'shape' in inputs.keys(): #hasattr(inputs,'shape'):
        DHshape = inputs["shape"]
    else:
        DHshape = 'circle' #default to a circular outer edge

    #convert opening angle to radians
    angRad = np.radians(angDeg)

    # Number of points across each axis. Crop the vertical (eta) axis if angDeg<180 degrees.
    # if Nxi is defined use that, if not calculate Nxi.
    if centering == "interpixel":
        if hasattr(inputs,'Nxi'):#if Nxi is defined 
            Nxi = inputs["Nxi"]
        else: #else calculate Nxi
            Nxi = falco.utils.ceil_even(2 * FOV * pixresFP)  # Number of points across the full FPM
            
        Neta = falco.utils.ceil_even(2 * FOV * pixresFP)
    else:
        if hasattr(inputs,'Nxi'):
            Nxi = inputs["Nxi"]
        else:
            Nxi = falco.utils.ceil_even(2 * (FOV * pixresFP + 0.5))  # Number of points across the full FPM  
            
        Neta = falco.utils.ceil_even(2 * (FOV * pixresFP + 0.5))

    # Focal Plane Coordinates
    deta = dxi = 1.0 / pixresFP
    if centering == "interpixel":
        xis = np.arange(-(Nxi - 1) / 2, (Nxi + 1) / 2) * dxi
        etas = np.arange(-(Neta - 1) / 2, (Neta + 1) / 2) * deta
    else:#pixel centering
        xis = np.arange(-Nxi / 2, Nxi / 2) * dxi
        etas = np.arange(-Neta / 2, Neta / 2) * deta

    [XIS, ETAS] = np.meshgrid(xis, etas)
    RHO = np.sqrt(XIS ** 2 + ETAS ** 2)
    THETAS = np.arctan2(ETAS,XIS)

    # Generate the Outer Mask
    # maskSW = 1.0 * (RHOS >= rhoInner) * (RHOS <= rhoOuter) * (THETAS <= angRad/2) * (THETAS >= -angRad/2)
    # maskSW = 1.0 * np.logical_and(RHOS>=rhoInner,RHOS<=rhoOuter)
    if DHshape in ('square'):
        maskSW0 = np.logical_and(RHO>=rhoInner,np.logical_and(abs(XIS)<=rhoOuter,abs(ETAS)<=rhoOuter))
    else:
        maskSW0 = np.logical_and(RHO>=rhoInner,RHO<=rhoOuter)
    
    
    #--If the use doesn't pass the clocking angle 
    if not hasattr(inputs,'clockAngDeg'):
        if whichSide in ("L", "left"):
            clockAng = np.pi
        elif whichSide in ("R", "right"):
            clockAng = 0
        elif whichSide in ("T", "top"):
            clockAng = np.pi/2
        elif whichSide in ("B", "bottom"):
            clockAng = 3*np.pi/2
        else:
            clockAng = 0
    else:
        clockAng = inputs["clockAngDeg"]*np.pi/180    


    maskSW = np.logical_and(maskSW0,np.abs(np.angle(np.exp(1j*(THETAS-clockAng))))<=angRad/2)

    if whichSide in ('both'):
        clockAng = clockAng + np.pi;
        maskSW2 = np.logical_and(maskSW0, np.abs(np.angle(np.exp(1j*(THETAS-clockAng))))<=angRad/2)
        maskSW = np.logical_or(maskSW,maskSW2)

    return maskSW, xis, etas


def falco_gen_pupil_WFIRSTcycle6_LS(Nbeam, Dbeam, ID, OD, strut_width, centering, rot180deg=False):
    """
    Quick Description here

    Detailed description here

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    Returns
    -------
    TBD
        Return value descriptio here
    """ 
    
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


def falco_gen_annular_FPM(inputs):
    """
    Function to generate an annular FPM in Matlab using PROPER.
    
    Outside the outer ring is opaque.If rhoOuter = infinity, then the outer 
    ring is omitted and the mask is cropped down to the size of the inner spot.
    The inner spot has a specifyable amplitude value. The output array is the 
    smallest size that fully contains the mask.    

    Parameters
    ----------
     pixresFPM:  resolution in pixels per lambda_c/D
     rhoInner:   radius of inner FPM amplitude spot (in lambda_c/D)
     rhoOuter:   radius of outer opaque FPM ring (in lambda_c/D). Set to
                 infinity for an occulting-spot only FPM
     FPMampFac:  amplitude transmission of inner FPM spot
     centering:  pixel centering 
    
    Returns
    -------
     mask:    cropped-down, 2-D FPM representation. amplitude only 
     
    """

    # Set default values of input parameters
    flagRot180deg = False


    #unfolding **kwargs
    class Struct(object):
        def __init__(self, **entries):
            self.__dict__.update(entries)

    #inputs=Struct(**kwargs)

    pixresFPM=inputs["pixresFPM"]
    rhoInner=inputs["rhoInner"]
    rhoOuter=inputs["rhoOuter"] 
    FPMampFac=inputs["FPMampFac"] 
    centering=inputs["centering"] 

    if hasattr(inputs, 'flagRot180deg'):
        flagRot180deg = inputs["flagRot180deg"]


    dxiUL = 1.0 / pixresFPM  # lambda_c/D per pixel. "UL" for unitless
    if np.isinf(rhoOuter):
        if centering == "interpixel":
            # number of points across the inner diameter of the FPM.
            Narray = falco.utils.ceil_even((2 * rhoInner / dxiUL))
        else:
            # number of points across the inner diameter of the FPM. Another half pixel added for pixel-centered masks.
            Narray = falco.utils.ceil_even(2 * (rhoInner / dxiUL + 0.5))
    else:
        if centering == "interpixel":
            # number of points across the outer diameter of the FPM.
            Narray = falco.utils.ceil_even(2 * rhoOuter / dxiUL)
        else:
            # number of points across the outer diameter of the FPM. Another half pixel added for pixel-centered masks.
            Narray = falco.utils.ceil_even(2 * (rhoOuter / dxiUL + 0.5))

    xshift = 0  # translation in x of FPM (in lambda_c/D)
    yshift = 0  # translation in y of FPM (in lambda_c/D)

    Darray = Narray * dxiUL  # width of array in lambda_c/D
    diam = Darray
    wl_dummy = 1e-6  # wavelength (m); Dummy value--no propagation here, so not used.

    if centering == "interpixel":
        cshift = -diam / 2 / Narray
    elif flagRot180deg: #rot180
        cshift = -diam / Narray
    else:
        cshift = 0

    #--INITIALIZE PROPER. Note that:  bm.dx = diam / bdf / np;
    wf = proper.prop_begin(diam, wl_dummy, Narray, 1.0)

    if not np.isinf(rhoOuter):
        # Outer opaque ring of FPM
        cx_OD = 0 + cshift + xshift
        cy_OD = 0 + cshift + yshift
        proper.prop_circular_aperture(wf, rhoOuter, cx_OD, cy_OD)

    # Inner spot of FPM (Amplitude transmission can be nonzero)
    cx_ID = 0 + cshift + xshift
    cy_ID = 0 + cshift + yshift
    innerSpot = proper.prop_ellipse(wf, rhoInner, rhoInner, cx_ID,
                                    cy_ID, DARK=True) * (1 - FPMampFac) + FPMampFac

    mask = np.fft.ifftshift(np.abs(wf.wfarr))  # undo PROPER's fftshift
    return mask * innerSpot  # Include the inner FPM spot


def falco_gen_bowtie_LS(inputs):

    Nbeam   = inputs["Nbeam"] # number of points across the incoming beam           
    ID = inputs["ID"]   # inner diameter of mask (in pupil diameters)
    OD = inputs["OD"]   # outer diameter of mask (in pupil diameters)
    ang = inputs["ang"] # opening angle of the upper and lower bowtie wedges [degrees]

    Dbeam = 1. #inputs.Dbeam; #--diameter of the beam at the mask [pupil diameters]
    dx = Dbeam/Nbeam
    Dmask = Dbeam # width of the beam (so can have zero padding if LS is undersized) [meters]

    #--Optional inputs and their defaults
    centering = inputs['centering'] if 'centering' in inputs.keys() else 'pixel'  #--Default to pixel centering
    xShear = inputs['xShear'] if 'xShear' in inputs.keys() else  0. #--x-axis shear of mask [pupil diameters]
    yShear = inputs['yShear'] if 'yShear' in inputs.keys() else  0. #--y-axis shear of mask [pupil diameters]
    clocking = inputs['clocking'] if 'clocking' in inputs.keys() else 0. #--Clocking of the mask [degrees]
    magfac = inputs['magfac'] if 'magfac' in inputs.keys() else 1. #--magnification factor of the pupil diameter

    if(centering=='pixel'):
        Narray = falco.utils.ceil_even(magfac*Nbeam + 1 + 2*Nbeam*np.max(np.abs(np.array([xShear,yShear]))))  #--number of points across output array. Sometimes requires two more pixels when pixel centered.
    else:
        Narray = falco.utils.ceil_even(magfac*Nbeam + 2*Nbeam*np.max(np.abs(np.array([xShear,yShear]))))    #--number of points across output array. Same size as width when interpixel centered.

    Darray = Narray*dx  #--width of the output array [meters]
    bdf = Dmask/Darray  #--beam diameter factor in output array
    wl_dummy   = 1e-6   # wavelength (m); Dummy value--no propagation here, so not used.

    # No shift for pixel-centered pupil, or -Dbeam/Narray/2 shift for inter-pixel centering
    cshift = -dx/2 if 'interpixel' in centering else 0.

    # % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

    #--INITIALIZE PROPER
    bm = proper.prop_begin(Dmask, wl_dummy, Narray,bdf)
    
    #--PRIMARY MIRROR (OUTER DIAMETER)
    ra_OD = (Dbeam*OD/2.)*magfac
    cx_OD = cshift + xShear
    cy_OD = cshift + yShear
    proper.prop_circular_aperture(bm, ra_OD,cx_OD,cy_OD)

    #--SECONDARY MIRROR (INNER DIAMETER)
    ra_ID = (Dbeam*ID/2.)*magfac
    cx_ID = cshift + xShear
    cy_ID = cshift + yShear
    proper.prop_circular_obscuration(bm, ra_ID,cx_ID,cy_ID)

    mask = np.fft.ifftshift(np.abs(bm.wfarr))

    #--Create the bowtie region
    if(ang<180.):
        ang2 = 90.-ang/2.
        bm2 = bm
        Lside = 1.1*ra_OD # Have the triangle go a little past the edge of the circle

        yvert0 = np.array([0., Lside*falco.utils.sind(ang2), Lside*falco.utils.sind(ang2), -Lside*falco.utils.sind(ang2), -Lside*falco.utils.sind(ang2), 0.])

        #--Right triangular obscuration
        xvert0 = np.array([0., Lside*falco.utils.cosd(ang2), Lside,             Lside,             Lside*falco.utils.cosd(ang2), 0.])
        xvert = xvert0.copy()
        yvert = yvert0.copy()
        for ii in range(len(xvert0)):
            xy = np.array([[falco.utils.cosd(clocking),falco.utils.sind(clocking)],[ -falco.utils.sind(clocking),falco.utils.cosd(clocking)]]) @ np.array([xvert0[ii], yvert0[ii]]).reshape(2,1)
            xvert[ii] = xy[0]
            yvert[ii] = xy[1]
            pass
        bowtieRight = proper.prop_irregular_polygon( bm, cshift+xShear+xvert, cshift+yShear+yvert,DARK=True)
   
        #--Left triangular obscuration
        xvert0 = -np.array([0., Lside*falco.utils.cosd(ang2), Lside,             Lside,             Lside*falco.utils.cosd(ang2), 0.])
        xvert = xvert0.copy()
        yvert = yvert0.copy()
        for ii in range(len(xvert0)):
            xy = np.array([[falco.utils.cosd(clocking),falco.utils.sind(clocking)], [-falco.utils.sind(clocking),falco.utils.cosd(clocking)]]) @ np.array([xvert0[ii], yvert0[ii]]).reshape(2,1)
            xvert[ii] = xy[0]
            yvert[ii] = xy[1]
            pass
        bowtieLeft = proper.prop_irregular_polygon( bm2, cshift+xShear+xvert, cshift+yShear+yvert,DARK=True)
        

        mask = mask*bowtieRight*bowtieLeft
        pass
    
    return mask


def falco_gen_pupil_LUVOIR_A_final(inputs,**kwargs):

#    #--Function to generate the LUVOIR Design A (Final) telescope pupil from 
#    % 2018 in Matlab using PROPER
#    % Coordinates and dimensions of the primary, secondary, and hex segments
#    %   are from Matthew Bolcar (NASA GSFC).
#    % Coordinates and dimensions of the secondary mirror support struts were a
#    %   best-fit match by A.J. Riggs by matching PROPER-made rectangles to the 
#    %   pupil file from Matthew Bolcar (NASA GSFC).
#    %
#    % Modified on 2018-10-09 by Carl Coker from
#    % falco_gen_pupil_LUVOIR_A_5_mag_trans to falco_gen_pupil_LUVOIR_A_final to
#    % have struts without kinks.
#    % Corrected on 2018-08-16 by A.J. Riggs to compute 'beam_diam_fraction' correctly.
#    % Modified on 2018-02-25 by A.J. Riggs to be for LUVOIR A aperture 5. 
#    % Written on  2017-09-07 by A.J. Riggs to generate the first proposed LUVOIR pupil. 
#    %   Values for the geometry were provided by Matthew Bolcar at NASA GSFC.
#    %
#    #--Coordinates of hex segments to skip:
#    % 1 13 114 115 126 127
#    % 1 12 113 114 125 126
#    
#    function mask = falco_gen_pupil_LUVOIR_A_final(inputs,varargin)
    
    #--Optional inputs and their defaults
    flagRot180deg = True if 'ROT180' in kwargs and kwargs["ROT180"]==True else False
    centering = inputs['centering'] if 'centering' in inputs.keys() else 'pixel'  #--Default to pixel centering
    magfacD = inputs['magfacD'] if 'magfacD' in inputs.keys() else 1.0  #--Default to no magnification of the pupil
    hexgap0 = inputs['wGap_m'] if 'wGap_m' in inputs.keys() else 6.0e-3  #--Default to 6 mm between segments
    
    # % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    
    #--USER INPUTS
    Nbeam = inputs["Nbeam"] # number of points across the incoming beam  
    Nap = Nbeam # number of points across FULL usable pupil
    width_hex0 = 1.2225 #-- flat-to-flat (m)
    Dap = (12*width_hex0 + 11*hexgap0) #(12*width_hex0 + 12*hexgap0);
    dx = Dap/Nap
    dx_drawing = 1.2225/158 # (m) #--In actual drawing, 158 pixels across the 1.2225m, so dx_pixel =
    #dx_drawing. strut in drawing is 19 pixels across, so dx_strut =
    # 19*dx_drawing = 0.1470m
    
    if("pixel" == centering):
        Narray = falco.utils.ceil_even(Nbeam/magfacD+1) #--number of points across output array. Sometimes requires two more pixels when pixel centered.
        cshift = -dx if flagRot180deg else 0. # shear of beam center relative to center pixel
        
    elif("interpixel" == centering):
        Narray = falco.utils.ceil_even(Nbeam/magfacD) #--number of points across output array. Same size as width when interpixel centered.
        cshift = -dx/2. # shear of beam center relative to center pixel
    else:
        pass
    
    Darray = Narray*dx
    
    #--For PROPER 
    wl_dummy = 1e-6 # wavelength (m)
    bdf = Nbeam/Narray #--beam diameter factor in output array
          
    dx_t = 0.
    dy_t = 0.
    
    width_hex = magfacD*width_hex0 #-- flat-to-flat (m)
    nrings = 6
    hexrad = 2./np.sqrt(3.)*width_hex/2.
    hexgap = magfacD*hexgap0 # (m)
    hexsep = width_hex + hexgap # distance from center to center of neighboring segments
    wStrut = 0.15*magfacD # meters
    
    #-------- Generate the input pupil for LUVOIR
    bm = proper.prop_begin(Dap, wl_dummy, Narray, bdf);
    
    # Subtract the inner ring from all the rings
#    ap = proper.prop_hex_aperture(bm,nrings,hexrad,hexsep,cshift-dx_t,cshift-dy_t,DARK=True) #--Official Matlab PROPER from August 2017
    ap = falco_hex_aperture_LUVOIR_A(bm,nrings,hexrad,hexsep,cshift-dx_t,cshift-dy_t,DARK=True) #--Official Matlab PROPER from August 2017
    
    #--Add the struts
    proper.prop_rectangular_obscuration(bm, wStrut, 7*width_hex, cshift-dx_t,cshift-dy_t + magfacD*Dap/4.);
    
    len_1b = (np.sqrt(93)+0.5)*hexrad
    proper.prop_rectangular_obscuration(bm, wStrut, len_1b, cshift-dx_t + 1.5*hexrad, cshift-dy_t - 11*np.sqrt(3)/4*hexrad, ROTATION=12.7)
    proper.prop_rectangular_obscuration(bm, wStrut, len_1b, cshift-dx_t - 1.5*hexrad, cshift-dy_t - 11*np.sqrt(3)/4*hexrad, ROTATION=-12.7)
    
    mask = np.fft.ifftshift(np.abs(bm.wfarr))*ap
    
    mask[mask>1] = 1. #--Get rid of overlapping segment edges at low resolution if the gap size is zero.
    
    if(flagRot180deg):
        mask = np.rot90(mask,2)

    return mask


def falco_gen_pupil_LUVOIR_A_final_Lyot(inputs,**kwargs):

    #--Optional inputs and their defaults
    flagRot180deg = True if 'ROT180' in kwargs and kwargs["ROT180"]==True else False
    centering = inputs['centering'] if 'centering' in inputs.keys() else 'pixel'  #--Default to pixel centering
    magfacD = inputs['magfacD'] if 'magfacD' in inputs.keys() else 1.0  #--Default to no magnification of the pupil
    hexgap0 = inputs['wGap_m'] if 'wGap_m' in inputs.keys() else 6.0e-3  #--Default to 6 mm between segments
    
    # % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    
    #--USER INPUTS
    Nbeam = inputs["Nbeam"] # number of points across the incoming beam  
    ID = inputs["ID"] # inner diameter of mask (in pupil diameters)
    OD = inputs["OD"] # outer diameter of mask (in pupil diameters)
    wStrut = inputs["wStrut"] # width of the struts (in pupil diameters)
    wStrut = wStrut*Dbeam # now in meters

    Nap = Nbeam # number of points across FULL usable pupil
    width_hex0 = 1.2225 #-- flat-to-flat (m)
    Dap = (12*width_hex0 + 11*hexgap0) #(12*width_hex0 + 12*hexgap0);
    dx = Dap/Nap
    dx_drawing = 1.2225/158 # (m) #--In actual drawing, 158 pixels across the 1.2225m, so dx_pixel =
    #dx_drawing. strut in drawing is 19 pixels across, so dx_strut =
    # 19*dx_drawing = 0.1470m
    
    if("pixel" == centering):
        Narray = falco.utils.ceil_even(Nbeam/magfacD+1) #--number of points across output array. Sometimes requires two more pixels when pixel centered.
        cshift = -dx if flagRot180deg else 0. # shear of beam center relative to center pixel
        
    elif("interpixel" == centering):
        Narray = falco.utils.ceil_even(Nbeam/magfacD) #--number of points across output array. Same size as width when interpixel centered.
        cshift = -dx/2. # shear of beam center relative to center pixel
    else:
        pass
    
    Darray = Narray*dx
    
    #--For PROPER 
    wl_dummy = 1e-6 # wavelength (m)
    bdf = Nbeam/Narray #--beam diameter factor in output array
          
    dx_t = 0.
    dy_t = 0.
    
    width_hex = magfacD*width_hex0 #-- flat-to-flat (m)
    nrings = 6
    hexrad = 2./np.sqrt(3.)*width_hex/2.
    hexgap = magfacD*hexgap0 # (m)
    hexsep = width_hex + hexgap # distance from center to center of neighboring segments
    wStrut = 0.15*magfacD # meters
    
    #-------- Generate the input pupil for LUVOIR
    bm = proper.prop_begin(Dap, wl_dummy, Narray, bdf);

    #--Outer pupil edge
    proper.prop_circular_aperture(bm, OD/2, cshift-dx_t, cshift-dy_t)

    # Central Obscuration
    proper.prop_circular_obscuration(bm, ID/2., cshift-dx_t, cshift-dy_t)
    
#    # Subtract the inner ring from all the rings
#    ap = falco_hex_aperture_LUVOIR_A(bm,nrings,hexrad,hexsep,cshift-dx_t,cshift-dy_t,DARK=True) #--Official Matlab PROPER from August 2017
    
    #--Add the struts
    proper.prop_rectangular_obscuration(bm, wStrut, 7*width_hex, cshift-dx_t,cshift-dy_t + magfacD*Dap/4.);
    len_1b = (np.sqrt(93)+0.5)*hexrad
    proper.prop_rectangular_obscuration(bm, wStrut, len_1b, cshift-dx_t + 1.5*hexrad, cshift-dy_t - 11*np.sqrt(3)/4*hexrad, ROTATION=12.7)
    proper.prop_rectangular_obscuration(bm, wStrut, len_1b, cshift-dx_t - 1.5*hexrad, cshift-dy_t - 11*np.sqrt(3)/4*hexrad, ROTATION=-12.7)
    
    mask = np.fft.ifftshift(np.abs(bm.wfarr)) #*ap
    
    mask[mask>1] = 1. #--Get rid of overlapping segment edges at low resolution if the gap size is zero.
    
    if(flagRot180deg):
        mask = np.rot90(mask,2)

    return mask

def falco_hex_aperture_LUVOIR_A(wf, nrings, hexrad, hexsep, xc = 0.0, yc = 0.0, **kwargs):
    """
    Return an image containing a hexagonal mask consisting of multiple hexagons.
    This is used for generating the primary mirror for the LUVOIR telescope. 
    The hexagons have antialiased edges. This routine does not modify the wavefront.
    
    Parameters
    ----------
    wf : object
        WaveFront class object
        
    nrings : int
        Number of rings of hexagons in aperture (e.g. 1 = a central hexagon 
        surrounded by a ring of hexagons)
        
    hexrad : float
        The distance in meters from the center of a hexagon segment to a vertex.
        
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
        The counterclockwise rotation in degrees of the aperture about its center.
        
    Returns
    -------
        numpy ndarray
        A hexagonal mask
    """

    ngrid = wf.ngrid
    
    ap = np.zeros([ngrid, ngrid], dtype = np.float64)
    
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
                if not any(counter==np.array([1, 12, 113, 114, 125, 126])):
                    ap = ap + proper.prop_polygon(wf, 6, hexrad, xhex, yhex, rotation = angle)
            
            if (iring != 0):
                xhex = -x * np.cos(angle_rad) - y * np.sin(angle_rad) + xc
                yhex = -x * np.sin(angle_rad) + y * np.cos(angle_rad) + yc
                counter += 1
                if not any(counter==np.array([1, 12, 113, 114, 125, 126])):
                    ap = ap + proper.prop_polygon(wf, 6, hexrad, xhex, yhex, rotation = angle)
                
            y += hexsep
            
    return ap


def falco_gen_pupil_customHex( input ):
#%gen_pupil_SCDA Generates a simple pupil.
#%   Function may be used to generate circular, annular, and simple on-axis 
#%   telescopes with radial struts. 

    hg_expon = 1000 # hyper-gaussian exponent for anti-aliasing 
    hg_expon_spider = 100 # hyper-gaussian exponent for anti-aliasing 

    N = input["Npad"] #Number of samples in NxN grid 
    OD = input["OD"] # pupil outer diameter, can be < 1
    ID = input["ID"] # central obscuration radius 
    apRad = input["Nbeam"]/2. # aperture radius in samples 
    wStrut = input["wStrut"] # width of strut [pupil diameters], float
    Nstrut = input["Nstrut"] # Number of radial struts, numpy ndarray
    angStruts = input["angStrut"] # Azimuthal locations of the radial struts, numpy ndarray
    
    # Create coordinates
    [X,Y] = np.meshgrid(np.arange(-N/2,N/2),np.arange(-N/2,N/2))
    [THETA,RHO] = falco.utils.cart2pol(X,Y)
   
    input["apDia"] = input["Nbeam"];
    if('pistons' in input.keys()):
        PUPIL0 = falco.segmentutils.hexSegMirror_getField( input )
    else:
        PUPIL0 = falco.segmentutils.hexSegMirror_getSupport( input )
    
    # Create inner and outer circles
    if(ID > 0):
        PUPIL = np.exp(-(RHO/(apRad*OD))**hg_expon) - np.exp(-(RHO/(apRad*ID))**hg_expon)
    else:
        PUPIL = np.exp(-(RHO/(apRad*OD))**hg_expon)
    
    PUPIL = PUPIL*PUPIL0
    
    # Create spiders 
    if(wStrut > 0):
        
        if not(Nstrut==angStruts.size):
            raise ValueError("Pupil generation error: angStrut should be an array of length Nstrut.")
        
        halfwidth = wStrut*2.*apRad
        for ang in angStruts:
           PUPIL = PUPIL*(1.-np.exp(-(RHO*np.sin(THETA-ang*np.pi/180.)/halfwidth)**hg_expon_spider)*
               (RHO*np.cos(THETA-ang*np.pi/180.)>0))
                           
    return PUPIL   
                  

def falco_gen_pupil_LUVOIR_B(Nbeam):

    D = 7.989 #--meters, circumscribed. The segment size is 0.955 m, flat-to-flat, and the gaps are 6 mm. 
    wGap = 6e-3/D # Fractional width of segment gaps
    
    input = {} # initialize dictionary
    input["Nbeam"] = Nbeam/0.925 # number of points across the pupil diameter
    input["wGap"] = wGap*Nbeam # number of samples across segment gaps
    input["numRings"] = 4 # Number of rings in hexagonally segmented mirror 
    input["Npad"] = int(2**(falco.utils.nextpow2(Nbeam)))
    input["ID"] = 0 # central obscuration radius 
    input["OD"] = 1 # pupil outer diameter, can be < 1
    input["Nstrut"] = 0 # Number of struts 
    input["angStrut"] = np.array([]) # Angles of the struts (deg)
    input["wStrut"] = np.array([]) # Width of the struts (fraction of pupil diam.)

    missingSegments = np.ones(falco.segmentutils.hexSegMirror_numSegments(input["numRings"]),)
    for index in range(6): #= 0:5
        missingSegments[38+index*4 - 1] = 0

    input["missingSegments"] = missingSegments;

    pupil = falco_gen_pupil_customHex(input)
        
    return pupil
        
def falco_gen_vortex_mask( charge, N ):
#% REQUIRED INPUTS: 
#% charge  = charge of the vortex
#% N   = number of points across the array 
#%
#% OUTPUTS:
#%  V:     2-D square array of the vortex E-field
#%
#% Created in 2018 by Garreth Ruane.
#
#function V = falco_gen_vortex_mask( charge, N )
#%   Detailed explanation goes here

    [X,Y] = np.meshgrid(np.arange(-N/2,N/2),np.arange(-N/2,N/2))
    V = np.exp(1j*charge*np.arctan2(Y,X))
    
    return V

def falco_gen_pupil_Simple( input ):
#% Copyright 2018, by the California Institute of Technology. ALL RIGHTS
#% RESERVED. United States Government Sponsorship acknowledged. Any
#% commercial use must be negotiated with the Office of Technology Transfer
#% at the California Institute of Technology.
#% -------------------------------------------------------------------------
#%
#% Inputs: 
#% inputs.Nbeam - Number of samples across the beam 
#% inputs.OD - Outer diameter (fraction of Nbeam)
#% inputs.ID - Inner diameter (fraction of Nbeam)
#% inputs.Nstrut - Number of struts
#% inputs.angStrut - Array of struct angles (deg)
#% inputs.wStrut - Strut widths (fraction of Nbeam)
#% inputs.stretch - Create an elliptical aperture by changing Nbeam along
#%                   the horizontal direction by a factor of stretch (PROPER
#%                   version isn't implemented as of March 2019).
#
#function PUPIL = falco_gen_pupil_Simple( input )
#%falco_gen_pupil_SCDA Generates a simple pupil.
#%   Function may be used to generate circular, annular, and simple on-axis 
#%   telescopes with radial struts. 
    pass

    hg_expon = 1000 #hyper-gaussian exponent for anti-aliasing 
    hg_expon_spider = 100 # hyper-gaussian exponent for anti-aliasing 

    if('centering' in input.keys()):
        centering = input["centering"]
    else:
        centering = 'pixel'
    
    N = input["Npad"] #Number of samples in NxN grid 
    OD = input["OD"] # pupil outer diameter, can be < 1
    ID = input["ID"] # central obscuration radius 
    apRad = input["Nbeam"]/2 # aperture radius in samples 
    b = input["stretch"] if('stretch' in input.keys()) else 1.0
    
    # Create coordinates
    if centering == 'pixel':
        [X,Y] = np.meshgrid( np.arange(-N/2,N/2) )
    elif centering == 'pixel':
        [X,Y] = np.meshgrid( np.arange(-(N-1)/2,(N-1)/2+1) )
            
    [THETA,RHO] = utils.cart2pol(X/b,Y)
    
    # Make sure the inputs make sense
    if(ID > OD):
        raise ValueError("Pupil generation error: Inner diameter larger than outer diameter.")
    
    # Create inner and outer circles
    if(ID > 0):
        PUPIL = np.exp(-(RHO/(apRad*OD))**hg_expon) - np.exp(-(RHO/(apRad*ID))**hg_expon)
    else:
        PUPIL = np.exp(-(RHO/(apRad*OD))**hg_expon)
    
    #--OVERWRITE if PROPER is specified
    if('flagPROPER' in input.keys()):
        if(input["flagPROPER"]==True):

            # Create inner and outer circles in PROPER

            #--INITIALIZE PROPER
            Dbeam = 1 #--diameter of beam (normalized to itself)
            dx = Dbeam/input["Nbeam"]
            Narray = N
            Darray = Narray*dx
            wl_dummy = 1e-6 #--dummy value
            bdf = Dbeam/Darray #--beam diameter fraction
            xshift = 0 #--x-shear of pupil
            yshift = 0 #--y-shear of pupil
            bm = proper.prop_begin(Dbeam, wl_dummy, Narray,bdf);

            if(centering=='pixel'):
                cshift = 0
            elif(centering=='interpixel'):
                cshift = -dx/2.

            #--PRIMARY MIRROR (OUTER DIAMETER)
            ra_OD = OD/2
            cx_OD = 0 + cshift + xshift
            cy_OD = 0 + cshift + yshift
            proper.prop_circular_aperture(bm, ra_OD,cx_OD,cy_OD)

            if(ID > 0):
                #--SECONDARY MIRROR (INNER DIAMETER)
                ra_ID = ID/2.
                cx_ID = 0 + cshift + xshift
                cy_ID = 0 + cshift + yshift
                proper.prop_circular_obscuration(bm, ra_ID,cx_ID,cy_ID)

            PUPIL = np.fft.ifftshift(np.abs(bm.wfarr))           


    # Create spiders 
    if(input["wStrut"] > 0):
        
        numSpiders = input["Nstrut"]
        angs = input["angStrut"]
        
        if not (numSpiders==angs.size):
            raise ValueError('Pupil generation error: angStrut should be an array of length Nstrut');
        
        halfwidth = input["wStrut"]*2*apRad
        for ang in angs:
           PUPIL = PUPIL*(1-np.exp(-(RHO*np.sin(THETA-ang*np.pi/180)/halfwidth)**hg_expon_spider)*
                          (RHO*np.cos(THETA-ang*np.pi/180)>0));

    return PUPIL
