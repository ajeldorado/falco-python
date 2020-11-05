"""Functions for setting up and generating HLC occulters."""

import numpy as np
from math import radians, sin, cos
from astropy.io import fits
from scipy.interpolate import griddata, RectBivariateSpline
import matplotlib.pyplot as plt

import falco
from falco.util import ceil_even, pad_crop, cart2pol
from falco.check import real_positive_scalar


def gen_fpm_from_LUT(mp, si, wi, modelType):
    """
    Produce the complex transmission of an HLC focal plane mask.

    Parameters
    ----------
    mp : TYPE
        DESCRIPTION.
    si : TYPE
        DESCRIPTION.
    wi : TYPE
        DESCRIPTION.
    modelType : {'compact', 'full'}
        String denoting whether to use the output is for the compact or full
        optical model.

    Returns
    -------
    fpm : numpy ndarray
        2-D, complex-valued representation of the HLC FPM.

    Notes
    -----
    In Matlab version is called falco_gen_fpm_from_LUT.
    """
    # Different variables for compact and full models
    if modelType.lower() == 'compact':
        # size = (Ndiel, Nmetal, mp.Nsbp)
        complexTransCube = mp.complexTransCompact
        # Index of the wavelength in mp.sbp_centers
        ilam = si
    elif modelType.lower() == 'full':
        # size = (Ndiel,Nmetal,mp.Nsbp*mp.Nwpsbp)
        complexTransCube = mp.complexTransFull
        # Index of the wavelength in mp.lam_array
        ilam = (si-1)*mp.Nwpsbp + wi
    else:
        raise ValueError('modelType must specify full or compact model.')

    t_diel_bias = mp.t_diel_bias_nm*1e-9  # bias thickness of dielectric [m]

    # Generate thickness profiles of each layer
    # Metal layer profile [m]
    DM8surf = falco.hlc.gen_fpm_surf_from_cube(mp.dm8, modelType)
    # Dielectric layer profile [m]
    DM9surf = t_diel_bias + falco.hlc.gen_fpm_surf_from_cube(mp.dm9, modelType)
    Nxi = DM8surf.shape[1]
    Neta = DM8surf.shape[0]

    if not (DM8surf.size == DM9surf.size):
        raise ValueError('FPM surf arrays for DM 8 and 9 must be same size!')

    # Obtain indices of nearest thickness values in the complex tran datacube.
    DM8transInd = falco.hlc.discretize_fpm_surf(DM8surf, mp.t_metal_nm_vec,
                                      mp.dt_metal_nm)
    DM9transInd = falco.hlc.discretize_fpm_surf(DM9surf, mp.t_diel_nm_vec,
                                      mp.dt_diel_nm)

    # Look up values
    fpm = np.zeros((Neta, Nxi), dtype=complex)
    for ix in range(Nxi):
        for iy in range(Neta):
            ind_metal = DM8transInd[iy, ix]
            ind_diel = DM9transInd[iy, ix]
            fpm[iy, ix] = complexTransCube[ind_diel, ind_metal, ilam]

    return fpm


def gen_fpm_cube_from_LUT(mp, modelType):
    """
    Produce the complex transmission of an HLC focal plane mask.

    Parameters
    ----------
    mp : TYPE
        DESCRIPTION.
    modelType : {'compact', 'full'}
        String denoting whether to use the output is for the compact or full
        optical model.

    Returns
    -------
    fpmCube : numpy ndarray
        Stack of 2-D, complex-valued HLC FPMs at each wavelength.

    Notes
    -----
    In Matlab version is called falco_gen_HLC_FPM_complex_trans_cube.
    """
    # Different variables for compact and full models
    if modelType.lower() == 'compact':
        # size = (Ndiel, Nmetal, mp.Nsbp)
        complexTransCube = mp.complexTransCompact
        lamVec = mp.sbp_centers
    elif modelType.lower() == 'full':
        # size = (Ndiel,Nmetal,mp.Nsbp*mp.Nwpsbp)
        complexTransCube = mp.complexTransFull
        lamVec = mp.full.lambdas
    else:
        raise ValueError('modelType must specify full or compact model.')
    Nlam = len(lamVec)

    t_diel_bias = mp.t_diel_bias_nm*1e-9  # bias thickness of dielectric [m]

    # Generate thickness profiles of each layer
    # Metal layer profile [m]
    DM8surf = falco.hlc.gen_fpm_surf_from_cube(mp.dm8, modelType)
    # Dielectric layer profile [m]
    DM9surf = t_diel_bias + falco.hlc.gen_fpm_surf_from_cube(mp.dm9,
                                                                 modelType)
    Neta, Nxi = DM8surf.shape

    if not (DM8surf.shape == DM9surf.shape):
        raise ValueError('FPM surf arrays for DM 8 and 9 must be same size!')

    # Obtain indices of nearest thickness values in the complex tran datacube.
    DM8transInd = falco.hlc.discretize_fpm_surf(DM8surf, mp.t_metal_nm_vec,
                                                mp.dt_metal_nm)
    DM9transInd = falco.hlc.discretize_fpm_surf(DM9surf, mp.t_diel_nm_vec,
                                                mp.dt_diel_nm)

    # Look up values
    fpmCube = np.zeros((Neta, Nxi, Nlam), dtype=complex)
    for ilam in range(Nlam):
        for ix in range(Nxi):
            for iy in range(Neta):
                ind_metal = DM8transInd[iy, ix]
                ind_diel = DM9transInd[iy, ix]
                fpmCube[iy, ix, ilam] = complexTransCube[ind_diel, ind_metal,
                                                         ilam]

    return fpmCube, DM8surf, DM9surf


def discretize_fpm_surf(FPMsurf, t_nm_vec, dt_nm):
    """
    Discretize the surface profiles of the FPM materials.

    Parameters
    ----------
    FPMsurf : numpy ndarray
        2-D array of the FPM material layer
    t_nm_vec : numpy ndarray
        The vector of allowed thickness values for the material layer
    dt_nm : float
        Scalar value of the step size between allowed thickness values
    
    Returns
    -------
    DMtransInd : numpy ndarray, dtype = int
        Array index of the metal/dielectric layer for each DM8/9 surface value
        in the complex transmission matrix.

    """
    # Convert surface profiles from meters to nanometers
    FPMsurf = 1e9*FPMsurf

    # Stay within the thickness range since material properties are not defined
    # outside it
    FPMsurf[FPMsurf < np.min(t_nm_vec)] = np.min(t_nm_vec)
    FPMsurf[FPMsurf > np.max(t_nm_vec)] = np.max(t_nm_vec)

    # Discretize to find the index the the complex transmission array
    DMtransInd = 0 + np.round(1/dt_nm*(FPMsurf - np.min(t_nm_vec)))

    return DMtransInd.astype(int)


def gen_fpm_surf_from_cube(dm, modelType):
    """
    Produce a FPM surface (in meters) with linear superposition.

    Function to produce a FPM surface (in meters). Uses linear superposition
    of a datacube of influence functions.

    Parameters
    ----------
    dm : TYPE
        DESCRIPTION.
    modelType : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    fpmSurf : TYPE
        DESCRIPTION.

    Notes
    -----
    In Matlab version is called falco_gen_HLC_FPM_surf_from_cube.

    """
    if modelType.lower() == 'compact':
        dmFullOrCompact = dm.compact
    elif modelType.lower() == 'full':
        dmFullOrCompact = dm
    else:
        raise ValueError('modelType must be full or compact.')

    fpmSurf = np.zeros((dmFullOrCompact.NdmPad, dmFullOrCompact.NdmPad))
    for iact in range(dm.NactTotal):
        if np.sum(np.abs(dmFullOrCompact.inf_datacube[:, :, iact])) >= 1e-8 \
            and np.sum(np.abs(dm.VtoH.flatten()[iact])) >= 1e-12:
            x_box_ind = np.arange(dmFullOrCompact.xy_box_lowerLeft[0, iact],
                                  dmFullOrCompact.xy_box_lowerLeft[0, iact]
                                  + dmFullOrCompact.Nbox, dtype=np.int)
            y_box_ind = np.arange(dmFullOrCompact.xy_box_lowerLeft[1, iact],
                                  dmFullOrCompact.xy_box_lowerLeft[1, iact]
                                  + dmFullOrCompact.Nbox, dtype=np.int)
            fpmSurf[np.ix_(y_box_ind, x_box_ind)] += dm.V.flatten()[iact] *\
            dm.VtoH.flatten()[iact]*dmFullOrCompact.inf_datacube[:, :, iact]

    # Non-negative values only
    fpmSurf[fpmSurf < 0] = 0.

    return fpmSurf


"""-------------------------------------------------------------------------"""


def setup_fpm_cosine(mp):
    
    if mp.F3.full.res != mp.F3.compact.res:
        raise ValueError('Resolution at F3 must be same for cosine basis set.')
    
    # Centering of DM surfaces on array
    mp.dm8.centering = mp.centering
    mp.dm9.centering = mp.centering

    mp.dm9.compact = mp.dm9
    
    mp.dm9.dxi = (mp.fl*mp.lambda0/mp.P2.D)/mp.F3.full.res  # width of a pixel at the FPM in the full model (meters)
    mp.dm9.compact.dxi = (mp.fl*mp.lambda0/mp.P2.D)/mp.F3.compact.res  # width of a pixel at the FPM in the compact model (meters)

    drCos = 1/mp.dm9.actres  # Width and double-separation of the cosine rings [lambda0/D]
    Nrad = int(np.ceil(2*mp.dm9.actres*mp.F3.Rin))

    # Generate datacube of influence functions, which are rings with radial cosine profile
    # Compact model
    mp.dm9.compact.NdmPad = ceil_even(1+2*mp.F3.Rin*mp.F3.compact.res)
    NbeamCompact = mp.dm9.compact.NdmPad
    mp.dm9.NdmPad = NbeamCompact
    mp.dm9.compact.Nbox = mp.dm9.compact.NdmPad # the modes take up the full array.
    # Normalized coordinates: Compact model
    if mp.centering == 'pixel':
        xc = np.arange(-mp.dm9.compact.NdmPad/2, mp.dm9.compact.NdmPad/2)/mp.F3.compact.res
    elif mp.centering == 'interpixel':
        xc = np.arange(-(mp.dm9.compact.NdmPad-1)/2, (mp.dm9.compact.NdmPad+1)/2)/mp.F3.compact.res
        
    [Xc, Yc] = np.meshgrid(xc, xc)
    Rc, THETAc = cart2pol(Xc, Yc)
    
    # Hyper-gaussian rolloff at edge of occulter
    hg_expon = 44  # Found empirically
    apRad = mp.F3.Rin/(mp.F3.Rin+0.1)  # Found empirically
    OD = 1
    mask = Rc <= mp.F3.Rin
    windowFull = mask*np.exp(-(Rc/mp.F3.Rin/(apRad*OD))**hg_expon)
    
    drSep = drCos/2
    min_azimSize = mp.min_azimSize_dm9  # [microns]
    pow_arr = np.arange(2, 62, 2)*6

    numdivCos = 2
    countCos = 0
    start_rad = int(np.floor(mp.dm9.actres/2))+1
    for ri in range(start_rad, Nrad+1):
        for _iter in range(numdivCos+1):
            countCos += 1
    numdivSin = 3
    countSin = 0
    start_rad = int(np.floor(mp.dm9.actres/2))+1
    for ri in range(start_rad, Nrad+1):
        for _iter in range(numdivSin+1):
            countSin += 1
    mp.dm9.NactTotal = Nrad + countCos + countSin
    mp.dm9.compact.inf_datacube = np.zeros((mp.dm9.compact.NdmPad,
                                    mp.dm9.compact.NdmPad, mp.dm9.NactTotal))
    
    # Compute the ring influence functions
    for ri in range(1, Nrad+1):
        modeTemp = windowFull * (1 + (-1)**((ri+1) % 2) *
                                 np.cos(2*np.pi*(Rc*mp.dm9.actres-0.5)))/2
        rMin = drSep*(ri - 1)
        rMax = drSep*(ri + 1)
        if ri == 1:  # Fill in the center
            modeTemp[Rc < drSep] = 1
        else:
            modeTemp[Rc < rMin] = 0
        modeTemp[Rc > rMax] = 0
        mp.dm9.compact.inf_datacube[:, :, ri-1] = modeTemp
        
    # for ri in range(1, Nrad+1):
    #     infFunc = mp.dm9.compact.inf_datacube[:, :, ri-1]
    #     plt.imshow(infFunc); plt.colorbar(); plt.pause(0.05)
    
    beamRad = NbeamCompact/2
    if mp.centering == 'pixel':
        x = np.arange(-beamRad, beamRad)
    elif mp.centering == 'interpixel':
        x = np.arange((NbeamCompact-1)/2, (NbeamCompact+1)/2)
    X, Y = np.meshgrid(x, x)
    RHO, THETA = cart2pol(X, Y)
    THETA2 = THETA+np.pi/3*2
    THETA3 = THETA+np.pi/3*4
    THETA4 = THETA+np.pi/3
    THETA5 = THETA+np.pi/3*3
    THETA6 = np.fliplr(THETA)
    apRad = mp.F3.Rin/(mp.F3.Rin+0.1)  # Found empirically
    OD = 1
    mask = Rc <= mp.F3.Rin
    
    # Cosine basis
    numdiv = 2
    count = 0  # index counter
    for ri in np.arange(start_rad, Nrad+1):
        modeTemp = windowFull * (1 + (-1)**((ri+1) % 2) *
                                 np.cos(2*np.pi*(Rc*mp.dm9.actres-0.5)))/2
        rMin = drSep*(ri - 1)
        rMax = drSep*(ri + 1)
        if ri == 1:  # Fill in the center
            modeTemp[Rc < drSep] = 1
        else:
            modeTemp[Rc < rMin] = 0
        
        modeTemp[Rc > rMax] = 0
        for II in range(numdiv+1):
            # Choose power for number of lobes
            powmin = 2 * np.pi * rMin / min_azimSize * 18
            aux = pow_arr - powmin
            aux[aux < 0] = np.Inf
            ind_mi = np.argmin(aux)
            power = pow_arr[ind_mi]
            #
            cosFull = np.cos(THETA*power) + 1
            numdiv = int(power/6)
            dth = 2*np.pi/power
            th_arr = np.linspace(np.pi/2, np.pi/2+np.pi/3, numdiv+1)
            th_rev_arr = np.linspace(np.pi/2+np.pi/3, np.pi/2, numdiv+1)
    
            th = th_arr[II]
            th_rev = th_rev_arr[II]
            ind = np.logical_and(THETA < (th+dth/2), THETA > (th-dth/2))
            ind_rev = np.logical_and(THETA4 < (th_rev+dth/2),
                                     THETA4 > (th_rev-dth/2))
            ind2 = np.logical_and(THETA2 < (th+dth/2),
                                  THETA2 > (th-dth/2))
            ind2_rev = np.logical_and(THETA5 < (th_rev+dth/2),
                                      THETA5 > (th_rev-dth/2))
            ind3 = np.logical_and(THETA3 < (th+dth/2), THETA3 > (th-dth/2))
            ind3_rev = np.logical_and(THETA6 < (th+dth/2-np.pi/2-np.pi/3/2),
                                      THETA6 > (th-dth/2-np.pi/2-np.pi/3/2))
            indTot = np.logical_or(ind, ind2)
            indTot = np.logical_or(indTot, ind3)
            indTot = np.logical_or(indTot, ind_rev)
            indTot = np.logical_or(indTot, ind2_rev)
            indTot = np.logical_or(indTot, ind3_rev)
            cosII = cosFull * indTot * modeTemp / 2
            mp.dm9.compact.inf_datacube[:, :, Nrad+count] = cosII
            count += 1

    # Sin basis
    numdiv = 3
    for ri in np.arange(start_rad, Nrad+1):
        modeTemp = windowFull * (1 + (-1)**((ri+1) % 2) *
                                 np.cos(2*np.pi*(Rc*mp.dm9.actres-0.5)))/2
        rMin = drSep*(ri - 1)
        rMax = drSep*(ri + 1)
        if ri == 1:  # Fill in the center
            modeTemp[Rc < drSep] = 1
        else:
            modeTemp[Rc < rMin] = 0
        modeTemp[Rc > rMax] = 0
        for II in range(numdiv+1):
            # Choose power for number of lobes
            powmin = 2*np.pi*rMin/min_azimSize*18
            aux = pow_arr - powmin
            aux[aux < 0] = np.Inf
            ind_mi = np.argmin(aux)
            power = pow_arr[ind_mi]
            # disp(['Number of lobes',num2str(pow)])
            cosFull = -np.cos(THETA*power)+1
            
            dth = 2*np.pi/power
            th_arr = np.linspace(np.pi/2, np.pi/2+np.pi/3+np.pi/6, numdiv+1)

            th = th_arr[II] + np.pi/12
            if th < np.pi:
                ind = np.logical_and(THETA < (th+dth/2), THETA > (th-dth/2))
            else:
                ind = np.fliplr(np.logical_and(THETA < (np.pi-th+dth/2),
                                               THETA > (np.pi-th-dth/2)))
            ind2 = np.logical_and(THETA2 < (th+dth/2),
                                  THETA2 > (th-dth/2))
            ind3 = np.logical_and(THETA3 < (th+dth/2),
                                  THETA3 > (th-dth/2))
            indTotsin = np.logical_or(ind, ind2)
            indTotsin = np.logical_or(indTotsin, ind3)
            cosII = cosFull * indTotsin * modeTemp / 2
            mp.dm9.compact.inf_datacube[:, :, Nrad+count] = cosII
            count += 1
            
    #         numdiv = int(power/6)
    #         dth = 2*np.pi/power
    #         th_arr = np.linspace(np.pi/2-np.pi/2/power, np.pi/2+np.pi/3-np.pi/2/power, numdiv+1)
    #         th_rev_arr = np.linspace(np.pi/2+np.pi/3+np.pi/2/power, np.pi/2+np.pi/2/power, numdiv+1)
    
    #         th = th_arr[II]
    #         th_rev = th_rev_arr[II]
    #         ind = np.logical_and((THETA)<(th+dth/2), (THETA)>(th-dth/2))
    #         ind_rev = np.logical_and((THETA4)<(th_rev+dth/2),
    #                                  (THETA4)>(th_rev-dth/2))
    #         ind2 = np.logical_and((THETA2)<(th+dth/2), (THETA2)>(th-dth/2))
    #         ind2_rev = np.logical_and((THETA5)<(th_rev+dth/2),
    #                                   (THETA5)>(th_rev-dth/2))
    #         ind3 = np.logical_and((THETA3)<(th+dth/2), (THETA3)>(th-dth/2))
    #         ind3_rev = np.logical_and((THETA6)<(th+dth/2-np.pi/2-np.pi/3/2),
    #                                   (THETA6)>(th-dth/2-np.pi/2-np.pi/3/2))
    #         indTot0 = np.logical_or(ind, ind2)
    #         indTot0 = np.logical_or(indTot0, ind3);
    #         indTot_rev = np.logical_or(ind_rev, ind2_rev)
    #         indTot_rev = np.logical_or(indTot_rev, ind3_rev)
    #     # %     cosII = zeros(N);
    #         sinII = sinFull*indTot0*modeTemp + sinFull_rev*indTot_rev*modeTemp
    # # %         figure(102);imagesc(sinII);axis image; set(gca,'YDir', 'normal')
    # # %         pause(0.1)
    #         mp.dm9.compact.inf_datacube[:, :, Nrad+count] = sinII
    #         count += 1
    
    mp.dm9.inf_datacube = mp.dm9.compact.inf_datacube
    
    mp.dm9.NactTotal = mp.dm9.inf_datacube.shape[2]
    mp.dm9.VtoH = mp.dm9.VtoHavg*np.ones(mp.dm9.NactTotal)
    
    # for ri in range(mp.dm9.NactTotal):
    #     infFunc = mp.dm9.compact.inf_datacube[:, :, ri]
    #     plt.imshow(infFunc); plt.colorbar(); plt.pause(0.01)
        
    # infSum = np.sum(mp.dm9.inf_datacube[:, :, Nrad+countCos:-1], 2)  # sin
    # infSum = np.sum(mp.dm9.inf_datacube[:, :, Nrad:Nrad+countCos], 2)  # cos
    # infSum = np.sum(mp.dm9.inf_datacube[:, :, 0:Nrad], 2)  # rings
    # infSum = np.sum(mp.dm9.inf_datacube[:, :, :], 2)  # all
    # plt.figure(); plt.imshow(infSum); plt.colorbar(); plt.pause(1)
    
    # Lower-left pixel coordinates are all (1,1) since the Zernikes take up the full array.
    mp.dm9.xy_box_lowerLeft = np.zeros((2, mp.dm9.NactTotal))
    mp.dm9.compact.xy_box_lowerLeft = np.zeros((2, mp.dm9.NactTotal))
    mp.dm9.compact.Nbox = NbeamCompact
    mp.dm9.Nbox = NbeamCompact
    
    # Coordinates for the full FPM array [meters]
    if mp.centering == 'pixel':
        mp.dm9.compact.x_pupPad = np.arange(-mp.dm9.compact.NdmPad/2,
                                            (mp.dm9.compact.NdmPad/2)) * \
                                    mp.dm9.compact.dxi
    elif mp.centering == 'interpixel':
        mp.dm9.compact.x_pupPad = np.arange(-(mp.dm9.compact.NdmPad-1)/2,
                                            (mp.dm9.compact.NdmPad+1)/2) * \
                                    mp.dm9.compact.dxi
    mp.dm9.compact.y_pupPad = mp.dm9.compact.x_pupPad
    
    # Initial DM9 voltages
    if not hasattr(mp.dm9, 'V'):
        mp.dm9.V = np.zeros(mp.dm9.NactTotal)
        mp.dm9.V[0:Nrad] = mp.dm9.V0coef * np.ones(Nrad)
    else:
        mp.dm9.V = mp.DM9V0
    mp.dm9.Vmin = np.min(mp.t_diel_nm_vec)  # minimum thickness of FPM dielectric layer (nm)
    mp.dm9.Vmax = np.max(mp.t_diel_nm_vec)  # maximum thickness (from one actuator, not of the facesheet) of FPM dielectric layer (nm)

    # OPTIONS FOR DEFINING DM8 (FPM Metal)
    mp.dm8.VtoHavg = 1e-9  # gain of DM8 (meters/Volt)
    mp.dm8.Vmin = np.min(mp.t_metal_nm_vec)  # minimum thickness of FPM metal layer (nm)
    mp.dm8.Vmax = np.max(mp.t_metal_nm_vec)  # maximum thickness (from one actuator, not of the facesheet) of FPM metal layer (nm)

    # DM8 Option 2: Set basis as a single nickel disk.
    mp.dm8.NactTotal = 1
    mp.dm8.act_ele = 1
    print('%d actuators in DM8.' % mp.dm8.NactTotal)
    mp.dm8.VtoH = mp.dm8.VtoHavg * np.ones(mp.dm8.NactTotal)  # Gains: volts to meters in surface height;
    mp.dm8.xy_box_lowerLeft = np.array([0, 0]).reshape((2, 1))
    mp.dm8.compact = mp.dm8
    if not hasattr(mp.dm8, 'V'):  # Initial DM8 voltages
        mp.dm8.V = mp.dm8.V0coef * np.ones(mp.dm8.NactTotal)
    else:
        mp.dm8.V = mp.DM8V0
        
    # Don't define extra actuators and time:
    if not mp.F3.Rin == mp.F3.RinA:
        raise ValueError('Change mp.F3.Rin and mp.F3.RinA to be equal to avoid wasting time.')
        
    # Copy over some common values from DM9:
    mp.dm8.dxi = mp.dm9.dxi  # Width of a pixel at the FPM in full model (meters)
    mp.dm8.NdmPad = mp.dm9.NdmPad
    mp.dm8.Nbox = mp.dm8.NdmPad
    mp.dm8.compact.dxi = mp.dm9.compact.dxi  # Width of a pixel at the FPM in compact model (meters)
    mp.dm8.compact.NdmPad = mp.dm9.compact.NdmPad
    mp.dm8.compact.Nbox = mp.dm8.compact.NdmPad

    # Make or read in DM8 disk for the full model
    FPMgenInputs = {}
    FPMgenInputs['pixresFPM'] = mp.F3.full.res  # pixels per lambda_c/D
    FPMgenInputs['rhoInner'] = mp.F3.Rin  # radius of inner FPM amplitude spot (in lambda_c/D)
    FPMgenInputs['rhoOuter'] = np.Inf  # radius of outer opaque FPM ring (in lambda_c/D)
    FPMgenInputs['FPMampFac'] = 0  # amplitude transmission of inner FPM spot
    FPMgenInputs['centering'] = mp.centering
    diskFull = np.round(pad_crop(1-falco.mask.gen_annular_FPM(FPMgenInputs),
                                 mp.dm8.NdmPad))
    mp.dm8.inf_datacube = np.zeros((diskFull.shape[0], diskFull.shape[1], 1))
    mp.dm8.inf_datacube[:, :, 0] = diskFull
    # Make or read in DM8 disk for the compact model
    FPMgenInputs['pixresFPM'] = mp.F3.compact.res  # pixels per lambda_c/D
    diskCompact = np.round(pad_crop(1-falco.mask.gen_annular_FPM(FPMgenInputs),
                                    mp.dm8.compact.NdmPad))
    mp.dm8.compact.inf_datacube = np.zeros((diskCompact.shape[0],
                                            diskCompact.shape[1], 1))
    mp.dm8.compact.inf_datacube[:, :, 0] = diskCompact
    pass


def setup_fpm(mp):

    mp.dm9.Nact = ceil_even(2*mp.F3.Rin*mp.dm9.actres)  # number of actuators across DM9 (if not in a hex grid)

    if mp.dm9.inf0name.lower() in '3x3':
        mp.dm9.inf0 = 1/4 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])  # influence function
        mp.dm9.dx_inf0_act = 1/2  # number of inter-actuator widths per pixel 
        # FPM resolution (pixels per lambda0/D) in the compact and full models.
        mp.F3.compact.res = mp.dm9.actres / mp.dm9.dx_inf0_act
        mp.F3.full.res = mp.dm9.actres / mp.dm9.dx_inf0_act

    elif mp.dm9.inf0name.lower() in 'lanczos3':
        N = 91
        xs = np.arange(-(N-1)/2, (N+1)/2)/N*10*0.906  #(-(N-1)/2:(N-1)/2)/N *10*0.906

        a = 3
        Lx0 = a*np.sin(np.pi*xs)*np.sin(np.pi*xs/a)/(np.pi*xs)**2
        Lx0[xs == 0] = 1
        Lx = Lx0
        Lx[xs >= a] = 0
        Lx[xs <= -a] = 0
        Lxy = Lx.T @ Lx  # The 2-D Lanczos kernel
        Nhalf = np.ceil(N/2)
        Nrad = 30
        LxyCrop = Lxy[Nhalf-Nrad-1:Nhalf+Nrad, Nhalf-Nrad-1:Nhalf+Nrad]

        mp.dm9.inf0 = LxyCrop  # influence function
        mp.dm9.dx_inf0_act = 1/10  # number of inter-actuator widths per pixel 

    elif mp.dm9.inf0name.lower() in 'xinetics':
        mp.dm9.inf0 = 1*fits.getdata('influence_dm5v2.fits')
        mp.dm9.dx_inf0_act = 1/10  # number of inter-actuator widths per pixel 
    
    
    # DM8 and DM9 (Optimizable FPM) Setup

    # Centering of DM surfaces on array
    mp.dm8.centering = mp.centering
    mp.dm9.centering = mp.centering

    mp.dm9.compact = mp.dm9

    if hasattr(mp, 'flagDM9inf3x3'):
        mp.dm9.xcent_dm = mp.dm9.Nact/2 - 1/2
        mp.dm9.ycent_dm = mp.dm9.Nact/2 - 1/2
        if mp.centering in 'interpixel':
           raise ValueError('The 3x3 influence function for DM9 requires a pixel-centered coordinate system.')
    else:
        mp.dm9.xcent_dm = mp.dm9.Nact/2 - 1/2
        mp.dm9.ycent_dm = mp.dm9.Nact/2 - 1/2
    
    mp.dm9.dm_spacing = 1/mp.dm9.actres*(mp.fl*mp.lambda0/mp.P2.D)  # meters, pitch of DM actuators
    mp.dm9.compact = mp.dm9

    mp.dm9.compact.dm_spacing = mp.dm9.dm_spacing  # meters, pitch of DM actuators

    mp.dm9.dx_inf0 = (mp.dm9.dx_inf0_act)*mp.dm9.dm_spacing  # meters, sampling of the influence function 

    mp.dm9.compact.dx_inf0 = (mp.dm9.compact.dx_inf0_act)*mp.dm9.compact.dm_spacing  # meters, sampling of the influence function 

    mp.dm9.dxi = (mp.fl*mp.lambda0/mp.P2.D)/mp.F3.full.res  # width of a pixel at the FPM in the full model (meters)
    mp.dm9.compact.dxi = (mp.fl*mp.lambda0/mp.P2.D)/mp.F3.compact.res  # width of a pixel at the FPM in the compact model (meters)

    if mp.dm9.inf0.shape[0] == 3:
        fpm_inf_cube_3x3(mp.dm9)
        fpm_inf_cube_3x3(mp.dm9.compact)
    else:
        fpm_inf_cube(mp.dm9)
        fpm_inf_cube(mp.dm9.compact)
        
    # Zero out DM9 actuators too close to the outer edge (within mp.dm9.FPMbuffer lambda0/D of edge)
    r_cent_lam0D = mp.dm9.r_cent_act*mp.dm9.dm_spacing/(mp.dm9.dxi)/mp.F3.full.res

    ##
    mp.F3.RinA_inds = np.array([], dtype=int)
    mp.F3.RinAB_inds = np.array([], dtype=int)
    for ii in range(mp.dm9.NactTotal):
        # Zero out FPM actuators beyond the allowed radius (mp.F3.Rin)
        if r_cent_lam0D[ii] > mp.F3.Rin-mp.dm9.FPMbuffer:
           mp.dm9.inf_datacube[:, :, ii] = np.zeros_like(mp.dm9.inf_datacube[:, :, ii])
           mp.dm9.compact.inf_datacube[:, :, ii] = np.zeros_like(mp.dm9.compact.inf_datacube[:, :, ii])

        # Get the indices for the actuators within radius mp.F3.RinA
        if r_cent_lam0D[ii] <= mp.F3.RinA-mp.dm9.FPMbuffer:
            mp.F3.RinA_inds = np.append(mp.F3.RinA_inds, ii)
        else:  # Get the indices for the actuators between radii mp.F3.RinA and mp.F3.Rin
            mp.F3.RinAB_inds = np.append(mp.F3.RinAB_inds, ii)

    print('%d actuators in DM9.' % mp.dm9.NactTotal)

    mp.dm9.ABfac = 1  # Gain factor between inner and outer FPM regions
    mp.dm9.VtoHavg = 1e-9  # gain of DM9 (meters/Volt)
    mp.dm9.VtoH = mp.dm9.VtoHavg * np.ones(mp.dm9.NactTotal)  # Gains: volts to meters in surface height;
    mp.dm9.VtoH[mp.F3.RinAB_inds] = mp.dm9.ABfac * mp.dm9.VtoH[mp.F3.RinAB_inds]

    if not hasattr(mp.dm9, 'V'):  # Initial DM9 voltages
        mp.dm9.V = np.zeros(mp.dm9.NactTotal)
        mp.dm9.V[mp.F3.RinA_inds] = mp.dm9.V0coef
    else:
        mp.dm9.V = mp.DM9V0

    mp.dm9.Vmin = np.min(mp.t_diel_nm_vec)  # minimum thickness of FPM dielectric layer (nm)
    mp.dm9.Vmax = np.max(mp.t_diel_nm_vec)  # maximum thickness (from one actuator, not of the facesheet) of FPM dielectric layer (nm)

    # -OPTIONS FOR DEFINING DM8 (FPM Metal)
    mp.dm8.VtoHavg = 1e-9  # gain of DM8 (meters/Volt)
    mp.dm8.Vmin = np.min(mp.t_metal_nm_vec)  # minimum thickness of FPM metal layer (nm)
    mp.dm8.Vmax = np.max(mp.t_metal_nm_vec)  # maximum thickness (from one actuator, not of the facesheet) of FPM metal layer (nm)

    # DM8 Option 2: Set basis as a single nickel disk.
    mp.dm8.NactTotal = 1
    mp.dm8.act_ele = 1
    print('%d actuators in DM8.' % mp.dm8.NactTotal)
    mp.dm8.VtoH = mp.dm8.VtoHavg * np.ones(mp.dm8.NactTotal)  # Gains: volts to meters in surface height;
    mp.dm8.xy_box_lowerLeft = np.array([0, 0]).reshape((2, 1))
    mp.dm8.compact = mp.dm8
    if not hasattr(mp.dm8, 'V'):  # Initial DM8 voltages
        mp.dm8.V = mp.dm8.V0coef * np.ones(mp.dm8.NactTotal)
    else:
        mp.dm8.V = mp.DM8V0
        
    # Don't define extra actuators and time:
    if not mp.F3.Rin == mp.F3.RinA:
        raise ValueError('Change mp.F3.Rin and mp.F3.RinA to be equal to avoid wasting time.')
        
    # Copy over some common values from DM9:
    mp.dm8.dxi = mp.dm9.dxi  # Width of a pixel at the FPM in full model (meters)
    mp.dm8.NdmPad = mp.dm9.NdmPad
    mp.dm8.Nbox = mp.dm8.NdmPad
    mp.dm8.compact.dxi = mp.dm9.compact.dxi  # Width of a pixel at the FPM in compact model (meters)
    mp.dm8.compact.NdmPad = mp.dm9.compact.NdmPad
    mp.dm8.compact.Nbox = mp.dm8.compact.NdmPad

    # Make or read in DM8 disk for the full model
    FPMgenInputs = {}
    FPMgenInputs['pixresFPM'] = mp.F3.full.res  # pixels per lambda_c/D
    FPMgenInputs['rhoInner'] = mp.F3.Rin  # radius of inner FPM amplitude spot (in lambda_c/D)
    FPMgenInputs['rhoOuter'] = np.Inf  # radius of outer opaque FPM ring (in lambda_c/D)
    FPMgenInputs['FPMampFac'] = 0  # amplitude transmission of inner FPM spot
    FPMgenInputs['centering'] = mp.centering
    diskFull = np.round(pad_crop(1-falco.mask.gen_annular_FPM(FPMgenInputs),
                                 mp.dm8.NdmPad))
    mp.dm8.inf_datacube = np.zeros((diskFull.shape[0], diskFull.shape[1], 1))
    mp.dm8.inf_datacube[:, :, 0] = diskFull
    # Make or read in DM8 disk for the compact model
    FPMgenInputs['pixresFPM'] = mp.F3.compact.res  # pixels per lambda_c/D
    diskCompact = np.round(pad_crop(1-falco.mask.gen_annular_FPM(FPMgenInputs),
                                    mp.dm8.compact.NdmPad))
    mp.dm8.compact.inf_datacube = np.zeros((diskCompact.shape[0],
                                            diskCompact.shape[1], 1))
    mp.dm8.compact.inf_datacube[:, :, 0] = diskCompact
    
    # Zero out parts of DM9 actuators that go outside the nickel disk.
    # Also apply the grayscale edge.
    DM8windowFull = diskFull
    DM8windowCompact = diskCompact
    for iact in range(mp.dm9.NactTotal):
        if np.sum(np.abs(mp.dm9.inf_datacube[:, :, iact])) >= 1e-8 and np.abs(mp.dm9.VtoH[iact]) >= 1e-13:
            y_box_ind = np.arange(mp.dm9.xy_box_lowerLeft[0, iact], mp.dm9.xy_box_lowerLeft[0, iact]+mp.dm9.Nbox, dtype=int)  # x-indices in pupil arrays for the box
            x_box_ind = np.arange(mp.dm9.xy_box_lowerLeft[1, iact], mp.dm9.xy_box_lowerLeft[1, iact]+mp.dm9.Nbox, dtype=int)  # y-indices in pupil arrays for the box
            mp.dm9.inf_datacube[:, :, iact] = DM8windowFull[np.ix_(x_box_ind, y_box_ind)] * mp.dm9.inf_datacube[:, :, iact]

            y_box_ind = np.arange(mp.dm9.compact.xy_box_lowerLeft[0, iact], mp.dm9.compact.xy_box_lowerLeft[0, iact]+mp.dm9.compact.Nbox, dtype=int)  # x-indices in pupil arrays for the box
            x_box_ind = np.arange(mp.dm9.compact.xy_box_lowerLeft[1, iact], mp.dm9.compact.xy_box_lowerLeft[1, iact]+mp.dm9.compact.Nbox, dtype=int)  # y-indices in pupil arrays for the box
            mp.dm9.compact.inf_datacube[:, :, iact] = DM8windowCompact[np.ix_(x_box_ind, y_box_ind)] * mp.dm9.compact.inf_datacube[:, :, iact]

    pass


def gen_fpm_poke_cube(dm, mp, dx_dm, **kwargs):
    """
    Compute the datacube of each influence function.
    
    Influence functions are cropped down or padded up
    to the best size for angular spectrum propagation.

    Parameters
    ----------
    dm : ModelParameters
        Structure containing parameter values for the DM
    mp: falco.config.ModelParameters
        Structure of model parameters
    dx_dm : float
        Pixel width [meters] at the DM plane

    Other Parameters
    ----------------
    NOCUBE : bool
       Switch that tells function not to compute the datacube of influence
       functions.

    Returns
    -------
    None
        modifies structure "dm" by reference

    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    real_positive_scalar(dx_dm, 'dx_dm', TypeError)
          
    if "NOCUBE" in kwargs and kwargs["NOCUBE"]:
        flagGenCube = False
    else:
        flagGenCube = True

    # Define this flag if it doesn't exist in the older code for square actuator arrays only
    if not hasattr(dm, 'flag_hex_array'):
        dm.flag_hex_array = False

    # Set the order of operations
    XYZ = True
    if(hasattr(dm, 'flagZYX')):
        if(dm.flagZYX):
            XYZ = False

    # Compute sampling of the pupil. Assume that it is square.
    dm.dx_dm = dx_dm
    dm.dx = dx_dm

    # Default to being centered on a pixel if not specified
    if not(hasattr(dm, 'centering')):
        dm.centering = 'pixel'
        
    # Compute coordinates of original influence function
    Ninf0 = dm.inf0.shape[0]  # Number of points across inf func at native res
    x_inf0 = np.linspace(-(Ninf0-1)/2., (Ninf0-1)/2., Ninf0)*dm.dx_inf0
    # True for even- or odd-sized influence function maps as long as they are
    # centered on the array.
    [Xinf0, Yinf0] = np.meshgrid(x_inf0, x_inf0)
    
    # Number of points across the DM surface at native inf func resolution
    Ndm0 = falco.util.ceil_even(Ninf0 + (dm.Nact - 1)*(dm.dm_spacing/dm.dx_inf0))
    # Number of points across the (un-rotated) DM surface at new, desired res.
    dm.NdmMin = falco.util.ceil_even(Ndm0*(dm.dx_inf0/dm.dx))+2.
    # Number of points across the array to fully contain the DM surface at new
    # desired resolution and z-rotation angle.
    dm.Ndm = int(ceil_even((abs(np.array([np.sqrt(2.)*np.cos(radians(45.-dm.zrot)),
            np.sqrt(2.)*np.sin(radians(45.-dm.zrot))])).max())*Ndm0*(dm.dx_inf0/dm.dx))+2)
    
    # Compute list of initial actuator center coordinates (in actutor widths).
    if(dm.flag_hex_array):  # Hexagonal, hex-packed grid
        raise ValueError('flag_hex_array option not implemented yet.')
#     Nrings = dm.Nrings;
#     x_vec = [];
#     y_vec = [];
#     % row number (rowNum) is 1 for the center row and 2 is above it, etc.
#     % Nacross is the total number of segments across that row
#     for rowNum = 1:Nrings
#         Nacross = 2*Nrings - rowNum  # Number of actuators across at that row (for hex tiling in a hex shape)
#         yval = sqrt(3)/2*(rowNum-1);
#         bx = Nrings - (rowNum+1)/2  # x offset from origin
# 
#         xs = (0:Nacross-1).' - bx  # x values are 1 apart
#         ys = yval*ones(Nacross,1)  # same y-value for the entire row
# 
#         if(rowNum==1)
#             x_vec = [x_vec;xs];
#             y_vec = [y_vec;ys]; 
#         else
#             x_vec = [x_vec;xs;xs];
#             y_vec = [y_vec;ys;-ys]  # rows +/-n have +/- y coordinates
#         end
#     end
    else:  # Square grid [actuator widths]
        [dm.Xact, dm.Yact] = np.meshgrid(np.arange(dm.Nact) - 
                                         dm.xc, np.arange(dm.Nact)-dm.yc)
#        # Use order='F' to compare the final datacube to Matlab's output.
#        #  Otherwise, use C ordering for Python FALCO.
#        x_vec = dm.Xact.reshape(dm.Nact*dm.Nact,order='F')
#        y_vec = dm.Yact.reshape(dm.Nact*dm.Nact,order='F')
        x_vec = dm.Xact.reshape(dm.Nact*dm.Nact)
        y_vec = dm.Yact.reshape(dm.Nact*dm.Nact)
    
    dm.NactTotal = x_vec.shape[0]  # Total number of actuators in the 2-D array
    dm.xy_cent_act = np.zeros((2, dm.NactTotal))  # Initialize

    # Compute the rotation matrix to apply to the influence function and 
    #  actuator center locations
    tlt = np.zeros(3)
    tlt[0] = radians(dm.xtilt)
    tlt[1] = radians(dm.ytilt)
    tlt[2] = radians(-dm.zrot)

    sa = sin(tlt[0])
    ca = cos(tlt[0])
    sb = sin(tlt[1])
    cb = cos(tlt[1])
    sg = sin(tlt[2])
    cg = cos(tlt[2])

    if XYZ:
        Mrot = np.array([[cb * cg, sa * sb * cg - ca * sg, ca * sb * cg + sa * sg, 0.0],
                 [cb * sg, sa * sb * sg + ca * cg, ca * sb * sg - sa * cg, 0.0],
                [-sb,      sa * cb,                ca * cb,                0.0],
                     [0.0,                    0.0,                    0.0, 1.0]])
    else:
        Mrot = np.array([               [cb * cg,               -cb * sg,       sb, 0.0],
                [ca * sg + sa * sb * cg, ca * cg - sa * sb * sg, -sa * cb, 0.0],
                [sa * sg - ca * sb * cg, sa * cg + ca * sb * sg,  ca * cb, 0.0],
                                   [0.0,                    0.0,      0.0, 1.0]])
    
    # Compute the actuator center coordinates in units of actuator spacings
    for iact in range(dm.NactTotal):
        xyzVals = np.array([x_vec[iact], y_vec[iact], 0., 1.])
        xyzValsRot = Mrot @ xyzVals
        dm.xy_cent_act[0, iact] = xyzValsRot[0].copy()
        dm.xy_cent_act[1, iact] = xyzValsRot[1].copy()

    N0 = dm.inf0.shape[0]
    Npad = falco.util.ceil_odd(np.sqrt(2.)*N0)
    inf0pad = np.zeros((Npad, Npad))
    inf0pad[int(np.ceil(Npad/2.)-np.floor(N0/2.)-1):int(np.ceil(Npad/2.)+np.floor(N0/2.)),
            int(np.ceil(Npad/2.)-np.floor(N0/2.)-1):int(np.ceil(Npad/2.)+np.floor(N0/2.))] = dm.inf0

    ydim = inf0pad.shape[0]
    xdim = inf0pad.shape[1]
    
    xd2 = np.fix(xdim / 2.) + 1
    yd2 = np.fix(ydim / 2.) + 1
    cx = np.arange(xdim) + 1. - xd2
    cy = np.arange(ydim) + 1. - yd2
    [Xs0, Ys0] = np.meshgrid(cx, cy)

    xsNewVec = np.zeros(xdim*xdim)
    ysNewVec = np.zeros(ydim*ydim)
    Xs0Vec = Xs0.reshape(xdim*xdim)
    Ys0Vec = Ys0.reshape(ydim*ydim)
    
    for ii in range(Xs0.size):
        xyzVals = np.array([Xs0Vec[ii], Ys0Vec[ii], 0., 1.])
        xyzValsRot = Mrot @ xyzVals
        xsNewVec[ii] = xyzValsRot[0]
        ysNewVec[ii] = xyzValsRot[1]
    
    # Calculate the interpolated DM grid at the new resolution
    # (set extrapolated values to 0.0)
    dm.infMaster = griddata((xsNewVec, ysNewVec), inf0pad.reshape(Npad*Npad),
                            (Xs0, Ys0), method='cubic', fill_value=0.)

    # Crop down the influence function until it has no zero padding left
    infSum = np.sum(dm.infMaster)
    infDiff = 0.
    counter = 0
    while(abs(infDiff) <= 1e-7):
        counter = counter + 2
        infDiff = infSum - np.sum(abs(dm.infMaster[int(counter/2):int(-counter/2),
                                                   int(counter/2):int(-counter/2)]))

    # Subtract an extra 2 to negate the extra step that overshoots.
    counter = counter - 2
    Ninf0pad = dm.infMaster.shape[0]-counter
    if counter == 0:
        infMaster2 = dm.infMaster.copy()
    else:
        # The cropped-down influence function
        infMaster2 = dm.infMaster[int(counter/2):int(-counter/2),
                                  int(counter/2):int(-counter/2)].copy()
        dm.infMaster = infMaster2
    
    Npad = Ninf0pad

    # True for even- or odd-sized influence function maps as long as they are
    # centered on the array.
    x_inf0 = np.linspace(-(Npad-1)/2, (Npad-1)/2., Npad)*dm.dx_inf0
    [Xinf0, Yinf0] = np.meshgrid(x_inf0, x_inf0)

    # Translate and resample the master influence function to be at each 
    # actuator's location in the pixel grid 

    # Compute the size of the postage stamps.
    # Number of points across the influence function array at the DM plane's
    # resolution. Want as even
    Nbox = falco.util.ceil_even(Ninf0pad*dm.dx_inf0/dx_dm)
    dm.Nbox = Nbox
    # Also compute their padded sizes for the angular spectrum (AS) propagation
    # between P2 and DM1 or between DM1 and DM2
    # Minimum number of points across for accurate angular spectrum propagation
    Nmin = falco.util.ceil_even(np.max(mp.sbp_centers)*np.max(np.abs(np.array(
        [mp.d_P2_dm1, mp.d_dm1_dm2, (mp.d_P2_dm1+mp.d_dm1_dm2)])))/dx_dm**2)
    # Use a larger array if the max sampling criterion for angular spectrum
    # propagation is violated
    dm.NboxAS = np.max(np.array([Nbox, Nmin]))

    # Pad the pupil to at least the size of the DM(s) surface(s) to allow all
    # actuators to be located outside the pupil.
    # (Same for both DMs)

    # Find actuator farthest from center:
    dm.r_cent_act = np.sqrt(dm.xy_cent_act[0, :]**2 + dm.xy_cent_act[1, :]**2)
    dm.rmax = np.max(np.abs(dm.r_cent_act))
    NpixPerAct = dm.dm_spacing/dx_dm
    if(dm.flag_hex_array):
        # padded 2 actuators past the last actuator center to avoid trying to
        # index outside the array
        dm.NdmPad = falco.util.ceil_even((2.*(dm.rmax+2))*NpixPerAct + 1)
    else:
        # DM surface array padded by the width of the padded influence function
        # to prevent indexing outside the array. 
        # The 1/2 term is because the farthest actuator center is still half an
        # actuator away from the nominal array edge. 
        dm.NdmPad = falco.util.ceil_even((dm.NboxAS + 2.0*(1 + (np.max(
        np.abs(dm.xy_cent_act.reshape(2*dm.NactTotal)))+0.5)*NpixPerAct)))

    # Compute coordinates (in meters) of the full DM array
    if(dm.centering == 'pixel'):
        # meters, coords for the full DM arrays. Origin is centered on a pixel
        dm.x_pupPad = np.linspace(-dm.NdmPad/2., (dm.NdmPad/2. - 1),
                                  dm.NdmPad)*dx_dm
    else:
        # meters, coords for the full DM arrays. Origin is interpixel centered
        dm.x_pupPad = np.linspace(-(dm.NdmPad-1)/2., (dm.NdmPad-1)/2.,
                                  dm.NdmPad)*dx_dm

    dm.y_pupPad = dm.x_pupPad

    # Make NboxPad-sized postage stamps for each actuator's influence function
    if(flagGenCube):
        if not dm.flag_hex_array:
            print("  Influence function padded from %d to %d points for A.S. propagation." % (Nbox,dm.NboxAS))

        print('Computing datacube of DM influence functions... ', end='')

        # Find the locations of the postage stamps arrays in the larger pupilPad array
        dm.xy_cent_act_inPix = dm.xy_cent_act*(dm.dm_spacing/dx_dm)  # Convert units to pupil-plane pixels
        dm.xy_cent_act_inPix = dm.xy_cent_act_inPix + 0.5  # For the half-pixel offset if pixel centered. 
        dm.xy_cent_act_box = np.round(dm.xy_cent_act_inPix)  # Center locations of the postage stamps (in between pixels), in actuator widths
        dm.xy_cent_act_box_inM = dm.xy_cent_act_box*dx_dm  # now in meters
        dm.xy_box_lowerLeft = dm.xy_cent_act_box + (dm.NdmPad-Nbox)/2 - 0  # index of pixel in lower left of the postage stamp within the whole pupilPad array. +0 for Python, +1 for Matlab

        # Starting coordinates (in actuator widths) for updated master influence function.
        # This is interpixel centered, so do not translate!
        dm.x_box0 = np.linspace(-(Nbox-1)/2., (Nbox-1)/2., Nbox)*dx_dm
        [dm.Xbox0, dm.Ybox0] = np.meshgrid(dm.x_box0, dm.x_box0)  # meters, interpixel-centered coordinates for the master influence function

        # (Allow for later) Limit the actuators used to those within 1 actuator width of the pupil
        r_cent_act_box_inM = np.sqrt(dm.xy_cent_act_box_inM[0, :]**2 + dm.xy_cent_act_box_inM[1, :]**2)
        # Compute and store all the influence functions:
        dm.inf_datacube = np.zeros((Nbox, Nbox, dm.NactTotal))  # initialize array of influence function "postage stamps"
        dm.act_ele = np.arange(dm.NactTotal)  # Initialize as including all actuators

        inf_datacube = np.zeros((dm.NactTotal, Nbox, Nbox))

        interp_spline = RectBivariateSpline(x_inf0, x_inf0, dm.infMaster)  # RectBivariateSpline is faster in 2-D than interp2d
        # Refer to https://scipython.com/book/chapter-8-scipy/examples/two-dimensional-interpolation-with-scipyinterpolaterectbivariatespline/

        for iact in range(dm.NactTotal):
           xbox = dm.x_box0 - (dm.xy_cent_act_inPix[0, iact]-dm.xy_cent_act_box[0, iact])*dx_dm # X = X0 -(x_true_center-x_box_center)
           ybox = dm.x_box0 - (dm.xy_cent_act_inPix[1, iact]-dm.xy_cent_act_box[1, iact])*dx_dm # Y = Y0 -(y_true_center-y_box_center)
           dm.inf_datacube[:, :, iact] = interp_spline(ybox, xbox)
           inf_datacube[iact, :, :] = interp_spline(ybox, xbox)

        print('done.')

    else:
        dm.act_ele = np.arange(dm.NactTotal)
   

def fpm_inf_cube_3x3(dm):
    print('Computing datacube of FPM influence functions... ')
    
    # Compute sampling of the pupil. Assume that it is square.
    dm.dx_dm = dm.dxi
    
    # Default to being centered on a pixel (FFT-style)
    if not hasattr(dm, 'centering'):
        dm.centering = 'pixel'
    
    # Compute coordinates of original influence function
    Ninf0 = dm.inf0.shape[0]
    x_inf0 = np.arange(-(Ninf0-1)/2, (Ninf0+1)/2) * dm.dx_inf0  # True for even- or odd-sized influence function maps as long as they are centered on the array.
    [Xinf0, Yinf0] = np.meshgrid(x_inf0, x_inf0)
    
    # Compute list of initial actuator center coordinates (in actuator widths).
    # Square grid
    [dm.Xact, dm.Yact] = np.meshgrid(np.arange(0, dm.Nact)-dm.xcent_dm, np.arange(0, dm.Nact)-dm.ycent_dm)  # in actuator widths
    x_vec = dm.Xact.flatten()
    y_vec = dm.Yact.flatten()
    
    dm.NactTotal = x_vec.size  # Total number of actuators in the 2-D array
    
    dm.infMaster = dm.inf0
    Nbox = dm.inf0.shape[0]
    dm.Nbox = Nbox
    print('FPM influence function size =\t%dx%d ' % (Nbox, Nbox))
    
    dm.xy_cent_act = np.vstack((x_vec, y_vec))  # in actuator widths
    
    # Pad the pupil to at least the size of the DM(s) surface(s) to allow all
    # actuators to be located outside the pupil.
    # (Same for both DMs)
    
    # Find actuator farthest from center:
    dm.r_cent_act = np.sqrt(dm.xy_cent_act[0, :]**2 + dm.xy_cent_act[1, :]**2)
    dm.rmax = np.max(np.abs(dm.r_cent_act))
    dm.absxymax = np.max(np.abs(dm.xy_cent_act))
    NpixPerActWidth = dm.dm_spacing / dm.dx_dm
    
    dm.NdmPad = ceil_even((dm.Nact+2)*NpixPerActWidth)  # prevent indexing outside the array
    
    # Compute coordinates (in meters) of the full DM array
    if dm.centering in 'pixel':
        dm.x_pupPad = np.arange(-dm.NdmPad/2, dm.NdmPad/2)*dm.dx_dm  # meters, coords for the full DM arrays. Origin is centered on a pixel
    else:
        dm.x_pupPad = np.arange(-(dm.NdmPad-1)/2, (dm.NdmPad+1)/2)*dm.dx_dm  # meters, coords for the full DM arrays. Origin is centered between pixels for an even-sized array

    dm.y_pupPad = dm.x_pupPad
    
    # DM: (use NboxPad-sized postage stamps,
    
    # Find the locations of the postage stamps arrays in the larger pupilPad array
    dm.xy_cent_act_inPix = dm.xy_cent_act*(dm.dm_spacing/dm.dx_dm)  # Convert units to pupil-file pixels
    if not dm.centering in 'pixel':
        raise ValueError('Not adapted for non-pixel centering.')
    
    dm.xy_cent_act_box = np.round(dm.xy_cent_act_inPix)  # Center locations of the postage stamps (in between pixels), in actuator widths
    dm.xy_cent_act_box_inM = dm.xy_cent_act_box*dm.dx_dm  # now in meters
    if Nbox % 2 == 0:
        dm.xy_box_lowerLeft = dm.xy_cent_act_box + (dm.NdmPad-Nbox)/2  # indices of pixel in lower left of the postage stamp within the whole pupilPad array
    else:
        dm.xy_box_lowerLeft = dm.xy_cent_act_box + (dm.NdmPad)/2 - np.floor(Nbox/2)  # indices of pixel in lower left of the postage stamp within the whole pupilPad array
    
    # Starting coordinates (in actuator widths) for updated influence function. This is
    # interpixel centered, so do not translate!
    dm.x_box0 = np.arange(-(Nbox-1)/2, (Nbox+1)/2) * dm.dx_dm
    [dm.Xbox0, dm.Ybox0] = np.meshgrid(dm.x_box0, dm.x_box0)  # meters, interpixel-centered coordinates for the master influence function
    
    # Limit the actuators used to those within 1 actuator width of the pupil
    r_cent_act_box_inM = np.sqrt(dm.xy_cent_act_box_inM[0, :]**2 + dm.xy_cent_act_box_inM[1, :]**2)
    # Compute and store all the influence functions:
    dm.inf_datacube = np.zeros((Nbox, Nbox, dm.NactTotal))  # initialize array of influence function "postage stamps"
    dm.act_ele = np.array([])  # Indices of nonzero-ed actuators
    for iact in range(dm.NactTotal):
       dm.act_ele = np.append(dm.act_ele, iact)  # Add actuator index to the keeper list
       dm.inf_datacube[:, :, iact] = dm.inf0
    
    print('done.')
    
    
def gen_fpm(mp):  #RENAME

    # Number of points across the FPM in the full model
    mp.F3.full.Nxi = mp.dm9.NdmPad
    mp.F3.full.Neta = mp.F3.full.Nxi
    # Number of points across the FPM in the compact model
    mp.F3.compact.Nxi = mp.dm9.compact.NdmPad
    mp.F3.compact.Neta = mp.F3.compact.Nxi

    # Starting DM8 and DM9 surfaces
    if mp.flagPlot:
        DM8surf = gen_fpm_surf_from_cube(mp.dm8, 'full')
        plt.figure(); plt.imshow(DM8surf); plt.colorbar(); plt.title('FPM Metal Thickness'); plt.pause(1)

        DM9surf = gen_fpm_surf_from_cube(mp.dm9, 'full')
        plt.figure(); plt.imshow(DM9surf); plt.colorbar(); plt.title('FPM Dielectric Thickness'); plt.pause(1)
        
    pass

def fpm_inf_cube(dm):
    pass