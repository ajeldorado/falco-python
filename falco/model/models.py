"""Compact and full diffractive optical models."""
import copy

import multiprocessing
import numpy as np
import scipy.ndimage as ndimage
import pdb
import matplotlib.pyplot as plt

from . import jacobians
import falco
from falco import check
from falco.util import pad_crop
import falco.proper as proper
import logging
log = logging.getLogger(__name__)


def full(mp, modvar, isNorm=True):
    """
    Truth model used to generate images in simulation.

    Truth model used to generate images in simulation. Can include
    aberrations/errors that are unknown to the estimator and controller. This
    function is the wrapper for full models of any coronagraph type.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    modvar : ModelVariables
        Structure containing temporary optical model variables
    isNorm : bool
        If False, return an unnormalized image. If True, return a
        normalized image with the currently stored norm value.

    Returns
    -------
    Eout : numpy ndarray
        2-D electric field in final focal plane
    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')

    if isNorm:
        if hasattr(modvar, 'sbpIndex') and hasattr(modvar, 'wpsbpIndex'):
            normFac = mp.Fend.full.I00[modvar.sbpIndex, modvar.wpsbpIndex]
        else:
            raise ValueError("modvar must have attributes 'sbpIndex' and "
                             "'wpsbpIndex'.")
    else:
        normFac = 0

    # Set the wavelength
    if hasattr(modvar, 'wvl'):  # For FALCO or standalone use of full model
        wvl = modvar.wvl
    elif hasattr(modvar, 'sbpIndex'):  # For use in FALCO
        wvl = mp.full.lambdasMat[modvar.sbpIndex, modvar.wpsbpIndex]
    else:
        raise ValueError('Need to specify value or indices for wavelength.')

    # %% Input E-fields

    # Include the tip/tilt in the input wavefront
    iStar = modvar.starIndex
    xiOffset = mp.star.xiOffsetVec[iStar]
    etaOffset = mp.star.etaOffsetVec[iStar]
    starWeight = mp.star.weights[iStar]
    TTphase = (-1)*(2*np.pi*(xiOffset*mp.P2.full.XsDL + etaOffset*mp.P2.full.YsDL))
    Ett = np.exp(1j*TTphase*mp.lambda0/wvl)
    Ein = np.sqrt(starWeight) * Ett * \
        np.squeeze(mp.P1.full.E[:, :, modvar.wpsbpIndex, modvar.sbpIndex])

    if modvar.whichSource.lower() == 'offaxis':  # Use for thput calculations
        TTphase = (-1.)*(2*np.pi*(modvar.x_offset*mp.P2.full.XsDL +
                                  modvar.y_offset*mp.P2.full.YsDL))
        Ett = np.exp(1j*TTphase*mp.lambda0/wvl)
        Ein *= Ett

    # Shift the source off-axis to compute the intensity normalization value.
    # This replaces the previous way of taking the FPM out in the optical model
    if normFac == 0:
        source_x_offset = mp.source_x_offset_norm  # source offset in lambda0/D
        source_y_offset = mp.source_y_offset_norm  # source offset in lambda0/D
        TTphase = (-1)*(2*np.pi*(source_x_offset*mp.P2.full.XsDL +
                                 source_y_offset*mp.P2.full.YsDL))
        Ett = np.exp(1j*TTphase*mp.lambda0/wvl)
        Ein *= Ett

    # Apply a Zernike (in amplitude) at input pupil if specified
    # if not (hasattr(modvar, 'zernIndex')):
    #     modvar.zernIndex = 1
    if not modvar.zernIndex == 1:
        indsZnoll = modvar.zernIndex  # Just send in 1 Zernike mode
        zernMat = np.squeeze(falco.zern.gen_norm_zern_maps(mp.P1.full.Nbeam,
                                                           mp.centering,
                                                           indsZnoll))
        zernMat = pad_crop(zernMat, mp.P1.full.Narr)
        Ein = Ein*zernMat*(2*np.pi/wvl)*mp.jac.Zcoef[mp.jac.zerns ==
                                                     modvar.zernIndex]

    # %% Pre-compute the FPM first for HLC
    if mp.Nsbp > 1 and mp.Nwpsbp > 1:
        # Weird indexing is because interior wavelengths at
        # edges of sub-bands are the same, and the fpmCube
        # contains only the minimal set of masks.
        ilam = (modvar.sbpIndex-2)*mp.Nwpsbp + modvar.wpsbpIndex + \
            (mp.Nsbp-modvar.sbpIndex+1)
    elif mp.Nsbp == 1 and mp.Nwpsbp > 1:
        ilam = modvar.wpsbpIndex
    elif mp.Nwpsbp == 1:
        ilam = modvar.sbpIndex

    if mp.coro.upper() == 'HLC':
        if mp.layout.lower() == 'fourier' or mp.layout.lower() == 'proper':
            if hasattr(mp.full, 'fpmCube'):
                mp.F3.full.mask = mp.full.fpmCube[:, :, ilam]
            else:
                mp.F3.full.mask = falco.hlc.gen_fpm_from_LUT(
                    mp, modvar.sbpIndex, modvar.wpsbpIndex, 'full')
        elif mp.layout.lower() == 'fpm_scale':
            mp.F3.full.mask = mp.full.fpmCube[:, :, ilam]

    # %% Select which optical layout's full model to use.
    if mp.layout.lower() == 'fourier':
        Eout = full_Fourier(mp, wvl, Ein, normFac)

    elif mp.layout.lower() == 'fpm_scale':  # FPM scales with wavelength
        Eout = full_Fourier(mp, wvl, Ein, normFac, flagScaleFPM=True)

    elif mp.layout.lower() == 'proper':

        optval = copy.copy(vars(mp.full))

        if any(mp.dm_ind == 1):
            optval['use_dm1'] = True
            optval['dm1'] = mp.dm1.V*mp.dm1.VtoH + mp.full.dm1FlatMap

        if any(mp.dm_ind == 2):
            optval['use_dm2'] = True
            optval['dm2'] = mp.dm2.V*mp.dm2.VtoH + mp.full.dm2FlatMap

        if normFac == 0:
            optval['xoffset'] = -mp.source_x_offset_norm
            optval['yoffset'] = -mp.source_y_offset_norm
            if mp.coro.upper() in ('VORTEX', 'VC', 'AVC'):
                optval['use_fpm'] = False
                optval['xoffset'] = 0
                optval['yoffset'] = 0
                pass

        # wavelength needs to be in microns instead of meters for PROPER
        [Eout, sampling_m] = proper.prop_run(mp.full.prescription, wvl*1e6,
                                             mp.P1.full.Narr, QUIET=True,
                                             PASSVALUE=optval)
        if not normFac == 0:
            Eout = Eout/np.sqrt(normFac)

        del optval

    elif mp.layout.lower() in ('wfirst_phaseb_proper', 'roman_phasec_proper'):
        optval = copy.copy(vars(mp.full))
        optval['use_dm1'] = True
        optval['use_dm2'] = True
        optval['dm1_m'] = mp.dm1.V*mp.dm1.VtoH + mp.full.dm1.flatmap
        optval['dm2_m'] = mp.dm2.V*mp.dm2.VtoH + mp.full.dm2.flatmap
        if normFac == 0:
            optval['source_x_offset'] = -mp.source_x_offset_norm
            optval['source_y_offset'] = -mp.source_y_offset_norm

        if mp.layout.lower() == 'wfirst_phaseb_proper':
            fn_proper = 'wfirst_phaseb'
        elif mp.layout.lower() == 'roman_phasec_proper':
            fn_proper = 'roman_phasec'

        [Eout, sampling] = proper.prop_run(
            fn_proper, wvl*1e6,
            int(2**falco.util.nextpow2(mp.Fend.Nxi)),
            QUIET=True, PASSVALUE=optval)
        Eout = pad_crop(Eout, (mp.Fend.Nxi, mp.Fend.Nxi))

        if not normFac == 0:
            Eout = Eout/np.sqrt(normFac)

    return Eout


def full_Fourier(mp, wvl, Ein, normFac, flagScaleFPM=False):
    """
    Truth model with a simple layout used to generate images in simulation.

    Truth model used to generate images in simulation. Can include
    aberrations/errors that are unknown to the estimator and controller.
    This function uses the simplest model (FTs except for angular spectrum
    between DMs) for several coronagraph types.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    wvl : Wavelength
        Scalar value for the wavelength of the light in meters
    Ein : Electric field input
        2-D electric field in the input pupil
    normFac : Normalization factor
        Scalar value of the PSF peak normalization factor to apply to the whole
        image.
    flagScaleFPM : bool, optional
        Whether to scale the diameter of the FPM inversely with wavelength.

    Returns
    -------
    Eout : numpy ndarray
        2-D electric field in final focal plane

    """
    check.is_bool(flagScaleFPM, 'flagScaleFPM')

    mirrorFac = 2  # Phase change is twice the DM surface height in reflection
    NdmPad = int(mp.full.NdmPad)

    if mp.flagRotation:
        NrelayFactor = 1
    else:
        NrelayFactor = 0  # zero out the number of relays

    if flagScaleFPM:
        fpmScaleFac = wvl/mp.lambda0
    else:
        fpmScaleFac = 1.0

    """ Masks and DM surfaces """
    if any(mp.dm_ind == 1):
        DM1surf = falco.dm.gen_surf_from_act(mp.dm1, mp.dm1.dx, NdmPad)
    else:
        DM1surf = np.zeros((NdmPad, NdmPad))

    if any(mp.dm_ind == 2):
        DM2surf = falco.dm.gen_surf_from_act(mp.dm2, mp.dm2.dx, NdmPad)
    else:
        DM2surf = np.zeros((NdmPad, NdmPad))

    pupil = pad_crop(mp.P1.full.mask, NdmPad)
    Ein = pad_crop(Ein, NdmPad)

    if mp.flagDM1stop:
        DM1stop = pad_crop(mp.dm1.full.mask, NdmPad)
    else:
        DM1stop = np.ones((NdmPad, NdmPad))

    if mp.flagDM2stop:
        DM2stop = pad_crop(mp.dm2.full.mask, NdmPad)
    else:
        DM2stop = np.ones((NdmPad, NdmPad))

    if mp.flagDMwfe:
        pass
        # if(any(mp.dm_ind==1));  Edm1WFE = exp(2*np.pi*1j/wvl*pad_crop(mp.dm1.wfe,NdmPad,'extrapval',0)); else; Edm1WFE = ones(NdmPad); end
        # if(any(mp.dm_ind==2));  Edm2WFE = exp(2*np.pi*1j/wvl*pad_crop(mp.dm2.wfe,NdmPad,'extrapval',0)); else; Edm2WFE = ones(NdmPad); end
    else:
        Edm1WFE = np.ones((NdmPad, NdmPad))
        Edm2WFE = np.ones((NdmPad, NdmPad))

    """ Propagation: entrance pupil, 2 DMs, (optional) apodizer, FPM, LS,
    and final focal plane """

    # Define pupil P1 and Propagate to pupil P2
    EP1 = pupil*Ein  # E-field at pupil plane P1
    EP2 = falco.prop.relay(EP1, NrelayFactor*mp.Nrelay1to2, mp.centering)

    # Propagate from P2 to DM1, and apply DM1 surface and aperture stop
    if not abs(mp.d_P2_dm1) == 0:
        Edm1 = falco.prop.ptp(EP2, mp.P2.full.dx*NdmPad, wvl, mp.d_P2_dm1)
    else:
        Edm1 = EP2   # E-field arriving at DM1
    Edm1b = Edm1*Edm1WFE*DM1stop*np.exp(mirrorFac*2*np.pi*1j*DM1surf/wvl)

    # Propagate from DM1 to DM2, and apply DM2 surface and aperture stop
    Edm2 = falco.prop.ptp(Edm1b, mp.P2.full.dx*NdmPad, wvl, mp.d_dm1_dm2)
    Edm2 *= Edm2WFE*DM2stop*np.exp(mirrorFac*2*np.pi*1j*DM2surf/wvl)

    # Back-propagate to pupil P2
    if(mp.d_P2_dm1 + mp.d_dm1_dm2 == 0):
        EP2eff = Edm2  # Do nothing if zero distance
    else:
        EP2eff = falco.prop.ptp(Edm2, mp.P2.full.dx*NdmPad, wvl,
                                -1*(mp.d_dm1_dm2 + mp.d_P2_dm1))

    # Re-image to pupil P3
    EP3 = falco.prop.relay(EP2eff, NrelayFactor*mp.Nrelay2to3, mp.centering)

    # Apply the apodizer mask (if there is one)
    if(mp.flagApod):
        EP3 = mp.P3.full.mask*pad_crop(EP3, mp.P3.full.Narr)

    # Propagations Specific to the Coronagraph Type
    if mp.coro.upper() in ('LC', 'APLC'):
        # MFT from apodizer plane to FPM (i.e., P3 to F3)
        EF3inc = falco.prop.mft_p2f(EP3, mp.fl, wvl, mp.P2.full.dx,
                                    fpmScaleFac*mp.F3.full.dxi, mp.F3.full.Nxi,
                                    fpmScaleFac*mp.F3.full.deta,
                                    mp.F3.full.Neta, mp.centering)
        # Apply (1-FPM) for Babinet's principle later
        EF3 = (1.-mp.F3.full.mask)*EF3inc
        #  Use Babinet's principle at the Lyot plane.
        EP4noFPM = falco.prop.relay(EP3, NrelayFactor*mp.Nrelay3to4, mp.centering)
        #  MFT from FPM to Lyot Plane (i.e., F3 to P4)
        EP4subtrahend = falco.prop.mft_f2p(EF3, mp.fl, wvl,
                                           fpmScaleFac*mp.F3.full.dxi,
                                           mp.F3.full.deta,
                                           fpmScaleFac*mp.P4.full.dx,
                                           mp.P4.full.Narr, mp.centering)
        # Babinet's principle at P4
        EP4 = pad_crop(EP4noFPM, mp.P4.full.Narr) - EP4subtrahend

    elif mp.coro.upper() == 'HLC':
        # Complex transmission of the points outside the FPM (just fused silica
        # with optional dielectric and no metal).

        if hasattr(mp.full, 'fpmCube'):
            transOuterFPM = mp.F3.full.mask[0, 0]
        else:
            t_Ti_base = 0
            t_Ni_vec = [0]
            t_PMGI_vec = [1e-9 * mp.t_diel_bias_nm]  # [meters]
            pol = 2
            transOuterFPM, rCoef = falco.thinfilm.calc_complex_trans_matrix(
                mp.F3.substrate, mp.F3.metal, mp.F3.dielectric, wvl, mp.aoi,
                t_Ti_base, t_Ni_vec, t_PMGI_vec, wvl*mp.F3.d0fac, pol)

        # MFT from apodizer plane to FPM (i.e., P3 to F3)
        EF3inc = falco.prop.mft_p2f(EP3, mp.fl, wvl, mp.P2.full.dx,
                            fpmScaleFac*mp.F3.full.dxi, mp.F3.full.Nxi, 
                            fpmScaleFac*mp.F3.full.deta,
                            mp.F3.full.Neta, mp.centering)
        # Apply (transOuterFPM-FPM) for Babinet's principle later
        EF3 = (transOuterFPM - mp.F3.full.mask) * EF3inc
        # Use Babinet's principle at the Lyot plane.
        # Propagate forward another pupil plane
        EP4noFPM = falco.prop.relay(EP3, NrelayFactor*mp.Nrelay3to4, mp.centering)
        # Apply the change from the FPM's outer complex transmission.
        EP4noFPM = transOuterFPM * pad_crop(EP4noFPM, mp.P4.full.Narr)
        # MFT from FPM to Lyot Plane (i.e., F3 to P4)
        # Subtrahend term for Babinet's principle
        EP4subtra = falco.prop.mft_f2p(EF3, mp.fl, wvl, fpmScaleFac*mp.F3.full.dxi,
                fpmScaleFac*mp.F3.full.deta, mp.P4.full.dx, mp.P4.full.Narr, mp.centering)
        # Babinet's principle at P4
        EP4 = EP4noFPM - EP4subtra

    elif(mp.coro.upper() == 'SPLC' or mp.coro.upper() == 'FLC'):
        # MFT from apodizer plane to FPM (i.e., P3 to F3)
        EF3inc = falco.prop.mft_p2f(EP3, mp.fl, wvl, mp.P2.full.dx,
                                    fpmScaleFac*mp.F3.full.dxi, mp.F3.full.Nxi,
                                    fpmScaleFac*mp.F3.full.deta,
                                    mp.F3.full.Neta, mp.centering)
        EF3 = mp.F3.full.mask * EF3inc

        # MFT from FPM to Lyot Plane (i.e., F3 to P4)
        EP4 = falco.prop.mft_f2p(EF3, mp.fl, wvl, fpmScaleFac*mp.F3.full.dxi,
                                 fpmScaleFac*mp.F3.full.deta, mp.P4.full.dx,
                                 mp.P4.full.Narr, mp.centering)
        EP4 = falco.prop.relay(EP4, NrelayFactor*mp.Nrelay3to4-1, mp.centering)

    elif mp.coro.upper() in ('VORTEX', 'VC'):
        # Get FPM charge
        if isinstance(mp.F3.VortexCharge, np.ndarray):
            # Passing an array for mp.F3.VortexCharge with
            # corresponding wavelengths mp.F3.VortexCharge_lambdas
            # represents a chromatic vortex FPM
            if mp.F3.VortexCharge.size == 1:
                charge = mp.F3.VortexCharge
            else:
                np.interp(wvl, mp.F3.VortexCharge_lambdas, mp.F3.VortexCharge,
                          'linear', 'extrap')

        elif isinstance(mp.F3.VortexCharge, (int, float)):
            # single value indicates fully achromatic mask
            charge = mp.F3.VortexCharge
        else:
            raise TypeError("mp.F3.VortexCharge must be an int, float, or\
                            numpy ndarray.")
            pass
        EP4 = falco.prop.mft_p2v2p(EP3, charge, mp.P1.full.Nbeam/2., 0.3, 5)
        EP4 = pad_crop(EP4, mp.P4.full.Narr)

        # Undo the rotation inherent to falco.prop.mft_p2v2p.m
        if not mp.flagRotation:
            EP4 = falco.prop.relay(EP4, -1, mp.centering)

    else:
        log.warning('The chosen coronagraph type is not included yet.')
        raise ValueError("Value of mp.coro not recognized.")
        pass

    # Remove FPM completely if normalization value being found for vortex
    if normFac == 0:
        if mp.coro.upper() in ('VORTEX', 'VC', 'AVC'):
            EP4 = falco.prop.relay(EP3, NrelayFactor*mp.Nrelay3to4, mp.centering)
            EP4 = pad_crop(EP4, mp.P4.full.Narr)
            pass
        pass

    """ Back to common propagation any coronagraph type """
    # Apply the (cropped-down) Lyot stop
    EP4 *= mp.P4.full.croppedMask

    # MFT from Lyot Stop to final focal plane (i.e., P4 to Fend)
    EP4 = falco.prop.relay(EP4, NrelayFactor*mp.NrelayFend, mp.centering)
    EFend = falco.prop.mft_p2f(EP4, mp.fl, wvl, mp.P4.full.dx, mp.Fend.dxi,
                               mp.Fend.Nxi, mp.Fend.deta, mp.Fend.Neta,
                               mp.centering)

    # Don't apply FPM if normalization value is being found
    if normFac == 0:
        Eout = EFend  # Don't normalize if normalization value is being found
    else:
        Eout = EFend/np.sqrt(normFac)  # Apply normalization

    return Eout

def compact_reverse_gradient(command_vec, mp, EestAll, EFendPrev, log10reg):
    """
    Simplified (aka compact) model used by estimator and controller.

    Simplified (aka compact) model used by estimator and controller. Does not
    include unknown aberrations of the full, "truth" model. This function is
    the wrapper for compact models of any coronagraph type.

    Parameters
    ----------
    command_vec : array_like
        Vector of vectorized and concatenated DM commands. They are vectorized
        for use in an optimizer.
    mp : ModelParameters
        Structure containing optical model parameters
    log10reg : float
        The log10() of the regularization to use when computing the control command.

    Returns
    -------
    total_cost : 
        Summed total intensity in the dark hole acting as the cost function.
    gradient : array_like
        The output command vector, consisting of the vectorized and
        concatenated commands to both DMs.

    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')

    mirrorFac = 2.  # Phase change is twice the DM surface height.
    NdmPad = int(mp.compact.NdmPad)

    # isNorm = True  # Leave unnormalized
    isEvalMode = False
    useFPM = True

    # Complex trans of points outside FPM
    if mp.coro.upper() == 'HLC':
        transOuterFPM = mp.F3.compact.mask[0, 0]  
    else:
        transOuterFPM = 1.

    if mp.flagRotation:
        NrelayFactor = 1
    else:
        NrelayFactor = 0  # zero out the number of relays

    total_cost = 0
    dmSurf1_bar_tot = 0
    dmSurf2_bar_tot = 0
    
    smooth = 0  # TODO hard-coded at 0 for now
    # flag_debug = False  #  hard-coded for now

    # Store initial cumulative DM commands
    mp.dm1.V0 = mp.dm1.V.copy()
    mp.dm2.V0 = mp.dm2.V.copy()

    # # TODO: Change to be more generic, e.g. for one DM at a time.
    # mp.dm1.V = command_vec[0:mp.dm1.NactTotal].reshape([mp.dm1.Nact, mp.dm1.Nact])
    # mp.dm2.V = command_vec[mp.dm2.NactTotal::].reshape([mp.dm2.Nact, mp.dm2.Nact])

    # # TODO: Change to be more generic, e.g. for one DM at a time.
    # mp.dm1.V = command_vec[0:mp.dm1.NactTotal].reshape([mp.dm1.Nact, mp.dm1.Nact])
    # mp.dm2.V = command_vec[mp.dm2.NactTotal::].reshape([mp.dm2.Nact, mp.dm2.Nact])

    for iMode in range(mp.jac.Nmode):

        modvar = falco.config.ModelVariables()
        modvar.whichSource = 'star'
        modvar.sbpIndex = mp.jac.sbp_inds[iMode]
        modvar.zernIndex = mp.jac.zern_inds[iMode]
        modvar.starIndex = mp.jac.star_inds[iMode]

        wvl = mp.sbp_centers[modvar.sbpIndex]
        normFac = mp.Fend.compact.I00[modvar.sbpIndex]
        # normFacFull = np.mean(mp.Fend.full.I00[modvar.sbpIndex, :])
        EestVec = EestAll[:, iMode]
        Eest2D = np.zeros_like(mp.Fend.corr.maskBool, dtype=complex)
        Eest2D[mp.Fend.corr.maskBool] = EestVec #* np.sqrt(normFacFull)  # Remove normalization
        normFacAD = np.sum(np.abs(EestVec)**2)

        mp.dm1.V += command_vec[0:mp.dm1.NactTotal].reshape([mp.dm1.Nact, mp.dm1.Nact])
        mp.dm2.V += command_vec[mp.dm2.NactTotal::].reshape([mp.dm2.Nact, mp.dm2.Nact])
                    
        # With delta DM commands applied
        EFendB, Edm1inc, Edm2inc, DM1surfB, DM2surfB = compact(
           mp, modvar, isNorm=True, isEvalMode=isEvalMode, useFPM=useFPM, 
           forRevGradModel=True)        
        
        # Reset
        mp.dm1.V = mp.dm1.V0.copy()
        mp.dm2.V = mp.dm2.V0.copy()
        
        EFendA = EFendPrev[iMode]
        dEend = EFendB - EFendA
        DM1surf = DM1surfB
        DM2surf = DM2surfB

        # DH = EFend[mp.Fend.corr.maskBool]
        EdhNew = Eest2D + dEend
        DH = EdhNew[mp.Fend.corr.maskBool]
        int_in_dh = np.sum(np.absolute(DH)**2) #* mp.ctrl.ad.cost_func_scale_fac
        # total_cost += normFac*int_in_dh/mp.Nsbp
        total_cost += mp.jac.weights[iMode] * int_in_dh / normFacAD

        g1 = np.exp(1j*mirrorFac*2*np.pi*DM1surf/wvl)
        g2 = np.exp(1j*mirrorFac*2*np.pi*DM2surf/wvl)
                
        # Gradient
        Fend_masked = 2/normFacAD*EdhNew*np.real(mp.Fend.corr.maskBool.astype(float)*mp.ctrl.ad.cost_func_scale_fac)
        # Fend_masked = 2/normFacAD*EFend*np.real(mp.Fend.corr.maskBool.astype(float)*mp.ctrl.ad.cost_func_scale_fac)
        
#        plt.figure(); plt.imshow(np.abs(Fend_masked)); plt.colorbar(); plt.magma(); plt.title('abs(Fend)'); plt.savefig('/Users/ajriggs/Downloads/fig_abs_Fend.png', format='png')
#        plt.figure(); plt.imshow(np.angle(Fend_masked)); plt.colorbar(); plt.hsv(); plt.title('angle(Fend)'); plt.savefig('/Users/ajriggs/Downloads/fig_angle_Fend.png', format='png')
        
        EP4_grad = falco.prop.mft_f2p(Fend_masked, -mp.fl, wvl, mp.Fend.dxi, mp.Fend.deta, mp.P4.compact.dx, mp.P4.compact.Narr, mp.centering)                                 
        EP4_grad = falco.prop.relay(EP4_grad, NrelayFactor*mp.NrelayFend, mp.centering)
        EP4LS_grad = EP4_grad * pad_crop(mp.P4.compact.croppedMask, mp.P4.compact.Narr)
      
#        plt.figure(); plt.imshow(np.abs(EP4_grad)); plt.colorbar(); plt.magma(); plt.title('abs(EP4)'); plt.savefig('/Users/ajriggs/Downloads/fig_abs_EP4.png', format='png')
#        plt.figure(); plt.imshow(np.angle(EP4_grad)); plt.colorbar(); plt.hsv(); plt.title('angle(EP4)'); plt.savefig('/Users/ajriggs/Downloads/fig_angle_EP4.png', format='png')

#        plt.figure(); plt.imshow(np.abs(EP4LS_grad)); plt.colorbar(); plt.magma(); plt.title('abs(EP4LS)'); plt.savefig('/Users/ajriggs/Downloads/fig_abs_EP4LS.png', format='png')
#        plt.figure(); plt.imshow(np.angle(EP4LS_grad)); plt.colorbar(); plt.hsv(); plt.title('angle(EP4LS)'); plt.savefig('/Users/ajriggs/Downloads/fig_angle_EP4LS.png', format='png')

#        plt.figure(); plt.imshow(np.real(EP4LS_grad)); plt.colorbar(); plt.magma(); plt.title('real(EP4LS)'); plt.savefig('/Users/ajriggs/Downloads/fig_real_EP4LS.png', format='png')
#        plt.figure(); plt.imshow(np.imag(EP4LS_grad)); plt.colorbar(); plt.magma(); plt.title('imag(EP4LS)'); plt.savefig('/Users/ajriggs/Downloads/fig_imag_EP4LS.png', format='png')

        if mp.coro.upper() in ('VORTEX', 'VC', 'AVC'):

            # Undo the rotation inherent to falco.prop.mft_p2v2p.m
            if not mp.flagRotation:
                EP4LS_grad = falco.prop.relay(EP4LS_grad, -1, mp.centering)

            # Get FPM charge
            if isinstance(mp.F3.VortexCharge, np.ndarray):
                # Passing an array for mp.F3.VortexCharge with
                # corresponding wavelengths mp.F3.VortexCharge_lambdas
                # represents a chromatic vortex FPM
                if mp.F3.VortexCharge.size == 1:
                    charge = mp.F3.VortexCharge
                else:
                    np.interp(wvl, mp.F3.VortexCharge_lambdas,
                              mp.F3.VortexCharge, 'linear', 'extrap')

            elif isinstance(mp.F3.VortexCharge, (int, float)):
                # single value indicates fully achromatic mask
                charge = mp.F3.VortexCharge
            else:
                raise TypeError("mp.F3.VortexCharge must be int, float or numpy ndarray.")
            
            EP4LS_grad = pad_crop(EP4LS_grad, mp.P1.compact.Narr)
            EP3_grad = falco.prop.mft_p2v2p(EP4LS_grad, charge, mp.P1.compact.Nbeam/2., 0.3, 5, reverseGradient=True)

        elif mp.coro.upper() == 'FLC' or mp.coro.upper() == 'SPLC':

            EP4LS_grad = falco.prop.relay(EP4LS_grad, NrelayFactor*mp.Nrelay3to4-1, mp.centering)
            EF3_grad = falco.prop.mft_p2f(EP4LS_grad, -mp.fl, wvl, mp.P2.compact.dx, mp.F3.compact.dxi, mp.F3.compact.Nxi, mp.F3.compact.deta, mp.F3.compact.Neta, mp.centering)  # E-field incident upon the FPM
            EF3_grad = np.conj(mp.F3.compact.mask) * EF3_grad
            EP3_grad = falco.prop.mft_f2p(EF3_grad, -mp.fl, wvl, mp.F3.compact.dxi, mp.F3.compact.deta, mp.P4.compact.dx, NdmPad, mp.centering) # Subtrahend term for Babinet's principle 
            # EP3subtr_grad_plot = falco.propcustom.propcustom_mft_FtoP(EF3_grad, -mp.fl, wvl, mp.F3.compact.dxi, mp.F3.compact.deta, mp.P4.compact.dx, NdmPad*10, mp.centering) # Subtrahend term for Babinet's principle 
            
    #        plt.figure(); plt.imshow(np.abs(auxEP4subtr_grad)); plt.colorbar(); plt.magma(); plt.title('abs(EF3bab)'); plt.savefig('/Users/ajriggs/Downloads/fig_abs_EF3bab.png', format='png')
    #        plt.figure(); plt.imshow(np.angle(auxEP4subtr_grad)); plt.colorbar(); plt.hsv(); plt.title('angle(EF3bab)'); plt.savefig('/Users/ajriggs/Downloads/fig_angle_EF3bab.png', format='png')
    
        elif mp.coro.upper() in ('LC', 'APLC', 'HLC'):
            # MFT Method
            EP4noFPM_grad = EP4LS_grad
            EP3noFPM_grad = pad_crop(falco.prop.relay(EP4noFPM_grad, NrelayFactor*mp.Nrelay3to4, mp.centering), NdmPad)
            EP4subtr_grad = -EP4LS_grad
            EP4subtr_grad = falco.prop.relay(EP4subtr_grad, NrelayFactor*mp.Nrelay3to4-1, mp.centering)
            # EP4subtr_grad = falco.propcustom.propcustom_relay(EP4subtr_grad, mp.Nrelay3to4 - 1, mp.centering)
            auxEP4subtr_grad = falco.prop.mft_p2f(EP4subtr_grad, -mp.fl, wvl, mp.P2.compact.dx, mp.F3.compact.dxi, mp.F3.compact.Nxi, mp.F3.compact.deta, mp.F3.compact.Neta, mp.centering)  # E-field incident upon the FPM
    #        auxEP4subtr_grad_plot = falco.propcustom.propcustom_mft_PtoF(EP4subtr_grad, -mp.fl,wvl,mp.P2.compact.dx,mp.F3.compact.dxi,mp.F3.compact.Nxi*10,mp.F3.compact.deta,mp.F3.compact.Neta*10,mp.centering) #--E-field incident upon the FPM
            EF3_grad = np.conj(transOuterFPM - mp.F3.compact.mask) * auxEP4subtr_grad  # TODO check if np.conj() usage is correct here
            EP3subtr_grad = falco.prop.mft_f2p(EF3_grad, -mp.fl, wvl, mp.F3.compact.dxi, mp.F3.compact.deta, mp.P4.compact.dx, NdmPad, mp.centering) # Subtrahend term for Babinet's principle 
            #EP3subtr_grad_plot = falco.propcustom.propcustom_mft_FtoP(EF3_grad, -mp.fl, wvl, mp.F3.compact.dxi, mp.F3.compact.deta, mp.P4.compact.dx, NdmPad*10, mp.centering) # Subtrahend term for Babinet's principle 
            
    #        plt.figure(); plt.imshow(np.abs(auxEP4subtr_grad)); plt.colorbar(); plt.magma(); plt.title('abs(EF3bab)'); plt.savefig('/Users/ajriggs/Downloads/fig_abs_EF3bab.png', format='png')
    #        plt.figure(); plt.imshow(np.angle(auxEP4subtr_grad)); plt.colorbar(); plt.hsv(); plt.title('angle(EF3bab)'); plt.savefig('/Users/ajriggs/Downloads/fig_angle_EF3bab.png', format='png')
            
            EP3_grad = EP3noFPM_grad + EP3subtr_grad

        else:
            raise ValueError('%s value of mp.coro not supported yet' % mp.coro)
        
#        plt.figure(); plt.imshow(np.abs(EP3subtr_grad)); plt.colorbar(); plt.magma(); plt.title('abs(EP3bab)'); plt.savefig('/Users/ajriggs/Downloads/fig_abs_EP3bab.png', format='png')
#        plt.figure(); plt.imshow(np.angle(EP3subtr_grad)); plt.colorbar(); plt.hsv(); plt.title('angle(EP3bab)'); plt.savefig('/Users/ajriggs/Downloads/fig_angle_EP3bab.png', format='png')
#                
#        plt.figure(); plt.imshow(np.abs(EP3_grad)); plt.colorbar(); plt.magma(); plt.title('abs(EP3)'); plt.savefig('/Users/ajriggs/Downloads/fig_abs_EP3.png', format='png')
#        plt.figure(); plt.imshow(np.angle(EP3_grad)); plt.colorbar(); plt.hsv(); plt.title('angle(EP3)'); plt.savefig('/Users/ajriggs/Downloads/fig_angle_EP3.png', format='png')
        
        if mp.flagApod:
            EP3_grad = pad_crop(mp.P3.compact.mask, EP3_grad.shape) * EP3_grad
        
        EP2eff_grad = falco.prop.relay(EP3_grad, NrelayFactor*mp.Nrelay2to3, mp.centering)
        EP2eff_grad = pad_crop(EP2eff_grad, NdmPad)
     
#        plt.figure(); plt.imshow(np.abs(EP2eff_grad)); plt.colorbar(); plt.magma(); plt.title('abs(EP2eff)'); plt.savefig('/Users/ajriggs/Downloads/fig_abs_EP2eff.png', format='png')
#        plt.figure(); plt.imshow(np.angle(EP2eff_grad)); plt.colorbar(); plt.hsv(); plt.title('angle(EP2eff)'); plt.savefig('/Users/ajriggs/Downloads/fig_angle_EP2eff.png', format='png')  
#        plt.figure(); plt.imshow(np.real(EP2eff_grad)); plt.colorbar(); plt.magma(); plt.title('real(EP2eff)'); plt.savefig('/Users/ajriggs/Downloads/fig_real_EP2eff.png', format='png')
#        plt.figure(); plt.imshow(np.imag(EP2eff_grad)); plt.colorbar(); plt.hsv(); plt.title('imag(EP2eff)'); plt.savefig('/Users/ajriggs/Downloads/fig_imag_EP2eff.png', format='png')
#        
        
        if(mp.d_P2_dm1 + mp.d_dm1_dm2 == 0):
            Edm2_grad = EP2eff_grad
        else:
            Edm2_grad = falco.prop.ptp(EP2eff_grad, mp.P2.compact.dx*NdmPad, wvl, (mp.d_dm1_dm2 + mp.d_P2_dm1))
    
#        plt.figure(); plt.imshow(np.abs(Edm2_grad)); plt.colorbar(); plt.magma(); plt.title('abs(Edm2)'); plt.savefig('/Users/ajriggs/Downloads/fig_abs_Edm2.png', format='png')
#        plt.figure(); plt.imshow(np.angle(Edm2_grad)); plt.colorbar(); plt.hsv(); plt.title('angle(Edm2)'); plt.savefig('/Users/ajriggs/Downloads/fig_angle_Edm2.png', format='png')
#        plt.figure(); plt.imshow(np.real(Edm2_grad)); plt.colorbar(); plt.magma(); plt.title('real(Edm2)'); plt.savefig('/Users/ajriggs/Downloads/fig_real_Edm2.png', format='png')
#        plt.figure(); plt.imshow(np.imag(Edm2_grad)); plt.colorbar(); plt.hsv(); plt.title('imag(Edm2)'); plt.savefig('/Users/ajriggs/Downloads/fig_imag_Edm2.png', format='png')
#        
        #--To DM2
        g2_bar = np.conj(Edm2inc) * Edm2_grad
        phase_DM2_bar = pad_crop(np.imag(g2_bar*np.conj(g2)), mp.P1.compact.Nbeam)
        
        if smooth != 0:
            phase_DM2_bar = ndimage.gaussian_filter(phase_DM2_bar, sigma=(smooth, smooth), order=0)

        # plt.figure(); plt.imshow(phase_DM2_bar); plt.colorbar(); plt.magma(); plt.title('DM2 Phase'); plt.savefig('/Users/ajriggs/Downloads/fig_DM2_phase.png', format='png')
        # Vout2 = -falco.dms.falco_fit_dm_surf(mp.dm2, dmSurf2_bar)
         
        #--To DM1
        Edm2_bar = np.conj(g2) * Edm2_grad
        Edm1_grad = falco.prop.ptp(Edm2_bar, mp.P2.compact.dx*NdmPad, wvl, -mp.d_dm1_dm2)        
#        Edm1_grad = falco.propcustom.propcustom_PTP_grad(Edm2_bar, mp.P2.compact.dx*NdmPad, wvl, mp.d_P2_dm1)
        # Edm1_grad = EP2eff_grad # ALTERNATE WAY
        
#        plt.figure(); plt.imshow(np.abs(Edm1_grad)); plt.colorbar(); plt.magma(); plt.title('abs(Edm1)'); plt.savefig('/Users/ajriggs/Downloads/fig_abs_Edm1.png', format='png')
#        plt.figure(); plt.imshow(np.angle(Edm1_grad)); plt.colorbar(); plt.hsv(); plt.title('angle(Edm1)'); plt.savefig('/Users/ajriggs/Downloads/fig_angle_Edm1.png', format='png')
#        plt.figure(); plt.imshow(np.real(Edm1_grad)); plt.colorbar(); plt.magma(); plt.title('real(Edm1)'); plt.savefig('/Users/ajriggs/Downloads/fig_real_Edm1.png', format='png')
#        plt.figure(); plt.imshow(np.imag(Edm1_grad)); plt.colorbar(); plt.hsv(); plt.title('imag(Edm1)'); plt.savefig('/Users/ajriggs/Downloads/fig_imag_Edm1.png', format='png')

        g1_bar = np.conj(Edm1inc) * Edm1_grad
        phase_DM1_bar = pad_crop(np.imag(g1_bar*np.conj(g1)), mp.P1.compact.Nbeam)
#        phase_DM1_bar = pad_crop(np.imag(g1_bar*np.conj(g1)), NdmPad)
        
#        plt.figure(); plt.imshow(phase_DM1_bar); plt.colorbar(); plt.magma(); plt.title('DM1 Phase'); plt.savefig('/Users/ajriggs/Downloads/fig_DM1_phase.png', format='png')
        
        if smooth != 0:
            phase_DM1_bar = ndimage.gaussian_filter(phase_DM1_bar, sigma=(smooth, smooth), order=0)

#        phase_DM1_bar_tot = phase_DM1_bar_tot+phase_DM1_bar
#        phase_DM1_bar_tot = phase_DM1_bar_tot/mp.Nsbp
#        phase_DM2_bar_tot = phase_DM2_bar_tot/mp.Nsbp
#        dmSurf2_bar = (4*np.pi/wvl*np.mean(mp.dm2.VtoH))*phase_DM2_bar_tot
#        dmSurf1_bar = (4*np.pi/wvl*np.mean(mp.dm1.VtoH))*phase_DM1_bar_tot
        dmSurf2_bar = (4*np.pi/wvl*np.mean(mp.dm2.VtoH))*phase_DM2_bar
        dmSurf1_bar = (4*np.pi/wvl*np.mean(mp.dm1.VtoH))*phase_DM1_bar

        dmSurf1_bar_tot += mp.jac.weights[iMode] * dmSurf1_bar
        dmSurf2_bar_tot += mp.jac.weights[iMode] * dmSurf2_bar     
        # dmSurf1_bar_tot = dmSurf1_bar_tot + dmSurf1_bar/mp.Nsbp
        # dmSurf2_bar_tot = dmSurf2_bar_tot + dmSurf2_bar/mp.Nsbp
        
    # Vout1 = -quick_fit_dm_surf(mp.dm1.compact, dmSurf1_bar_tot)
    # Vout2 = -quick_fit_dm_surf(mp.dm2.compact, dmSurf2_bar_tot)
    Vout1 = -falco.dm.fit_surf_to_act(mp.dm1.compact, dmSurf1_bar_tot) #+ 2*(10.0**log10reg)*mp.dm1.V
    Vout2 = -falco.dm.fit_surf_to_act(mp.dm2.compact ,dmSurf2_bar_tot) #+ 2*(10.0**log10reg)*mp.dm2.V

    gradient = np.concatenate((Vout1.reshape([mp.dm1.NactTotal]), Vout2.reshape([mp.dm2.NactTotal])), axis=None)
    
#     if flag_debug:
# #        return total_cost, gradient, np.abs(EP2eff_grad), np.abs(EP3subtr_grad), np.abs(auxEP4subtr_grad_plot) #dmSurf2_bar
#         return total_cost, gradient, np.abs(EP2eff_grad), dmSurf1_bar, dmSurf2_bar #g1, g1_bar, dmSurf1_bar_tot
#     else:
#         return total_cost, gradient

    return total_cost, gradient


def compact(mp, modvar, isNorm=True, isEvalMode=False, useFPM=True,
            forRevGradModel=False):
    """
    Simplified (aka compact) model used by estimator and controller.

    Simplified (aka compact) model used by estimator and controller. Does not
    include unknown aberrations of the full, "truth" model. This function is
    the wrapper for compact models of any coronagraph type.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    modvar : ModelVariables
        Structure containing temporary optical model variables
    isNorm : bool
        If False, return an unnormalized image. If True, return a
        normalized image with the currently stored norm value.
    isEvalMode : bool
        If set, uses a higher resolution in the focal plane for
        measuring performance metrics such as throughput.
    useFPM : bool
        Whether to include the FPM in the model
    forRevGradModel : bool
        Whether to compute the reverse gradient model for algorithmic
        differentiation instead of the regular forward model.

    Returns
    -------
    Eout : array_like
        2-D electric field in final focal plane

    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')

    # use a different res at final focal plane for eval
    flagEval = isEvalMode

    # Is image already normalized?
    if not isNorm:
        normFac = 0.

    elif isNorm and isEvalMode:
        normFac = mp.Fend.eval.I00[modvar.sbpIndex]

    else:
        normFac = mp.Fend.compact.I00[modvar.sbpIndex]

    # Set the wavelength
    if hasattr(modvar, 'wvl'):
        wvl = modvar.wvl
    else:
        wvl = mp.sbp_centers[modvar.sbpIndex]

    # %% Input E-fields

    # Include the star position and weight in the starting wavefront
    iStar = modvar.starIndex
    xiOffset = mp.compact.star.xiOffsetVec[iStar]
    etaOffset = mp.compact.star.etaOffsetVec[iStar]
    starWeight = mp.compact.star.weights[iStar]
    TTphase = (-1)*(2*np.pi*(xiOffset*mp.P2.compact.XsDL +
                             etaOffset*mp.P2.compact.YsDL))
    Ett = np.exp(1j*TTphase*mp.lambda0/wvl)
    Ein = np.sqrt(starWeight) * Ett * mp.P1.compact.E[:, :, modvar.sbpIndex]

    if modvar.whichSource.lower() == 'offaxis':  # Use for throughput calc
        TTphase = (-1)*(2*np.pi*(modvar.x_offset*mp.P2.compact.XsDL +
                                 modvar.y_offset*mp.P2.compact.YsDL))
        Ett = np.exp(1j*TTphase*mp.lambda0/wvl)
        Ein *= Ett

    # Shift the source off-axis to compute the intensity normalization value.
    # This replaces the previous way of taking the FPM out in optical model.
    if normFac == 0:
        TTphase = (-1)*(2*np.pi*(mp.source_x_offset_norm*mp.P2.compact.XsDL +
                                 mp.source_y_offset_norm*mp.P2.compact.YsDL))
        Ett = np.exp(1j*TTphase*mp.lambda0/wvl)
        Ein *= Ett
        # Ein = Ett*mp.P1.compact.E[:, :, modvar.sbpIndex]

    # Apply a Zernike (in amplitude) at input pupil if specified
    if not hasattr(modvar, 'zernIndex'):
        modvar.zernIndex = 1

    # Only used for Zernike sensitivity control, which requires the perfect
    # E-field of the differential Zernike term.
    if not (modvar.zernIndex == 1):
        indsZnoll = modvar.zernIndex  # Just send in 1 Zernike mode
        zernMat = np.squeeze(falco.zern.gen_norm_zern_maps(mp.P1.compact.Nbeam,
                                                           mp.centering,
                                                           indsZnoll))
        zernMat = pad_crop(zernMat, mp.P1.compact.Narr)
        Ein *= zernMat*(2*np.pi*1j/wvl)*mp.jac.Zcoef[mp.jac.zerns ==
                                                     modvar.zernIndex]

    # Define what the complex-valued FPM is if the coro is some type of HLC.
    if mp.coro.upper() in ('HLC',):
        if mp.layout.lower() == 'fourier':
            if hasattr(mp.compact, 'fpmCube'):
                mp.F3.compact.mask = mp.compact.fpmCube[:, :, modvar.sbpIndex]
            else:
                mp.F3.compact.mask = falco.hlc.gen_fpm_from_LUT(
                    mp, modvar.sbpIndex, -1, 'compact')
        elif mp.layout.lower() in ('roman_phasec_proper',
                                   'wfirst_phaseb_proper',
                                   'fpm_scale', 'proper'):
            mp.F3.compact.mask = mp.compact.fpmCube[:, :, modvar.sbpIndex]
        else:
            raise ValueError('Incompatible values of mp.layout and mp.coro.')

    # Select which optical layout's compact model to use and get E-field
    if mp.layout.lower() == 'fourier':
        Eout = compact_general(mp, wvl, Ein, normFac, flagEval, useFPM=useFPM,
                               forRevGradModel=forRevGradModel)

    elif mp.layout.lower() in ('roman_phasec_proper', 'wfirst_phaseb_proper',
                               'proper', 'fpm_scale'):
        if mp.coro.upper() == 'HLC':
            Eout = compact_general(mp, wvl, Ein, normFac, flagEval,
                                   flagScaleFPM=True, useFPM=useFPM,
                                   forRevGradModel=forRevGradModel)
        else:
            Eout = compact_general(mp, wvl, Ein, normFac, flagEval,
                                   useFPM=useFPM,
                                   forRevGradModel=forRevGradModel)
    return Eout


def compact_general(mp, wvl, Ein, normFac, flagEval, flagScaleFPM=False,
                    useFPM=True, forRevGradModel=False):
    """
    Compact model with a general-purpose optical layout.

    Used by estimator and controller.

    Simplified (aka compact) model used by estimator and controller. Does not
    include unknown aberrations of the full, "truth" model. This has a general
    optical layout that should work for most applications.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    wvl : float
        Wavelength of light [meters]
    Ein : numpy ndarray
        2-D input electric field
    normFac : float
        Intensity normalization factor
    flagEval : bool
        Whether to use a higher resolution in final image plane for evaluation.
    flagScaleFPM : bool
        Whether to scale the diameter of the FPM inversely with wavelength.
    useFPM : bool
        Whether to include the FPM in the model.
    forRevGradModel : bool
        Whether different outputs should be given for a controller that uses
        algorithmic differentiation.

    Returns
    -------
    Eout : numpy ndarray
        2-D electric field in final focal plane

    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    check.is_bool(flagEval, 'flagEval')
    check.is_bool(flagScaleFPM, 'flagScaleFPM')

    mirrorFac = 2.  # Phase change is twice the DM surface height.
    NdmPad = int(mp.compact.NdmPad)
    
    # Complex trans of points outside FPM
    if mp.coro.upper() == 'HLC':
        transOuterFPM = mp.F3.compact.mask[0, 0]  
    else:
        transOuterFPM = 1.

    if mp.flagRotation:
        NrelayFactor = 1
    else:
        NrelayFactor = 0  # zero out the number of relays

    if flagScaleFPM:
        fpmScaleFac = wvl/mp.lambda0
    else:
        fpmScaleFac = 1.0

    if(flagEval):  # Higher resolution at final focal plane for eval
        dxi = mp.Fend.eval.dxi
        Nxi = mp.Fend.eval.Nxi
        deta = mp.Fend.eval.deta
        Neta = mp.Fend.eval.Neta
    else:  # Otherwise use the detector resolution
        dxi = mp.Fend.dxi
        Nxi = mp.Fend.Nxi
        deta = mp.Fend.deta
        Neta = mp.Fend.Neta

    """ Masks and DM surfaces """
    # ompute the DM surfaces for the current DM commands
    if any(mp.dm_ind == 1):
        DM1surf = falco.dm.gen_surf_from_act(mp.dm1, mp.dm1.compact.dx,
                                             NdmPad)
    else:
        DM1surf = np.zeros((NdmPad, NdmPad))
    if any(mp.dm_ind == 2):
        DM2surf = falco.dm.gen_surf_from_act(mp.dm2, mp.dm2.compact.dx,
                                             NdmPad)
    else:
        DM2surf = np.zeros((NdmPad, NdmPad))

    pupil = pad_crop(mp.P1.compact.mask, NdmPad)
    Ein = pad_crop(Ein, NdmPad)

    if(mp.flagDM1stop):
        DM1stop = pad_crop(mp.dm1.compact.mask, NdmPad)
    else:
        DM1stop = np.ones((NdmPad, NdmPad))
    if(mp.flagDM2stop):
        DM2stop = pad_crop(mp.dm2.compact.mask, NdmPad)
    else:
        DM2stop = np.ones((NdmPad, NdmPad))

    # if mp.useGPU:
    #     log.warning('GPU support not yet implemented. Proceeding without GPU.')

    # This block is for BMC surface error testing
    if mp.flagDMwfe:
        if any(mp.dm_ind == 1):
            Edm1WFE = np.exp(2*np.pi*1j/wvl *
                             pad_crop(mp.dm1.compact.wfe,
                                      NdmPad, 'extrapval', 0))
        else:
            Edm1WFE = np.ones((NdmPad, NdmPad))
        if any(mp.dm_ind == 2):
            Edm2WFE = np.exp(2*np.pi*1j/wvl *
                             pad_crop(mp.dm2.compact.wfe,
                                      NdmPad, 'extrapval', 0))
        else:
            Edm2WFE = np.ones((NdmPad, NdmPad))
    else:
        Edm1WFE = np.ones((NdmPad, NdmPad))
        Edm2WFE = np.ones((NdmPad, NdmPad))

    """Propagation from P1 to P3"""

    # Define pupil P1 and Propagate to pupil P2
    EP1 = pupil*Ein  # E-field at pupil plane P1
    EP2 = falco.prop.relay(EP1, NrelayFactor*mp.Nrelay1to2, mp.centering)

    # Propagate from P2 to DM1, and apply DM1 surface and aperture stop
    if not (abs(mp.d_P2_dm1) == 0):  # E-field arriving at DM1
        Edm1inc = falco.prop.ptp(EP2, mp.P2.compact.dx*NdmPad, wvl, mp.d_P2_dm1)
    else:
        Edm1inc = EP2

    # E-field leaving DM1
    Edm1inc *= DM1stop
    Edm1 = Edm1inc*Edm1WFE*np.exp(mirrorFac*2*np.pi*1j*DM1surf/wvl)

    # Propagate from DM1 to DM2, and apply DM2 surface and aperture stop
    Edm2inc = DM2stop * falco.prop.ptp(Edm1, mp.P2.compact.dx*NdmPad, wvl, mp.d_dm1_dm2)
    Edm2 = Edm2WFE * Edm2inc*np.exp(mirrorFac*2*np.pi*1j*DM2surf/wvl)

    # Back-propagate to pupil P2
    if(mp.d_P2_dm1 + mp.d_dm1_dm2 == 0):
        EP2eff = Edm2
    else:
        EP2eff = falco.prop.ptp(Edm2, mp.P2.compact.dx*NdmPad, wvl, -1 *
                                (mp.d_dm1_dm2 + mp.d_P2_dm1))

    # Re-image to pupil P3
    EP3 = falco.prop.relay(EP2eff, NrelayFactor*mp.Nrelay2to3, mp.centering)

    # Apply apodizer mask.
    if mp.flagApod:
        EP3 = mp.P3.compact.mask*pad_crop(EP3, mp.P3.compact.Narr)


    """Propagation from P3 to P4 depends on coronagraph type."""
    # Remove FPM completely if normalization value is being found for vortex
    if normFac == 0:
        if mp.coro.upper() in ('VORTEX', 'VC', 'AVC'):
            useFPM = False

    if useFPM:

        if mp.coro.upper() in ('VORTEX', 'VC', 'AVC'):

            # Get FPM charge
            if isinstance(mp.F3.VortexCharge, np.ndarray):
                # Passing an array for mp.F3.VortexCharge with
                # corresponding wavelengths mp.F3.VortexCharge_lambdas
                # represents a chromatic vortex FPM
                if mp.F3.VortexCharge.size == 1:
                    charge = mp.F3.VortexCharge
                else:
                    np.interp(wvl, mp.F3.VortexCharge_lambdas,
                              mp.F3.VortexCharge, 'linear', 'extrap')

            elif isinstance(mp.F3.VortexCharge, (int, float)):
                # single value indicates fully achromatic mask
                charge = mp.F3.VortexCharge
            else:
                raise TypeError("mp.F3.VortexCharge must be int, float or numpy\
                                ndarray.")
                pass
            EP4 = falco.prop.mft_p2v2p(EP3, charge, mp.P1.compact.Nbeam/2., 0.3, 5)
            EP4 = pad_crop(EP4, mp.P4.compact.Narr)

            # Undo the rotation inherent to falco.prop.mft_p2v2p.m
            if not mp.flagRotation:
                EP4 = falco.prop.relay(EP4, -1, mp.centering)

        elif mp.coro.upper() == 'FLC' or mp.coro.upper() == 'SPLC':
            # MFT from SP to FPM (i.e., P3 to F3)
            # E-field incident upon the FPM
            EF3inc = falco.prop.mft_p2f(
                EP3, mp.fl, wvl, mp.P2.compact.dx, fpmScaleFac*mp.F3.compact.dxi,
                mp.F3.compact.Nxi, fpmScaleFac*mp.F3.compact.deta,
                mp.F3.compact.Neta, mp.centering)

            # Apply FPM
            EF3 = mp.F3.compact.mask * EF3inc

            # MFT from FPM to Lyot Plane (i.e., F3 to P4)
            EP4 = falco.prop.mft_f2p(EF3, mp.fl, wvl, fpmScaleFac*mp.F3.compact.dxi,
                                     fpmScaleFac*mp.F3.compact.deta, mp.P4.compact.dx,
                                     mp.P4.compact.Narr, mp.centering)
            EP4 = falco.prop.relay(EP4, NrelayFactor*mp.Nrelay3to4-1, mp.centering)

        elif mp.coro.upper() in ('LC', 'APLC', 'HLC'):
            # MFT from SP to FPM (i.e., P3 to F3)
            # E-field incident upon the FPM
            EF3inc = falco.prop.mft_p2f(EP3, mp.fl, wvl, mp.P2.compact.dx,
                                        fpmScaleFac*mp.F3.compact.dxi,
                                        mp.F3.compact.Nxi,
                                        fpmScaleFac*mp.F3.compact.deta,
                                        mp.F3.compact.Neta, mp.centering)

            EF3 = (transOuterFPM - mp.F3.compact.mask)*EF3inc

            # Use Babinet's principle at the Lyot plane.
            EP4noFPM = falco.prop.relay(EP3, NrelayFactor*mp.Nrelay3to4, mp.centering)
            EP4noFPM = pad_crop(EP4noFPM, mp.P4.compact.Narr)
            if mp.coro.upper() == 'HLC':
                EP4noFPM = transOuterFPM*EP4noFPM
            # MFT from FPM to Lyot Plane (i.e., F3 to P4)
            # Subtrahend term for Babinet's principle
            EP4sub = falco.prop.mft_f2p(EF3, mp.fl, wvl,
                                        fpmScaleFac*mp.F3.compact.dxi,
                                        fpmScaleFac*mp.F3.compact.deta,
                                        mp.P4.compact.dx, mp.P4.compact.Narr,
                                        mp.centering)
            EP4subRelay = falco.prop.relay(EP4sub, NrelayFactor*mp.Nrelay3to4-1, mp.centering)
            # Babinet's principle at P4
            EP4 = (EP4noFPM-EP4subRelay)

    else:  # No FPM

        EP4 = falco.prop.relay(EP3, NrelayFactor*mp.Nrelay3to4, mp.centering)
        EP4 *= transOuterFPM

        # Downsample the beam if Lyot plane has lower resolution
        if mp.P4.compact.Nbeam != mp.P1.compact.Nbeam:
            # Make sure array is oversized before downsampling
            padFac = 1.2
            EP4 = pad_crop(EP4,
                           falco.util.ceil_even(padFac*mp.P1.compact.Nbeam))

            EP4tempReal = falco.mask.rotate_shift_downsample_pupil_mask(
                np.real(EP4), mp.P1.compact.Nbeam, mp.P4.compact.Nbeam, 0, 0, 0)
            EP4tempImag = falco.mask.rotate_shift_downsample_pupil_mask(
                np.imag(EP4), mp.P1.compact.Nbeam, mp.P4.compact.Nbeam, 0, 0, 0)

            EP4 = EP4tempReal + 1j*EP4tempImag
            # Preserve summed intensity in the pupil:
            EP4 *= mp.P1.compact.Nbeam/mp.P4.compact.Nbeam

        EP4 = pad_crop(EP4, mp.P4.compact.Narr)

    """  Back to common propagation any coronagraph type   """
    # Apply the Lyot stop
    EP4 = mp.P4.compact.croppedMask*EP4

    # MFT to camera
    EP4 = falco.prop.relay(EP4, NrelayFactor*mp.NrelayFend, mp.centering)
    EFend = falco.prop.mft_p2f(EP4, mp.fl, wvl, mp.P4.compact.dx, dxi, Nxi,
                               deta, Neta, mp.centering)

    # Don't apply FPM if normalization value is being found
    if normFac == 0:
        Eout = EFend  # Don't normalize if normalization value is being found
    else:
        Eout = EFend/np.sqrt(normFac)  # Apply normalization

    if forRevGradModel:
        return Eout, Edm1inc, Edm2inc, DM1surf, DM2surf
    else:
        return Eout


def jacobian(mp):
    """
    Compute the control Jacobian used for EFC.

    Wrapper function for the function model_Jacobian_middle_layer, which gets
    the DM response matrix, aka the control Jacobian for each specified DM.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters

    Returns
    -------
    jacStruct : ModelParameters
        Structure containing the Jacobians for each specified DM.

    """
    jacStruct = falco.config.Object()  # Initialize the new structure

    # Pre-compute the DM surfaces to save time
    NdmPad = int(mp.compact.NdmPad)
    if any(mp.dm_ind == 1):
        mp.dm1.compact.surfM = falco.dm.gen_surf_from_act(
            mp.dm1, mp.dm1.compact.dx, NdmPad)
    else:
        mp.dm1.compact.surfM = np.zeros((NdmPad, NdmPad))
    if any(mp.dm_ind == 2):
        mp.dm2.compact.surfM = falco.dm.gen_surf_from_act(
            mp.dm2, mp.dm2.compact.dx, NdmPad)
    else:
        mp.dm2.compact.surfM = np.zeros((NdmPad, NdmPad))

    # Pre-compute the HLC FPM at each wavelength to save time
    if mp.layout.lower() == 'fourier':
        if mp.coro.upper() == 'HLC':
            if not hasattr(mp.compact, 'fpmCube'):
                mp.compact.fpmCube, mp.dm8.surf, mp.dm9.surf = \
                    falco.hlc.gen_fpm_cube_from_LUT(mp, 'compact')

    # Initialize the Jacobians for each DM
    jacStruct.G1 = np.zeros((mp.Fend.corr.Npix, mp.dm1.Nele, mp.jac.Nmode),
                            dtype=complex)
    jacStruct.G2 = np.zeros((mp.Fend.corr.Npix, mp.dm2.Nele, mp.jac.Nmode),
                            dtype=complex)
    jacStruct.G8 = np.zeros((mp.Fend.corr.Npix, mp.dm8.Nele, mp.jac.Nmode),
                            dtype=complex)
    jacStruct.G9 = np.zeros((mp.Fend.corr.Npix, mp.dm9.Nele, mp.jac.Nmode),
                            dtype=complex)

    # Calculate the Jacobian in parallel or serial
    if mp.flagParallel:
        # print('Computing control Jacobian matrices in parallel via multiprocessing.Process...', end='')
        # with falco.util.TicToc():
        #      ##results_order = [pool.apply(_func_Jac_ordering, args=(im,idm)) for im,idm in zip(*map(np.ravel, np.meshgrid(np.arange(mp.jac.Nmode,dtype=int),mp.dm_ind))) ]       
        #     results_order = [(im, idm) for idm in mp.dm_ind for im in np.arange(mp.jac.Nmode,dtype=int)] # Use for assigning parts of the Jacobian list to the correct DM and mode

        #     print(zip(*map(np.ravel, np.meshgrid(np.arange(mp.jac.Nmode, dtype=int), mp.dm_ind))))

        #     output = multiprocessing.Queue()
        #     processes = [multiprocessing.Process(target=_jac_middle_layer_process,
        #                  args=(mp,im,idm,output)) for im,idm in zip(*map(np.ravel, np.meshgrid(np.arange(mp.jac.Nmode,dtype=int), mp.dm_ind)))]

        #     # if __name__ == '__main__':
        #     jobs = []
        #     for p in processes:
        #         jobs.append(p)
        #         p.start()

        #     # for j in jobs:
        #     #     j.join()

        #     for p in processes:
        #         p.terminate()

        #     for p in processes:
        #         p.join()

        #     results_Jac = [output.get() for p in processes]

        #     # Reorder Jacobian by mode and DM from the list
        #     for ii in range(mp.jac.Nmode*mp.dm_ind.size):
        #         im = results_order[ii][0]
        #         idm = results_order[ii][1]
        #         if idm == 1:
        #             jacStruct.G1[:, :, im] = results_Jac[ii]
        #         if idm == 2:
        #             jacStruct.G2[:, :, im] = results_Jac[ii]

        print('Computing control Jacobian matrices in parallel...', end='')
        pool = multiprocessing.Pool(processes=mp.Nthreads)

        with falco.util.TicToc():
            results_order = [(im, idm) for idm in mp.dm_ind for im in range(mp.jac.Nmode)]
            results = pool.starmap(
                _jac_middle_layer,
                [(mp, im, idm)for im, idm in zip(*map(np.ravel, np.meshgrid(np.arange(mp.jac.Nmode,dtype=int), mp.dm_ind)))])
            results_Jac = results
            pool.close()
            pool.join()

            # Reorder Jacobian by mode and DM from the list
            for ii in range(mp.jac.Nmode*mp.dm_ind.size):
                im = results_order[ii][0]
                idm = results_order[ii][1]
                if idm == 1:
                    jacStruct.G1[:, :, im] = results_Jac[ii]
                if idm == 2:
                    jacStruct.G2[:, :, im] = results_Jac[ii]
                if idm == 8:
                    jacStruct.G8[:, :, im] = results_Jac[ii]
                if idm == 9:
                    jacStruct.G9[:, :, im] = results_Jac[ii]

            print('done.')

    else:
        print('Computing control Jacobian matrices in serial:\n  ', end='')
        with falco.util.TicToc():
            for im in range(mp.jac.Nmode):
                if any(mp.dm_ind == 1):
                    print('mode%ddm%d...' % (im, 1), end='')
                    jacStruct.G1[:, :, im] = _jac_middle_layer(mp, im, 1)
                if any(mp.dm_ind == 2):
                    print('mode%ddm%d...' % (im, 2), end='')
                    jacStruct.G2[:, :, im] = _jac_middle_layer(mp, im, 2)
                if any(mp.dm_ind == 8):
                    print('mode%ddm%d...' % (im, 8), end='')
                    jacStruct.G8[:, :, im] = _jac_middle_layer(mp, im, 8)
                if any(mp.dm_ind == 9):
                    print('mode%ddm%d...' % (im, 9), end='')
                    jacStruct.G9[:, :, im] = _jac_middle_layer(mp, im, 9)
            print('done.')

    # TIED ACTUATORS

    # Handle tied actuators by adding the 2nd actuator's Jacobian column to the
    # first actuator's column, and then zeroing out the 2nd actuator's column.
    if any(mp.dm_ind == 1):
        # Update the sets of tied actuators
        mp.dm1 = falco.dm.enforce_constraints(mp.dm1)
        for ti in range(mp.dm1.tied.shape[0]):
            Index1all = mp.dm1.tied[ti, 0]  # Index of first tied actuator in whole actuator set. 
            Index2all = mp.dm1.tied[ti, 1]  # Index of second tied actuator in whole actuator set. 
            Index1subset = np.nonzero(mp.dm1.act_ele == Index1all)[0]  # Index of first tied actuator in subset of used actuators. 
            Index2subset = np.nonzero(mp.dm1.act_ele == Index2all)[0]  # Index of second tied actuator in subset of used actuators. 
            jacStruct.G1[:, Index1subset, :] += jacStruct.G1[:, Index2subset, :]  # adding the 2nd actuators Jacobian column to the first actuator's column
            jacStruct.G1[:, Index2subset, :] = 0*jacStruct.G1[:, Index2subset, :]  # zero out the 2nd actuator's column.

    if any(mp.dm_ind == 2):
        mp.dm2 = falco.dm.enforce_constraints(mp.dm2)  # Update the sets of tied actuators
        for ti in range(mp.dm2.tied.shape[0]):
            Index1all = mp.dm2.tied[ti, 0]  # Index of first tied actuator in whole actuator set. 
            Index2all = mp.dm2.tied[ti, 1]  # Index of second tied actuator in whole actuator set. 
            Index1subset = np.nonzero(mp.dm2.act_ele == Index1all)[0]  # Index of first tied actuator in subset of used actuators. 
            Index2subset = np.nonzero(mp.dm2.act_ele == Index2all)[0]  # Index of second tied actuator in subset of used actuators. 
            jacStruct.G2[:, Index1subset, :] += jacStruct.G2[:, Index2subset, :]  # adding the 2nd actuators Jacobian column to the first actuator's column
            jacStruct.G2[:, Index2subset, :] = 0*jacStruct.G2[:, Index2subset, :]  # zero out the 2nd actuator's column.

    return jacStruct


def _func_Jac_ordering(im, idm):
    """Order modes for parallelized Jacobian calculation."""
    return (im, idm)


def _jac_middle_layer(mp, im, idm):
    """
    Select which optical layout's Jacobian model to use and get E-field.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters

    Returns
    -------
    jacMode : numpy ndarray
        Complex-valued, 2-D array containing the Jacobian for the specified DM.

    """
    if mp.layout.lower() in ('fourier', 'proper', 'fpm_scale'):
        if mp.coro.upper() in ('LC', 'APLC', 'HLC', 'FLC', 'SPLC'):
            jacMode = jacobians.lyot(mp, im, idm)
        elif mp.coro.upper() in ('VC', 'AVC', 'VORTEX'):
            jacMode = jacobians.vortex(mp, im, idm)
            
    elif mp.layout.lower() in ('wfirst_phaseb_proper', 'roman_phasec_proper'):
        if mp.coro.upper() in ('HLC', 'SPC', 'SPLC'):
            jacMode = jacobians.lyot(mp, im, idm)
        else:
            raise ValueError('%s not recognized as value for mp.coro' %
                             mp.coro)
            
    else:
        raise ValueError('mp.layout.lower not recognized')

    return jacMode


def _jac_middle_layer_process(mp, im, idm, output):
    """
    Select which optical layout's Jacobian model to use and get E-field.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters

    Returns
    -------
    jacMode : numpy ndarray
        Complex-valued, 2-D array containing the Jacobian for the specified DM.

    """
    if mp.layout.lower() in ('fourier', 'proper'):
        if mp.coro.upper() in ('LC', 'APLC', 'FLC', 'SPLC'):
            jacMode = jacobians.lyot(mp, im, idm)
        elif mp.coro.upper() in ('VC', 'AVC', 'VORTEX'):
            jacMode = jacobians.vortex(mp, im, idm)
    elif mp.layout.lower() in ('wfirst_phaseb_proper', 'roman_phasec_proper'):
        if mp.coro.upper() in ('HLC', 'SPC', 'SPLC'):
            jacMode = jacobians.lyot(mp, im, idm)
        else:
            raise ValueError('%s not recognized as value for mp.coro' %
                             mp.coro)
    else:
        raise ValueError('mp.layout.lower not recognized')

    output.put(jacMode)
