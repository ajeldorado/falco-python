from falco.config import ModelParameters, ModelVariables, DeformableMirrorParameters
import numpy as np
import falco
import logging

log = logging.getLogger(__name__)


def model_compact_LC(mp, DM, modvar):
    """
    Blind model used by the estimator and controller. Does not include unknown aberrations/errors
    that are in the full model. This function is for the Lyot coronagraph, DMLC, and APLC.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    DM : DeformableMirrorParameters
        Structure containing deformable mirror parameters
    modvar : ModelVariables
        Structure containing optical model variables

    Returns
    -------
    Eout : array_like
        Electric field in final focal plane

    """
    lambda_ = mp.sbp_center_vec[modvar.sbpIndex]  # Center wavelength
    mirrorFac = 2  # Phase change from surface is doubled in reflection mode
    NdmPad = DM.compact.NdmPad

    """ Input E-fields """

    # Include tip/tilt in input wavefront, if any
    try:
        x_offset = mp.ttx[modvar.ttIndex]
        y_offset = mp.tty[modvar.ttIndex]
        TTphase = -2 * np.pi * (x_offset * mp.P2.compact.XsDL + y_offset * mp.P2.compact.YsDL)

        # Scale phase by lambda0/lambda because ttx and tty are in lambda0/D units
        Ett = np.exp(1j * TTphase * mp.lambda0 / lambda_)
        Ein = Ett * mp.P1.compact.E[:, :, modvar.sbpIndex]

    except AttributeError:  # No tip/tilt information specified
        Ein = mp.P1.compact.E[:, :, modvar.sbpIndex]

    """ Masks and DM surfaces """

    # Compute DM surfaces for the current DM commands
    # TODO: possibly better control structure depending on implementation of DM.dm_ind
    if any(DM.dm_ind == 1):
        DM1surf = falco.lib.dm.falco_gen_dm_surf(DM.dm1, DM.dm1.compact.dx, NdmPad)
    else:
        DM1surf = 0

    if any(DM.dm_ind == 2):
        DM2surf = falco.lib.dm.falco_gen_dm_surf(DM.dm2, DM.dm2.compact.dx, NdmPad)
    else:
        DM2surf = 0

    pupil = falco.padOrCropEven(mp.P1.compact.mask, NdmPad)
    Ein = falco.padOrCropEven(Ein, DM.compact.NdmPad)

    if mp.flagDM1stop:
        DM1stop = falco.padOrCropEven(mp.dm1.compact.mask, NdmPad)
    else:
        DM1stop = 1

    if mp.flagDM2stop:
        DM2stop = falco.padOrCropEven(mp.dm2.compact.mask, NdmPad)
    else:
        DM2stop = 1

    if mp.useGPU:
        log.warning('GPU support not yet implemented. Proceeding without GPU.')

    """ Propagation: 2 DMs, apodizer, binary-amplitude FPM, LS, and final focal plane """

    # Define pupil P1 and propagation to pupil P2
    EP1 = pupil * Ein  # E-field at pupil plane P1
    EP2 = falco.propcustom.propcustom_2FT(EP1, mp.centering)  # Rotate 180 deg to propagate to P2

    # Propagate from P2 to DM1, apply DM1 surface and aperture stop
    if np.abs(mp.d_P2_dm1):
        Edm1 = falco.propcustom.propcustom_PTP(EP2, mp.P2.compact.dx * NdmPad, lambda_, mp.d_P2_dm1)
    else:
        Edm1 = EP2

    Edm1 *= DM1stop * np.exp(mirrorFac * 1j * 2 * np.pi * DM1surf / lambda_)

    # Propagate from DM1 to DM2, and apply DM2 surface and aperture stop
    Edm2 = falco.propcustom.propcustom_PTP(Edm1, mp.P2.compact.dx * NdmPad, lambda_, mp.d_dm1_dm2)
    Edm2 *= DM2stop * np.exp(mirrorFac * 1j * 2 * np.pi * DM2surf / lambda_)

    # Backpropagate to pupil P2
    if mp.d_P2_dm1 + mp.d_dm1_dm2 == 0:
        EP2eff = Edm2
    else:
        EP2eff = falco.propcustom.propcustom_PTP(Edm2, mp.P2.compact.dx * NdmPad, lambda_,
                                                 -1 * (mp.d_dm1_dm2 + mp.d_P2_dm1))

    # Rotate 180 degrees to propagate to pupil P3
    EP3 = falco.propcustom.propcustom_2FT(EP2eff, mp.centering)

    # Apply apodizer mask
    if mp.flagApod:
        EP3 = mp.P3.compact.mask * falco.padOrCropEven(EP3, mp.P3.compact.Narr)

    # MFT from SP to FPM (i.e. P3 to F3)
    EF3inc = falco.propcustom.propcustom_mft_PtoF(EP3, mp.fl, lambda_, mp.P2.compact.dx,
                                                  mp.F3.compact.dxi, mp.F3.compact.Nxi,
                                                  mp.F3.compact.deta, mp.F3.compact.Neta,
                                                  mp.centering)
    EF3 = (1 - mp.F3.compact.mask.amp) * EF3inc  # Apply (1 - FPM) for Babinet's principle

    # MFT from FPM to Lyot stop plane (i.e. F3 to P4)- compute subtrahend term for Babinet
    EP4sub = falco.propcustom.propcustom_mft_FtoP(EF3, mp.fl, lambda_, mp.F3.compact.dxi,
                                                  mp.F3.compact.deta, mp.P4.compact.dx,
                                                  mp.P4.compact.Narr, mp.centering)

    # Use Babinet's principle at the Lyot stop plane
    EP4noFPM = falco.propcustom.propcustom_2FT(EP3, mp.centering)  # Propagate forward another pupil
    EP4noFPM = falco.padOrCropEven(EP4noFPM, mp.P4.compact.Narr)  # Crop down to Lyot stop opening
    EP4 = mp.P4.compact.croppedMask * (EP4noFPM - EP4sub)

    try:
        if modvar.flagGetNormVal:
            EP4 = mp.P4.compact.croppedMask * EP4noFPM  # No FPM, so just model DMs and Lyot stop
    except AttributeError:
        pass

    # DFT to camera
    EF4 = falco.propcustom.propcustom_mft_PtoF(EP4, mp.fl, lambda_,
                                               mp.P4.compact.dx, mp.F4.compact.dxi,
                                               mp.F4.compact.Nxi, mp.F4.compact.deta,
                                               mp.F4.compact.Neta, mp.centering)

    # Don't apply FPM if in normalization mode, if flag doesn't exist (for testing only)
    Eout = EF4

    if hasattr(modvar, 'flagGetNormVal'):
        if not modvar.flagGetNormVal:
            Eout = EF4 / np.sqrt(mp.F4.compact.I00[modvar.sbpIndex])  # Apply normalization
    elif hasattr(mp.F4.compact, 'I00'):
        Eout = EF4 / np.sqrt(mp.F4.compact.I00[modvar.sbpIndex])  # Apply normalization

    return Eout

