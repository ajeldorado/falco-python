"""Estimation functions for WFSC."""

import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import falco
from . import check


def wrapper(mp, ev, jacStruct):
    """
    Select and run the chosen estimator.

    The estimates are placed in the ev object.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    ev : FALCO object
        Structure containing estimation variables.
    jacStruct : ModelParameters
        Structure containing control Jacobians for each specified DM.

    Returns
    -------
    None.

    """
    if mp.estimator == 'perfect':

        falco.est.perfect(mp, ev)
        with falco.util.TicToc('Getting updated summed image'):
            ev.Im = falco.imaging.get_summed_image(mp)

    elif mp.estimator in ['pairwise', 'pairwise-square', 'pairwise-rect',
                          'pwp-bp-square', 'pwp-bp', 'pwp-kf']:

        if mp.est.flagUseJac:  # Send in the Jacobian if true
            falco.est.pairwise_probing(mp, ev, jacStruct=jacStruct)
        else:  # Otherwise don't pass the Jacobian
            falco.est.pairwise_probing(mp, ev)

    elif mp.estimator == 'ekf_maintenance':
        if ev.Itr == 1:
            disp('starting ekf initialization')
            initialize_ekf_maintenance(mp, ev, jacStruct)
            disp('done ekf initialization')


        falco_est_ekf_maintenance(mp, ev, jacStruct=jacStruct)


    return None


def perfect(mp, ev):
    """
    Return the perfect-knowledge E-field from the full model.

    Optionally add Zernikes at the input pupil.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    ev : FALCO object
        Structure containing estimation variables.

    Returns
    -------
    None

    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')

    # Polarization states as defined in the WFIRST or Roman PROPER models
    # Use the average polarization state for the perfect estimation.
    if hasattr(mp.full, 'pol_conds'):
        if set(mp.full.pol_conds) in ({-2, -1, 1, 2}, {10}):  # X and Y out
            mp.full.polaxis = 10
        elif set(mp.full.pol_conds) in ({-1, 1}, {5}):  # X out
            mp.full.polaxis = 5
        elif set(mp.full.pol_conds) in ({-2, 2}, {6}):  # Y out
            mp.full.polaxis = 6
        elif set(mp.full.pol_conds) == {0}:  # Y out
            mp.full.polaxis = 0
        else:
            raise ValueError('Invalid value of mp.full.pol_conds.')

    # ev = falco.config.Object()
    ev.imageArray = np.zeros((mp.Fend.Neta, mp.Fend.Nxi, 1, mp.Nsbp))
    Eest = np.zeros((mp.Fend.corr.Npix, mp.jac.Nmode), dtype=complex)

    if mp.flagParallel:

        inds_list = [(x, y) for x in range(mp.jac.Nmode)
                     for y in range(mp.Nwpsbp)]
        Nvals = mp.jac.Nmode*mp.Nwpsbp

        pool = multiprocessing.Pool(processes=mp.Nthreads)
        results = pool.starmap(
            _est_perfect_Efield_with_Zernikes_in_parallel,
            [(mp, ilist, inds_list) for ilist in range(Nvals)]
        )
        pool.close()
        pool.join()

        # # Convert from a list to arrays:
        # for ni in range(Nvals):
        #     InormVec[ni] = results_ctrl[ni][0]
        #     if any(mp.dm_ind == 1):
        #         dDM1V_store[:, :, ni] = results_ctrl[ni][1].dDM1V

        # # Loop over all modes and wavelengths
        # inds_list = [(x, y) for x in range(mp.jac.Nmode) for y in range(mp.Nwpsbp)]
        # Nvals = mp.jac.Nmode*mp.Nwpsbp

        # pool = multiprocessing.Pool(processes=mp.Nthreads)
        # resultsRaw = [pool.apply_async(_est_perfect_Efield_with_Zernikes_in_parallel,
        #                                args=(mp, ilist, inds_list)) for ilist in range(Nvals)]
        # results = [p.get() for p in resultsRaw]  # All the images in a list
        # pool.close()
        # pool.join()

        # Re-order for easier indexing
        EmatArray = np.zeros((mp.Fend.Nxi, mp.Fend.Neta, mp.jac.Nmode, mp.Nwpsbp), dtype=complex)
        EvecArray = np.zeros((mp.Fend.corr.Npix, mp.jac.Nmode, mp.Nwpsbp), dtype=complex)
        for iv in range(Nvals):
            iMode = inds_list[iv][0]  # Index of the Jacobian mode
            wi = inds_list[iv][1]   # Index of the wavelength in the sub-bandpass
            E2D = results[iv]
            EmatArray[:, :, iMode, wi] = E2D
            EvecArray[:, iMode, wi] = E2D[mp.Fend.corr.maskBool] 
        # Eest = np.mean(EvecArray, axis=2)  # Average over wavelengths in the subband

        # # Average the E-field over the wavelengths in a subband
        # for iMode in range(mp.jac.Nmode):

        #     iSubband = mp.jac.sbp_inds[iMode]
        #     iZernike = mp.jac.zern_inds[iMode]
        #     # iStar = mp.jac.star_inds[iMode]
        #     isPiston = (iZernike == 1)
        #     if isPiston:
        #         ev.imageArray[:, :, 0, iSubband] += np.abs(np.mean(EmatArray, axis=3)[:, :, iMode])**2

        # Average the E-field over the wavelengths in a subband
        # counter = 0
        Eest = np.zeros((mp.Fend.corr.Npix, mp.jac.Nmode), dtype=complex)
        for iMode in range(mp.jac.Nmode):
            EsubbandMeanVec = np.zeros(mp.Fend.corr.Npix, dtype=complex)
            EsubbandMeanMat = np.zeros((mp.Fend.Neta, mp.Fend.Nxi), dtype=complex)
            for wi in range(mp.Nwpsbp):
                EsubbandMeanVec += mp.full.lambda_weights[wi] * EvecArray[:, iMode, wi]
                EsubbandMeanMat += mp.full.lambda_weights[wi] * EmatArray[:, :, iMode, wi]
                # counter += 1

            Eest[:, iMode] = EsubbandMeanVec

            iSubband = mp.jac.sbp_inds[iMode]
            iZernike = mp.jac.zern_inds[iMode]
            # iStar = mp.jac.star_inds[iMode]
            isPiston = (iZernike == 1)
            if isPiston:
                # imageTemp = np.zeros((mp.Fend.Neta, mp.Fend.Nxi), dtype=float)
                # imageTemp[mp.Fend.corr.maskBool] = np.abs(EsubbandMean)**2
                ev.imageArray[:, :, 0, iSubband] += np.abs(EsubbandMeanMat)**2

    else:

        modvar = falco.config.ModelVariables()

        for iMode in range(mp.jac.Nmode):
            modvar.sbpIndex = mp.jac.sbp_inds[iMode]
            modvar.zernIndex = mp.jac.zern_inds[iMode]
            modvar.starIndex = mp.jac.star_inds[iMode]
            modvar.whichSource = 'star'
            isPiston = (modvar.zernIndex == 1)

            # Take the mean over the wavelengths within the sub-bandpass
            EmatSubband = np.zeros((mp.Fend.Neta, mp.Fend.Nxi), dtype=complex)
            for wi in range(mp.Nwpsbp):
                modvar.wpsbpIndex = wi
                E2D = falco.model.full(mp, modvar)
                EmatSubband += mp.full.lambda_weights[wi] * E2D

            if isPiston:
                ev.imageArray[:, :, 0, modvar.sbpIndex] = np.abs(EmatSubband)**2

            Eest[:, iMode] = EmatSubband[mp.Fend.corr.maskBool]

    ev.Eest = Eest
    ev.IincoEst = np.zeros(Eest.shape, dtype=float)

    return ev


# Extra function needed to use parfor (because parfor can have only a
# single changing input argument).
def _est_perfect_Efield_with_Zernikes_in_parallel(mp, ilist, inds_list):

    iMode = inds_list[ilist][0]  # Index of the Jacobian mode
    wi = inds_list[ilist][1]   # Index of the wavelength in the sub-bandpass

    modvar = falco.config.ModelVariables()
    modvar.sbpIndex = mp.jac.sbp_inds[iMode]
    modvar.zernIndex = mp.jac.zern_inds[iMode]
    modvar.starIndex = mp.jac.star_inds[iMode]
    modvar.wpsbpIndex = wi
    modvar.whichSource = 'star'

    E2D = falco.model.full(mp, modvar)

    return E2D


def pairwise_probing(mp, ev, jacStruct=np.array([])):
    """
    Estimate the dark hole E-field with pair-wise probing.

    Parameters
    ----------
    mp : falco.config.ModelParameter
        Object containing all model parameters.
    ev : falco.config.Object()
    jacStruct : array_like, optional
        Array containing the control Jacobian. Default is an empty array.

    Returns
    -------
    None
        Outputs are included in the object ev.

    """
    Itr = ev.Itr
    whichDM = mp.est.probe.whichDM

    # # Reset "ev" if not using a Kalman filter. 
    # if mp.estimator != 'pwp-kf':
    #     # ev = falco.config.Object()
    #     ev.dm1 = falco.config.Object()
    #     ev.dm2 = falco.config.Object()

    # If scheduled, change some aspects of the probe.
    # None values mean they are not scheduled.
    if mp.est.probeSchedule.xOffsetVec is not None:
        check.oneD_array(mp.est.probeSchedule.xOffsetVec)
        if len(mp.est.probeSchedule.xOffsetVec) < mp.Nitr:
            raise ValueError('mp.est.probeSchedule.xOffsetVec must have '
                             'enough values for all WFSC iterations.')
        mp.est.probe.xOffset = mp.est.probeSchedule.xOffsetVec[Itr]

    if mp.est.probeSchedule.yOffsetVec is not None:
        check.oneD_array(mp.est.probeSchedule.yOffsetVec)
        if len(mp.est.probeSchedule.yOffsetVec) < mp.Nitr:
            raise ValueError('mp.est.probeSchedule.yOffsetVec must have '
                             'enough values for all WFSC iterations.')
        mp.est.probe.yOffset = mp.est.probeSchedule.yOffsetVec[Itr]

    if mp.est.probeSchedule.rotationVec is not None:
        check.oneD_array(mp.est.probeSchedule.rotationVec)
        if len(mp.est.probeSchedule.rotationVec) < mp.Nitr:
            raise ValueError('mp.est.probeSchedule.rotationVec must have '
                             'enough values for all WFSC iterations.')
        mp.est.probe.rotation = mp.est.probeSchedule.rotationVec[Itr]

    if mp.est.probeSchedule.InormProbeVec is not None:
        check.oneD_array(mp.est.probeSchedule.InormProbeVec)
        if len(mp.est.probeSchedule.InormProbeVec) < mp.Nitr:
            raise ValueError('mp.est.probeSchedule.InormProbeVec must have '
                             'enough values for all WFSC iterations.')

    # Temporarily augment which DMs are used if the probing DM isn't used for control.
    mp.dm_ind_init = mp.dm_ind.copy()
    mp.dm_ind = list(mp.dm_ind)
    if whichDM == 1 and ~np.any(mp.dm_ind == 1):
        mp.dm_ind = np.concatenate((mp.dm_ind, [1]))
    elif whichDM == 2 and ~np.any(mp.dm_ind == 2):
        mp.dm_ind = np.concatenate((mp.dm_ind, [2]))

    # Select number of actuators across based on chosen DM for the probing
    if whichDM == 1:
        Nact = mp.dm1.Nact
    elif whichDM == 2:
        Nact = mp.dm2.Nact
    else:
        raise ValueError('mp.est.probe.whichDM must equal 1 or 2.')

    # Store the initial DM commands
    if np.any(mp.dm_ind == 1):
        DM1Vnom = mp.dm1.V
    else:
        DM1Vnom = np.zeros((mp.dm1.Nact, mp.dm1.Nact))

    if np.any(mp.dm_ind == 2):
        DM2Vnom = mp.dm2.V
    else:
        DM2Vnom = np.zeros((mp.dm2.Nact, mp.dm2.Nact))

    # Initialize output arrays
    Npairs = mp.est.probe.Npairs  # Number of image PAIRS
    ev.imageArray = np.zeros((mp.Fend.Neta, mp.Fend.Nxi, 1+2*Npairs, mp.Nsbp))
    ev.Eest = np.zeros((mp.Fend.corr.Npix, mp.Nsbp*mp.compact.star.count),
                       dtype=complex)
    ev.IincoEst = np.zeros((mp.Fend.corr.Npix, mp.Nsbp*mp.compact.star.count))
    ev.IprobedMean = 0
    ev.Im = np.zeros((mp.Fend.Neta, mp.Fend.Nxi))
    ev.dm1 = falco.config.Object()
    ev.dm2 = falco.config.Object()
    if np.any(mp.dm_ind == 1):
        ev.dm1.Vall = np.zeros((mp.dm1.Nact, mp.dm1.Nact, 1+2*Npairs, mp.Nsbp))
    if np.any(mp.dm_ind == 2):
        ev.dm2.Vall = np.zeros((mp.dm2.Nact, mp.dm2.Nact, 1+2*Npairs, mp.Nsbp))

    # Generate evenly spaced probes along the complex unit circle
    # NOTE: Nprobes=Npairs*2
    probePhaseVec = np.array([0, Npairs])
    for k in range(Npairs-1):
        probePhaseVec = np.append(probePhaseVec, probePhaseVec[-1]-(Npairs-1))
        probePhaseVec = np.append(probePhaseVec, probePhaseVec[-1]+Npairs)
    probePhaseVec = probePhaseVec*np.pi/(Npairs)

    # Only for square probe region centered on the star
    if mp.estimator in ('pairwise', 'pairwise-square', 'pwp-bp-square'):

        badAxisVec = ''
        if mp.est.probe.axis.lower() == 'y':
            for _iter in range(2*Npairs):
                badAxisVec += 'y'
        elif mp.est.probe.axis.lower() == 'x':
            for _iter in range(2*Npairs):
                badAxisVec += 'x'
        elif mp.est.probe.axis.lower() in ('alt', 'xy', 'alternate'):

            # Change probe ordering for odd- vs even-numbered WFSC iterations.
            if Itr % 2 == 0:

                for iPair in range(2*Npairs):
                    if (iPair+1) % 4 == 1 or (iPair+1) % 4 == 2:
                        badAxisVec += 'x'
                    elif (iPair+1) % 4 == 3 or (iPair+1) % 4 == 0:
                        badAxisVec += 'y'

            else:

                for iPair in range(2*Npairs):
                    if (iPair+1) % 4 == 1 or (iPair+1) % 4 == 2:
                        badAxisVec += 'y'
                    elif (iPair+1) % 4 == 3 or (iPair+1) % 4 == 0:
                        badAxisVec += 'x'

        elif mp.est.probe.axis.lower() in ('m', 'multi'):
            for _iter in range(2*Npairs):
                badAxisVec += 'm'
        else:
            raise ValueError('Invalid value for mp.est.probe.axis')

    # Get images and perform estimates in each sub-bandpass
    print('Estimating electric field with batch process estimation ...')

    for iStar in range(mp.compact.star.count):

        modvar = falco.config.ModelVariables() # Initialize the new structure
        modvar.starIndex = iStar
        modvar.whichSource = 'star'

        for iSubband in range(mp.Nsbp):

            modvar.sbpIndex = iSubband
            print('Wavelength: %u/%u ... ' % (iSubband, mp.Nsbp-1))
            modeIndex = iStar*mp.Nsbp + iSubband #(iStar-1)*mp.Nsbp + iSubband
            print('Mode: %u/%u ... ' % (modeIndex, mp.jac.Nmode-1))

            # Measure current contrast level average
            # Reset DM commands to the unprobed state:
            mp.dm1.V = DM1Vnom
            mp.dm2.V = DM2Vnom
            # Separate out image values at DH pixels and delta DM voltage settings
            Iplus = np.zeros((mp.Fend.corr.Npix, Npairs))
            Iminus = np.zeros((mp.Fend.corr.Npix, Npairs))
            DM1Vplus = np.zeros((Nact, Nact, Npairs))
            DM1Vminus = np.zeros((Nact, Nact, Npairs))
            DM2Vplus = np.zeros((Nact, Nact, Npairs))
            DM2Vminus = np.zeros((Nact, Nact, Npairs))

            # Compute probe shapes and take probed images:

            # Take initial, unprobed image (for unprobed DM settings).
            whichImage = 1
            I0 = falco.imaging.get_sbp_image(mp, iSubband)
            I0vec = I0[mp.Fend.corr.maskBool]  # Vectorize the dark hole

            # Image already includes all stars, so don't sum over star loop
            if iStar == 0:
                ev.Im += mp.sbp_weights[iSubband]*I0  # image for plotting

                # Store values for first image and its DM commands
                ev.imageArray[:, :, whichImage, iSubband] = I0
                if np.any(mp.dm_ind == 1):
                    ev.dm1.Vall[:, :, whichImage, iSubband] = mp.dm1.V
                if np.any(mp.dm_ind == 2):
                    ev.dm2.Vall[:, :, whichImage, iSubband] = mp.dm2.V

            # Compute the average Inorm in the scoring and correction regions
            ev.corr = falco.config.Object()
            ev.corr.Inorm = np.mean(I0[mp.Fend.corr.maskBool])
            ev.score = falco.config.Object()
            ev.score.Inorm = np.mean(I0[mp.Fend.score.maskBool])
            print('Measured unprobed Inorm (Corr / Score): %.2e \t%.2e' %
                  (ev.corr.Inorm, ev.score.Inorm))

            # Set (approximate) probe intensity based on current measured Inorm
            if mp.est.probeSchedule.InormProbeVec is None:
                ev.InormProbeMax = mp.est.InormProbeMax
                InormProbe = np.min([np.sqrt(np.max(I0vec)*1e-5),
                                     ev.InormProbeMax])
                print('Chosen probe intensity: %.2e' % InormProbe)
            else:
                InormProbe = mp.est.probeSchedule.InormProbeVec(Itr)
                print('Scheduled probe intensity: %.2e' % InormProbe)

            # Perform the probing
            iOdd = 0  # Initialize index counters
            iEven = 0
            for iProbe in range(2*Npairs):

                # Generate the command map for the probe
                if mp.estimator in ('pairwise', 'pairwise-square',
                                    'pwp-bp-square'):
                    probeCmd = gen_pairwise_probe_square(
                        mp, InormProbe, probePhaseVec[iProbe],
                        badAxisVec[iProbe], mp.est.probe.rotation,
                    )
                elif mp.estimator in ('pairwise-rect', 'pwp-bp', 'pwp-kf'):
                    probeCmd = gen_pairwise_probe(
                        mp, InormProbe, probePhaseVec[iProbe], iStar,
                        mp.est.probe.rotation,
                    )

                # Select which DM to use for probing. Allocate probe to that DM
                if whichDM == 1:
                    dDM1Vprobe = probeCmd/mp.dm1.VtoH  # Now in volts
                    dDM2Vprobe = 0
                elif whichDM == 2:
                    dDM1Vprobe = 0
                    dDM2Vprobe = probeCmd/mp.dm2.VtoH  # Now in volts
                else:
                    raise ValueError('DM for probing must be 1 or 2.')

                if np.any(mp.dm_ind == 1):
                    mp.dm1.V = DM1Vnom + dDM1Vprobe
                if np.any(mp.dm_ind == 2):
                    mp.dm2.V = DM2Vnom + dDM2Vprobe

                # Take probed image
                Im = falco.imaging.get_sbp_image(mp, iSubband)

                # ImNonneg = Im
                # ImNonneg[Im < 0] = 0
                whichImage = 1+iProbe  # Increment image counter
                # Inorm averaged over all the probed images
                ev.IprobedMean = ev.IprobedMean + \
                    np.mean(Im[mp.Fend.corr.maskBool]) / (2*Npairs)

                # Store probed image and its DM settings
                ev.imageArray[:, :, whichImage, iSubband] = Im
                if np.any(mp.dm_ind == 1):
                    ev.dm1.Vall[:, :, whichImage, iSubband] = mp.dm1.V
                if np.any(mp.dm_ind == 2):
                    ev.dm2.Vall[:, :, whichImage, iSubband] = mp.dm2.V

                # Report results
                probeSign = '-+'
                print('Actual Probe %d%s Contrast is: %.2e' %
                      (np.floor(iProbe/2), probeSign[(iProbe+1) % 2],
                       np.mean(Im[mp.Fend.corr.maskBool])))

                # Assign image to positive or negative probe collection:
                if (iProbe+1) % 2 == 1:  # Odd; for plus probes
                    if whichDM == 1:
                        DM1Vplus[:, :, iOdd] = dDM1Vprobe + DM1Vnom
                    if whichDM == 2:
                        DM2Vplus[:, :, iOdd] = dDM2Vprobe + DM2Vnom
                    Iplus[:, iOdd] = Im[mp.Fend.corr.maskBool]
                    iOdd += 1
                elif (iProbe+1) % 2 == 0:  # Even; for minus probes
                    if whichDM == 1:
                        DM1Vminus[:, :, iEven] = dDM1Vprobe + DM1Vnom
                    if whichDM == 2:
                        DM2Vminus[:, :, iEven] = dDM2Vprobe + DM2Vnom
                    Iminus[:, iEven] = Im[mp.Fend.corr.maskBool]
                    iEven += 1

            # Calculate probe amplitudes and measurement vector.
            # (Refer again to Give'on+ SPIE 2011 to undersand why.)
            ampSq = (Iplus+Iminus)/2 - np.tile(I0vec.reshape((-1, 1)), (1, Npairs))  # square of probe E-field amplitudes
            ampSq[ampSq < 0] = 0  # If probe amplitude is zero, set amp = 0
            amp = np.sqrt(ampSq)  # E-field amplitudes, dimensions: [mp.Fend.corr.Npix, Npairs]
            isnonzero = np.all(amp, 1)
            zAll = ((Iplus-Iminus)/4).T  # Measurement vector, dimensions: [Npairs,mp.Fend.corr.Npix]
            ampSq2Dcube = np.zeros((mp.Fend.Neta, mp.Fend.Nxi, mp.est.probe.Npairs))
            for iProbe in range(Npairs):  # Display the actual probe intensity
                ampSq2D = np.zeros((mp.Fend.Neta, mp.Fend.Nxi))
                ampSq2D[mp.Fend.corr.maskBool] = ampSq[:, iProbe]
                ampSq2Dcube[:, :, iProbe] = ampSq2D
                print('*** Mean measured Inorm for probe #%d  =\t%.3e' %
                      (iProbe, np.mean(ampSq2D[mp.Fend.corr.maskBool])))

            # Plot relevant data for all the probes
            ev.iStar = iStar
            dDMVplus = np.zeros_like(DM1Vplus)
            for ii in range(Npairs):
                dDMVplus[:, :, ii] = DM1Vplus[:, :, ii] - DM1Vnom
            falco.plot.pairwise_probes(mp, ev, dDMVplus, ampSq2Dcube, iSubband)

            # ################# Perform the estimation #######################

            # Use Jacobian for estimation. This is fully model-based if the
            # Jacobian is purely model-based, or it is better if the Jacobian
            # is adaptive based on empirical data.
            if mp.est.flagUseJac:

                dEplus = np.zeros_like(Iplus, dtype=complex)
                for iProbe in range(Npairs):
                    if whichDM == 1:
                        dV = DM1Vplus[:, :, iProbe] - DM1Vnom
                        dEplus[:, iProbe] = \
                            np.squeeze(jacStruct.G1[:, :, iSubband]) * \
                            dV[mp.dm1.act_ele]
                    elif whichDM == 2:
                        dV = DM2Vplus[:, :, iProbe] - DM2Vnom
                        dEplus[:, iProbe] = \
                            np.squeeze(jacStruct.G2[:, :, iSubband]) * \
                            dV[mp.dm2.act_ele]

            # Get the probe phase from the model
            # and the probe amplitude from the measurements
            else:

                # For unprobed field based on model:
                if np.any(mp.dm_ind == 1):
                    mp.dm1.V = DM1Vnom
                if np.any(mp.dm_ind == 2):
                    mp.dm2.V = DM2Vnom

                E0 = falco.model.compact(mp, modvar)
                E0vec = E0[mp.Fend.corr.maskBool]

                # For probed fields based on model:
                Eplus = np.zeros_like(Iplus, dtype=complex)
                Eminus = np.zeros_like(Iminus, dtype=complex)
                for iProbe in range(Npairs):
                    # For plus probes:
                    if whichDM == 1:
                        mp.dm1.V = np.squeeze(DM1Vplus[:, :, iProbe])
                    if whichDM == 2:
                        mp.dm2.V = np.squeeze(DM2Vplus[:, :, iProbe])
                    Etemp = falco.model.compact(mp, modvar)
                    Eplus[:, iProbe] = Etemp[mp.Fend.corr.maskBool]

                    # For minus probes:
                    if whichDM == 1:
                        mp.dm1.V = np.squeeze(DM1Vminus[:, :, iProbe])
                    if whichDM == 2:
                        mp.dm2.V = np.squeeze(DM2Vminus[:, :, iProbe])
                    Etemp = falco.model.compact(mp, modvar)
                    Eminus[:, iProbe] = Etemp[mp.Fend.corr.maskBool]

                # Create delta E-fields for each probe image.
                # Then create Npairs phase angles.
                dEplus = Eplus - np.tile(E0vec.reshape((-1, 1)), (1, Npairs))
                dEminus = Eminus - np.tile(E0vec.reshape((-1, 1)), (1, Npairs))
                dphdm = np.zeros((mp.Fend.corr.Npix, Npairs))  # phases
                for iProbe in range(Npairs):
                    dphdm[:, iProbe] = np.arctan2(
                        np.imag(dEplus[:, iProbe]) - np.imag(dEminus[:, iProbe]),
                        np.real(dEplus[:, iProbe]) - np.real(dEminus[:, iProbe]))

            # Batch process the measurements to estimate the electric field in the
            # dark hole. Done pixel by pixel.

            if (mp.estimator in ('pairwise', 'pairwise-square','pairwise-rect', 'pwp-bp', 'pwp-bp-square')) or \
                (mp.estimator == 'pwp-kf' and ev.Itr < mp.est.ItrStartKF):

                Eest = np.zeros((mp.Fend.corr.Npix,), dtype=complex)
                zerosCounter = 0  # number of zeroed-out dark hole pixels
                for ipix in range(mp.Fend.corr.Npix):

                    if mp.est.flagUseJac:
                        dE = dEplus[ipix, :].T
                        H = np.array([np.real(dE), np.imag(dE)])
                    else:
                        H = np.zeros([Npairs, 2])  # Observation matrix
                        # Leave Eest for a pixel as zero if any probe amp is 0
                        if isnonzero[ipix] == 1:
                            for iProbe in range(Npairs):
                                H[iProbe, :] = amp[ipix, iProbe] * \
                                    np.array([np.cos(dphdm[ipix, iProbe]),
                                              np.sin(dphdm[ipix, iProbe])])
                        else:
                            zerosCounter += 1

                    Epix = np.linalg.pinv(H) @ zAll[:, ipix]  # Batch processing
                    Eest[ipix] = Epix[0] + 1j*Epix[1]

                # If estimate is too bright, the estimate was probably bad.
                Eest[np.abs(Eest)**2 > mp.est.Ithreshold] = 0.0

                print('%d of %d pixels were given zero probe amplitude.' %
                      (zerosCounter, mp.Fend.corr.Npix))

                # Initialize the state and state covariance estimates for the
                # Kalman filter. The state is the real and imag parts of the
                # E-field.
                if mp.estimator == 'pwp-kf':
                    # Re-organize the batch-processed E-field estimate into the
                    # 1st state estimate for the Kalman filter
                    xOld = np.zeros((2*mp.Fend.corr.Npix, 1))
                    for ii in range(mp.Fend.corr.Npix):
                        xOld[2*(ii-1)+0:2*(ii-1)+1] = \
                            np.array([np.real(Eest[ii]), np.imag(Eest[ii])])
                    ev.xOld = xOld  # Save out for returning later

                    # Initialize the state covariance matrix
                    # (2x2 for each dark hole pixel)
                    ev.Pold_KF_array = np.tile(mp.est.Pcoef0*np.eye(2),
                                               (mp.Fend.corr.Npix, 1, mp.Nsbp))

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # Begin Kalman Filter Update
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #
            # #TODO
            #
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # End Kalman Filter Update
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            # Save out the estimates
            ev.Eest[:, modeIndex] = Eest
            ev.IincoEst[:, modeIndex] = I0vec - np.abs(Eest)**2  # incoherent

    # Other data to save out
    ev.ampSqMean = np.mean(ampSq)  # Mean probe intensity
    ev.ampNorm = amp/np.sqrt(InormProbe)  # Normalized probe amplitude maps

    # # Calculate the mean normalized intensity over the whole dark hole at all
    # # wavelengths.
    # ev.InormEst = np.mean(Iest)

    if mp.flagPlot:
        Eest2D = np.zeros((mp.Fend.Neta, mp.Fend.Nxi), dtype=complex)
        Eest2D[mp.Fend.corr.maskBool] = Eest
        # figure(701); imagesc(real(Eest2D)); title('real(Eest)', 'Fontsize', 18); set(gca, 'Fontsize', 18); axis xy equal tight; colorbar;
        # figure(702); imagesc(imag(Eest2D)); title('imag(Eest)', 'Fontsize', 18); set(gca, 'Fontsize', 18); axis xy equal tight; colorbar;
        # figure(703); imagesc(log10(abs(Eest2D).^2)); title('abs(Eest)^2', 'Fontsize', 18); set(gca, 'Fontsize', 18); axis xy equal tight; colorbar;

    # Reset nominal values that were changed
    if np.any(mp.dm_ind == 1):
        mp.dm1.V = DM1Vnom
    if np.any(mp.dm_ind == 2):
        mp.dm2.V = DM2Vnom
    mp.dm_ind = mp.dm_ind_init

    print('Completed pairwise probing estimation.')

    return None


def gen_pairwise_probe_square(mp, InormDes, psi, badAxis, rotation):
    """
    Generate delta DM commands that probe the dark hole.

    Parameters
    ----------
    mp : falco.config.ModelParameter
        Object containing all model parameters.
    InormDes : float
        Desired normalized intensity of the probes in the image.
    psi : float
        phase angle of the sinusoidal part of the probe. Units of radians.
    badAxis : str
        The axis along which to have the phase discontinuity

    Returns
    -------
    probeCmd : array_like
         Nact x Nact array of delta DM actuator commands to make a probe.
    """
    check.real_positive_scalar(InormDes, 'InormDes', ValueError)
    check.real_scalar(psi, 'psi', TypeError)
    if badAxis.lower() not in ('x', 'y', 'm'):
        raise ValueError('Invalid value for badAxis.')

    # Number of actuators across DM surface
    # (independent of beam diameter for time being)
    if mp.est.probe.whichDM == 1:
        Nact = mp.dm1.Nact
        dm = mp.dm1
    elif mp.est.probe.whichDM == 2:
        Nact = mp.dm2.Nact
        dm = mp.dm2

    # Coordinates in actuator space
    xs = np.arange(-(Nact-1)/2, (Nact+1)/2)/Nact - \
        np.round(mp.est.probe.xOffset)/Nact
    ys = np.arange(-(Nact-1)/2, (Nact+1)/2)/Nact - \
        np.round(mp.est.probe.yOffset)/Nact
    [XS, YS] = np.meshgrid(xs, ys)

    # Rotate the coordinates
    if np.abs(rotation) > 10*np.finfo(float).eps:
        RS = np.sqrt(XS**2 + YS**2)
        THETAS = np.arctan2(YS, XS)
        rotRad = np.radians(rotation)
        XS = RS*np.cos(THETAS-rotRad)
        YS = RS*np.sin(THETAS-rotRad)
        

    # Restrict the probing region if it is not possible to achieve
    if mp.est.probe.radius > Nact/2.0:
        mp.est.probe.radius = Nact/2.0

    # Generate the DM command for the probe
    magn = 4*np.pi*mp.lambda0*np.sqrt(InormDes)  # surface height to get desired intensity [meters]
    if badAxis.lower() == 'y':
        mX = mp.est.probe.radius
        mY = 2*mp.est.probe.radius
        omegaX = mp.est.probe.radius/2
        probeSurf = magn*np.sinc(mX*XS)*np.sinc(mY*YS)*np.cos(2*np.pi*omegaX*XS + psi)

    elif badAxis.lower() == 'x':
        mX = 2*mp.est.probe.radius
        mY = mp.est.probe.radius
        omegaY = mp.est.probe.radius/2
        probeSurf = magn*np.sinc(mX*XS)*np.sinc(mY*YS)*np.cos(2*np.pi*omegaY*YS + psi)

    elif badAxis.lower() in ('m'):  # alternate between x and y
        omegaX = mp.est.probe.Xloc/2
        omegaY = mp.est.probe.Yloc/2
        probeSurf = np.zeros_like(XS)
        for i in range(mp.Fend.Nfiber):
            probeSurf = probeSurf + \
                magn*np.sin(2*np.pi*omegaX(i)*XS + 2*np.pi*omegaY(i)*YS + psi)

    # Option to use just the sincs for a zero phase shift. This avoids the
    # phase discontinuity along one axis (for this probe only!).
    if psi == 0:
        m = 2*mp.est.probe.radius
        probeSurf = magn*np.sinc(m*XS)*np.sinc(m*YS)

    probeCmd = falco.dm.fit_surf_to_act(dm, probeSurf)

    # Scale the probe amplitude empirically if needed
    probeCmd = mp.est.probe.gainFudge*probeCmd

    return probeCmd


def gen_pairwise_probe(mp, InormDes, phaseShift, starIndex, rotation):
    """
    Compute rectangular pair-wise probe commands.

    Compute a pair-wise probe shape for batch process estimation of the
    electric field in the final focal plane. The rectangular dark
    hole region is specified by its position on one half of the focal plane.

    Parameters
    ----------
    mp : falco.config.ModelParameter
        Object containing all model parameters.
    InormDes : float
        Desired normalized intensity of the probes in the image.
    psi : float
        phase angle of the sinusoidal part of the probe. Units of radians.
    starIndex : int
        Index of the star for which to perform probing around

    Returns
    -------
    probeCmd : array_like
         Nact x Nact array of delta DM actuator commands to make a probe.

    """
    if mp.est.probe.xiOffset[starIndex] == 0 and mp.est.probe.etaOffset[starIndex] == 0:
        raise ValueError("Probed region's center must be offset from the star location.")

    # Number of actuators across DM surface
    if mp.est.probe.whichDM == 1:
        dm = mp.dm1
    elif mp.est.probe.whichDM == 2:
        dm = mp.dm2
    Nact = dm.Nact
    NactPerBeam = mp.P2.D / dm.dm_spacing

    # Coordinates in actuator space
    xs = np.linspace(-(Nact-1)/2, (Nact-1)/2, Nact)/Nact - \
        float(round(mp.est.probe.xOffset))/Nact
    ys = np.linspace(-(Nact-1)/2, (Nact-1)/2, Nact)/Nact - \
        float(round(mp.est.probe.yOffset))/Nact
    [XS, YS] = np.meshgrid(xs, ys)

    # Rotate the coordinates
    if np.abs(rotation) > 10*np.finfo(float).eps:
        RS = np.sqrt(XS**2 + YS**2)
        THETAS = np.arctan2(YS, XS)
        rotRad = np.radians(rotation)
        XS = RS*np.cos(THETAS-rotRad)
        YS = RS*np.sin(THETAS-rotRad)

    # Convert units from lambda/D to actuators
    lamDIntoAct = Nact / NactPerBeam
    xiOffset = mp.est.probe.xiOffset[starIndex] * lamDIntoAct
    etaOffset = mp.est.probe.etaOffset[starIndex] * lamDIntoAct
    width = mp.est.probe.width[starIndex] * lamDIntoAct
    height = mp.est.probe.height[starIndex] * lamDIntoAct

    maxSpatialFreq = Nact / 2
    if (xiOffset + width/2) > maxSpatialFreq or (etaOffset + height/2) > maxSpatialFreq:
        print('*** WARNING: SPECIFIED PROBING REGION IN DARK HOLE IS NOT FULLY CONTROLLABLE. ***')

    # Generate the DM command for the probe
    surfMax = 4*np.pi*mp.lambda0*np.sqrt(InormDes)  # [meters]
    probeHeight = surfMax * np.sinc(width*XS) * np.sinc(height*YS) * np.cos(2*np.pi*(xiOffset*XS + etaOffset*YS) + phaseShift)
    probeCmd = falco.dm.fit_surf_to_act(dm, probeHeight)
    probeCmd = mp.est.probe.gainFudge[starIndex] * probeCmd  # Scale the probe amplitude empirically if needed

    return probeCmd





