"""Estimation functions for WFSC."""

import numpy as np
import multiprocessing
# from astropy.io import fits
import matplotlib.pyplot as plt
import falco
from . import check

def perfect(mp):
    """
    Return the perfect-knowledge E-field from the full model.
    
    Optionally add Zernikes at the input pupil.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
        
    Returns
    -------
    Emat : numpy ndarray
        2-D array with the vectorized, complex E-field of the dark hole pixels
        for each mode included in the control Jacobian.
    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    
    if mp.flagMultiproc:
        
        Emat = np.zeros((mp.Fend.corr.Npix, mp.jac.Nmode), dtype=complex)
        
        # Loop over all modes and wavelengths
        inds_list = [(x, y) for x in range(mp.jac.Nmode) for y in range(mp.Nwpsbp)]
        Nvals = mp.jac.Nmode*mp.Nwpsbp

        pool = multiprocessing.Pool(processes=mp.Nthreads)
        resultsRaw = [pool.apply_async(_est_perfect_Efield_with_Zernikes_in_parallel,
                                       args=(mp, ilist, inds_list)) for ilist in range(Nvals)]
        results = [p.get() for p in resultsRaw]  # All the images in a list
        pool.close()
        pool.join()
        
        # Re-order for easier indexing
        Ecube = np.zeros((mp.Fend.corr.Npix, mp.jac.Nmode, mp.Nwpsbp), dtype=complex)
        for iv in range(Nvals):
            im = inds_list[iv][0]  # Index of the Jacobian mode
            wi = inds_list[iv][1]   # Index of the wavelength in the sub-bandpass
            Ecube[:, im, wi] = results[iv]
        Emat = np.mean(Ecube, axis=2)  # Average over wavelengths in the subband
  
#        EmatAll = np.zeros((mp.Fend.corr.Npix, Nval))
#        for iv in range(Nval):
#            EmatAll[:, iv] = results[iv]
#
#        counter = 0;
#        for im=1:mp.jac.Nmode
#            EsbpMean = 0;
#            for wi=1:mp.Nwpsbp
#                counter = counter + 1;
#                EsbpMean = EsbpMean + EmatAll(:,counter) * \
#                     mp.full.lambda_weights(wi);
#            end
#            Emat(:,im) = EsbpMean;
#        end
    
    else:
    
        Emat = np.zeros((mp.Fend.corr.Npix, mp.jac.Nmode), dtype=complex)
        modvar = falco.config.Object()
        
        for im in range(mp.jac.Nmode):
            modvar.sbpIndex = mp.jac.sbp_inds[im]
            modvar.zernIndex = mp.jac.zern_inds[im]
            modvar.whichSource = 'star'
            
            # Take the mean over the wavelengths within the sub-bandpass
            EmatSbp = np.zeros((mp.Fend.corr.Npix, mp.Nwpsbp), dtype=complex)
            for wi in range(mp.Nwpsbp):
                modvar.wpsbpIndex = wi
                E2D = falco.model.full(mp, modvar)
                # Actual field in estimation area. Apply spectral weight
                # within the sub-bandpass
                EmatSbp[:, wi] = mp.full.lambda_weights[wi] * \
                    E2D[mp.Fend.corr.maskBool]
            Emat[:, im] = np.sum(EmatSbp, axis=1)
            
    return Emat
    

# Extra function needed to use parfor (because parfor can have only a
# single changing input argument).
def _est_perfect_Efield_with_Zernikes_in_parallel(mp, ilist, inds_list):

    im = inds_list[ilist][0]  # Index of the Jacobian mode
    wi = inds_list[ilist][1]   # Index of the wavelength in the sub-bandpass
    
    modvar = falco.config.Object()
    modvar.sbpIndex = mp.jac.sbp_inds[im]
    modvar.zernIndex = mp.jac.zern_inds[im]
    modvar.wpsbpIndex = wi
    modvar.whichSource = 'star'
    
    E2D = falco.model.full(mp, modvar)

    # Actual field in estimation area. Don't apply spectral weight here.
    return E2D[mp.Fend.corr.maskBool]


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
    # "ev" is passed in only for the Kalman filter. Reset it for the batch
    # process to avoid accidentally using old data.
    # if 'pwp-bp' == mp.estimator.lower():
    #     ev = falco.config.Object()
    
    # Select number of actuators across based on chosen DM for the probing
    if mp.est.probe.whichDM == 1:
        Nact = mp.dm1.Nact
    elif mp.est.probe.whichDM == 2:
        Nact = mp.dm2.Nact
    else:
        raise ValueError('mp.est.probe.whichDM must equal 1 or 2.')
    
    # Store the initial DM commands
    if np.any(mp.dm_ind == 1):
        DM1Vnom = mp.dm1.V
    
    if np.any(mp.dm_ind == 2):
        DM2Vnom = mp.dm2.V
    else:
        DM2Vnom = np.zeros_like(mp.dm1.V)
    
    # Definitions:
    Npairs = mp.est.probe.Npairs  # Number of image PAIRS
    ev.Icube = np.zeros((mp.Fend.Neta, mp.Fend.Nxi, 1+2*Npairs))
    if np.any(mp.dm_ind == 1):
        ev.Vcube1 = np.zeros((mp.dm1.Nact, mp.dm1.Nact, 1+2*Npairs))
    if np.any(mp.dm_ind == 2):
        ev.Vcube2 = np.zeros((mp.dm2.Nact, mp.dm2.Nact, 1+2*Npairs))
    
    # Generate evenly spaced probes along the complex unit circle
    # NOTE: Nprobes=Npairs*2
    probePhaseVec = np.array([0, Npairs])
    for k in range(Npairs-1):
        probePhaseVec = np.append(probePhaseVec, probePhaseVec[-1]-(Npairs-1))
        probePhaseVec = np.append(probePhaseVec, probePhaseVec[-1]+Npairs)
    probePhaseVec = probePhaseVec*np.pi/(Npairs)
    
    badAxisVec = ''
    if mp.est.probe.axis.lower() == 'y':
        for _iter in range(2*Npairs):
            badAxisVec += 'y'
    elif mp.est.probe.axis.lower() == 'x':
        for _iter in range(2*Npairs):
            badAxisVec += 'x'
    elif mp.est.probe.axis.lower() in ('alt', 'xy', 'alternate'):
        for iPair in range(2*Npairs):
            if (iPair+1) % 4 == 1 or (iPair+1) % 4 == 2:
                badAxisVec += 'x'
            elif (iPair+1) % 4 == 3 or (iPair+1) % 4 == 0:
                badAxisVec += 'y'
    elif mp.est.probe.axis.lower() in ('m', 'multi'):
        for _iter in range(2*Npairs):
            badAxisVec += 'm'
    else:
        raise ValueError('Incorrect value for mp.est.probe.axis')
    
    # Initialize output arrays
    ev.Eest = np.zeros((mp.Fend.corr.Npix, mp.Nsbp), dtype=complex)
    ev.IincoEst = np.zeros((mp.Fend.corr.Npix, mp.Nsbp))
    ev.I0mean = 0
    ev.IprobedMean = 0
    
    # Get images and perform estimates in each sub-bandpass
    print('Estimating electric field with batch process estimation ...')
    
    for si in range(mp.Nsbp):
        print('Wavelength: %u/%u ... ' % (si, mp.Nsbp-1))
    
        # Valid for all calls to model_compact.m:
        modvar = falco.config.Object()  # Initialize the new structure
        modvar.sbpIndex = si
        modvar.whichSource = 'star'
    
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
        whichImg = 1
        I0 = falco.imaging.get_sbp_image(mp, si)
        I0vec = I0[mp.Fend.corr.maskBool]  # Vectorize the correction region
        ev.I0mean = ev.I0mean+I0/mp.Nsbp  # Getting Inorm for whole bandpass
    
        # Store values for first image and its DM commands
        ev.Icube[:, :, whichImg] = I0
        if np.any(mp.dm_ind == 1):
            ev.Vcube1[:, :, whichImg] = mp.dm1.V
        if np.any(mp.dm_ind == 2):
            ev.Vcube2[:, :, whichImg] = mp.dm2.V
    
        # Compute the average Inorm in the scoring and correction regions
        ev.InormScore = np.mean(I0[mp.Fend.score.maskBool])
        ev.InormCorr = np.mean(I0[mp.Fend.corr.maskBool])
        print('Measured unprobed Inorm (Corr / Score): %.2e \t%.2e \n' %
              (ev.InormCorr, ev.InormScore))
    
        # Set (approximate) probe intensity based on current measured Inorm
        if mp.flagFiber:
            ev.InormProbeMax = 1e-5
            InormProbe = np.min([np.sqrt(np.max(I0)*1e-8), ev.InormProbeMax])
        else:
            ev.InormProbeMax = 1e-4
            InormProbe = np.min([np.sqrt(np.max(I0vec)*1e-5),
                                 ev.InormProbeMax])
            # Change this to a high percentile value (e.g., 90%) instead of the
            # max to avoid being tricked by noise
        print('Chosen probe intensity: %.2e' % InormProbe)
    
        # Perform the probing
        iOdd = 0  # Initialize index counters
        iEven = 0
        for iProbe in range(2*Npairs):

            # Generate the command map for the probe
            probeCmd = gen_pairwise_probe(mp, InormProbe,
                                probePhaseVec[iProbe], badAxisVec[iProbe])
    
            # Select which DM to use for probing. Allocate probe to that DM
            if mp.est.probe.whichDM == 1:
                dDM1Vprobe = probeCmd/mp.dm1.VtoH  # Now in volts
                dDM2Vprobe = 0
            elif mp.est.probe.whichDM == 2:
                dDM1Vprobe = 0
                dDM2Vprobe = probeCmd/mp.dm1.VtoH  # Now in volts
            else:
                raise ValueError('DM for probing must be 1 or 2.')
                
            if np.any(mp.dm_ind == 1):
                mp.dm1.V = DM1Vnom + dDM1Vprobe
            if np.any(mp.dm_ind == 2):
                mp.dm2.V = DM2Vnom + dDM2Vprobe
                    
            # Take probed image
            if mp.flagFiber:
                Im = falco.imaging.get_sbp_image_fiber(mp, si)
            else:
                Im = falco.imaging.get_sbp_image(mp, si)
            
            # plt.imshow(np.log10(Im)); plt.title('Probed Image %d' % iProbe); plt.colorbar(); plt.pause(1e-2);
            
            ImNonneg = Im
            ImNonneg[Im < 0] = 0
            whichImg = 1+iProbe  # Increment image counter
            # Inorm averaged over all the probed images
            ev.IprobedMean = ev.IprobedMean + \
                np.mean(Im[mp.Fend.corr.maskBool]) / (2*Npairs)
    
            # Store probed image and its DM settings
            ev.Icube[:, :, whichImg] = Im
            if np.any(mp.dm_ind == 1):
                ev.Vcube1[:, :, whichImg] = mp.dm1.V
            if np.any(mp.dm_ind == 2):
                ev.Vcube2[:, :, whichImg] = mp.dm2.V
    
            # Report results
            probeSign = '-+'
            print('Actual Probe %d%s Contrast is: %.2e' % (
                    np.floor(iProbe/2), probeSign[(iProbe+1) % 2],
            np.mean(Im[mp.Fend.corr.maskBool])))

            # Assign image to positive or negative probe collection:
            if (iProbe+1) % 2 == 1:  # Odd; for plus probes
                if np.any(mp.dm_ind == 1):
                    DM1Vplus[:, :, iOdd] = dDM1Vprobe + DM1Vnom
                if np.any(mp.dm_ind == 2):
                    DM2Vplus[:, :, iOdd] = dDM2Vprobe + DM2Vnom
                Iplus[:, iOdd] = Im[mp.Fend.corr.maskBool]
                iOdd += 1
            elif (iProbe+1) % 2 == 0:  # Even; for minus probes
                if np.any(mp.dm_ind == 1):
                    DM1Vminus[:, :, iEven] = dDM1Vprobe + DM1Vnom
                if np.any(mp.dm_ind == 2):
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
        # falco_plot_pairwi1se_probes(mp, ev,
        #         DM1Vplus-repmat(DM1Vnom, [1,1,size(DM1Vplus,3)]),
        #         ampSq2Dcube)
    
        # ################# Perform the estimation ############################
        
        # Use Jacobian for estimation. This is fully model-based if the
        # Jacobian is purely model-based, or it is better if the Jacobian is
        # adaptive based on empirical data.
        if mp.est.flagUseJac:
            
            dEplus = np.zeros_like(Iplus, dtype=complex)
            for iProbe in range(Npairs):
                if mp.est.probe.whichDM == 1:
                    dV = DM1Vplus[:, :, iProbe] - DM1Vnom
                    dEplus[:, iProbe] = np.squeeze(jacStruct.G1[:, :, si]) * \
                        dV[mp.dm1.act_ele]
                elif mp.est.probe.whichDM == 2:
                    dV = DM2Vplus[:, :, iProbe] - DM2Vnom
                    dEplus[:, iProbe] = np.squeeze(jacStruct.G2[:, :, si]) * \
                        dV[mp.dm2.act_ele]

        # Get the probe phase from the model and measure the probe amplitude
        else:

            # For unprobed field based on model:
            if np.any(mp.dm_ind == 1):
                mp.dm1.V = DM1Vnom
            if np.any(mp.dm_ind == 2):
                mp.dm2.V = DM2Vnom
            if mp.flagFiber:
                pass
                # [~, E0] = model_compact(mp, modvar)
            else:
                E0 = falco.model.compact(mp, modvar)

            E0vec = E0[mp.Fend.corr.maskBool]
    
            # For probed fields based on model:
            Eplus = np.zeros_like(Iplus, dtype=complex)
            Eminus = np.zeros_like(Iminus, dtype=complex)
            for iProbe in range(Npairs):
                # For plus probes:
                if np.any(mp.dm_ind == 1):
                    mp.dm1.V = np.squeeze(DM1Vplus[:, :, iProbe])
                if np.any(mp.dm_ind == 2):
                    mp.dm2.V = np.squeeze(DM2Vplus[:, :, iProbe])
                if(mp.flagFiber):
                    pass
                    # [~, Etemp] = model_compact(mp, modvar);
                else:
                    Etemp = falco.model.compact(mp, modvar)
                Eplus[:, iProbe] = Etemp[mp.Fend.corr.maskBool]
                
                # For minus probes:
                if np.any(mp.dm_ind == 1):
                    mp.dm1.V = np.squeeze(DM1Vminus[:, :, iProbe])
                if np.any(mp.dm_ind == 2):
                    mp.dm2.V = np.squeeze(DM2Vminus[:, :, iProbe])
                if mp.flagFiber:
                    pass
                    # [~, Etemp] = model_compact(mp, modvar)
                else:
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
        
        if (mp.estimator.lower() == 'pwp-bp') or \
            (mp.estimator.lower() == 'pwp-kf' and ev.Itr < mp.est.ItrStartKF):
    
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
            # !!!!!!!!!!!!!!BE VERY CAREFUL WITH THIS HARD-CODED VALUE!!!!!!!!!
            Eest[np.abs(Eest)**2 > 1e-2] = 0.0
            
            print('%d of %d pixels were given zero probe amplitude.' %
                    (zerosCounter, mp.Fend.corr.Npix))
        
            # Initialize the state and state covariance estimates for Kalman
            # filter. The state is the real and imag parts of the E-field.
            if mp.estimator.lower() == 'pwp-kf':
                # Re-organize the batch-processed E-field estimate into the 1st
                # state estimate for the Kalman filter
                xOld = np.zeros((2*mp.Fend.corr.Npix, 1))
                for ii in range(mp.Fend.corr.Npix):
                    xOld[2*(ii-1)+0:2*(ii-1)+1] = np.array([np.real(Eest[ii]),
                                                            np.imag(Eest[ii])])
                ev.xOld = xOld  # Save out for returning later
    
                # Initialize the state covariance matrix
                # (2x2 for each dark hole pixel)
                ev.Pold_KF_array = np.tile(mp.est.Pcoef0*np.eye(2),
                                           (mp.Fend.corr.Npix, 1, mp.Nsbp))
    
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Begin Kalman Filter Update
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #
        # To be completed later...
        #
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # End Kalman Filter Update
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        # Save out the estimates
        ev.Eest[:, si] = Eest
        Iest = np.abs(Eest)**2
        ev.IincoEst[:, si] = I0vec - Iest  # incoherent light
    
    # Other data to save out
    ev.ampSqMean = np.mean(ampSq)  # Mean probe intensity
    ev.ampNorm = amp/np.sqrt(InormProbe)  # Normalized probe amplitude maps
    
    # Calculate the mean normalized intensity over the whole dark hole at all
    # wavelengths.
    ev.InormEst = np.mean(Iest)
    
    # Reset DM commands to their values before probing
    if np.any(mp.dm_ind == 1):
        mp.dm1.V = DM1Vnom
    
    if np.any(mp.dm_ind == 2):
        mp.dm2.V = DM2Vnom
    
    print('Completed pairwise probing estimation.')
 
    pass
    # return None


def gen_pairwise_probe(mp, InormDes, psi, badAxis):
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
    badAxis : TYPE
        DESCRIPTION.

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
        np.round(mp.est.probe.offsetX)/Nact
    ys = np.arange(-(Nact-1)/2, (Nact+1)/2)/Nact - \
        np.round(mp.est.probe.offsetY)/Nact
    [XS, YS] = np.meshgrid(xs, ys)
    
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
    
    elif badAxis.lower() == 'm':
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
