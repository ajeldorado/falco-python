"""Functions for generating images in FALCO."""
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
# from concurrent.futures import ProcessPoolExecutor as PoolExecutor
import matplotlib.pyplot as plt
from pyHCIT_hwControl.interface import TestbedInterface

import falco
from falco import check


def add_noise_to_subband_image(mp, imageIn, iSubband):
    """
    Add noise (photon shot, dark current, & read) to a simulated image.

    Parameters
    ----------
    mp : ModelParameters
        structure of model parameters.
    imageIn : array_like
        2-Dnoiseless starting image for a given subband [normalized intensity]
    iSubband : int
        index of subband in which the image was taken

    Returns
    -------
    imageOut : array_like
        2-D noisy image [normalized intensity]
    """
    check.twoD_array(imageIn, 'imageIn', ValueError)
    check.nonnegative_scalar_integer(iSubband, 'iSubband', ValueError)

    peakCounts = (mp.detector.peakFluxVec[iSubband] *
                  mp.detector.tExpVec[iSubband])
    peakElectrons = mp.detector.gain * peakCounts

    imageInElectrons = peakElectrons * imageIn

    imageInCounts = 0
    for iExp in range(mp.detector.Nexp):

        # Add photon shot noise
        noisyImageInElectrons = np.random.poisson(imageInElectrons)

        # Compute dark current
        darkCurrent = (mp.detector.darkCurrentRate *
                       mp.detector.tExpVec[iSubband] *
                       np.ones_like(imageIn))
        darkCurrent = np.random.poisson(darkCurrent)

        # Compute Gaussian read noise
        readNoise = (mp.detector.readNoiseStd *
                     np.random.randn(imageIn.shape[0], imageIn.shape[1]))

        # Convert back from e- to counts and then discretize
        imageInCounts = (imageInCounts +
                         np.round((noisyImageInElectrons +
                                   darkCurrent + readNoise) /
                                  mp.detector.gain)/mp.detector.Nexp)

    # Convert back from counts to normalized intensity
    imageOut = imageInCounts / peakCounts

    return imageOut


def calc_thput(mp):
    """
    Calculate the off-axis throughput of the coronagraph.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters

    Returns
    -------
    thput : float
        Off-axis throughput of the coronagraph at the specified field location.
    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')

    ImSimOffaxis = falco.imaging.get_sim_offaxis_image_compact(
        mp, mp.thput_eval_x, mp.thput_eval_y, isEvalMode=True)

    # Absolute energy within half-max isophote(s)
    if mp.thput_metric.lower() == 'hmi':
        maskHM = np.zeros(mp.Fend.eval.RHOS.shape, dtype=bool)
        maskHM[ImSimOffaxis >= 0.5*np.max(ImSimOffaxis)] = True
        thput = (np.sum(ImSimOffaxis[maskHM == 1]) /
                 mp.sumPupil *
                 np.mean(mp.Fend.eval.I00))
        print('Core throughput within the half-max isophote(s) = %.2f%% \tat '
              'separation = (%.1f, %.1f) lambda0/D.' %
              (100*thput, mp.thput_eval_x, mp.thput_eval_y))

    # Absolute energy encircled within a given radius
    elif mp.thput_metric.lower() in ('ee', 'e.e.'):
        # (x,y) location [lambda0/D] in dark hole at which to evaluate thput
        maskEE = np.zeros(mp.Fend.eval.RHOS.shape, dtype=bool)
        maskEE[mp.Fend.eval.RHOS <= mp.thput_radius] = True
        thput = (np.sum(ImSimOffaxis[maskEE == 1]) /
                 mp.sumPupil *
                 np.mean(mp.Fend.eval.I00))
        print('E.E. throughput within a %.2f lambda/D radius = %.2f%% \tat '
              'separation = (%.1f, %.1f) lambda/D.' %
              (mp.thput_radius, 100*thput, mp.thput_eval_x, mp.thput_eval_y))

    return thput, ImSimOffaxis


def calc_psf_norm_factor(mp):
    """
    Get the intensity normalization factor for each model at each sub-band.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters

    Returns
    -------
    None
        Changes are made by reference to the structure mp

    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')

    mp.sumPupil = np.sum(np.abs(
        mp.P1.compact.mask * falco.util.pad_crop(
            np.mean(mp.P1.compact.E, axis=2), mp.P1.compact.mask.shape))**2)

    # Initialize Model Normalizations
    if not hasattr(mp.Fend, 'compact'):
        mp.Fend.compact = falco.config.Object()
    if not hasattr(mp.Fend, 'eval'):
        mp.Fend.eval = falco.config.Object()
    if not hasattr(mp.Fend, 'full'):
        mp.Fend.full = falco.config.Object()
    mp.Fend.compact.I00 = np.ones(mp.Nsbp)
    mp.Fend.eval.I00 = np.ones(mp.Nsbp)
    mp.Fend.full.I00 = np.ones((mp.Nsbp, mp.Nwpsbp))

    modvar = falco.config.ModelVariables()
    modvar.zernIndex = 1
    modvar.whichSource = 'star'
    modvar.starIndex = 0  # Always use first star for image normalization

    # Compact Model Normalizations
    for si in range(mp.Nsbp):
        modvar.sbpIndex = si
        Ecompact = falco.model.compact(mp, modvar, isNorm=False)
        mp.Fend.compact.I00[si] = (np.abs(Ecompact)**2).max()

    # Compact Evaluation Model Normalizations
    for si in range(mp.Nsbp):
        modvar.sbpIndex = si
        Eeval = falco.model.compact(mp, modvar, isNorm=False, isEvalMode=True)
        mp.Fend.eval.I00[si] = (np.abs(Eeval)**2).max()

    # Full Model Normalizations (at points for entire-bandpass evaluation)
    if mp.flagSim:
        if mp.flagParallel:
            # Make all combinations of the values
            inds_list = [(x, y) for x in range(mp.Nsbp)
                         for y in range(mp.Nwpsbp)]
            Nvals = mp.Nsbp*mp.Nwpsbp

            # pool = multiprocessing.Pool(processes=mp.Nthreads)
            # resultsRaw = pool.starmap(
            #     _model_full_norm_wrapper,
            #     [(mp, ilist, inds_list) for ilist in range(Nvals)]
            # )
            # pool.close()
            # pool.join()

            with PoolExecutor(max_workers=mp.Nthreads) as executor:
                result = executor.map(
                    lambda p: _model_full_norm_wrapper(*p),
                    [(mp, ilist, inds_list) for ilist in range(Nvals)]
                )
            resultsRaw = tuple(result)

            I00list = resultsRaw

            for ilist in range(Nvals):
                si = inds_list[ilist][0]
                wi = inds_list[ilist][1]
                mp.Fend.full.I00[si, wi] = I00list[ilist]

        else:
            for si in range(mp.Nsbp):
                for wi in range(mp.Nwpsbp):
                    modvar.sbpIndex = si
                    modvar.wpsbpIndex = wi
                    Efull = falco.model.full(mp, modvar, isNorm=False)
                    mp.Fend.full.I00[si, wi] = np.max(np.abs(Efull)**2)

    # Visually verify the normalized coronagraphic PSF
    if mp.flagPlot:
        modvar = falco.config.ModelVariables()  # reset
        modvar.sbpIndex = mp.si_ref
        modvar.wpsbpIndex = mp.wi_ref
        modvar.whichSource = 'star'
        modvar.starIndex = 0
        modvar.zernIndex = 1

        E0c = falco.model.compact(mp, modvar)
        I0c = np.abs(E0c)**2

        plt.figure(501)
        plt.clf()
        plt.imshow(np.log10(I0c))
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.title('Compact Model Normalization')
        plt.pause(1e-2)

        E0f = falco.model.full(mp, modvar)
        I0f = np.abs(E0f)**2

        plt.figure(502)
        plt.clf()
        plt.imshow(np.log10(I0f))
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.title('Full Model Normalization')
        plt.pause(1e-2)


def _model_full_norm_wrapper(mp, ilist, inds_list):
    """Use only with calc_psf_norm_factor for parallel processing."""
    si = inds_list[ilist][0]
    wi = inds_list[ilist][1]

    modvar = falco.config.ModelVariables()
    modvar.sbpIndex = si  # mp.full.indsLambdaMat[ilam, 0]
    modvar.wpsbpIndex = wi  # mp.full.indsLambdaMat[ilam, 1]
    modvar.zernIndex = 1
    modvar.whichSource = 'star'
    modvar.starIndex = 0

    Etemp = falco.model.full(mp, modvar, isNorm=False)
    return np.max(np.abs(Etemp)**2)


def get_summed_image(mp, tb = None):
    """
    Get the broadband image over the entire bandpass.

    Get a broadband image over the entire bandpass by getting the sub-bandpass
    images and doing a weighted sum.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    tb: pyHCIT_hwControl.interface.TestbedInterface
        (Optional) control interface for a physical testbed

    Returns
    -------
    Imean : numpy ndarray
        band-averaged image in units of normalized intensity

    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')

    if not (mp.flagParallel and mp.flagSim):
        summedImage = 0
        for si in range(mp.Nsbp):
            summedImage += mp.sbp_weights[si] * get_sbp_image(mp, si, tb)

    else:  # Compute simulated images in parallel

        # Initializations
        vals_list = [(ilam, pol) for ilam in range(mp.full.NlamUnique)
                     for pol in mp.full.pol_conds]
        Nvals = mp.full.NlamUnique*len(mp.full.pol_conds)

#         result = map(
#             lambda p: _get_single_sim_full_image(*p),
#             [(mp, ilist, vals_list) for ilist in range(Nvals)]
#         )
#         result_image = tuple(result)

        # mp.vals_list = vals_list
        # with PoolExecutor(max_workers=mp.Nthreads) as executor:
        #     result = executor.map(_get_single_sim_full_image_one_arg, range(Nvals))
        # result_image = tuple(result)

        # # # Failing numerical test
        # mp.vals_list = vals_list
        # with PoolExecutor(max_workers=mp.Nthreads) as executor:
        #     result = executor.map(
        #         _get_single_sim_full_image_one_arg,
        #         [mp for mp.ilist in range(Nvals)]
        #     )
        # result_image = tuple(result)

        with PoolExecutor(max_workers=mp.Nthreads) as executor:
            result = executor.map(
                lambda p: _get_single_sim_full_image(*p),
                [(mp, ilist, vals_list) for ilist in range(Nvals)]
            )
        result_image = tuple(result)

        # # pool = multiprocessing.get_context("spawn").Pool(processes=mp.Nthreads)
        # pool = multiprocessing.Pool(processes=mp.Nthreads)
        # result_image = pool.starmap(_get_single_sim_full_image,
        #                         [(mp, ilist, vals_list) for ilist in range(Nvals)])
        # pool.close()
        # pool.join()

        # Apply the spectral weights and sum
        summedImage = 0
        for ilist in np.arange(Nvals, dtype=int):
            ilam = vals_list[ilist][0]
            # pol = vals_list[ilist][1]
            summedImage += mp.full.lambda_weights_all[ilam] / \
                len(mp.full.pol_conds) * result_image[ilist]

    return summedImage


def _get_single_sim_full_image_one_arg(mp):
    """Use only with get_summed_image."""
    ilist = mp.ilist
    vals_list = mp.vals_list
    ilam = vals_list[ilist][0]
    pol = vals_list[ilist][1]

    modvar = falco.config.ModelVariables()  # Initialize the new structure
    modvar.sbpIndex = mp.full.indsLambdaMat[mp.full.indsLambdaUnique[ilam], 0]
    modvar.wpsbpIndex = mp.full.indsLambdaMat[mp.full.indsLambdaUnique[ilam],
                                              1]
    mp.full.polaxis = pol  # mp.full.pol_conds[ipol]
    modvar.whichSource = 'star'
    modvar.starIndex = 0
    Estar = falco.model.full(mp, modvar)

    return np.abs(Estar)**2  # Apply spectral weighting outside this function


def _get_single_sim_full_image(mp, ilist, vals_list):
    """Use only with get_summed_image."""
    ilam = vals_list[ilist][0]
    pol = vals_list[ilist][1]

    modvar = falco.config.ModelVariables()  # Initialize the new structure
    modvar.sbpIndex = mp.full.indsLambdaMat[mp.full.indsLambdaUnique[ilam], 0]
    modvar.wpsbpIndex = mp.full.indsLambdaMat[mp.full.indsLambdaUnique[ilam],
                                              1]
    mp.full.polaxis = pol  # mp.full.pol_conds[ipol]
    modvar.whichSource = 'star'
    modvar.starIndex = 0
    Estar = falco.model.full(mp, modvar)

    return np.abs(Estar)**2  # Apply spectral weighting outside this function


def get_sbp_image(mp, si, tb = None):
    """
    Get an image in the specified sub-bandpass.

    Get an image in the specified sub-bandpass. Wrapper for functions to get a
    simulated image or a testbed image.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    si: int
        Index of sub-bandpass for which to take the image
    tb: pyHCIT_hwControl.interface.TestbedInterface or None
        (Optional) Control interface for a physical testbed

    Returns
    -------
    Isbp : numpy ndarray
        Sub-bandpass image in units of normalized intensity

    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    check.nonnegative_scalar_integer(si, 'si', TypeError)

    if mp.flagSim:
        Isbp = get_sim_sbp_image(mp, si)
    else:
        if type(tb) is not TestbedInterface:
            raise TypeError('Input "tb" must be of type TestbedInterface')
        tb.dm.apply(mp.dm1.V)
        Isbp = tb.get_sbp_image(si)

    return Isbp


def get_sim_sbp_image(mp, si):
    """
    Get a simulated image in the specified sub-bandpass.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    si: int
        Index of sub-bandpass for which to take the image

    Returns
    -------
    subbandImage
        Simulated sub-bandpass image in units of normalized intensity
    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    check.nonnegative_scalar_integer(si, 'si', TypeError)

    Npol = len(mp.full.pol_conds)  # Number of polarization states used

    # Loop over all wavelengths and polarizations and stars
    inds_list = [(x, y, z) for x in range(mp.Nwpsbp) for y in range(Npol)
                 for z in range(mp.star.count)]
    Ncombos = mp.Nwpsbp*Npol*mp.star.count

    Iall = np.zeros((Ncombos, mp.Fend.Neta, mp.Fend.Nxi))
    if mp.flagParallel:

        # pool = multiprocessing.Pool(processes=mp.Nthreads)
        # resultsRaw = pool.starmap(_get_subband_image_component,
        #                           [(mp, si, ilist, inds_list)
        #                            for ilist in range(Ncombos)])
        # results = resultsRaw
        # pool.close()
        # pool.join()

        # with PoolExecutor(max_workers=mp.Nthreads) as executor:
        #     result = executor.map(
        #         _get_single_sim_full_image_one_arg,
        #         [mp for mp.ilist in range(Ncombos)],
        #     )
        # results = tuple(result)

        with PoolExecutor(max_workers=mp.Nthreads) as executor:
            result = executor.map(
                lambda p: _get_subband_image_component(*p),
                [(mp, si, ilist, inds_list) for ilist in range(Ncombos)]
            )
        results = tuple(result)

        for ilist in range(Ncombos):
            Iall[ilist, :, :] = results[ilist]

    else:
        for iCombo in range(Ncombos):
            Iall[iCombo, :, :] = _get_subband_image_component(
                mp, si, iCombo, inds_list)

    subbandImage = 0
    for iCombo in range(Ncombos):
        subbandImage += np.squeeze(Iall[iCombo, :, :])

    if mp.flagImageNoise:
        subbandImage = add_noise_to_subband_image(mp, subbandImage, si)

    return subbandImage


def _get_subband_image_component(mp, si, ilist, inds_list):
    """
    Use only with get_sim_sbp_image for parallel processing.

    Function to return the weighted, normalized intensity image at a
    given wavelength in the specified sub-bandpass.
    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    check.nonnegative_scalar_integer(si, 'si', TypeError)

    wi = inds_list[ilist][0]
    ipol = inds_list[ilist][1]
    iStar = inds_list[ilist][2]

    # Get the starlight image
    modvar = falco.config.ModelVariables()
    modvar.sbpIndex = si
    modvar.wpsbpIndex = wi
    mp.full.polaxis = mp.full.pol_conds[ipol]
    modvar.whichSource = 'star'
    modvar.starIndex = iStar
    Estar = falco.model.full(mp, modvar)

    # Apply weight within the sub-bandpass.
    # Assume polarizations are evenly weighted.
    Iout = mp.full.lambda_weights[wi]/len(mp.full.pol_conds)*np.abs(Estar)**2

    return Iout


def get_expected_summed_image(mp, cvar, dDM):
    """
    Generate the expected broadband image after a new DM command.

    Function to generate the expected broadband image over the entire bandpass
    by adding the model-based delta electric field on top of the current
    E-field estimate in each sub-bandpass.

    Parameters
    ----------
    mp : falco.config.ModelParameters
        Structure of model parameters
    cvar : ModelParameters
        Structure of controller variables
    dDM : ModelParameters
        Structure of delta DM commands from the controller

    Returns
    -------
    Ibandavg : numpy ndarray
        Expected bandpass-averaged image in units of normalized intensity
    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')

    if(any(mp.dm_ind == 1)):
        DM1V0 = mp.dm1.V.copy()
    if(any(mp.dm_ind == 2)):
        DM2V0 = mp.dm2.V.copy()
    if(any(mp.dm_ind == 8)):
        DM8V0 = mp.dm8.V.copy()
    if(any(mp.dm_ind == 9)):
        DM9V0 = mp.dm9.V.copy()

    # Initialize variables
    Ibandavg = 0
    EnewTempVecArray = np.zeros((mp.Fend.corr.Npix, mp.Nsbp), dtype=complex)
    EoldTempVecArray = np.zeros((mp.Fend.corr.Npix, mp.Nsbp), dtype=complex)

    # Generate the model-based E-field with the new DM setting
    modvar = falco.config.ModelVariables()  # Initialize the new structure
    modvar.whichSource = 'star'
    for si in range(mp.Nsbp):
        modvar.sbpIndex = si
        Etemp = falco.model.compact(mp, modvar)
        EnewTempVecArray[:, si] = Etemp[mp.Fend.corr.maskBool]

    # Revert to the previous DM commands
    if any(mp.dm_ind == 1):
        mp.dm1.V = mp.dm1.V - dDM.dDM1V
    if any(mp.dm_ind == 2):
        mp.dm2.V = mp.dm2.V - dDM.dDM2V
    if any(mp.dm_ind == 8):
        mp.dm8.V = mp.dm8.V - dDM.dDM8V
    if any(mp.dm_ind == 9):
        mp.dm9.V = mp.dm9.V - dDM.dDM9V

    # Generate the model-based E-field with the previous DM setting
    modvar.whichSource = 'star'
    modvar.wpsbpIndex = 0  # Dummy, placeholder value
    for si in range(mp.Nsbp):
        modvar.sbpIndex = si
        Etemp = falco.model.compact(mp, modvar)
        EoldTempVecArray[:, si] = Etemp[mp.Fend.corr.maskBool]

    # Compute the expected new 2-D intensity image
    for si in range(mp.Nsbp):
        EexpectedVec = cvar.Eest[:, si] + \
            (EnewTempVecArray[:, si] - EoldTempVecArray[:, si])
        Eexpected2D = np.zeros((mp.Fend.Neta, mp.Fend.Nxi), dtype=complex)
        Eexpected2D[mp.Fend.corr.maskBool] = EexpectedVec

        Ibandavg += mp.sbp_weights[si]*np.abs(Eexpected2D)**2

    # Reset voltage commands in mp
    if any(mp.dm_ind == 1):
        mp.dm1.V = DM1V0
    if any(mp.dm_ind == 2):
        mp.dm2.V = DM2V0
    if any(mp.dm_ind == 8):
        mp.dm8.V = DM8V0
    if any(mp.dm_ind == 9):
        mp.dm9.V = DM9V0

    return Ibandavg


def get_sim_offaxis_image_compact(mp, x_offset, y_offset, isEvalMode=False):
    """
    Return the broadband intensity for the compact model.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    x_offset: int
        lateral offset (in xi) of the stellar PSF in the focal plane.
        [lambda0/D]
    y_offset: int
        vertical offset (in eta) of the stellar PSF in the focal plane.
        [lambda0/D]
    isEvalMode : bool
       Switch that tells function to run at a higher final focal plane
       resolution when evaluating throughput.

    Returns
    -------
    Iout : numpy ndarray
        Simulated bandpass-averaged intensity from the compact model
    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    check.real_scalar(x_offset, 'x_offset', TypeError)
    check.real_scalar(y_offset, 'y_offset', TypeError)
    if not isinstance(isEvalMode, bool):
        raise TypeError('isEvalMode must be a bool')

    modvar = falco.config.ModelVariables()
    modvar.whichSource = 'offaxis'
    modvar.x_offset = x_offset
    modvar.y_offset = y_offset

    Iout = 0.
    for iStar in range(mp.star.count):
        modvar.starIndex = iStar

        for si in range(mp.Nsbp):
            modvar.sbpIndex = si
            modvar.zernIndex = 1

            E2D = falco.model.compact(mp, modvar, isEvalMode=isEvalMode)
            Iout += (np.abs(E2D)**2) * mp.jac.weightMat[si, 0]

    return Iout


def get_testbed_sbp_image(mp, si):
    """
    Get an image in the specified sub-bandpass from a testbed.

    This function calls an equivalent sub-function depending on mp.testbed.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    si: int
        Index of sub-bandpass for which to take the image

    Returns
    -------
    TBD
        Normalized intensity in the sub-bandpass
               (i.e. approximate raw contrast but normalized
           by a photometry measurement at a single offset)

    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    check.nonnegative_scalar_integer(si, 'si', TypeError)

    raise NotImplementedError('Testbed functionality not implemented yet.')

    return None
