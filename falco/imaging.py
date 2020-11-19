"""Functions for generating images in FALCO."""
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

import falco
from falco import check


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

    ImSimOffaxis = falco.imaging.get_sim_offaxis_image_compact(mp,
                            mp.thput_eval_x, mp.thput_eval_y, isEvalMode=True)

    if mp.thput_metric.lower() == 'hmi':  # Absolute energy within half-max isophote(s)
        maskHM = np.zeros(mp.Fend.eval.RHOS.shape, dtype=bool)
        maskHM[ImSimOffaxis >= 0.5*np.max(ImSimOffaxis)] = True
        thput = np.sum(ImSimOffaxis[maskHM == 1])/mp.sumPupil*np.mean(mp.Fend.eval.I00)
        print('Core throughput within the half-max isophote(s) = %.2f%% \tat separation = (%.1f, %.1f) lambda0/D.' %
              (100*thput, mp.thput_eval_x, mp.thput_eval_y))
            
    elif (mp.thput_metric.lower() == 'ee') or (mp.thput_metric.lower() == 'e.e.'):  # Absolute energy encircled within a given radius
        # (x,y) location [lambda_c/D] in dark hole at which to evaluate throughput
        maskEE = np.zeros(mp.Fend.eval.RHOS.shape, dtype=bool)
        maskEE[mp.Fend.eval.RHOS <= mp.thput_radius] = True
        thput = np.sum(ImSimOffaxis[maskEE == 1])/mp.sumPupil*np.mean(mp.Fend.eval.I00)
        print('E.E. throughput within a %.2f lambda/D radius = %.2f%% \tat separation = (%.1f, %.1f) lambda/D.' %
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

    modvar = falco.config.Object()
    modvar.zernIndex = 1
    modvar.whichSource = 'star'
    
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
    if(mp.flagSim):
        if mp.flagMultiproc:
            # Make all combinations of the values
            inds_list = [(x, y) for x in range(mp.Nsbp)
                         for y in range(mp.Nwpsbp)]
            Nvals = mp.Nsbp*mp.Nwpsbp
            
            pool = multiprocessing.Pool(processes=mp.Nthreads)
            resultsRaw = pool.starmap(_model_full_norm_wrapper, [(mp, ilist, inds_list) for ilist in range(Nvals)])
            I00list = resultsRaw
            pool.close()
            pool.join()
        
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
                    mp.Fend.full.I00[si, wi] = (np.abs(Efull)**2).max()
    
    # Visually verify the normalized coronagraphic PSF
    if mp.flagPlot:
        modvar = falco.config.Object()  # reset
        modvar.ttIndex = 1
        modvar.sbpIndex = mp.si_ref
        # modvar.wpsbpIndex = mp.wi_ref
        modvar.whichSource = 'star'
        E0c = falco.model.compact(mp, modvar)
        I0c = np.abs(E0c)**2
    
        plt.figure(501)
        plt.clf()
        plt.imshow(np.log10(I0c)); plt.colorbar()
        plt.gca().invert_yaxis()
        plt.title('Compact Model Normalization'); plt.pause(1e-2)
       
        modvar = falco.config.Object()  # reset
        modvar.ttIndex = 1
        modvar.sbpIndex = mp.si_ref
        modvar.wpsbpIndex = mp.wi_ref
        modvar.whichSource = 'star'
        E0f = falco.model.full(mp, modvar)
        I0f = np.abs(E0f)**2
        plt.figure(502);
        plt.clf()
        plt.imshow(np.log10(I0f)); plt.colorbar()
        plt.gca().invert_yaxis()
        plt.title('Full Model Normalization'); plt.pause(1e-2)


def _model_full_norm_wrapper(mp, ilist, inds_list):
    """Use only with calc_psf_norm_factor for parallel processing."""
    si = inds_list[ilist][0]
    wi = inds_list[ilist][1]
    
    modvar = falco.config.Object()
    modvar.sbpIndex = si  # mp.full.indsLambdaMat[ilam, 0]
    modvar.wpsbpIndex = wi  # mp.full.indsLambdaMat[ilam, 1]
    modvar.zernIndex = 1
    modvar.whichSource = 'star'
    
    Etemp = falco.model.full(mp, modvar, isNorm=False)
    return np.max(np.abs(Etemp)**2)


def get_summed_image(mp):
    """
    Get the broadband image over the entire bandpass.
    
    Get a broadband image over the entire bandpass by getting the sub-bandpass
    images and doing a weighted sum.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters

    Returns
    -------
    Imean : numpy ndarray
        band-averaged image in units of normalized intensity

    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
   
    if not (mp.flagMultiproc and mp.flagSim):
        Imean = 0
        for si in range(0, mp.Nsbp):
            Imean += mp.sbp_weights[si]*get_sbp_image(mp, si)
            
    else:  # Compute simulated images in parallel
        
        # Initializations
        vals_list = [(ilam, pol) for ilam in range(mp.full.NlamUnique) for pol in mp.full.pol_conds]
        Nvals = mp.full.NlamUnique*len(mp.full.pol_conds)
            
        pool = multiprocessing.Pool(processes=mp.Nthreads)
        results = pool.starmap(_get_single_sim_full_image, [(mp, ilist, vals_list) for ilist in range(Nvals)])
        results_img = results
        pool.close()
        pool.join()
        
        # Apply the spectral weights and sum
        Imean = 0
        for ilist in np.arange(Nvals, dtype=int):
            ilam = vals_list[ilist][0]
            # pol = vals_list[ilist][1]
            Imean += mp.full.lambda_weights_all[ilam] / \
                len(mp.full.pol_conds) * results_img[ilist]
            
    return Imean
   
         
def _get_single_sim_full_image(mp, ilist, vals_list):
    """Use only with get_summed_image."""
    ilam = vals_list[ilist][0]
    pol = vals_list[ilist][1]
    
    modvar = falco.config.Object()  # Initialize the new structure
    modvar.sbpIndex = mp.full.indsLambdaMat[mp.full.indsLambdaUnique[ilam], 0]
    modvar.wpsbpIndex = mp.full.indsLambdaMat[mp.full.indsLambdaUnique[ilam], 1]
    mp.full.polaxis = pol  # mp.full.pol_conds[ipol]
    modvar.whichSource = 'star'
    Estar = falco.model.full(mp, modvar)
    
    return np.abs(Estar)**2  # Apply spectral weighting outside this function


def get_sbp_image(mp, si):
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
        Isbp = falco_get_testbed_sbp_image(mp, si)

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
    Isbp
        Simulated sub-bandpass image in units of normalized intensity
    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    check.nonnegative_scalar_integer(si, 'si', TypeError)

    Npol = len(mp.full.pol_conds)  # Number of polarization states used
    
    # Loop over all wavelengths and polarizations
    inds_list = [(x, y) for x in range(mp.Nwpsbp) for y in range(Npol)]
    Nvals = mp.Nwpsbp*Npol
        
    Iall = np.zeros((Nvals, mp.Fend.Neta, mp.Fend.Nxi))
    if mp.flagMultiproc:
        pool = multiprocessing.Pool(processes=mp.Nthreads)
        #resultsRaw = [pool.apply_async(_get_single_sbp_image_wvlPol,args=(mp, si, ilist, inds_list)) for ilist in range(Nvals)]
        #results = [p.get() for p in resultsRaw]  # All the images in a list
        resultsRaw = pool.starmap(_get_single_sbp_image_wvlPol,[(mp, si, ilist, inds_list) for ilist in range(Nvals)])
        results = resultsRaw
        
        pool.close()
        pool.join()

        for ilist in range(Nvals):
            Iall[ilist, :, :] = results[ilist]
            
    else:
        for ilist in range(Nvals):
            Iall[ilist, :, :] = _get_single_sbp_image_wvlPol(mp, si, ilist, inds_list)

    # Apply the spectral weights and sum
    Isbp = 0
    for ilist in range(Nvals):
        Isbp = Isbp + np.squeeze(Iall[ilist, :, :])

#    # Loop over all wavelengths to get the starlight image
#    Isbp = 0 # Initialize the image sum in the sub-bandpass
#    modvar = falco.config.Object() # Initialize the new structure
#    for wi in range(mp.Nwpsbp):
#        modvar.sbpIndex   = si
#        modvar.wpsbpIndex = wi
#        modvar.whichSource = 'star'
#        Estar = falco.model.full(mp, modvar)
#        Iout = np.abs(Estar)**2 # Apply spectral weighting outside this function
#
#        # Optionally include the planet PSF
#        if(mp.planetFlag):
#            modvar.whichSource = 'exoplanet'
#            Eplanet = falco.model.full(mp,modvar)
#            Iout = Iout + np.abs(Eplanet)**2 # Apply spectral weighting outside this function
#
#        # Apply weight within the sub-bandpass. Assume polarizations are evenly weigted.
#        Iout = mp.full.lambda_weights[wi]*Iout #mp.full.lambda_weights(wi)/length(mp.full.pol_conds)*Iout;
#        Isbp += Iout
        
    return Isbp

    
def _get_single_sbp_image_wvlPol(mp, si, ilist, inds_list):
    """
    Called only by get_sim_sbp_image for parallel processing.
    
    Function to return the weighted, normalized intensity image at a
    given wavelength in the specified sub-bandpass.
    
    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    check.nonnegative_scalar_integer(si, 'si', TypeError)
    
    wi = inds_list[ilist][0]
    ipol = inds_list[ilist][1]

    # Get the starlight image
    modvar = falco.config.Object()
    modvar.sbpIndex = si
    modvar.wpsbpIndex = wi
    mp.full.polaxis = mp.full.pol_conds[ipol]
    modvar.whichSource = 'star'
    Estar = falco.model.full(mp, modvar)
    Iout = np.abs(Estar)**2  # Apply spectral weighting outside this function

    # Optionally include the planet PSF
    if(mp.planetFlag):
        modvar.whichSource = 'exoplanet'
        Eplanet = falco.model.full(mp, modvar)
        # Apply spectral weighting outside this function
        Iout = Iout + np.abs(Eplanet)**2

    # Apply weight within the sub-bandpass.
    # Assume polarizations are evenly weighted.
    Iout = mp.full.lambda_weights[wi]/len(mp.full.pol_conds)*Iout
    
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
        
    if(any(mp.dm_ind == 1)): DM1V0 = mp.dm1.V.copy()
    if(any(mp.dm_ind == 2)): DM2V0 = mp.dm2.V.copy()
    if(any(mp.dm_ind == 8)): DM8V0 = mp.dm8.V.copy()
    if(any(mp.dm_ind == 9)): DM9V0 = mp.dm9.V.copy()
        
    # Initialize variables
    Ibandavg = 0
    EnewTempVecArray = np.zeros((mp.Fend.corr.Npix, mp.Nsbp), dtype=complex)
    EoldTempVecArray = np.zeros((mp.Fend.corr.Npix, mp.Nsbp), dtype=complex)

    # Generate the model-based E-field with the new DM setting
    modvar = falco.config.Object()  # Initialize the new structure
    modvar.whichSource = 'star'
    modvar.wpsbpIndex = 0  # Dummy, placeholder value
    for si in range(mp.Nsbp):
        modvar.sbpIndex = si
        Etemp = falco.model.compact(mp, modvar)
        EnewTempVecArray[:, si] = Etemp[mp.Fend.corr.maskBool]
    
    # Revert to the previous DM commands
    if(any(mp.dm_ind == 1)): mp.dm1.V = mp.dm1.V - dDM.dDM1V
    if(any(mp.dm_ind == 2)): mp.dm2.V = mp.dm2.V - dDM.dDM2V
    if(any(mp.dm_ind == 8)): mp.dm8.V = mp.dm8.V - dDM.dDM8V
    if(any(mp.dm_ind == 9)): mp.dm9.V = mp.dm9.V - dDM.dDM9V
        
    # Generate the model-based E-field with the previous DM setting
    modvar.whichSource = 'star'
    modvar.wpsbpIndex = 0  # Dummy, placeholder value
    for si in range(mp.Nsbp):
        modvar.sbpIndex = si
        Etemp = falco.model.compact(mp, modvar)
        EoldTempVecArray[:, si] = Etemp[mp.Fend.corr.maskBool]
    
    # Compute the expected new 2-D intensity image
    for si in range(mp.Nsbp):
        EexpectedVec = cvar.EfieldVec[:, si] + \
            (EnewTempVecArray[:, si] - EoldTempVecArray[:, si])
        Eexpected2D = np.zeros((mp.Fend.Neta, mp.Fend.Nxi), dtype=complex)
        Eexpected2D[mp.Fend.corr.maskBool] = EexpectedVec
        
        Ibandavg += mp.sbp_weights[si]*np.abs(Eexpected2D)**2
    
    # Reset voltage commands in mp
    if(any(mp.dm_ind == 1)): mp.dm1.V = DM1V0
    if(any(mp.dm_ind == 2)): mp.dm2.V = DM2V0
    if(any(mp.dm_ind == 8)): mp.dm8.V = DM8V0
    if(any(mp.dm_ind == 9)): mp.dm9.V = DM9V0
    
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
          
    modvar = falco.config.Object()
    modvar.whichSource = 'offaxis'
    modvar.x_offset = x_offset
    modvar.y_offset = y_offset
    modvar.zernIndex = 1
    modvar.wpsbpIndex = mp.wi_ref
    
    Iout = 0.
    for si in range(mp.Nsbp):
        modvar.sbpIndex = si
        E2D = falco.model.compact(mp, modvar, isEvalMode=isEvalMode)
        Iout = Iout + (np.abs(E2D)**2)*mp.jac.weightMat[si, 0]

    return Iout


def falco_get_sbp_image_fiber(mp, si):
    """
    Get an image in the specified sub-bandpass from an optical fiber.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    si: int
        Index of sub-bandpass for which to take the image

    Returns
    -------
    TBD
        Sub-bandpass image in units of normalized intensity
    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    check.nonnegative_scalar_integer(si, 'si', TypeError)

    if mp.flagSim:
        ImNI = falco_get_sim_sbp_image_fiber(mp, si)
    else:
        raise NotImplementedError(
            'Testbed functionality not implemented for fibers yet.')

    return ImNI


def falco_get_sim_sbp_image_fiber(mp, si):
    """
    Get an image in the specified sub-bandpass.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    si: int
        Index of sub-bandpass for which to take the image

    Returns
    -------
    TBD
        Sub-bandpass image in units of normalized intensity

    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    check.nonnegative_scalar_integer(si, 'si', TypeError)

    pass


def falco_get_summed_image_fiber(mp):
    """
    Get a summed image from the back end of a single-mode optical fiber(s).

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters

    Returns
    -------
    TBD
        Total intensity across the bandpass from all fibers.

    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')

    pass


def falco_get_testbed_sbp_image(mp, si):
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

    pass


def falco_get_gpct_sbp_image(mp, si):
    """
    Get an image in the specified sub-bandpass from the GPCT.

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
    pass


def falco_get_hcst_sbp_image(mp, si):
    """
    Get an image in the specified sub-bandpass from the Caltech HCST testbed.
    
    This function will need to be replaced in order to run on a
    different testbed. Note that the number of pixels per lambda*F# is
    predetermined.

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
    pass
