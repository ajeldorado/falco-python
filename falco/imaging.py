import falco
import numpy as np

def falco_get_PSF_norm_factor(mp):
    """
    Function to get the intensity normalization factor for each model at each sub-band.

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
        
    #--Initialize Model Normalizations
    if not hasattr(mp.Fend,'compact'):
         mp.Fend.compact = falco.config.Object() #--Initialize the new structure
    if not hasattr(mp.Fend,'eval'):
         mp.Fend.eval = falco.config.Object() #--Initialize the new structure
    if not hasattr(mp.Fend,'full'):
         mp.Fend.full = falco.config.Object() #--Initialize the new structure
    mp.Fend.compact.I00 = np.ones(mp.Nsbp) # Initial input before computing
    mp.Fend.eval.I00 =np.ones(mp.Nsbp) # Initial input before computing
    mp.Fend.full.I00 = np.ones((mp.Nsbp,mp.Nwpsbp)) # Initial input before computing

    modvar = falco.config.Object() #--Initialize the new structure
    modvar.zernIndex = 1
    modvar.whichSource = 'star'
    
    #--Compact Model Normalizations
    for si in range(mp.Nsbp):
        modvar.sbpIndex = si
        Etemp = falco.models.model_compact(mp, modvar,GETNORM=True)
        mp.Fend.compact.I00[si] = (np.abs(Etemp)**2).max()

    #--Compact Evaluation Model Normalizations
    for si in range(mp.Nsbp):
        modvar.sbpIndex = si
        Etemp = falco.models.model_compact(mp, modvar,GETNORM=True,EVAL=True)
        mp.Fend.eval.I00[si] = (np.abs(Etemp)**2).max()

    #--Full Model Normalizations (at points for entire-bandpass evaluation)
    if(mp.flagSim):
        for si in range(mp.Nsbp):
            for wi in range(mp.Nwpsbp):
                modvar.sbpIndex = si
                modvar.wpsbpIndex = wi
                Etemp = falco.models.model_full(mp, modvar,GETNORM=True)
                mp.Fend.full.I00[si,wi] = (np.abs(Etemp)**2).max()


def falco_get_summed_image(mp):
    """
    Function to get the broadband image over the entire bandpass.
    
    Function to get a broadband image over the entire bandpass by getting the sub-bandpass 
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

    ### Compute the DM surfaces outside the full model to save some time
    
    #--Loop over the function that gets the sbp images
    Imean = 0 # Initialize image
    for si in range(0,mp.Nsbp):
        Imean += mp.sbp_weights[si]*falco_get_sbp_image(mp,si)

    return Imean


def falco_get_sbp_image(mp, si):
    """
    Function to get an image in the specified sub-bandpass.
    
    Function to get an image in the specified sub-bandpass. Wrapper for functions to get a 
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

    if mp.flagSim:
        Isbp = falco_get_sim_sbp_image(mp, si)
    else:
        Isbp = falco_get_testbed_sbp_image(mp, si)

    return Isbp


def falco_get_sim_sbp_image(mp, si):
    """
    Function to get a simulated image in the specified sub-bandpass.

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

    #--Compute the DM surfaces outside the full model to save lots of time

    #--Loop over all wavelengths to get the starlight image
    Isbp = 0 #--Initialize the image sum in the sub-bandpass
    modvar = falco.config.Object() #--Initialize the new structure
    for wi in range(mp.Nwpsbp):
        modvar.sbpIndex   = si
        modvar.wpsbpIndex = wi
        modvar.whichSource = 'star'
        Estar = falco.models.model_full(mp, modvar)
        Iout = np.abs(Estar)**2 #--Apply spectral weighting outside this function

        #--Optionally include the planet PSF
        if(mp.planetFlag):
            modvar.whichSource = 'exoplanet'
            Eplanet = falco.models.model_full(mp,modvar)
            Iout = Iout + np.abs(Eplanet)**2 #--Apply spectral weighting outside this function

        #--Apply weight within the sub-bandpass. Assume polarizations are evenly weigted.
        Iout = mp.full.lambda_weights[wi]*Iout #mp.full.lambda_weights(wi)/length(mp.full.pol_conds)*Iout;
        Isbp += Iout
        
    return Isbp  
    
    
    
def falco_get_expected_summed_image(mp, cvar):
    """
    Function to generate the expected broadband image after a new control command.
    
    Function to generate the expected broadband image over the entire bandpass by adding 
    the model-based delta electric field on top of the current E-field estimate in each 
    sub-bandpass.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    cvar: ModelParameters
        Structure of controller variables

    Returns
    -------
    TBD : numpy ndarray
        Expected band-averaged image in units of normalized intensity
    """

    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    pass

def falco_get_sbp_image_fiber(mp, si):
    """
    Function to get an image in the specified sub-bandpass from an optical fiber.

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

    if mp.flagSim:
        ImNI = falco_get_sim_sbp_image_fiber(mp, si)
    else:
        raise NotImplementedError('Testbed functionality not implemented for fibers yet.')

    return ImNI

def falco_get_sim_sbp_image_fiber(mp, si):
    """
    Function to get an image in the specified sub-bandpass.

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

    pass
    
def falco_get_summed_image_fiber(mp):
    """
    Function to get a summed image from the back end of a single-mode optical
    fiber(s).

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
    Function to get an image in the specified sub-bandpass from a testbed.
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

    pass

def falco_sim_image_compact_offaxis(mp, x_offset, y_offset, **kwargs):
    """
    Function to return the broadband intensity for the compact model.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    x_offset: int
        lateral offset (in xi) of the stellar PSF in the focal plane. [lambda0/D]
    y_offset: int
        vertical offset (in eta) of the stellar PSF in the focal plane. [lambda0/D]

    Other Parameters
    ----------------
    EVAL : bool
       Switch that tells function to run at a higher final focal plane resolution when 
       evaluating throughput.

    Returns
    -------
    Iout : numpy ndarray
        Simulated bandpass-averaged intensity from the compact model
    """

    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')

    #--Optional Keyword arguments
    if( ("EVAL" in kwargs and kwargs["EVAL"]) or ("eval" in kwargs and kwargs["eval"]) ):
        flagEval = True # flag to use a different (usually higher) resolution at final focal plane for evaluation
    else:
        flagEval = False 
          
    modvar = falco.config.Object()
    modvar.whichSource = 'offaxis'
    modvar.x_offset = x_offset
    modvar.y_offset = y_offset
    modvar.zernIndex = 1
    modvar.wpsbpIndex = mp.wi_ref
    
    Iout = 0. #--Initialize output
    for si in range(mp.Nsbp):
        modvar.sbpIndex = si
        E2D = falco.models.model_compact(mp, modvar, EVAL=flagEval )            
        Iout = Iout + (np.abs(E2D)**2)*mp.jac.weightMat[si,0]

    return Iout
    
def falco_get_gpct_sbp_image(mp, si):
    """
    Function to get an image in the specified sub-bandpass from the GPCT.

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
    Function to get an image in the specified sub-bandpass from the Caltech
    HCST testbed. This function will need to be replaced in order to run on a
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
