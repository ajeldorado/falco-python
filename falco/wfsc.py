import numpy as np
import falco
import os
import pickle
import math

#from falco import models

def falco_init_ws(config):

    with open(config, 'rb') as f:
        mp = pickle.load(f)

    mainPath = mp.path.falco;

    print('DM 1-to-2 Fresnel number (using radius) = ' + str((mp.P2.D/2)**2/(mp.d_dm1_dm2*mp.lambda0)))

    ## Intializations of structures (if they don't exist yet)
    mp.jac.dummy = 1;
    
    ## Optional/Hidden flags
    if not hasattr(mp,'flagSaveWS'):  
        mp.flagSaveWS = False  #--Whehter to save otu the entire workspace at the end of the trial. Can take up lots of space.
    if not hasattr(mp,'flagSaveEachItr'):  
        mp.flagSaveEachItr = False  #--Whether to save out the performance at each iteration. Useful for long trials in case it crashes or is stopped early.
    if not hasattr(mp,'flagSVD'):
        mp.flagSVD = False    #--Whether to compute and save the singular mode spectrum of the control Jacobian (each iteration)
    if not hasattr(mp,'flagFiber'):
        mp.flagFiber = False  #--Whether to couple the final image through lenslets and a single mode fiber.
    if not hasattr(mp,'flagDMwfe'):  
        mp.flagDMwfe = False  #--Temporary for BMC quilting study
    if not hasattr(mp,'flagTrainModel'):  
        mp.flagTrainModel = False  #--Whether to call the Expectation-Maximization (E-M) algorithm to improve the linearized model. 
    if not hasattr(mp,'flagUseLearnedJac'):  
        mp.flagUseLearnedJac = False #--Whether to load and use an improved Jacobian from the Expectation-Maximization (E-M) algorithm 
    if not hasattr(mp.est,'flagUseJac'):  
        mp.est.flagUseJac = False   #--Whether to use the Jacobian or not for estimation. (If not using Jacobian, model is called and differenced.)
    if not hasattr(mp.ctrl,'flagUseModel'):  
        mp.ctrl.flagUseModel = False #--Whether to perform a model-based (vs empirical) grid search for the controller
    
    ## Optional/Hidden variables
    if not hasattr(mp,'propMethodPTP'):
        mp.propMethodPTP = 'fft' #--Propagation method for postage stamps around the influence functions. 'mft' or 'fft'
    if not hasattr(mp,'SPname'):  
        mp.SPname = 'none' #--Apodizer name default

    ## File Paths
    
    #--Storage directories (data in these folders will not be synced via Git
    filesep = os.pathsep
    if not hasattr(mp.path,'ws'):
        mp.path.ws = mainPath + 'data' + filesep + 'ws' + filesep # Store final workspace data here
    if not hasattr(mp.path,'maps'): 
        mp.path.falcoaps = mainPath + 'maps' + filesep # Maps go here
    if not hasattr(mp.path,'jac'): 
        mp.path.jac = mainPath + 'data' + filesep + 'jac' + filesep # Store the control Jacobians here
    if not hasattr(mp.path,'images'): 
        mp.path.images = mainPath + 'data' + filesep + 'images' + filesep # Store all full, reduced images here
    if not hasattr(mp.path,'dm'): 
        mp.path.dm = mainPath + 'data' + filesep + 'DM' + filesep # Store DM command maps here
    if not hasattr(mp.path,'ws_inprogress'): 
        mp.path.ws_inprogress = mainPath + 'data' + filesep + 'ws_inprogress' + filesep # Store in progress workspace data here

    ## Loading previous DM commands as the starting point
    #--Stash DM8 and DM9 starting commands if they are given in the main script
    if hasattr(mp,'dm8'):
        if hasattr(mp.dm8,'V'): 
            mp.DM8V0 = mp.dm8.V
        if hasattr(mp.dm9,'V'): 
            mp.DM9V0 = mp.dm9.V

    ## Useful factor
    mp.mas2lam0D = 1/(mp.lambda0/mp.P1.D*180/np.pi*3600*1000); #--Conversion factor: milliarcseconds (mas) to lambda0/D
    
    ## Estimator
    if not hasattr(mp,'estimator'): 
        mp.estimator = 'perfect'

    ## Bandwidth and Wavelength Specs
    
    if not hasattr(mp,'Nwpsbp'):
        mp.Nwpsbp = 1;

    ### NOTE: I added this 
    if not hasattr(mp, 'full'):
        mp.full = falco.config.EmptyObject()

    mp.full.Nlam = mp.Nsbp*mp.Nwpsbp; #--Total number of wavelengths in the full model

    #--When in simulation and using perfect estimation, use end wavelengths in bandbass, which (currently) requires Nwpsbp=1. 
    if mp.estimator.lower() == 'perfect' and mp.Nsbp>1:
        if mp.Nwpsbp>1:
            print('* Forcing mp.Nwpsbp = 1 * \n')
            mp.Nwpsbp = 1; # number of wavelengths per sub-bandpass. To approximate better each finite sub-bandpass in full model with an average of images at these values. Only >1 needed when each sub-bandpass is too large (say >3#).
    
    #--Center-ish wavelength indices (ref = reference)
    mp.si_ref = math.ceil(mp.Nsbp/2);
    mp.wi_ref = math.ceil(mp.Nwpsbp/2);

    #--Wavelengths used for Compact Model (and Jacobian Model)
    mp.sbp_weights = np.ones((mp.Nsbp,1));
    
    if mp.estimator.lower() == 'perfect' and mp.Nsbp>1: #--For design or modeling without estimation: Choose ctrl wvls evenly between endpoints (inclusive) of the total bandpass
        mp.fracBWsbp = mp.fracBW/(mp.Nsbp-1);
        mp.sbp_centers = mp.lambda0*np.linspace(1-mp.fracBW/2,1+mp.fracBW/2,mp.Nsbp);
    else: #--For cases with estimation: Choose est/ctrl wavelengths to be at subbandpass centers.
        mp.fracBWsbp = mp.fracBW/mp.Nsbp;
        mp.fracBWcent2cent = mp.fracBW*(1-1/mp.Nsbp); #--Bandwidth between centers of endpoint subbandpasses.
        mp.sbp_centers = mp.lambda0*np.linspace(1-mp.fracBWcent2cent/2,1+mp.fracBWcent2cent/2,mp.Nsbp); #--Space evenly at the centers of the subbandpasses.
    mp.sbp_weights = mp.sbp_weights/np.sum(mp.sbp_weights); #--Normalize the sum of the weights
    
    print(' Using %d discrete wavelength(s) in each of %d sub-bandpasses over a %.1f# total bandpass \n'%(mp.Nwpsbp, mp.Nsbp,100*mp.fracBW));
    print('Sub-bandpasses are centered at wavelengths [nm]:\t ')
    print('%.2f  '%(1e9*mp.sbp_centers))
    print('\n\n');

    #--Wavelength factors/weights within sub-bandpasses in the full model
    mp.full.lambda_weights = np.ones((mp.Nwpsbp,1)); #--Initialize as all ones. Weights within a single sub-bandpass
    if mp.Nsbp==1:
        mp.full.sbp_facs = np.linspace(1-mp.fracBW/2,1+mp.fracBW/2,mp.Nwpsbp);
        if mp.Nwpsbp>2: #--Include end wavelengths with half weights
            mp.full.lambda_weights[0] = 1/2;
            mp.full.lambda_weights[-1] = 1/2;
    else: #--For cases with estimation (est/ctrl wavelengths at subbandpass centers). Full model only
        mp.full.sbp_facs = np.linspace(1-(mp.fracBWsbp/2)*(1-1/mp.Nwpsbp), \
                               1+(mp.fracBWsbp/2)*(1-1/mp.Nwpsbp), mp.Nwpsbp)
    if mp.Nwpsbp==1:  
        mp.full.sbp_facs = np.array([1]) #--Set factor to 1 if only 1 value.
    
    mp.full.lambda_weights = mp.full.lambda_weights/np.sum(mp.full.lambda_weights); #--Normalize sum of the weights

    #--Make vector of all wavelengths and weights used in the full model
    mp.full.lambdas = np.zeros((mp.Nsbp*mp.Nwpsbp,1));
    mp.full.weights = np.zeros((mp.Nsbp*mp.Nwpsbp,1));
    counter = 0;
    for si in range(0,mp.Nsbp):
        for wi in range(0,mp.Nwpsbp):
            mp.full.lambdas[counter] = mp.sbp_centers[si]*mp.full.sbp_facs[wi];
            mp.full.all_weights = mp.sbp_weights[si]*mp.full.lambda_weights[wi];
            counter = counter+1;

    ## Zernike and Chromatic Weighting of the Control Jacobian
    if not hasattr(mp.jac,'zerns'):  
        mp.jac.zerns = 1 #--Which Zernike modes to include in Jacobian [Noll index]. Always include 1 for piston term.
    if not hasattr(mp.jac,'Zcoef'):  
        mp.jac.Zcoef = 1e-9*np.ones((10,1)); end #--meters RMS of Zernike aberrations. (piston value is reset to 1 later for correct normalization)

    falco.configs.falco_config_jac_weights(mp)

    ## Pupil Masks
    falco.configs.falco_config_gen_chosen_pupil(mp) #--input pupil mask
    falco.configs.falco_config_gen_chosen_apodizer(mp) #--apodizer mask
    falco.configs.falco_config_gen_chosen_LS(mp) #--Lyot stop

    pass

def falco_wfsc_loop(mp):

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

    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    pass

    ## Sort out file paths and save the config file    
    
    #--Add the slash or backslash to the FALCO path if it isn't there.
    filesep = os.pathsep

    if mp.path.falco[-1] != '/' and mp.path.falco[-1] != '\\':
        mp.path.falco = mp.path.falco + filesep
    
    mp.path.dummy = 1; #--Initialize the folders structure in case it doesn't already exist
    
    #--Store minimal data to re-construct the data from the run: the config files and "out" structure after a trial go here
    if not hasattr(mp.path,'config'):
        mp.path.config = mp.path.falco + filesep + 'data' + filesep + 'config' + filesep
    
    #--Entire final workspace from FALCO gets saved here.
    if not hasattr(mp.path,'ws'):
        mp.path.ws = mp.path.falco + filesep + 'data' + filesep + 'ws' + filesep
    
    #--Save the config file
    fn_config = mp.path.config + mp.runLabel + '_config.pkl'
    #save(fn_config)
    print('Saved the config file: \t%s\n'%(fn_config))
    with open(fn_config, 'wb') as f:
        pickle.dump(mp, f)

    
    ## Get configuration data from a function file
    if not mp.flagSim:  
        bench = mp.bench #--Save the testbed structure "mp.bench" into "bench" so it isn't overwritten by falco_init_ws

    falco_init_ws(fn_config)
    #[mp,out] = falco_init_ws(fn_config);
    #if(~mp.flagSim);  mp.bench = bench;  end

    

def falco_est_perfect_Efield(mp, DM, which_model='full'):
    """
    Function to return the perfect-knowledge E-field and summed intensity for the full model.

    Parameters
    ----------
    mp : ModelParameters
        Parameter structure for current model.
    DM : DeformableMirrorParameters (placeholder class for now)
        Parameter structure for deformable mirrors
    which_model: string
        Coronagraph model used to compute the detector-plane electric field. Either 'full' or
        'compact'.
    Returns
    -------
    Emat : np.ndarray
        Exact electric field inside dark hole
    Isum2D : float
        Total intensity inside dark hole
    """

    if which_model == 'full':
        Icube_shape = (mp.F4.full.Neta, mp.F4.full.Nxim, mp.Nttlam)
        Emat_shape = (mp.F4.full.corr.inds.shape[0], mp.Nttlam)
        model = falco.models.model_full
    elif which_model == 'compact':
        Icube_shape = (mp.F4.compact.Neta, mp.F4.compact.Nxim, mp.Nttlam)
        Emat_shape = (mp.F4.compact.corr.inds.shape[0], mp.Nttlam)
        model = falco.models.model_compact
    else:
        raise ValueError('Invalid model specified.  Try \'full\' or \'compact\'.')

    Icube = np.zeros(Icube_shape, dtype=np.float64)
    Emat = np.zeros(Emat_shape, dtype=np.float64)

    # Initialize model variable dictionary
    modvar = {
        'flagCalcJac': 0,
        'wpsbpIndex': mp.wi_ref,
        'whichSource': 'star'
    }

    # Execute model to obtain electric field and total intensity
    for tsi in range(mp.Nttlam):
        modvar['sbpIndex'] = mp.Wttlam_si[tsi]
        modvar['ttIndex'] = mp.Wttlam_ti[tsi]

        E2D = model(mp, DM, modvar)
        Emat[:, tsi] = E2D[mp.F4.corr.inds]  # Exact field inside estimation area
        Icube[:, :, tsi] = (np.abs(E2D) ** 2) * mp.WttlamVec(tsi) / mp.Wsum

    Isum2D = Icube.sum(axis=2)
    return Emat, Isum2D
