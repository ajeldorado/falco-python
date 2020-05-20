import cupy as cp
import numpy as np
import falco
import os
import pickle
import scipy
import psutil # For checking number of cores available
import multiprocessing
from astropy.io import fits 
import matplotlib.pyplot as plt 
import copy

#def flesh_out_workspace(mp, config=None):
def flesh_out_workspace(mp):

#     if config:
#         with open(config, 'rb') as f:
#             mp = pickle.load(f)

    falco_set_optional_variables(mp) # Optional/hidden boolean flags and variables

    falco_set_spectral_properties(mp)
    falco_set_jacobian_weights(mp) # Zernike Modes and Subband Weighting of Control Jacobian

    #--Pupil Masks
    falco_gen_chosen_pupil(mp);
    falco_gen_chosen_apodizer(mp);
    falco_gen_chosen_lyot_stop(mp);
    falco_plot_superposed_pupil_masks(mp) #--Visually inspect relative pupil mask alignment

    #--Focal planes
    falco_gen_FPM(mp) #% Generate FPM if necessary. If HLC, uses DM8 and DM9.
    falco_get_FPM_coordinates(mp); 
    falco_get_Fend_resolution(mp); 
    falco_configure_dark_hole_region(mp) #% Software Mask for Correction (corr) and Scoring (score)
    falco_set_spatial_weights(mp) #--Spatial weighting for control Jacobian. 
    # mp.Fend.mask = ones(mp.Fend.Neta,mp.Fend.Nxi) #% Field Stop at Fend (as a software mask) (NOT INCLUDED YET)

    #--DM1 and DM2
    falco_configure_dm1_and_dm2(mp) #% Flesh out the dm1 and dm2 structures
    falco_gen_DM_stops(mp);
    falco_set_dm_surface_padding(mp) #% DM Surface Array Sizes for Angular Spectrum Propagation with FFTs

    falco_set_initial_Efields(mp)

    falco.imaging.falco_get_PSF_norm_factor(mp)
    # falco_gen_contrast_over_NI_map(mp) #--Contrast to Normalized Intensity Map Calculation (NOT INCLUDED YET)

    out = falco_init_storage_arrays(mp) #% Initialize Arrays to Store Performance History

    print('\nBeginning Trial %d of Series %d.\n' % (mp.TrialNum, mp.SeriesNum))    

    print('DM 1-to-2 Fresnel number (using radius) = ' + str((mp.P2.D/2)**2/(mp.d_dm1_dm2*mp.lambda0)))


    return out


########################################################################
    

def falco_set_optional_variables(mp):
    
    ## Intializations of structures (if they don't exist yet)
    if not hasattr(mp, "compact"):
        mp.compact = falco.config.Object()
    if not hasattr(mp, "full"):
        mp.full = falco.config.Object()
    if not hasattr(mp, "jac"):
        mp.jac = falco.config.Object()
    if not hasattr(mp, "est"):
        mp.est = falco.config.Object()

    ## File Paths for Data Storage (excluded from git)
    filesep = os.pathsep
    mainPath = mp.path.falco
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
    
    ## Number of threads to use if doing multiprocessing
    if not hasattr(mp, "Nthreads"):
        mp.Nthreads = psutil.cpu_count(logical=False) 
    
    ## Optional/hidden boolean flags
    #--Saving data
    if not hasattr(mp,'flagSaveWS'):  
        mp.flagSaveWS = False  #--Whehter to save out the entire workspace at the end of the trial. Can take up lots of space.
    if not hasattr(mp,'flagSaveEachItr'):  
        mp.flagSaveEachItr = False  #--Whether to save out the performance at each iteration. Useful for long trials in case it crashes or is stopped early.
    if not hasattr(mp,'flagSVD'):
        mp.flagSVD = False    #--Whether to compute and save the singular mode spectrum of the control Jacobian (each iteration)
    #--Jacobian or controller related
    if not hasattr(mp,'flagTrainModel'):  
        mp.flagTrainModel = False  #--Whether to call the Expectation-Maximization (E-M) algorithm to improve the linearized model. 
    if not hasattr(mp,'flagUseLearnedJac'):  
        mp.flagUseLearnedJac = False #--Whether to load and use an improved Jacobian from the Expectation-Maximization (E-M) algorithm 
    if not hasattr(mp.est,'flagUseJac'):  
        mp.est.flagUseJac = False   #--Whether to use the Jacobian or not for estimation. (If not using Jacobian, model is called and differenced.)
    if not hasattr(mp.ctrl,'flagUseModel'):  
        mp.ctrl.flagUseModel = False #--Whether to perform a model-based (vs empirical) grid search for the controller
    
    #--Model options (Very specialized cases--not for the average user)
    if not hasattr(mp,'flagFiber'):
        mp.flagFiber = False  #--Whether to couple the final image through lenslets and a single mode fiber.
    if not hasattr(mp,'flagLenslet'):
        mp.flagLenslet = False    #--Whether to propagate through a lenslet array placed in Fend before coupling light into fibers
    if not hasattr(mp,'flagDMwfe'):  
        mp.flagDMwfe = False  #--Temporary for BMC quilting study
        
    #--Whether to generate or load various masks: compact model
    if not hasattr(mp.compact,'flagGenPupil'):  
        mp.compact.flagGenPupil = True
    if not hasattr(mp.compact,'flagGenApod'):  
        mp.compact.flagGenApod = False #--Different! Apodizer generation defaults to false.
    if not hasattr(mp.compact,'flagGenFPM'):  
        mp.compact.flagGenFPM = True
    if not hasattr(mp.compact,'flagGenLS'):  
        mp.compact.flagGenLS = True
    #--Whether to generate or load various masks: full model
    if not hasattr(mp.full,'flagPROPER'):  
        mp.full.flagPROPER = False #--Whether to use a full model written in PROPER. If true, then load (don't generate) all masks for the full model
    if mp.full.flagPROPER:
        mp.full.flagGenPupil = False;
        mp.full.flagGenApod = False;
        mp.full.flagGenFPM = False;
        mp.full.flagGenLS = False;
    if not hasattr(mp.full,'flagGenPupil'):  
        mp.full.flagGenPupil = True
    if not hasattr(mp.full,'flagGenApod'):  
        mp.full.flagGenApod = False #--Different! Apodizer generation defaults to False.
    if not hasattr(mp.full,'flagGenFPM'):  
        mp.full.flagGenFPM = True
    if not hasattr(mp.full,'flagGenLS'):  
        mp.full.flagGenLS = True    
    
    ## Optional/Hidden variables
    if not hasattr(mp.full,'pol_conds'):  
        mp.full.pol_conds = cp.array([0])  #--Vector of which polarization state(s) to use when creating images from the full model. Currently only used with PROPER full models from John Krist.
    if not hasattr(mp,'propMethodPTP'):
        mp.propMethodPTP = 'fft' #--Propagation method for postage stamps around the influence functions. 'mft' or 'fft'
    if not hasattr(mp,'apodType'):  
        mp.apodType = 'none';  #--Type of apodizer. Only use this variable when generating the apodizer. Curr
    
    #--Sensitivities to Zernike-Mode Perturbations
    if not hasattr(mp.full,'ZrmsVal'): 
        mp.full.ZrmsVal = 1e-9 #--Amount of RMS Zernike mode used to calculate aberration sensitivities [meters]. WFIRST CGI uses 1e-9, and LUVOIR and HabEx use 1e-10. 
    if not hasattr(mp.full,'Rsens'): 
        mp.full.Rsens = cp.array([])
    if not hasattr(mp.full,'indsZnoll'): 
        mp.full.indsZnoll = cp.array([2, 3])
    
    ## DM Initialization
    if not hasattr(mp,'dm1'):
        mp.dm1 = falco.config.Object()
    if not hasattr(mp,'dm2'):
        mp.dm2 = falco.config.Object()
    if not hasattr(mp,'dm3'):
        mp.dm3 = falco.config.Object()
    if not hasattr(mp,'dm4'):
        mp.dm4 = falco.config.Object()
    if not hasattr(mp,'dm5'):
        mp.dm5 = falco.config.Object()
    if not hasattr(mp,'dm6'):
        mp.dm6 = falco.config.Object()
    if not hasattr(mp,'dm7'):
        mp.dm7 = falco.config.Object()
    if not hasattr(mp,'dm8'):
        mp.dm8 = falco.config.Object()
    if not hasattr(mp,'dm9'):
        mp.dm9 = falco.config.Object()  

    #--Initialize the number of actuators (NactTotal) and actuators used (Nele).
    mp.dm1.NactTotal=0; mp.dm2.NactTotal=0; mp.dm3.NactTotal=0; mp.dm4.NactTotal=0; mp.dm5.NactTotal=0; mp.dm6.NactTotal=0; mp.dm7.NactTotal=0; mp.dm8.NactTotal=0; mp.dm9.NactTotal=0; #--Initialize for bookkeeping later.
    mp.dm1.Nele=0; mp.dm2.Nele=0; mp.dm3.Nele=0; mp.dm4.Nele=0; mp.dm5.Nele=0; mp.dm6.Nele=0; mp.dm7.Nele=0; mp.dm8.Nele=0; mp.dm9.Nele=0; #--Initialize for Jacobian calculations later. 

    
    ## Deformable mirror settings
    #--DM1
    if not hasattr(mp.dm1,'Vmin'):  
        mp.dm1.Vmin = -1000. #--Min allowed voltage command
    if not hasattr(mp.dm1,'Vmax'):  
        mp.dm1.Vmax = 1000. #--Max allowed voltage command
    if not hasattr(mp.dm1,'pinned'):  
        mp.dm1.pinned = cp.array([]) #--Indices of pinned actuators
    if not hasattr(mp.dm1,'Vpinned'):  
        mp.dm1.Vpinned = cp.array([]) #--(Fixed) voltage commands of pinned actuators
    if not hasattr(mp.dm1,'tied'):  
        mp.dm1.tied = cp.zeros((0,2)) #--Indices of paired actuators. Two indices per row       
    if not hasattr(mp.dm1,'flagNbrRule'):
        mp.dm1.flagNbrRule = False #--Whether to set constraints on neighboring actuator voltage differences. If set to true, need to define mp.dm1.dVnbr
    #--DM2
    if not hasattr(mp.dm2,'Vmin'):  
        mp.dm2.Vmin = -1000. #--Min allowed voltage command
    if not hasattr(mp.dm2,'Vmax'):  
        mp.dm2.Vmax = 1000. #--Max allowed voltage command
    if not hasattr(mp.dm2,'pinned'):  
        mp.dm2.pinned = cp.array([]) #--Indices of pinned actuators
    if not hasattr(mp.dm2,'Vpinned'):  
        mp.dm2.Vpinned = cp.array([]) #--(Fixed) voltage commands of pinned actuators
    if not hasattr(mp.dm2,'tied'):  
        mp.dm2.tied = cp.zeros((0,2)) #--Indices of paired actuators. Two indices per row       
    if not hasattr(mp.dm2,'flagNbrRule'):  
        mp.dm2.flagNbrRule = False;  #--Whether to set constraints on neighboring actuator voltage differences. If set to true, need to define mp.dm2.dVnbr

    ## Loading previous DM commands as the starting point
    #--Stash DM8 and DM9 starting commands if they are given in the main script
    if hasattr(mp,'dm8'):
        if hasattr(mp.dm8,'V'): 
            mp.DM8V0 = mp.dm8.V
        if hasattr(mp.dm9,'V'): 
            mp.DM9V0 = mp.dm9.V

    ## Intialize delta DM voltages. Needed for Kalman filters.
    ##--Save the delta from the previous command
    if cp.any(mp.dm_ind==1):
        mp.dm1.dV = 0
    if cp.any(mp.dm_ind==2):  
        mp.dm2.dV = 0
    if cp.any(mp.dm_ind==3): 
        mp.dm3.dV = 0
    if cp.any(mp.dm_ind==4): 
        mp.dm4.dV = 0
    if cp.any(mp.dm_ind==5):  
        mp.dm5.dV = 0
    if cp.any(mp.dm_ind==6):
        mp.dm6.dV = 0
    if cp.any(mp.dm_ind==7): 
        mp.dm7.dV = 0
    if cp.any(mp.dm_ind==8):  
        mp.dm8.dV = 0
    if cp.any(mp.dm_ind==9):  
        mp.dm9.dV = 0
    
#    #--First delta DM settings are zero (for covariance calculation in Kalman filters or robust controllers)
#    mp.dm1.dV = cp.zeros((mp.dm1.Nact,mp.dm1.Nact)); # delta voltage on DM1;
#    mp.dm2.dV = cp.zeros((mp.dm2.Nact,mp.dm2.Nact)); # delta voltage on DM2;
#    mp.dm8.dV = cp.zeros((mp.dm8.NactTotal,1)); # delta voltage on DM8;
#    mp.dm9.dV = cp.zeros((mp.dm9.NactTotal,1)); # delta voltage on DM9;

    ## Control
    if not hasattr(mp, 'WspatialDef'):
        mp.WspatialDef = cp.array([]) #--spatial weight matrix for the Jacobian

    ## Performance Evaluation
    mp.mas2lam0D = 1/(mp.lambda0/mp.P1.D*180/cp.pi*3600*1000); #--Conversion factor: milliarcseconds (mas) to lambda0/D
    if not hasattr(mp.Fend, 'eval'): #--Initialize the structure if it doesn't exist.
        mp.Fend.eval = falco.config.Object()
    if not hasattr(mp.Fend.eval,'res'):  
        mp.Fend.eval.res = 10
        
    ## Off-axis, incoherent point source (exoplanet). Used if modvar.whichSource = 'exoplanet'
    if not mp.flagFiber:
        mp.c_planet = 1; # contrast of exoplanet
        mp.x_planet = 6; # x position of exoplanet in lambda0/D
        mp.y_planet = 0; # y position of exoplanet in lambda0/D
        
    pass


def falco_set_spectral_properties(mp):
    ## Bandwidth and Wavelength Specs: Compact Model

    #--Center-ish wavelength indices (ref = reference)(Only the center if
    #  an odd number of wavelengths is used.)
    mp.si_ref = cp.floor(mp.Nsbp/2).astype(int)

    #--Wavelengths used for Compact Model (and Jacobian Model)
    mp.sbp_weights = cp.ones((mp.Nsbp,1));
    if mp.Nwpsbp==1: #--Set ctrl wavelengths evenly between endpoints (inclusive) of the total bandpass.
        if mp.Nsbp==1:
            mp.sbp_centers = cp.array([mp.lambda0])
        else:
            mp.sbp_centers = mp.lambda0*cp.linspace(1-mp.fracBW/2,1+mp.fracBW/2,mp.Nsbp);
    else:#--For cases with multiple sub-bands: Choose wavelengths to be at subbandpass centers since the wavelength samples will span to the full extent of the sub-bands.
        mp.fracBWcent2cent = mp.fracBW*(1-1/mp.Nsbp); #--Bandwidth between centers of endpoint subbandpasses.
        mp.sbp_centers = mp.lambda0*cp.linspace(1-mp.fracBWcent2cent/2,1+mp.fracBWcent2cent/2,mp.Nsbp); #--Space evenly at the centers of the subbandpasses.
    mp.sbp_weights = mp.sbp_weights/cp.sum(mp.sbp_weights); #--Normalize the sum of the weights
    
    print(' Using %d discrete wavelength(s) in each of %d sub-bandpasses over a %.1f%% total bandpass \n'%(mp.Nwpsbp, mp.Nsbp,100*mp.fracBW));
    print('Sub-bandpasses are centered at wavelengths [nm]:\t ',end='')
    print(1e9*mp.sbp_centers)
 

    ## Bandwidth and Wavelength Specs: Full Model
    
    #--Center(-ish) wavelength indices (ref = reference). (Only the center if an odd number of wavelengths is used.)
    mp.wi_ref = cp.floor(mp.Nwpsbp/2).astype(int)

    #--Wavelength factors/weights within sub-bandpasses in the full model
    mp.full.lambda_weights = cp.ones((mp.Nwpsbp,1)); #--Initialize as all ones. Weights within a single sub-bandpass
    if mp.Nwpsbp==1:
        mp.full.dlam = 0; #--Delta lambda between every wavelength in the sub-band in the full model
    else:
        #--Spectral weighting in image
        mp.full.lambda_weights[0] = 1/2; #--Include end wavelengths with half weights
        mp.full.lambda_weights[-1] = 1/2; #--Include end wavelengths with half weights
        mp.fracBWsbp = mp.fracBW/mp.Nsbp; #--Bandwidth per sub-bandpass
        #--Indexing of wavelengths in each sub-bandpass
        sbp_facs = cp.linspace(1-mp.fracBWsbp/2,1+mp.fracBWsbp/2,mp.Nwpsbp); #--Factor applied to lambda0 only
        mp.full.dlam = (sbp_facs[1] - sbp_facs[0])*mp.lambda0; #--Delta lambda between every wavelength in the full model 
    
    mp.full.lambda_weights = mp.full.lambda_weights/cp.sum(mp.full.lambda_weights); #--Normalize sum of the weights (within the sub-bandpass)

    #--Make vector of all wavelengths and weights used in the full model
    lambdas = cp.zeros((mp.Nsbp*mp.Nwpsbp,))
    lambda_weights_all = cp.zeros((mp.Nsbp*mp.Nwpsbp,))
    mp.full.lambdasMat = cp.zeros((mp.Nsbp,mp.Nwpsbp))
    mp.full.indsLambdaMat = cp.zeros((mp.Nsbp*mp.Nwpsbp,2),dtype=int)
    counter = 0;
    for si in range(mp.Nsbp):
        if(mp.Nwpsbp==1):
            mp.full.lambdasMat[si,0] = mp.sbp_centers[si]
        else:
            mp.full.lambdasMat[si,:] = cp.arange(-(mp.Nwpsbp-1)/2,(mp.Nwpsbp+1)/2)*mp.full.dlam + mp.sbp_centers[si]
        cp.arange(-(mp.Nwpsbp-1)/2,(mp.Nwpsbp-1)/2)*mp.full.dlam# + mp.sbp_centers[si];
        for wi in range(mp.Nwpsbp):
            lambdas[counter] = mp.full.lambdasMat[si,wi];
            lambda_weights_all[counter] = mp.sbp_weights[si,0]*mp.full.lambda_weights[wi,0];
            mp.full.indsLambdaMat[counter,0] = si
            mp.full.indsLambdaMat[counter,1] = wi
            counter = counter+1;
            
    #--Get rid of redundant wavelengths in the complete list, and sum weights for repeated wavelengths
    # indices of unique wavelengths
    unused_1, inds_unique = cp.unique(cp.around(1e12*lambdas), return_index=True); #--Check equality at the picometer level for wavelength
    mp.full.indsLambdaUnique = inds_unique;
    # indices of duplicate wavelengths
    duplicate_inds = cp.asarray(np.setdiff1d( cp.asnumpy(cp.arange(len(lambdas),dtype=int)) , cp.asnumpy(inds_unique)));
    # duplicate weight values
    duplicate_values = lambda_weights_all[duplicate_inds]

    #--Shorten the vectors to contain only unique values. Combine weights for repeated wavelengths.
    mp.full.lambdas = lambdas[inds_unique];
    mp.full.lambda_weights_all = lambda_weights_all[inds_unique];
    for idup in range(len(duplicate_inds)):
        wvl = lambdas[duplicate_inds[idup]];
        weight = lambda_weights_all[duplicate_inds[idup]];
        ind = cp.where(cp.abs(mp.full.lambdas-wvl)<=1e-11)
        print(ind)
        mp.full.lambda_weights_all[ind] = mp.full.lambda_weights_all[ind] + weight;
    mp.full.NlamUnique = len(inds_unique)
    pass

def falco_set_jacobian_weights(mp):
    """
    Function to set the relative weights for the Jacobian modes based on
    wavelength and Zernike mode.

    Function to set the relative weights for the Jacobian modes. The weights are 
    formulated first in a 2-D array with rows for wavelengths and columns for Zernike 
    modes. The weights are then normalized in each column. The weight matrix is then 
    vectorized, with all zero weights removed.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    Returns
    -------
    nothing
        Values are added by reference into the mp structure.
    """

    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
                
    #--Initialize mp.jac if it doesn't exist
    if not hasattr(mp, 'jac'):
        mp.jac = falco.config.EmptyClass()
    
    #--Which Zernike modes to include in Jacobian. Given as a vector of Noll indices. 1 is the on-axis piston mode.
    if not hasattr(mp.jac, 'zerns'):
        mp.jac.zerns = cp.array([1])
        mp.jac.Zcoef = cp.array([1.])
        
    mp.jac.Nzern = cp.size(mp.jac.zerns)
    mp.jac.Zcoef[cp.nonzero(mp.jac.zerns==1)][0] = 1.; #--Reset coefficient for piston term to 1
    
    #--Initialize weighting matrix of each Zernike-wavelength mode for the controller
    mp.jac.weightMat = cp.zeros((mp.Nsbp,mp.jac.Nzern)); 
    for izern in range(0,mp.jac.Nzern):
        whichZern = mp.jac.zerns[izern];
        if whichZern==1:
            mp.jac.weightMat[:,0] = cp.ones(mp.Nsbp) #--Include all wavelengths for piston Zernike mode
        else: #--Include just middle and end wavelengths for Zernike mode 2 and up
            mp.jac.weightMat[0,izern] = 1
            mp.jac.weightMat[mp.si_ref,izern] = 1
            mp.jac.weightMat[mp.Nsbp-1,izern] = 1
        
    #--Half-weighting if endpoint wavelengths are used
    if mp.estimator.lower()=='perfect': #--For design or modeling without estimation: Choose ctrl wvls evenly between endpoints of the total bandpass
        mp.jac.weightMat[0,:] = 0.5*mp.jac.weightMat[0,:];
        mp.jac.weightMat[mp.Nsbp-1,:] = 0.5*mp.jac.weightMat[mp.Nsbp-1,:];
    
    #--Normalize the summed weights of each column separately
    for izern in range(mp.jac.Nzern):
        colSum = cp.double(sum(mp.jac.weightMat[:,izern]))
        mp.jac.weightMat[:,izern] = mp.jac.weightMat[:,izern]/colSum

    #--Zero out columns for which the RMS Zernike value is zero
    for izern in range(mp.jac.Nzern):
        if mp.jac.Zcoef[izern]==0:
            mp.jac.weightMat[:,izern] = 0*mp.jac.weightMat[:,izern]

    mp.jac.weightMat_ele = cp.nonzero(mp.jac.weightMat>0) #--Indices of the non-zero control Jacobian modes in the weighting matrix
    mp.jac.weights = mp.jac.weightMat[mp.jac.weightMat_ele] #--Vector of control Jacobian mode weights
    mp.jac.Nmode = cp.size(mp.jac.weights) #--Number of (Zernike-wavelength pair) modes in the control Jacobian

    #--Get the wavelength indices for the nonzero values in the weight matrix. 
    tempMat = cp.tile( cp.arange(mp.Nsbp).reshape((mp.Nsbp,1)), (1,mp.jac.Nzern) )
    mp.jac.sbp_inds = tempMat[mp.jac.weightMat_ele];

    #--Get the Zernike indices for the nonzero elements in the weight matrix. 
    tempMat = cp.tile(mp.jac.zerns,(mp.Nsbp,1));
    mp.jac.zern_inds = tempMat[mp.jac.weightMat_ele];

    pass


def falco_gen_chosen_pupil(mp):
    """
    Function to generate the apodizer representation based on configuration 
    settings.

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

    # Input pupil plane resolution, masks, and coordinates
    #--Resolution at input pupil and DM1 and DM2
    if not hasattr(mp.P2,'full'):
        mp.P2.full = falco.config.Object()
        
    mp.P2.full.dx = mp.P2.D/mp.P1.full.Nbeam
    
    if not hasattr(mp.P2,'compact'):
        mp.P2.compact = falco.config.Object()
        
    mp.P2.compact.dx = mp.P2.D/mp.P1.compact.Nbeam

    whichPupil = mp.whichPupil.upper()
    if whichPupil in ('SIMPLE', 'SIMPLEPROPER'):
                
        inputs = dict([
                ('OD', mp.P1.ODnorm),
                ('ID', mp.P1.IDnorm),
                ('Nstrut', mp.P1.Nstrut),
                ('angStrut', mp.P1.angStrut),
                ('wStrut', mp.P1.wStrut),
                ('stretch', mp.P1.stretch)
                ])
    
        if whichPupil in ('SIMPLEPROPER',):
            inputs['flagPROPER'] = True
        
        if mp.full.flagGenPupil:
            inputs['Nbeam'] = mp.P1.full.Nbeam
            inputs['Npad'] = 2**falco.nextpow2(mp.P1.full.Nbeam) 
            mp.P1.full.mask = falco.masks.falco_gen_pupil_Simple(inputs)
        
        #--Generate low-res input pupil for the 'compact' model
        if mp.compact.flagGenPupil:
            inputs['Nbeam'] = mp.P1.compact.Nbeam # number of points across usable pupil   
            inputs['Npad'] = 2**falco.nextpow2(mp.P1.compact.Nbeam) 
            mp.P1.compact.mask = falco.masks.falco_gen_pupil_Simple(inputs)    
    
    
    elif whichPupil == 'WFIRST20191009':
        if mp.full.flagGenPupil:
            mp.P1.full.mask = falco.masks.falco_gen_pupil_WFIRST_CGI_20191009(mp.P1.full.Nbeam, mp.centering)

        if mp.compact.flagGenPupil:
            mp.P1.compact.mask = falco.masks.falco_gen_pupil_WFIRST_CGI_20191009(mp.P1.compact.Nbeam, mp.centering)
    
    elif whichPupil == 'WFIRST180718':
        if mp.full.flagGenPupil:
            mp.P1.full.mask = falco.masks.falco_gen_pupil_WFIRST_CGI_180718(mp.P1.full.Nbeam, mp.centering)

        if mp.compact.flagGenPupil:
            mp.P1.compact.mask = falco.masks.falco_gen_pupil_WFIRST_CGI_180718(mp.P1.compact.Nbeam, mp.centering)
            
    elif whichPupil == 'LUVOIRAFINAL':
#        print('whichPupil = %s'%(whichPupil))
        pass
    elif whichPupil == 'LUVOIRA5':
#        print('whichPupil = %s'%(whichPupil))
        pass
    elif whichPupil == 'LUVOIRA0':
#        print('whichPupil = %s'%(whichPupil))
        pass
    elif whichPupil == 'LUVOIR_B_OFFAXIS':
#        print('whichPupil = %s'%(whichPupil))
        if mp.full.flagGenPupil:
            mp.P1.full.mask = falco.masks.falco_gen_pupil_LUVOIR_B(mp.P1.full.Nbeam)

        if mp.compact.flagGenPupil:
            mp.P1.compact.mask = falco.masks.falco_gen_pupil_LUVOIR_B(mp.P1.compact.Nbeam)
    
        pass
    elif whichPupil == 'DST_LUVOIRB':
#        print('whichPupil = %s'%(whichPupil))
        pass
    elif whichPupil == 'HABEX_B_OFFAXIS':
#        print('whichPupil = %s'%(whichPupil))
        pass
    elif whichPupil == 'ISAT':
#        print('whichPupil = %s'%(whichPupil))
        pass
    else:
#        print('whichPupil = %s'%(whichPupil))
        pass

    mp.P1.compact.Narr = len(mp.P1.compact.mask) #--Number of pixels across the array containing the input pupil in the compact model
    
    ##--NORMALIZED (in pupil diameter) coordinate grids in the input pupil for making the tip/tilted input wavefront within the compact model
    if mp.centering.lower() == ('interpixel'):
        mp.P2.compact.xsDL = cp.linspace(-(mp.P1.compact.Narr-1)/2, (mp.P1.compact.Narr-1)/2,mp.P1.compact.Narr)*mp.P2.compact.dx/mp.P2.D
    else:
        mp.P2.compact.xsDL = cp.linspace(-mp.P1.compact.Narr/2, (mp.P1.compact.Narr/2-1),mp.P1.compact.Narr)*mp.P2.compact.dx/mp.P2.D


    [mp.P2.compact.XsDL,mp.P2.compact.YsDL] = cp.meshgrid(mp.P2.compact.xsDL,mp.P2.compact.xsDL)

    if not hasattr(mp.P1.full, 'Narr'):    
        if(mp.full.flagPROPER):
            if mp.centering.lower() == 'interpixel':
                mp.P1.full.Narr = int(falco.utils.ceil_even(mp.P1.full.Nbeam))
            else:
                mp.P1.full.Narr = int(2**falco.utils.nextpow2(mp.P1.full.Nbeam+1)) #falco.utils.ceil_even(mp.P1.full.Nbeam+1)
        else:
            mp.P1.full.Narr = len(mp.P1.full.mask)  ##--Total number of pixels across array containing the pupil in the full model. Add 2 pixels to Nbeam when the beam is pixel-centered.


    #--NORMALIZED (in pupil diameter) coordinate grids in the input pupil for making the tip/tilted input wavefront within the full model
    if mp.centering.lower() == ('interpixel'):
        mp.P2.full.xsDL = cp.linspace(-(mp.P1.full.Narr-1)/2, (mp.P1.full.Narr-1)/2,mp.P1.full.Narr)*mp.P2.full.dx/mp.P2.D
    else:
        mp.P2.full.xsDL = cp.linspace(-mp.P1.full.Narr/2, (mp.P1.full.Narr/2-1),mp.P1.full.Narr)*mp.P2.full.dx/mp.P2.D

    [mp.P2.full.XsDL,mp.P2.full.YsDL] = cp.meshgrid(mp.P2.full.xsDL,mp.P2.full.xsDL)
    pass


def falco_gen_chosen_apodizer(mp):
    """
    Function to generate the apodizer representation based on configuration 
    settings.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    Returns
    -------
    NA
    """

    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    pass

    if not hasattr(mp.P3,'full'):
        mp.P3.full = falco.config.Object()
        
    if not hasattr(mp.P3,'compact'):
        mp.P3.compact = falco.config.Object()   

    
    if mp.flagApod:
        #--mp.apodType is used only when generating certain types of analytical apodizers
        if mp.apodType.lower() in ('simple'): #--A simple, circular aperture stop
            #--inputs common to both the compact and full models
            inputs = {}
            inputs["ID"] = mp.P3.IDnorm
            inputs["OD"] = mp.P3.ODnorm

            inputs["Nstrut"] = mp.P3.Nstrut
            inputs["angStrut"] = mp.P3.angStrut # Angles of the struts 
            inputs["wStrut"] = mp.P3.wStrut # spider width (fraction of the pupil diameter)
            inputs["stretch"] = mp.P3.stretch

            #--Full model only
            inputs["Nbeam"] = mp.P1.full.Nbeam # number of points across incoming beam 
            inputs["Npad"] = 2**(falco.utils.nextpow2(mp.P1.full.Nbeam)) 
            
            if(mp.full.flagGenApod):
                mp.P3.full.mask = falco.masks.falco_gen_pupil_Simple( inputs );
            else:
                disp('*** Simple aperture stop to be loaded instead of generated for full model. ***')
        
            # Compact model only 
            inputs["Nbeam"] = mp.P1.compact.Nbeam #--Number of pixels across the aperture or beam (independent of beam centering)
            inputs["Npad"] = 2**(falco.utils.nextpow2(mp.P1.compact.Nbeam))
            
            if(mp.compact.flagGenApod):
                mp.P3.compact.mask = falco.masks.falco_gen_pupil_Simple( inputs );
            else:
                disp('*** Simple aperture stop to be loaded instead of generated for compact model. ***')
            

    mp.P3.full.dummy = 1
    if hasattr(mp.P3.full,'mask'):   #==false || isfield(mp.P3.compact,'mask')==false)
        mp.P3.full.Narr = mp.P3.full.mask.shape[0]
    else:
        print('*** If not generated or loaded in a PROPER model, the apodizer must be loaded \n    in the main script or config file into the variable mp.P3.full.mask ***')

    
    mp.P3.compact.dummy = 1
    if hasattr(mp.P3.compact,'mask'):    #    ==false || isfield(mp.P3.compact,'mask')==false)
        mp.P3.compact.Narr = mp.P3.compact.mask.shape[0]
    else:
        print('*** If not generated, the apodizer must be loaded in the main script or config \n    file into the variable mp.P3.compact.mask ***')
    
   
    ##--Set the pixel width [meters]
    mp.P3.full.dx = mp.P2.full.dx
    mp.P3.compact.dx = mp.P2.compact.dx
    pass


def falco_gen_chosen_lyot_stop(mp):
    """
    Function to generate the Lyot stop representation based on configuration 
    settings.

    Detailed description here

    Created on 2018-05-29 by A.J. Riggs.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
        
    Returns
    -------
    NA
    """

    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    
    #--Resolution at Lyot Plane
    if(mp.full.flagPROPER==False):
        mp.P4.full.dx = mp.P4.D/mp.P4.full.Nbeam


    ### Changes to the pupil
    changes = {}
#    class Changes(object):
#        pass
#    
#    changes = Changes()
    
    
    """
    % switch mp.layout
    %     case{'wfirst_phaseb_simple','wfirst_phaseb_proper'}
    %         
    %     otherwise
    %         mp.P4.full.dx = mp.P4.D/mp.P4.full.Nbeam;
    % end
    """
    mp.P4.compact.dx = mp.P4.D/mp.P4.compact.Nbeam

    whichPupil = mp.whichPupil.upper()
    if whichPupil in ('SIMPLE','SIMPLEPROPER','DST_LUVOIRB','ISAT'):
        
        inputs = dict([
                ('OD', mp.P4.ODnorm),
                ('ID', mp.P4.IDnorm),
                ('Nstrut', mp.P4.Nstrut),
                ('angStrut', mp.P4.angStrut),
                ('wStrut', mp.P4.wStrut),
                ('stretch', mp.P4.stretch)
                ])
    
        if whichPupil in ('SIMPLEPROPER',):
            inputs['flagPROPER'] = True
        
        if mp.full.flagGenLS:
            inputs['Nbeam'] = mp.P4.full.Nbeam
            inputs['Npad'] = 2**falco.nextpow2(mp.P4.full.Nbeam) 
            mp.P4.full.mask = falco.masks.falco_gen_pupil_Simple(inputs)
        
        #--Generate low-res input pupil for the 'compact' model
        if mp.compact.flagGenLS:
            inputs['Nbeam'] = mp.P4.compact.Nbeam # number of points across usable pupil   
            inputs['Npad'] = 2**falco.nextpow2(mp.P4.compact.Nbeam) 
            mp.P4.compact.mask = falco.masks.falco_gen_pupil_Simple(inputs)    
        
        """
        if whichPupil in ('SIMPLEPROPER'):  
            inputs.flagPROPER = true
        
        inputs.Nbeam = mp.P4.full.Nbeam # number of points across incoming beam 
        inputs.Npad = 2^(falco.utils.nextpow2nextpow2(mp.P4.full.Nbeam))
        inputs.OD = mp.P4.ODnorm
        inputs.ID = mp.P4.IDnorm
        inputs.Nstrut = mp.P4.Nstrut
        inputs.angStrut = mp.P4.angStrut # Angles of the struts 
        inputs.wStrut = mp.P4.wStrut # spider width (fraction of the pupil diameter)

        mp.P4.full.mask = falco_gen_pupil_Simple(inputs)
        
        inputs.Nbeam = mp.P4.compact.Nbeam #--Number of pixels across the aperture or beam (independent of beam centering)
        inputs.Npad = 2^(nextpow2(mp.P4.compact.Nbeam))
        
        mp.P4.compact.mask = falco_gen_pupil_Simple(inputs)
        """
        pass
    

    elif whichPupil == 'WFIRST20191009':
        #--Define Lyot stop generator function inputs for the 'full' optical model
        if mp.compact.flagGenLS or mp.full.flagGenLS:
            changes["ID"] = mp.P4.IDnorm
            changes["OD"] = mp.P4.ODnorm
            changes["wStrut"] = mp.P4.wStrut
            changes["flagRot180"] = True
        
        #kwargs = changes.__dict__ #convert changes to dictionary to use as input to gen_pupil routine
        if(mp.full.flagGenLS):
            mp.P4.full.mask = falco.masks.falco_gen_pupil_WFIRST_CGI_20191009(mp.P4.full.Nbeam,mp.centering,changes)
        
        ##--Make or read in Lyot stop (LS) for the 'compact' model
        if(mp.compact.flagGenLS):
            mp.P4.compact.mask = falco.masks.falco_gen_pupil_WFIRST_CGI_20191009(mp.P4.compact.Nbeam,mp.centering,changes)
        
        if hasattr(mp, 'LSshape'):
            LSshape = mp.LSshape.lower()
            if LSshape in ('bowtie'):
                #--Define Lyot stop generator function inputs in a structure
                inputs = {}
                inputs["ID"] = mp.P4.IDnorm # (pupil diameters)
                inputs["OD"]= mp.P4.ODnorm # (pupil diameters)
                inputs["ang"] = mp.P4.ang # (degrees)
                inputs["centering"] = mp.centering # 'interpixel' or 'pixel'

#                #--Optional inputs and their defaults
#                centering = inputs['centering'] if 'centering' in inputs.keys() else 'pixel'  #--Default to pixel centering
#                xShear = inputs['xShear'] if 'xShear' in inputs.keys() else  0. #--x-axis shear of mask [pupil diameters]
#                yShear = inputs['yShear'] if 'yShear' in inputs.keys() else  0. #--y-axis shear of mask [pupil diameters]
#                clocking = inputs['clocking'] if 'clocking' in inputs.keys() else 0. #--Clocking of the mask [degrees]
#                magfac = inputs['magfac'] if 'magfac' in inputs.keys() else 1. #--magnification factor of the pupil diameter
                
                if(mp.full.flagGenLS):
                    inputs["Nbeam"] = mp.P4.full.Nbeam
                    mp.P4.full.mask = falco.masks.falco_gen_bowtie_LS(inputs)
                
                #--Make bowtie Lyot stop (LS) for the 'compact' model
                if(mp.compact.flagGenLS):
                    inputs["Nbeam"] = mp.P4.compact.Nbeam
                    mp.P4.compact.mask = falco.masks.falco_gen_bowtie_LS(inputs)


    elif whichPupil == 'WFIRST180718':
        #--Define Lyot stop generator function inputs for the 'full' optical model
        if mp.compact.flagGenLS or mp.full.flagGenLS:
            changes["ID"] = mp.P4.IDnorm
            changes["OD"] = mp.P4.ODnorm
            changes["wStrut"] = mp.P4.wStrut
            changes["flagRot180"] = True
        
        #kwargs = changes.__dict__ #convert changes to dictionary to use as input to gen_pupil routine
        if(mp.full.flagGenLS):
            mp.P4.full.mask = falco.masks.falco_gen_pupil_WFIRST_CGI_180718(mp.P4.full.Nbeam,mp.centering,changes)
        
        ##--Make or read in Lyot stop (LS) for the 'compact' model
        if(mp.compact.flagGenLS):
            mp.P4.compact.mask = falco.masks.falco_gen_pupil_WFIRST_CGI_180718(mp.P4.compact.Nbeam,mp.centering,changes)
        
        if hasattr(mp, 'LSshape'):
            LSshape = mp.LSshape.lower()
            if LSshape in ('bowtie'):
                #--Define Lyot stop generator function inputs in a structure
                inputs = {}
                inputs["ID"] = mp.P4.IDnorm # (pupil diameters)
                inputs["OD"]= mp.P4.ODnorm # (pupil diameters)
                inputs["ang"] = mp.P4.ang # (degrees)
                inputs["centering"] = mp.centering # 'interpixel' or 'pixel'

#                #--Optional inputs and their defaults
#                centering = inputs['centering'] if 'centering' in inputs.keys() else 'pixel'  #--Default to pixel centering
#                xShear = inputs['xShear'] if 'xShear' in inputs.keys() else  0. #--x-axis shear of mask [pupil diameters]
#                yShear = inputs['yShear'] if 'yShear' in inputs.keys() else  0. #--y-axis shear of mask [pupil diameters]
#                clocking = inputs['clocking'] if 'clocking' in inputs.keys() else 0. #--Clocking of the mask [degrees]
#                magfac = inputs['magfac'] if 'magfac' in inputs.keys() else 1. #--magnification factor of the pupil diameter
                
                if(mp.full.flagGenLS):
                    inputs["Nbeam"] = mp.P4.full.Nbeam
                    mp.P4.full.mask = falco.masks.falco_gen_bowtie_LS(inputs)
                
                #--Make bowtie Lyot stop (LS) for the 'compact' model
                if(mp.compact.flagGenLS):
                    inputs["Nbeam"] = mp.P4.compact.Nbeam
                    mp.P4.compact.mask = falco.masks.falco_gen_bowtie_LS(inputs)
       
    elif whichPupil in ('LUVOIRAFINAL'):
        pass
        """
        #--Define Lyot stop generator function inputs for the 'full' optical model
        inputs.Nbeam = mp.P4.full.Nbeam # number of points across incoming beam  
        inputs.Dbeam = mp.P1.D
        inputs.ID = mp.P4.IDnorm
        inputs.OD = mp.P4.ODnorm
        inputs.wStrut = mp.P4.wStrut
        inputs.centering = mp.centering
        #--Make or read in Lyot stop (LS) for the 'full' model
        mp.P4.full.mask = falco_gen_pupil_LUVOIR_A_final_Lyot(inputs,'ROT180');
        
        #--Make or read in Lyot stop (LS) for the 'compact' model
        inputs.Nbeam = mp.P4.compact.Nbeam;     # number of points across incoming beam           
        mp.P4.compact.mask = falco_gen_pupil_LUVOIR_A_final_Lyot(inputs,'ROT180')
        """
    elif whichPupil in ('LUVOIRA5','LUVOIRA0'):
        #--Define Lyot stop generator function inputs for the 'full' optical model
        pass
        """
        inputs.Nbeam = mp.P4.full.Nbeam # number of points across incoming beam  
        inputs.Dbeam = mp.P1.D;
        inputs.ID = mp.P4.IDnorm;
        inputs.OD = mp.P4.ODnorm;
        inputs.wStrut = mp.P4.wStrut;
        inputs.centering = mp.centering;
        %--Make or read in Lyot stop (LS) for the 'full' model
        mp.P4.full.mask = falco_gen_pupil_LUVOIR_A_5_Lyot_struts(inputs,'ROT180');
        
        %--Make or read in Lyot stop (LS) for the 'compact' model
        inputs.Nbeam = mp.P4.compact.Nbeam;     % number of points across incoming beam           
        mp.P4.compact.mask = falco_gen_pupil_LUVOIR_A_5_Lyot_struts(inputs,'ROT180');
        """
    elif whichPupil in ('LUVOIR_B_OFFAXIS','HABEX_B_OFFAXIS'):
        
        inputs = {} # initialize
        inputs["ID"] = mp.P4.IDnorm #- Outer diameter (fraction of Nbeam)
        inputs["OD"] = mp.P4.ODnorm#- Inner diameter (fraction of Nbeam)
#        inputs["Nstrut"] = 0 #- Number of struts
#        inputs["angStrut"] = cp.array([])#- Array of struct angles (deg)
#        inputs["wStrut"] = cp.array([]) #- Strut widths (fraction of Nbeam)
#        inputs["stretch"] = 1.#- Create an elliptical aperture by changing Nbeam along
        #                  the horizontal direction by a factor of stretch (PROPER
        #                  version isn't implemented as of March 2019).

        inputs["Nbeam"] = mp.P4.compact.Nbeam #- Number of samples across the beam 
        inputs["Npad"] = int(2**falco.utils.nextpow2( falco.utils.ceil_even(mp.P4.compact.Nbeam )))
        mp.P4.compact.mask = falco.masks.falco_gen_pupil_Simple( inputs )

        inputs["Nbeam"] = mp.P4.full.Nbeam #- Number of samples across the beam 
        inputs["Npad"] = int(2**falco.utils.nextpow2( falco.utils.ceil_even(mp.P4.full.Nbeam )))
        mp.P4.full.mask = falco.masks.falco_gen_pupil_Simple( inputs )


        pass
        """  
        #--Full model
        inputs.Nbeam = mp.P4.full.Nbeam # number of points across incoming beam 
        inputs.Npad = 2^(nextpow2(mp.P4.full.Nbeam));
        inputs.OD = mp.P4.ODnorm;
        inputs.ID = mp.P4.IDnorm;
        inputs.Nstrut = 0;
        inputs.angStrut = [] # Angles of the struts 
        inputs.wStrut = 0 # spider width (fraction of the pupil diameter)

        mp.P4.full.mask = falco_gen_pupil_Simple(inputs);
        
        pad_pct = mp.P4.padFacPct;
        if(pad_pct>0) %--Also apply an eroded/padded version of the segment gaps

            pupil0 = mp.P1.full.mask;
            Nbeam = inputs.Nbeam;
            Npad = inputs.Npad;

            xsD = (-Npad/2:(Npad/2-1))/Nbeam #--coordinates, normalized to the pupil diameter
            [XS,YS] = meshgrid(xsD);
            RS = sqrt(XS.^2 + YS.^2);
        
            pupil1 = 1-pupil0;

            spot = zeros(Npad);
            spot(RS <= pad_pct/100) = 1;

            pupil4 = ifftshift(ifft2(fft2(fftshift(pupil1)).*fft2(fftshift(spot))));
            pupil4 = abs(pupil4);
            pupil4 = pupil4/max(pupil4(:));

            pupil5 = 1-pupil4;

            thresh = 0.99;
            pupil5(pupil5<thresh) = 0;
            pupil5(pupil5>=thresh) = 1;

            mp.P4.full.mask = mp.P4.full.mask.*pupil5;            
        end
        
        #--Compact model
        inputs.Nbeam = mp.P4.compact.Nbeam #--Number of pixels across the aperture or beam (independent of beam centering)
        inputs.Npad = 2^(nextpow2(mp.P4.compact.Nbeam))
        
        mp.P4.compact.mask = falco_gen_pupil_Simple(inputs)
        
        if(pad_pct>0): #--Also apply an eroded/padded version of the segment gaps
            pupil0 = mp.P1.compact.mask
            Nbeam = inputs.Nbeam
            Npad = inputs.Npad

            xsD = (-Npad/2:(Npad/2-1))/Nbeam #--coordinates, normalized to the pupil diameter
            [XS,YS] = meshgrid(xsD)
            RS = sqrt(XS.^2 + YS.^2)

            pupil1 = 1-pupil0

            spot = zeros(Npad)
            spot(RS <= pad_pct/100) = 1

            pupil4 = ifftshift(ifft2(fft2(fftshift(pupil1)).*fft2(fftshift(spot))))
            pupil4 = abs(pupil4)
            pupil4 = pupil4/max(pupil4(:))

            pupil5 = 1-pupil4

            thresh = 0.99
            pupil5(pupil5<thresh) = 0
            pupil5(pupil5>=thresh) = 1

            mp.P4.compact.mask = mp.P4.compact.mask.*pupil5
    """


    ## Crop down the Lyot stop(s) to get rid of extra zero padding for the full model
    if(False): # mp.coro.upper() in ('VORTEX','VC','AVC'):
        mp.P4.full.Narr = mp.P4.full.mask.shape[0]
        mp.P4.full.croppedMask = mp.P4.full.mask
        mp.P4.compact.Narr = mp.P4.compact.mask.shape[0]
        mp.P4.compact.croppedMask = mp.P4.compact.mask
    else:
        if(mp.full.flagPROPER==False):
            #--Crop down the high-resolution Lyot stop to get rid of extra zero padding
            LSsum = cp.sum(mp.P4.full.mask)
            LSdiff = 0 
            counter = 2
            while(cp.abs(LSdiff) <= 1e-7):
                mp.P4.full.Narr = len(mp.P4.full.mask)-counter
                LSdiff = LSsum - cp.sum(falco.utils.padOrCropEven(mp.P4.full.mask, mp.P4.full.Narr-2)) #--Subtract an extra 2 to negate the extra step that overshoots.
                counter = counter + 2
            
            mp.P4.full.croppedMask = falco.utils.padOrCropEven(mp.P4.full.mask,mp.P4.full.Narr) #--The cropped-down Lyot stop for the full model. 
        
        ## --Crop down the low-resolution Lyot stop to get rid of extra zero padding. Speeds up the compact model.
        LSsum = cp.sum(mp.P4.compact.mask)
        LSdiff= 0
        counter = 2
        while(abs(LSdiff) <= 1e-7):
            mp.P4.compact.Narr = len(mp.P4.compact.mask)-counter #--Number of points across the cropped-down Lyot stop
            LSdiff = LSsum - cp.sum(falco.utils.padOrCropEven(mp.P4.compact.mask, mp.P4.compact.Narr-2))  #--Subtract an extra 2 to negate the extra step that overshoots.
            counter = counter + 2

        mp.P4.compact.croppedMask = falco.utils.padOrCropEven(mp.P4.compact.mask,mp.P4.compact.Narr) #--The cropped-down Lyot stop for the compact model
 
    #--(METERS) Lyot plane coordinates (over the cropped down to Lyot stop mask) for MFTs in the compact model from the FPM to the LS.
    if mp.centering == 'interpixel':
        #mp.P4.compact.xs = (-(mp.P4.compact.Narr-1)/2:(mp.P4.compact.Narr-1)/2)*mp.P4.compact.dx
        mp.P4.compact.xs = cp.linspace(-(mp.P4.compact.Narr-1)/2, (mp.P4.compact.Narr-1)/2,mp.P4.compact.Narr)*mp.P4.compact.dx
    else:
        #mp.P4.compact.xs = (-mp.P4.compact.Narr/2:(mp.P4.compact.Narr/2-1))*mp.P4.compact.dx
        mp.P4.compact.xs = cp.linspace(-mp.P4.compact.Narr/2, (mp.P4.compact.Narr/2-1),mp.P4.compact.Narr)*mp.P4.compact.dx
    
    mp.P4.compact.ys = cp.transpose(mp.P4.compact.xs) #transpose of array (196x1)
    pass


def falco_plot_superposed_pupil_masks(mp):
    ## Plot the pupil and Lyot stop on top of each other to make sure they are aligned correctly
    #--Only for coronagraphs using Babinet's principle, for which the input
    #pupil and Lyot plane have the same resolution.
    if mp.coro.upper() in ['FOHLC','HLC','LC','APLC','VC','AVC', 'VORTEX']:
        if mp.flagPlot:
            P4mask = falco.utils.padOrCropEven(mp.P4.compact.mask,mp.P1.compact.Narr)
            P4mask = cp.rot90(P4mask,2);
            if mp.centering.lower() == 'pixel':
               #P4mask = circshift(P4mask,[1 1]);
               P4mask = cp.roll(P4mask,(1,1),(0,1))
            P1andP4 = mp.P1.compact.mask + P4mask;
            plt.figure(301); plt.imshow(P1andP4); plt.colorbar(); plt.title('Pupil and LS Superimposed'); plt.pause(1e-2)

            if mp.flagApod:
                P1andP3 = mp.P1.compact.mask + falco.utils.padOrCropEven(mp.P3.compact.mask,len(mp.P1.compact.mask));
                plt.figure(302); plt.imshow(P1andP3); plt.colorbar(); plt.title('Pupil and Apod Superimposed'); plt.pause(1e-2)
    pass


def falco_gen_FPM(mp):
    #% Generate FPM if necessary. If HLC, uses DM8 and DM9.\
    ## HLC and EHLC FPM: Initialization and Generation
    
    print(mp.layout.lower())
    print(mp.coro.upper())
    if mp.layout.lower() == 'fourier':
        if mp.coro.upper() == 'HLC':
            if mp.dm9.inf0name == '3foldZern':
                falco_setup_FPM_HLC_3foldZern(mp);
            else:
                falco_setup_FPM_HLC(mp);
            falco.configs.falco_config_gen_FPM_HLC(mp);
    
            ##--Pre-compute the complex transmission of the allowed Ni+PMGI FPMs.
            if mp.coro in ('HLC',):
                [mp.complexTransCompact,mp.complexTransFull] = falco_gen_complex_trans_table(mp);

    ## Generate FPM
    if mp.coro.upper() in ['LC', 'APLC']: #--Occulting spot FPM (can be HLC-style and partially transmissive)
        falco_gen_FPM_LC(mp)
        #falco.configs.falco_config_gen_FPM_LC(mp);
    elif mp.coro.upper() in ['SPLC', 'FLC']:
        falco.setup.falco_gen_FPM_SPLC(mp);
    elif mp.coro.upper() == 'RODDIER':
        falco.setup.falco_gen_FPM_Roddier(mp);
    pass


def falco_get_FPM_coordinates(mp):
    ## FPM coordinates, [meters] and [dimensionless]
    
    if mp.coro.upper() in ['VORTEX','VC','AVC']: #--Nothing needed to run the vortex model
        pass
    elif mp.coro.upper() == 'SPHLC':
        pass
    else:
        if mp.layout in ['wfirst_phaseb_simple','wfirst_phaseb_proper']:
            pass
        else:
            #--FPM (at F3) Resolution [meters]
            mp.F3.full.dxi = (mp.fl*mp.lambda0/mp.P2.D)/mp.F3.full.res;
            mp.F3.full.deta = mp.F3.full.dxi;
        mp.F3.compact.dxi = (mp.fl*mp.lambda0/mp.P2.D)/mp.F3.compact.res;
        mp.F3.compact.deta = mp.F3.compact.dxi;
    
        #--Coordinates in FPM plane in the compact model [meters]
        if mp.centering.lower() == 'interpixel' or mp.F3.compact.Nxi%2 == 1:
            mp.F3.compact.xis  = cp.arange(-(mp.F3.compact.Nxi-1)/2,(mp.F3.compact.Nxi-1)/2)*mp.F3.compact.dxi;
            mp.F3.compact.etas = cp.arange(-(mp.F3.compact.Neta-1)/2,(mp.F3.compact.Neta-1)/2)*mp.F3.compact.deta;
        else:
            mp.F3.compact.xis  = cp.arange(-mp.F3.compact.Nxi/2, (mp.F3.compact.Nxi/2))*mp.F3.compact.dxi;
            mp.F3.compact.etas = cp.arange(-mp.F3.compact.Neta/2,(mp.F3.compact.Neta/2))*mp.F3.compact.deta;

        if mp.layout in ['wfirst_phaseb_simple','wfirst_phaseb_proper']:
            pass
        else:
            #--Coordinates (dimensionless [DL]) for the FPMs in the full model
            if mp.centering.lower() == 'interpixel' or mp.F3.full.Nxi%2==1:
                mp.F3.full.xisDL  = cp.arange(-(mp.F3.full.Nxi-1)/2,(mp.F3.full.Nxi-1)/2)/mp.F3.full.res;
                mp.F3.full.etasDL = cp.arange(-(mp.F3.full.Neta-1)/2, (mp.F3.full.Neta-1)/2)/mp.F3.full.res;
            else:
                mp.F3.full.xisDL  = cp.arange(-mp.F3.full.Nxi/2,(mp.F3.full.Nxi/2-1))/mp.F3.full.res;
                mp.F3.full.etasDL = cp.arange(-mp.F3.full.Neta/2,(mp.F3.full.Neta/2-1))/mp.F3.full.res;

        #--Coordinates (dimensionless [DL]) for the FPMs in the compact model
        if mp.centering.lower() == 'interpixel' or mp.F3.compact.Nxi%2==1:
            mp.F3.compact.xisDL  = cp.arange(-(mp.F3.compact.Nxi-1)/2,(mp.F3.compact.Nxi-1)/2)/mp.F3.compact.res;
            mp.F3.compact.etasDL = cp.arange(-(mp.F3.compact.Neta-1)/2,(mp.F3.compact.Neta-1)/2)/mp.F3.compact.res;
        else:
            mp.F3.compact.xisDL  = cp.arange(-mp.F3.compact.Nxi/2,(mp.F3.compact.Nxi/2-1))/mp.F3.compact.res;
            mp.F3.compact.etasDL = cp.arange(-mp.F3.compact.Neta/2,(mp.F3.compact.Neta/2-1))/mp.F3.compact.res;
    pass


def falco_get_Fend_resolution(mp):
    ## Sampling/Resolution and Scoring/Correction Masks for Final Focal Plane (Fend.
    
    mp.Fend.dxi = (mp.fl*mp.lambda0/mp.P4.D)/mp.Fend.res; # sampling at Fend.[meters]
    mp.Fend.deta = mp.Fend.dxi; # sampling at Fend.[meters]    
    
    if mp.flagFiber:
        mp.Fend.lenslet.D = 2*mp.Fend.res*mp.Fend.lensletWavRad*mp.Fend.dxi;
        mp.Fend.x_lenslet_phys = mp.Fend.dxi*mp.Fend.res*mp.Fend.x_lenslet;
        mp.Fend.y_lenslet_phys = mp.Fend.deta*mp.Fend.res*mp.Fend.y_lenslet;
    
        mp.F5.dxi = mp.lensletFL*mp.lambda0/mp.Fend.lenslet.D/mp.F5.res;
        mp.F5.deta = mp.F5.dxi;
    pass


def falco_configure_dark_hole_region(mp):
    #% Software Mask for Correction (corr) and Scoring (score)
    ## Software Mask for Correction (corr) and Scoring (score)
    
    #--Set inputs:
    maskCorr = {}
    maskCorr["pixresFP"] = mp.Fend.res;
    maskCorr["rhoInner"] = mp.Fend.corr.Rin #--lambda0/D
    maskCorr["rhoOuter"] = mp.Fend.corr.Rout  #--lambda0/D
    maskCorr["angDeg"] = mp.Fend.corr.ang #--degrees
    maskCorr["centering"] = mp.centering
    maskCorr["FOV"] = mp.Fend.FOV
    maskCorr["whichSide"] = mp.Fend.sides; #--which (sides) of the dark hole have open
    if hasattr(mp.Fend,'shape'):
        maskCorr.shape = mp.Fend.shape
    
    #--Compact Model: Generate Software Mask for Correction 
    mp.Fend.corr.mask, mp.Fend.xisDL, mp.Fend.etasDL = falco.masks.falco_gen_SW_mask(maskCorr);
    mp.Fend.corr.settings = maskCorr; #--Store values for future reference
    #--Size of the output image 
    #--Need the sizes to be the same for the correction and scoring masks
    mp.Fend.Nxi  = mp.Fend.corr.mask.shape[1] #size(mp.Fend.corr.mask,2);
    mp.Fend.Neta = mp.Fend.corr.mask.shape[0] #size(mp.Fend.corr.mask,1);

    XIS, ETAS = cp.meshgrid(mp.Fend.xisDL, mp.Fend.etasDL)
    mp.Fend.RHOS = cp.sqrt(XIS**2 + ETAS**2)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    #--Mask defining the area covered by the lenslet.  Only the immediate area
    #around the lenslet is propagated, saving computation time.  This lenslet
    #can then be moved around to different positions in Fend.
    if mp.flagFiber:
        maskLenslet["pixresFP"] = mp.Fend.res;
        maskLenslet["rhoInner"] = 0;
        maskLenslet["rhoOuter"] = mp.Fend.lensletWavRad;
        maskLenslet["angDeg"] = mp.Fend.corr.ang;
        maskLenslet["centering"] = mp.centering;
        maskLenslet["FOV"] = mp.Fend.FOV;
        maskLenslet["whichSide"] = mp.Fend.sides;
        mp.Fend.lenslet.mask, unused_1, unused_2 = falco.masks.falco_gen_SW_mask(**maskLenslet);
    
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
        #--Dummy mask to calculate the F5 coordinates correctly.
        maskF5["pixresFP"] = mp.F5.res;
        maskF5["rhoInner"] = 0;
        maskF5["rhoOuter"] = 1.22;
        maskF5["angDeg"] = 180;
        maskF5["centering"] = mp.centering;
        maskF5["FOV"] = mp.F5.FOV;
        maskF5["whichSide"] = mp.Fend.sides;
        mp.F5.mask, mp.F5.xisDL, mp.F5.etasDL = falco.masks.falco_gen_SW_mask(maskF5);
    
        #--Size of the output image in F5
        mp.F5.Nxi = mp.F5.mask.shape[1] #size(mp.F5.mask, 2);
        mp.F5.Neta = mp.F5.mask.shape[0]
    
        ## Set up the fiber mode in F5
    
        V = 2*pi/mp.lambda0*mp.fiber.a*mp.fiber.NA;
        W = 1.1428*V - 0.996;
        U = cp.sqrt(V**2 - W**2);
    
        maskFiberCore["pixresFP"] = mp.F5.res;
        maskFiberCore["rhoInner"] = 0;
        maskFiberCore["rhoOuter"] = mp.fiber.a;
        maskFiberCore["angDeg"] = 180;
        maskFiberCore["FOV"] = mp.F5.FOV;
        maskFiberCore["whichSide"] = mp.Fend.sides;
        mp.F5.fiberCore.mask, unused_1, unused_2 = falco.masks.falco_gen_SW_mask(maskFiberCore);
    
        maskFiberCladding["pixresFP"] = mp.F5.res;
        maskFiberCladding["rhoInner"] = mp.fiber.a;
        maskFiberCladding["rhoOuter"] = 10;
        maskFiberCladding["angDeg"] = 180;
        maskFiberCladding["FOV"] = mp.F5.FOV;
        maskFiberCladding["whichSide"] = mp.Fend.sides;
        mp.F5.fiberCladding.mask, unused_1, unused_2 = falco.masks.falco_gen_SW_mask(maskFiberCladding);
    
        F5XIS, F5ETAS = cp.meshgrid(mp.F5.xisDL, mp.F5.etasDL);
    
        mp.F5.RHOS = cp.sqrt((F5XIS - mp.F5.fiberPos[0])**2 + (F5ETAS - mp.F5.fiberPos[1])**2);
        mp.F5.fiberCore.mode = mp.F5.fiberCore.mask*scipy.special.j0(U*mp.F5.RHOS/mp.fiber.a)/scipy.special.j0(0,U);
        mp.F5.fiberCladding.mode = mp.F5.fiberCladding.mask*scipy.special.k0(W*mp.F5.RHOS/mp.fiber.a)/scipy.special.k0(W);
        mp.F5.fiberCladding.mode[cp.isnan(mp.F5.fiberCladding.mode)] = 0;
        mp.F5.fiberMode = mp.F5.fiberCore.mode + mp.F5.fiberCladding.mode;
        fiberModeNorm = cp.sqrt(cp.sum(cp.sum(cp.abs(mp.F5.fiberMode)**2)));
        mp.F5.fiberMode = mp.F5.fiberMode/fiberModeNorm;

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    #--Evaluation Model for Computing Throughput (same as Compact Model but
    # with different Fend.resolution)
    maskCorr["pixresFP"] = mp.Fend.eval.res; #--Assign the resolution
    mp.Fend.eval.mask, mp.Fend.eval.xisDL, mp.Fend.eval.etasDL = falco.masks.falco_gen_SW_mask(maskCorr);  #--Generate the mask
    mp.Fend.eval.Nxi  = mp.Fend.eval.mask.shape[1]
    mp.Fend.eval.Neta = mp.Fend.eval.mask.shape[0]
    mp.Fend.eval.dxi = (mp.fl*mp.lambda0/mp.P4.D)/mp.Fend.eval.res; # higher sampling at Fend.for evaulation [meters]
    mp.Fend.eval.deta = mp.Fend.eval.dxi; # higher sampling at Fend.for evaulation [meters]   
    
    # (x,y) location [lambda_c/D] in dark hole at which to evaluate throughput
    XIS,ETAS = cp.meshgrid(mp.Fend.eval.xisDL - mp.thput_eval_x, mp.Fend.eval.etasDL - mp.thput_eval_y);
    mp.Fend.eval.RHOS = cp.sqrt(XIS**2 + ETAS**2);
    
    #--Storage array for throughput at each iteration
    mp.thput_vec = cp.zeros((mp.Nitr+1,1));

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    #--Software Mask for Scoring Contrast 
    #--Set inputs
    maskScore = {}
    maskScore["rhoInner"] = mp.Fend.score.Rin; #--lambda0/D
    maskScore["rhoOuter"] = mp.Fend.score.Rout ; #--lambda0/D
    maskScore["angDeg"] = mp.Fend.score.ang; #--degrees
    maskScore["centering"] = mp.centering;
    maskScore["FOV"] = mp.Fend.FOV; #--Determines max dimension length
    maskScore["whichSide"] = mp.Fend.sides; #--which (sides) of the dark hole have open
    if hasattr(mp.Fend,'shape'):  
        maskScore["shape"] = mp.Fend.shape
    #--Compact Model: Generate Software Mask for Scoring Contrast 
    maskScore["pixresFP"] = mp.Fend.res;
    mp.Fend.score.mask, unused_1, unused_2 = falco.masks.falco_gen_SW_mask(maskScore);
    mp.Fend.score.settings = maskScore; #--Store values for future reference
    
    #--Number of pixels used in the dark hole
    mp.Fend.corr.Npix = int(cp.sum(cp.sum(mp.Fend.corr.mask)));
    mp.Fend.score.Npix = int(cp.sum(cp.sum(mp.Fend.score.mask)));
    
    #--Indices of dark hole pixels and logical masks
    if mp.flagFiber:
        #mp.Fend.corr.inds = find(cp.sum(mp.Fend.lenslet.mask,3)~=0);
        #mp.Fend.corr.maskBool = logical(mp.Fend.corr.mask);
        mp.Fend.corr.inds = cp.where(cp.sum(mp.Fend.lenslet.mask,3)!=0);
        mp.Fend.corr.maskBool = cp.array(mp.Fend.corr.mask, dtype=bool)
    else:
        #mp.Fend.corr.inds = find(mp.Fend.corr.mask~=0);
        #mp.Fend.corr.maskBool = logical(mp.Fend.corr.mask);
        mp.Fend.corr.inds = cp.where(mp.Fend.corr.mask!=0);
        mp.Fend.corr.maskBool = cp.array(mp.Fend.corr.mask, dtype=bool)
    
    #mp.Fend.score.inds = find(mp.Fend.score.mask~=0)
    #mp.Fend.score.maskBool = logical(mp.Fend.score.mask);
    mp.Fend.score.inds = cp.where(mp.Fend.score.mask!=0)
    mp.Fend.score.maskBool = cp.array(mp.Fend.score.mask, dtype=bool)
    pass


def falco_set_spatial_weights(mp):
    """
    Set up spatially-based weighting of the dark hole intensity.

    Set up spatially-based weighting of the dark hole intensity in annular zones centered 
    on the star. Zones are specified with rows of three values: zone inner radius [l/D],
    zone outer radius [l/D], and intensity weight. As many rows can be used as desired.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    Returns
    -------
    nothing
        Values are added by reference into the mp structure.
    """

    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')

    #--Define 2-D coordinate grid
    [XISLAMD,ETASLAMD] = cp.meshgrid(mp.Fend.xisDL, mp.Fend.etasDL)
    RHOS = cp.sqrt(XISLAMD**2+ETASLAMD**2)
    mp.Wspatial = mp.Fend.corr.mask #--Convert from boolean to float
    if hasattr(mp, 'WspatialDef'): #--Do only if spatial weights are defined
        if(np.size(mp.WspatialDef)>0): #--Do only if variable is not empty
            for kk in range(0,mp.WspatialDef.shape[0]): #--Increment through the rows
                Wannulus = 1. + (cp.sqrt(mp.WspatialDef[kk,2])-1.)*((RHOS>=mp.WspatialDef[kk,0]) & (RHOS<mp.WspatialDef[kk,1]))
                mp.Wspatial = mp.Wspatial*Wannulus
                
    mp.WspatialVec = mp.Wspatial[mp.Fend.corr.maskBool]
    if(mp.flagFiber and mp.flagLenslet):
        mp.WspatialVec = cp.ones((mp.Fend.Nlens,))
                
    pass


def falco_configure_dm1_and_dm2(mp):
    #% Flesh out the dm1 and dm2 structures
    ## Deformable Mirror (DM) 1 and 2 Parameters
    
    if hasattr(mp,'dm1'):
        # Read the influence function header data from the FITS file
        dx1 = None
        pitch1 = None
        mp.dm1.inf0 = None
        mp.dm1.dx_inf0 = None
        with fits.open(mp.dm1.inf_fn) as hdul:
            PrimaryData = hdul[0].header
            count = 0
            dx1 = PrimaryData['P2PDX_M'] # pixel width of the influence function IN THE FILE [meters];
            pitch1 = PrimaryData['C2CDX_M'] # actuator spacing x (m)
    
            mp.dm1.inf0 = cp.asarray(np.squeeze(np.float32(hdul[0].data))) #hdul[0].data[0,:,:]

        mp.dm1.dx_inf0 = mp.dm1.dm_spacing*(dx1/pitch1);
    
        if mp.dm1.inf_sign[0] in ['-', 'n', 'm']:
            mp.dm1.inf0 = -1*mp.dm1.inf0;
        elif mp.dm1.inf_sign[0] in ['+', 'p']:
            pass
        else:
            raise ValueError('Sign of influence function not recognized')

    if hasattr(mp,'dm2'):
        # Read the influence function header data from the FITS file
        dx2 = None
        pitch2 = None
        mp.dm2.inf0 = None
        mp.dm2.dx_inf0 = None
        with fits.open(mp.dm2.inf_fn) as hdul:
            PrimaryData = hdul[0].header
            count = 0
            dx2 = PrimaryData['P2PDX_M'] # pixel width of the influence function IN THE FILE [meters];
            pitch2 = PrimaryData['C2CDX_M'] # actuator spacing x (m)
    
            mp.dm2.inf0 = cp.asarray(np.squeeze(np.float32(hdul[0].data)))
        mp.dm2.dx_inf0 = mp.dm2.dm_spacing*(dx2/pitch2);
    
        if mp.dm2.inf_sign[0] in ['-','n','m']:
            mp.dm2.inf0 = -1*mp.dm2.inf0;
        elif mp.dm2.inf_sign[0] in ['+', 'p']:
            pass
        else:
            raise ValueError('Sign of influence function not recognized')

    ## DM1
    mp.dm1.centering = mp.centering
    mp.dm1.compact = falco.config.Object()
    mp.dm1.compact = copy.copy(mp.dm1)
    mp.dm1.dx = mp.P2.full.dx
    mp.dm1.compact.dx = mp.P2.compact.dx
    falco.dms.falco_gen_dm_poke_cube(mp.dm1, mp, mp.P2.full.dx, NOCUBE=True);
    if cp.any(mp.dm_ind==1):
        falco.dms.falco_gen_dm_poke_cube(mp.dm1.compact, mp, mp.P2.compact.dx)
    else:        
        falco.dms.falco_gen_dm_poke_cube(mp.dm1.compact, mp, mp.P2.compact.dx, NOCUBE=True)

    ## DM2
    mp.dm2.centering = mp.centering
    mp.dm2.compact = falco.config.Object()
    mp.dm2.compact = copy.copy(mp.dm2)
    mp.dm2.dx = mp.P2.full.dx
    mp.dm2.compact.dx = mp.P2.compact.dx
    falco.dms.falco_gen_dm_poke_cube(mp.dm2, mp, mp.P2.full.dx, NOCUBE=True)
    if cp.any(mp.dm_ind==2):
        falco.dms.falco_gen_dm_poke_cube(mp.dm2.compact, mp, mp.P2.compact.dx)
    else:    
        falco.dms.falco_gen_dm_poke_cube(mp.dm2.compact, mp, mp.P2.compact.dx, NOCUBE=True)

    #--Initial DM voltages
    if not hasattr(mp.dm1,'V'):
        mp.dm1.V = cp.zeros((mp.dm1.Nact,mp.dm1.Nact))
    if not hasattr(mp.dm2,'V'): 
        mp.dm2.V = cp.zeros((mp.dm2.Nact,mp.dm2.Nact))
    pass


def falco_gen_DM_stops(mp):
    if not hasattr(mp.dm2, 'full'):
        mp.dm2.full = falco.config.Object()
    if not hasattr(mp.dm2, 'compact'):
        mp.dm2.compact = falco.config.Object()

    if mp.flagDM1stop:
        mp.dm1.full.mask = falco.masks.falco_gen_DM_stop(mp.P2.full.dx,mp.dm1.Dstop,mp.centering);
        mp.dm1.compact.mask = falco.masks.falco_gen_DM_stop(mp.P2.compact.dx,mp.dm1.Dstop,mp.centering);
    if mp.flagDM2stop:
        mp.dm2.full.mask = falco.masks.falco_gen_DM_stop(mp.P2.full.dx,mp.dm2.Dstop,mp.centering);
        mp.dm2.compact.mask = falco.masks.falco_gen_DM_stop(mp.P2.compact.dx,mp.dm2.Dstop,mp.centering);
    pass


def falco_set_dm_surface_padding(mp):
    #% DM Surface Array Sizes for Angular Spectrum Propagation with FFTs
    ## Array Sizes for Angular Spectrum Propagation with FFTs
    
    #--Compact Model: Set nominal DM plane array sizes as a power of 2 for angular spectrum propagation with FFTs
    if cp.any(mp.dm_ind==1) and cp.any(mp.dm_ind==2):
        NdmPad = 2**np.ceil(1 + np.log2(np.max([mp.dm1.compact.NdmPad,mp.dm2.compact.NdmPad])));
    elif cp.any(mp.dm_ind==1):
        NdmPad = 2**np.ceil(1 + np.log2(np.dm1.compact.NdmPad));
    elif cp.any(mp.dm_ind==2):
        NdmPad = 2**np.ceil(1 + np.log2(np.dm2.compact.NdmPad));
    else:
        NdmPad = 2*mp.P1.compact.Nbeam;

    while (NdmPad < cp.min(mp.sbp_centers)*cp.abs(mp.d_dm1_dm2)/mp.P2.full.dx**2) or (NdmPad < cp.min(mp.sbp_centers)*cp.abs(mp.d_P2_dm1)/mp.P2.compact.dx**2): 
        #--Double the zero-padding until the angular spectrum sampling requirement is not violated
        NdmPad = 2*NdmPad;
    mp.compact.NdmPad = NdmPad;
    
    #--Full Model: Set nominal DM plane array sizes as a power of 2 for angular spectrum propagation with FFTs
    if cp.any(mp.dm_ind==1) and cp.any(mp.dm_ind==2):
        NdmPad = 2**np.ceil(1 + np.log2(np.max([mp.dm1.NdmPad,mp.dm2.NdmPad])));
    elif cp.any(mp.dm_ind==1):
        NdmPad = 2**np.ceil(1 + np.log2(mp.dm1.NdmPad))
    elif cp.any(mp.dm_ind==2):
        NdmPad = 2**np.ceil(1 + np.log2(mp.dm2.NdmPad))
    else:
        NdmPad = 2*mp.P1.full.Nbeam;
    while (NdmPad < cp.min(mp.full.lambdas)*cp.abs(mp.d_dm1_dm2)/mp.P2.full.dx**2) or (NdmPad < cp.min(mp.full.lambdas)*cp.abs(mp.d_P2_dm1)/mp.P2.full.dx**2): #--Double the zero-padding until the angular spectrum sampling requirement is not violated
        NdmPad = 2*NdmPad;
    mp.full.NdmPad = NdmPad;
    pass


def falco_set_initial_Efields(mp):
    ## Initial Electric Fields for Star and Exoplanet
    
    if not hasattr(mp.P1.full,'E'):
        mp.P1.full.E  = cp.ones((mp.P1.full.Narr,mp.P1.full.Narr,mp.Nwpsbp,mp.Nsbp),dtype=complex); # Input E-field at entrance pupil
    
    mp.Eplanet = mp.P1.full.E; #--Initialize the input E-field for the planet at the entrance pupil. Will apply the phase ramp later
    
    if not hasattr(mp.P1.compact,'E'):
        mp.P1.compact.E = cp.ones((mp.P1.compact.Narr,mp.P1.compact.Narr,mp.Nsbp),dtype=complex)
    mp.sumPupil = cp.sum(cp.sum(cp.abs(mp.P1.compact.mask*falco.utils.padOrCropEven(cp.mean(mp.P1.compact.E,2),mp.P1.compact.mask.shape[0] ))**2)); #--Throughput is computed with the compact model
    
    pass
                

def falco_init_storage_arrays(mp):
    #% Initialize Arrays to Store Performance History

    out = falco.config.Object()
    out.dm1 = falco.config.Object()
    out.dm2 = falco.config.Object()
    out.dm8 = falco.config.Object()
    out.dm9 = falco.config.Object()

    ## Storage Arrays for DM Metrics
    #--EFC regularization history
    out.log10regHist = cp.zeros((mp.Nitr,1))
    
    #--Peak-to-Valley DM voltages
    out.dm1.Vpv = cp.zeros((mp.Nitr,1))
    out.dm2.Vpv = cp.zeros((mp.Nitr,1))
    out.dm8.Vpv = cp.zeros((mp.Nitr,1))
    out.dm9.Vpv = cp.zeros((mp.Nitr,1))
    
    #--Peak-to-Valley DM surfaces
    out.dm1.Spv = cp.zeros((mp.Nitr,1))
    out.dm2.Spv = cp.zeros((mp.Nitr,1))
    out.dm8.Spv = cp.zeros((mp.Nitr,1))
    out.dm9.Spv = cp.zeros((mp.Nitr,1))
    
    #--RMS DM surfaces
    out.dm1.Srms = cp.zeros((mp.Nitr,1))
    out.dm2.Srms = cp.zeros((mp.Nitr,1))
    out.dm8.Srms = cp.zeros((mp.Nitr,1))
    out.dm9.Srms = cp.zeros((mp.Nitr,1))
    
    #--Zernike sensitivities to 1nm RMS
    if not hasattr(mp.eval,'Rsens'): 
        mp.eval.Rsens = []
    if not hasattr(mp.eval,'indsZnoll'):  
        mp.eval.indsZnoll = [1,2]
    Nannuli = mp.eval.Rsens.shape[0]
    Nzern = len(mp.eval.indsZnoll)
    out.Zsens = cp.zeros((Nzern,Nannuli,mp.Nitr));

    #--Store the DM commands at each iteration
    if hasattr(mp,'dm1'): 
        if hasattr(mp.dm1,'V'):  
            out.dm1.Vall = cp.zeros((mp.dm1.Nact,mp.dm1.Nact,mp.Nitr+1))
    if hasattr(mp,'dm2'): 
        if hasattr(mp.dm2,'V'):  
            out.dm2.Vall = cp.zeros((mp.dm2.Nact,mp.dm2.Nact,mp.Nitr+1))
    if hasattr(mp,'dm8'): 
        if hasattr(mp.dm8,'V'):  
            out.dm8.Vall = cp.zeros((mp.dm8.NactTotal,mp.Nitr+1))
    if hasattr(mp,'dm9'): 
        if hasattr(mp.dm9,'V'):  
            out.dm9.Vall = cp.zeros((mp.dm9.NactTotal,mp.Nitr+1))

    return out


def falco_gen_FPM_LC(mp):
    """
    Make or read in focal plane mask (FPM) amplitude for the full model.

    Detailed description here

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    Returns
    -------
    mp: falco.config.ModelParameters
        Structure of model parameters
    """

    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    

#    class FPMgenIn(object):
#        pass
    
    FPMgeninputs = {} #FPMgenIn()
    
    #--Make or read in focal plane mask (FPM) amplitude for the full model
    #FPMgeninputs.flagRot180deg
    FPMgeninputs["pixresFPM"] = mp.F3.full.res #--pixels per lambda_c/D
    FPMgeninputs["rhoInner"] = mp.F3.Rin # radius of inner FPM amplitude spot (in lambda_c/D)
    FPMgeninputs["rhoOuter"] = mp.F3.Rout # radius of outer opaque FPM ring (in lambda_c/D)
    FPMgeninputs["FPMampFac"] = mp.FPMampFac # amplitude transmission of inner FPM spot
    FPMgeninputs["centering"] = mp.centering
    #kwargs = FPMgeninputs.__dict__
    
    if not hasattr(mp.F3.full,'mask'):
        mp.F3.full.mask = falco.config.Object()
        
    mp.F3.full.ampMask = falco.masks.falco_gen_annular_FPM(FPMgeninputs)

    mp.F3.full.Nxi = mp.F3.full.ampMask.shape[1]
    mp.F3.full.Neta= mp.F3.full.ampMask.shape[0]  
    
    #--Number of points across the FPM in the compact model
    if(mp.F3.Rout==cp.inf):
        if mp.centering == 'pixel':
            mp.F3.compact.Nxi = falco.utils.ceil_even((2*(mp.F3.Rin*mp.F3.compact.res + 1/2)))
        else:
            mp.F3.compact.Nxi = falco.utils.ceil_even((2*mp.F3.Rin*mp.F3.compact.res))
            
    else:
        if mp.centering == 'pixel':
            mp.F3.compact.Nxi = falco.utils.ceil_even((2*(mp.F3.Rout*mp.F3.compact.res + 1/2)))
        else: #case 'interpixel'
            mp.F3.compact.Nxi = falco.utils.ceil_even((2*mp.F3.Rout*mp.F3.compact.res))

    mp.F3.compact.Neta = mp.F3.compact.Nxi
    
    #--Make or read in focal plane mask (FPM) amplitude for the compact model
    FPMgeninputs["pixresFPM"] = mp.F3.compact.res #--pixels per lambda_c/D
    #kwargs=FPMgeninputs.__dict__
    
    if not hasattr(mp.F3.compact,'mask'):
        mp.F3.compact.mask = falco.config.Object()
        
    mp.F3.compact.ampMask = falco.masks.falco_gen_annular_FPM(FPMgeninputs)
    

def falco_gen_FPM_SPLC(mp):

    if not hasattr(mp.F3,'ang'):
        mp.F3.ang = 180
    
    if(mp.full.flagGenFPM):
        #--Generate the FPM amplitude for the full model
        inputs = {}
        inputs["rhoInner"] = mp.F3.Rin # radius of inner FPM amplitude spot (in lambda_c/D)
        inputs["rhoOuter"] = mp.F3.Rout # radius of outer opaque FPM ring (in lambda_c/D)
        inputs["ang"] = mp.F3.ang # [degrees]
        inputs["centering"] = mp.centering;
        inputs["pixresFPM"] = mp.F3.full.res #--pixels per lambda_c/D
        mp.F3.full.ampMask = falco.masks.falco_gen_bowtie_FPM(inputs);
    
    if(mp.compact.flagGenFPM):
        #--Generate the FPM amplitude for the compact model
        inputs = {}
        inputs["rhoInner"] = mp.F3.Rin # radius of inner FPM amplitude spot (in lambda_c/D)
        inputs["rhoOuter"] = mp.F3.Rout # radius of outer opaque FPM ring (in lambda_c/D)
        inputs["ang"] = mp.F3.ang # [degrees]
        inputs["centering"] = mp.centering
        inputs["pixresFPM"] = mp.F3.compact.res
        mp.F3.compact.ampMask = falco.masks.falco_gen_bowtie_FPM(inputs)        
    
    if not mp.full.flagPROPER:
        mp.F3.full.Nxi = mp.F3.full.ampMask.shape[1]
        mp.F3.full.Neta= mp.F3.full.ampMask.shape[0]

    
    mp.F3.compact.Nxi = mp.F3.compact.ampMask.shape[1]
    mp.F3.compact.Neta= mp.F3.compact.ampMask.shape[0]
    pass
    
