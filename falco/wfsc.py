import numpy as np
import falco
import os
import pickle
import scipy
import psutil # For checking number of cores available
import multiprocessing
from astropy.io import fits 
import matplotlib.pyplot as plt 

def falco_init_ws(mp, config=None):

    if config:
        with open(config, 'rb') as f:
            mp = pickle.load(f)

    mainPath = mp.path.falco;

    print('DM 1-to-2 Fresnel number (using radius) = ' + str((mp.P2.D/2)**2/(mp.d_dm1_dm2*mp.lambda0)))

    #SFF NOTE
    if not hasattr(mp, "compact"):
        mp.compact = falco.config.Object()
    if not hasattr(mp, "full"):
        mp.full = falco.config.Object()

    ## Intializations of structures (if they don't exist yet)
    mp.jac.dummy = 1;
    mp.est.dummy = 1
    mp.compact.dummy = 1
    mp.full.dummy = 1
    
    ## Number of threads to use if doing multiprocessing
    if not hasattr(mp, "Nthreads"):
        mp.Nthreads = psutil.cpu_count(logical=False) 
    
    ## Optional/Hidden flags
    #--Saving data
    if not hasattr(mp,'flagSaveWS'):  
        mp.flagSaveWS = False  #--Whehter to save out the entire workspace at the end of the trial. Can take up lots of space.
    if not hasattr(mp,'flagSaveEachItr'):  
        mp.flagSaveEachItr = False  #--Whether to save out the performance at each iteration. Useful for long trials in case it crashes or is stopped early.
    if not hasattr(mp,'flagSVD'):
        mp.flagSVD = False    #--Whether to compute and save the singular mode spectrum of the control Jacobian (each iteration)
    if not hasattr(mp,'flagTrainModel'):  
        mp.flagTrainModel = False  #--Whether to call the Expectation-Maximization (E-M) algorithm to improve the linearized model. 
    if not hasattr(mp,'flagUseLearnedJac'):  
        mp.flagUseLearnedJac = False #--Whether to load and use an improved Jacobian from the Expectation-Maximization (E-M) algorithm 
    if not hasattr(mp.est,'flagUseJac'):  
        mp.est.flagUseJac = False   #--Whether to use the Jacobian or not for estimation. (If not using Jacobian, model is called and differenced.)
    if not hasattr(mp.ctrl,'flagUseModel'):  
        mp.ctrl.flagUseModel = False #--Whether to perform a model-based (vs empirical) grid search for the controller
    
    #--Deformable mirror actuator constraints or bounds
    if not hasattr(mp.dm1,'Vmin'):  
        mp.dm1.Vmin = -1000. #--Min allowed voltage command
    if not hasattr(mp.dm1,'Vmax'):  
        mp.dm1.Vmax = 1000. #--Max allowed voltage command
    if not hasattr(mp.dm1,'pinned'):  
        mp.dm1.pinned = np.array([]) #--Indices of pinned actuators
    if not hasattr(mp.dm1,'Vpinned'):  
        mp.dm1.Vpinned = np.array([]) #--(Fixed) voltage commands of pinned actuators
    if not hasattr(mp.dm1,'tied'):  
        mp.dm1.tied = np.zeros((0,2)) #--Indices of paired actuators. Two indices per row       
    if not hasattr(mp.dm1,'flagNbrRule'):
        mp.dm1.flagNbrRule = False #--Whether to set constraints on neighboring actuator voltage differences. If set to true, need to define mp.dm1.dVnbr

    if not hasattr(mp.dm2,'Vmin'):  
        mp.dm2.Vmin = -1000. #--Min allowed voltage command
    if not hasattr(mp.dm2,'Vmax'):  
        mp.dm2.Vmax = 1000. #--Max allowed voltage command
    if not hasattr(mp.dm2,'pinned'):  
        mp.dm2.pinned = np.array([]) #--Indices of pinned actuators
    if not hasattr(mp.dm2,'Vpinned'):  
        mp.dm2.Vpinned = np.array([]) #--(Fixed) voltage commands of pinned actuators
    if not hasattr(mp.dm2,'tied'):  
        mp.dm2.tied = np.zeros((0,2)) #--Indices of paired actuators. Two indices per row       
    if not hasattr(mp.dm2,'flagNbrRule'):  
        mp.dm2.flagNbrRule = False;  #--Whether to set constraints on neighboring actuator voltage differences. If set to true, need to define mp.dm2.dVnbr

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
    if not hasattr(mp.full,'ZrmsVal'): 
        mp.full.ZrmsVal = 1e-9 #--Amount of RMS Zernike mode used to calculate aberration sensitivities [meters]. WFIRST CGI uses 1e-9, and LUVOIR and HabEx use 1e-10. 
    if not hasattr(mp.full,'pol_conds'):  
        mp.full.pol_conds = np.array([0])  #--Vector of which polarization state(s) to use when creating images from the full model. Currently only used with PROPER full models from John Krist.
    if not hasattr(mp,'propMethodPTP'):
        mp.propMethodPTP = 'fft' #--Propagation method for postage stamps around the influence functions. 'mft' or 'fft'
    if not hasattr(mp,'apodType'):  
        mp.apodType = 'none';  #--Type of apodizer. Only use this variable when generating the apodizer. Curr

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

    ## Bandwidth and Wavelength Specs: Compact Model
    
    if not hasattr(mp,'Nwpsbp'):
        mp.Nwpsbp = 1;

    #--Center-ish wavelength indices (ref = reference)(Only the center if
    #  an odd number of wavelengths is used.)
    mp.si_ref = np.floor(mp.Nsbp/2).astype(int)

    #--Wavelengths used for Compact Model (and Jacobian Model)
    mp.sbp_weights = np.ones((mp.Nsbp,1));
    if mp.Nwpsbp==1: #--Set ctrl wavelengths evenly between endpoints (inclusive) of the total bandpass.
        if mp.Nsbp==1:
            mp.sbp_centers = np.array([mp.lambda0])
        else:
            mp.sbp_centers = mp.lambda0*np.linspace(1-mp.fracBW/2,1+mp.fracBW/2,mp.Nsbp);
    else:#--For cases with multiple sub-bands: Choose wavelengths to be at subbandpass centers since the wavelength samples will span to the full extent of the sub-bands.
        mp.fracBWcent2cent = mp.fracBW*(1-1/mp.Nsbp); #--Bandwidth between centers of endpoint subbandpasses.
        mp.sbp_centers = mp.lambda0*np.linspace(1-mp.fracBWcent2cent/2,1+mp.fracBWcent2cent/2,mp.Nsbp); #--Space evenly at the centers of the subbandpasses.
    mp.sbp_weights = mp.sbp_weights/np.sum(mp.sbp_weights); #--Normalize the sum of the weights
    
    print(' Using %d discrete wavelength(s) in each of %d sub-bandpasses over a %.1f%% total bandpass \n'%(mp.Nwpsbp, mp.Nsbp,100*mp.fracBW));
    print('Sub-bandpasses are centered at wavelengths [nm]:\t ',end='')
    print(1e9*mp.sbp_centers)
 

    ## Bandwidth and Wavelength Specs: Full Model
    
    #--Center(-ish) wavelength indices (ref = reference). (Only the center if an odd number of wavelengths is used.)
    mp.wi_ref = np.floor(mp.Nwpsbp/2).astype(int)

    #--Wavelength factors/weights within sub-bandpasses in the full model
    mp.full.lambda_weights = np.ones((mp.Nwpsbp,1)); #--Initialize as all ones. Weights within a single sub-bandpass
    if mp.Nwpsbp==1:
        mp.full.dlam = 0; #--Delta lambda between every wavelength in the sub-band in the full model
    else:
        #--Spectral weighting in image
        mp.full.lambda_weights[0] = 1/2; #--Include end wavelengths with half weights
        mp.full.lambda_weights[-1] = 1/2; #--Include end wavelengths with half weights
        mp.fracBWsbp = mp.fracBW/mp.Nsbp; #--Bandwidth per sub-bandpass
        #--Indexing of wavelengths in each sub-bandpass
        sbp_facs = np.linspace(1-mp.fracBWsbp/2,1+mp.fracBWsbp/2,mp.Nwpsbp); #--Factor applied to lambda0 only
        mp.full.dlam = (sbp_facs[1] - sbp_facs[0])*mp.lambda0; #--Delta lambda between every wavelength in the full model 
    
    mp.full.lambda_weights = mp.full.lambda_weights/np.sum(mp.full.lambda_weights); #--Normalize sum of the weights (within the sub-bandpass)

    #--Make vector of all wavelengths and weights used in the full model
    lambdas = np.zeros((mp.Nsbp*mp.Nwpsbp,))
    lambda_weights_all = np.zeros((mp.Nsbp*mp.Nwpsbp,))
    mp.full.lambdasMat = np.zeros((mp.Nsbp,mp.Nwpsbp))
    mp.full.indsLambdaMat = np.zeros((mp.Nsbp*mp.Nwpsbp,2),dtype=int)
    counter = 0;
    for si in range(mp.Nsbp):
        if(mp.Nwpsbp==1):
            mp.full.lambdasMat[si,0] = mp.sbp_centers[si]
        else:
            mp.full.lambdasMat[si,:] = np.arange(-(mp.Nwpsbp-1)/2,(mp.Nwpsbp+1)/2)*mp.full.dlam + mp.sbp_centers[si]
        np.arange(-(mp.Nwpsbp-1)/2,(mp.Nwpsbp-1)/2)*mp.full.dlam# + mp.sbp_centers[si];
        for wi in range(mp.Nwpsbp):
            lambdas[counter] = mp.full.lambdasMat[si,wi];
            lambda_weights_all[counter] = mp.sbp_weights[si]*mp.full.lambda_weights[wi];
            mp.full.indsLambdaMat[counter,:] = [si,wi]
            counter = counter+1;
            
    #--Get rid of redundant wavelengths in the complete list, and sum weights for repeated wavelengths
    # indices of unique wavelengths
    unused_1, inds_unique = np.unique(np.round(1e12*lambdas), return_index=True); #--Check equality at the picometer level for wavelength
    mp.full.indsLambdaUnique = inds_unique;
    # indices of duplicate wavelengths
    duplicate_inds = np.setdiff1d( np.arange(len(lambdas),dtype=int) , inds_unique);
    # duplicate weight values
    duplicate_values = lambda_weights_all[duplicate_inds]

    #--Shorten the vectors to contain only unique values. Combine weights for repeated wavelengths.
    mp.full.lambdas = lambdas[inds_unique];
    mp.full.lambda_weights_all = lambda_weights_all[inds_unique];
    for idup in range(len(duplicate_inds)):
        wvl = lambdas[duplicate_inds[idup]];
        weight = lambda_weights_all[duplicate_inds[idup]];
        ind = np.where(np.abs(mp.full.lambdas-wvl)<=1e-11)
        print(ind)
        mp.full.lambda_weights_all[ind] = mp.full.lambda_weights_all[ind] + weight;
    mp.full.NlamUnique = len(inds_unique);

    # Set the relative weights of the Jacobian modes
    falco.configs.falco_config_jac_weights(mp)

    ## Pupil Masks
    falco.configs.falco_config_gen_chosen_pupil(mp) #--input pupil mask
    falco.configs.falco_config_gen_chosen_apodizer(mp) #--apodizer mask
    falco.configs.falco_config_gen_chosen_LS(mp) #--Lyot stop

    ## Plot the pupil and Lyot stop on top of each other to make sure they are aligned correctly
    #--Only for coronagraphs using Babinet's principle, for which the input
    #pupil and Lyot plane have the same resolution.
    if mp.coro in ['FOHLC','HLC','LC','APLC','VC','AVC']:
        if mp.flagPlot:
            P4mask = falco.utils.padOrCropEven(mp.P4.compact.mask,mp.P1.compact.Narr)
            P4mask = np.rot90(P4mask,2);
            if mp.centering.lower() == 'pixel':
               #P4mask = circshift(P4mask,[1 1]);
               print(type(P4mask))
               P4mask = np.roll(P4mask,(1,1),(0,1))
            P1andP4 = mp.P1.compact.mask + P4mask;
            #figure(301); imagesc(P1andP4); axis xy equal tight; colorbar; set(gca,'Fontsize',20); title('Pupil and LS Superimposed','Fontsize',16');

            if mp.flagApod:
                P1andP3 = mp.P1.compact.mask + falco.utils.padOrCropEven(mp.P3.compact.mask,len(mp.P1.compact.mask));
                #figure(302); imagesc(P1andP3); axis xy equal tight; colorbar; set(gca,'Fontsize',20); title('Pupil and Apod Superimposed','Fontsize',16');

    ## DM Initialization
    mp.dm3 = falco.config.Object()   
    mp.dm4 = falco.config.Object()   
    mp.dm5 = falco.config.Object()   
    mp.dm6 = falco.config.Object()   
    mp.dm7 = falco.config.Object()   
    mp.dm8 = falco.config.Object()   
    mp.dm9 = falco.config.Object()   

    #--Initialize the number of actuators (NactTotal) and actuators used (Nele).
    mp.dm1.NactTotal=0; mp.dm2.NactTotal=0; mp.dm3.NactTotal=0; mp.dm4.NactTotal=0; mp.dm5.NactTotal=0; mp.dm6.NactTotal=0; mp.dm7.NactTotal=0; mp.dm8.NactTotal=0; mp.dm9.NactTotal=0; #--Initialize for bookkeeping later.
    mp.dm1.Nele=0; mp.dm2.Nele=0; mp.dm3.Nele=0; mp.dm4.Nele=0; mp.dm5.Nele=0; mp.dm6.Nele=0; mp.dm7.Nele=0; mp.dm8.Nele=0; mp.dm9.Nele=0; #--Initialize for Jacobian calculations later. 

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
        elif mp.coro.upper() == 'FOHLC':
            falco_setup_FPM_FOHLC(mp);
            falco.configs.falco_config_gen_FPM_FOHLC(mp);
            mp.compact.Nfpm = np.max([mp.dm8.compact.NdmPad,mp.dm9.compact.NdmPad]); #--Width of the FPM array in the compact model.
            mp.full.Nfpm = np.max([mp.dm8.NdmPad,mp.dm9.NdmPad]); #--Width of the FPM array in the full model.
        elif mp.coro.upper() == 'EHLC':
            falco_setup_FPM_EHLC(mp);
            falco.configs.falco_config_gen_FPM_EHLC(mp);
        elif mp.coro.upper() == 'SPHLC':
            falco.configs.falco_config_gen_FPM_SPHLC(mp);
    
            ##--Pre-compute the complex transmission of the allowed Ni+PMGI FPMs.
            if mp.coro in ['EHLC','HLC','SPHLC']:
                [mp.complexTransCompact,mp.complexTransFull] = falco_gen_complex_trans_table(mp);

    ## Generate FPM
    if mp.coro.upper() in ['LC', 'APLC']: #--Occulting spot FPM (can be HLC-style and partially transmissive)
        falco.configs.falco_config_gen_FPM_LC(mp);
    elif mp.coro.upper() in ['SPLC', 'FLC']:
        falco.configs.falco_config_gen_FPM_SPLC(mp);
    elif mp.coro.upper() == 'RODDIER':
        falco.configs.falco_config_gen_FPM_Roddier(mp);

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
            mp.F3.compact.xis  = np.arange(-(mp.F3.compact.Nxi-1)/2,(mp.F3.compact.Nxi-1)/2)*mp.F3.compact.dxi;
            mp.F3.compact.etas = np.arange(-(mp.F3.compact.Neta-1)/2,(mp.F3.compact.Neta-1)/2)*mp.F3.compact.deta;
        else:
            mp.F3.compact.xis  = np.arange(-mp.F3.compact.Nxi/2, (mp.F3.compact.Nxi/2))*mp.F3.compact.dxi;
            mp.F3.compact.etas = np.arange(-mp.F3.compact.Neta/2,(mp.F3.compact.Neta/2))*mp.F3.compact.deta;

        if mp.layout in ['wfirst_phaseb_simple','wfirst_phaseb_proper']:
            pass
        else:
            #--Coordinates (dimensionless [DL]) for the FPMs in the full model
            if mp.centering.lower() == 'interpixel' or mp.F3.full.Nxi%2==1:
                mp.F3.full.xisDL  = np.arange(-(mp.F3.full.Nxi-1)/2,(mp.F3.full.Nxi-1)/2)/mp.F3.full.res;
                mp.F3.full.etasDL = np.arange(-(mp.F3.full.Neta-1)/2, (mp.F3.full.Neta-1)/2)/mp.F3.full.res;
            else:
                mp.F3.full.xisDL  = np.arange(-mp.F3.full.Nxi/2,(mp.F3.full.Nxi/2-1))/mp.F3.full.res;
                mp.F3.full.etasDL = np.arange(-mp.F3.full.Neta/2,(mp.F3.full.Neta/2-1))/mp.F3.full.res;

        #--Coordinates (dimensionless [DL]) for the FPMs in the compact model
        if mp.centering.lower() == 'interpixel' or mp.F3.compact.Nxi%2==1:
            mp.F3.compact.xisDL  = np.arange(-(mp.F3.compact.Nxi-1)/2,(mp.F3.compact.Nxi-1)/2)/mp.F3.compact.res;
            mp.F3.compact.etasDL = np.arange(-(mp.F3.compact.Neta-1)/2,(mp.F3.compact.Neta-1)/2)/mp.F3.compact.res;
        else:
            mp.F3.compact.xisDL  = np.arange(-mp.F3.compact.Nxi/2,(mp.F3.compact.Nxi/2-1))/mp.F3.compact.res;
            mp.F3.compact.etasDL = np.arange(-mp.F3.compact.Neta/2,(mp.F3.compact.Neta/2-1))/mp.F3.compact.res;

    ## Sampling/Resolution and Scoring/Correction Masks for Final Focal Plane (Fend.
    
    mp.Fend.dxi = (mp.fl*mp.lambda0/mp.P4.D)/mp.Fend.res; # sampling at Fend.[meters]
    mp.Fend.deta = mp.Fend.dxi; # sampling at Fend.[meters]    
    
    if mp.flagFiber:
        mp.Fend.lenslet.D = 2*mp.Fend.res*mp.Fend.lensletWavRad*mp.Fend.dxi;
        mp.Fend.x_lenslet_phys = mp.Fend.dxi*mp.Fend.res*mp.Fend.x_lenslet;
        mp.Fend.y_lenslet_phys = mp.Fend.deta*mp.Fend.res*mp.Fend.y_lenslet;
    
        mp.F5.dxi = mp.lensletFL*mp.lambda0/mp.Fend.lenslet.D/mp.F5.res;
        mp.F5.deta = mp.F5.dxi;
    
    ## Software Mask for Correction (corr) and Scoring (score)
    
    #--Set Inputs:
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

    XIS, ETAS = np.meshgrid(mp.Fend.xisDL, mp.Fend.etasDL)
    mp.Fend.RHOS = np.sqrt(XIS**2 + ETAS**2)

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
        U = np.sqrt(V**2 - W**2);
    
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
    
        F5XIS, F5ETAS = np.meshgrid(mp.F5.xisDL, mp.F5.etasDL);
    
        mp.F5.RHOS = np.sqrt((F5XIS - mp.F5.fiberPos[0])**2 + (F5ETAS - mp.F5.fiberPos[1])**2);
        mp.F5.fiberCore.mode = mp.F5.fiberCore.mask*scipy.special.j0(U*mp.F5.RHOS/mp.fiber.a)/scipy.special.j0(0,U);
        mp.F5.fiberCladding.mode = mp.F5.fiberCladding.mask*scipy.special.k0(W*mp.F5.RHOS/mp.fiber.a)/scipy.special.k0(W);
        mp.F5.fiberCladding.mode[np.isnan(mp.F5.fiberCladding.mode)] = 0;
        mp.F5.fiberMode = mp.F5.fiberCore.mode + mp.F5.fiberCladding.mode;
        fiberModeNorm = np.sqrt(np.sum(np.sum(np.abs(mp.F5.fiberMode)**2)));
        mp.F5.fiberMode = mp.F5.fiberMode/fiberModeNorm;

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    #--Evaluation Model for Computing Throughput (same as Compact Model but
    # with different Fend.resolution)
    if not hasattr(mp.Fend, 'eval'): #--Initialize the structure if it doesn't exist.
        mp.Fend.eval = falco.config.Object()
    if not hasattr(mp.Fend.eval,'res'):  
        mp.Fend.eval.res = 10
    maskCorr["pixresFP"] = mp.Fend.eval.res; #--Assign the resolution
    mp.Fend.eval.mask, mp.Fend.eval.xisDL, mp.Fend.eval.etasDL = falco.masks.falco_gen_SW_mask(maskCorr);  #--Generate the mask
    mp.Fend.eval.Nxi  = mp.Fend.eval.mask.shape[1]
    mp.Fend.eval.Neta = mp.Fend.eval.mask.shape[0]
    mp.Fend.eval.dxi = (mp.fl*mp.lambda0/mp.P4.D)/mp.Fend.eval.res; # higher sampling at Fend.for evaulation [meters]
    mp.Fend.eval.deta = mp.Fend.eval.dxi; # higher sampling at Fend.for evaulation [meters]   
    
    # (x,y) location [lambda_c/D] in dark hole at which to evaluate throughput
    XIS,ETAS = np.meshgrid(mp.Fend.eval.xisDL - mp.thput_eval_x, mp.Fend.eval.etasDL - mp.thput_eval_y);
    mp.Fend.eval.RHOS = np.sqrt(XIS**2 + ETAS**2);
    
    #--Storage array for throughput at each iteration
    mp.thput_vec = np.zeros((mp.Nitr+1,1));

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    #--Software Mask for Scoring Contrast 
    #--Set Inputs
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
    mp.Fend.corr.Npix = int(np.sum(np.sum(mp.Fend.corr.mask)));
    mp.Fend.score.Npix = int(np.sum(np.sum(mp.Fend.score.mask)));
    
    #--Indices of dark hole pixels and logical masks
    if mp.flagFiber:
        #mp.Fend.corr.inds = find(np.sum(mp.Fend.lenslet.mask,3)~=0);
        #mp.Fend.corr.maskBool = logical(mp.Fend.corr.mask);
        mp.Fend.corr.inds = np.where(np.sum(mp.Fend.lenslet.mask,3)!=0);
        mp.Fend.corr.maskBool = np.array(mp.Fend.corr.mask, dtype=bool)
    else:
        #mp.Fend.corr.inds = find(mp.Fend.corr.mask~=0);
        #mp.Fend.corr.maskBool = logical(mp.Fend.corr.mask);
        mp.Fend.corr.inds = np.where(mp.Fend.corr.mask!=0);
        mp.Fend.corr.maskBool = np.array(mp.Fend.corr.mask, dtype=bool)
    
    #mp.Fend.score.inds = find(mp.Fend.score.mask~=0)
    #mp.Fend.score.maskBool = logical(mp.Fend.score.mask);
    mp.Fend.score.inds = np.where(mp.Fend.score.mask!=0)
    mp.Fend.score.maskBool = np.array(mp.Fend.score.mask, dtype=bool)
    
    if mp.flagFiber:
        mp.WspatialVec = np.ones((mp.Fend.Nlens,1));
    else:
        falco.configs.falco_config_spatial_weights(mp);
        #--Extract the vector of weights at the pixel locations of the dark hole pixels.
        mp.WspatialVec = mp.Wspatial[mp.Fend.corr.maskBool];
    
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
    
            mp.dm1.inf0 = np.squeeze(hdul[0].data) #hdul[0].data[0,:,:]
        mp.dm1.dx_inf0 = mp.dm1.dm_spacing*(dx1/pitch1);
    
        if mp.dm1.inf_sign[0] in ['-','n','m']:
            mp.dm1.inf0 = -1*mp.dm1.inf0;
        else:
            #--Leave coefficient as +1
            pass

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
    
            mp.dm2.inf0 = np.squeeze(hdul[0].data)
        mp.dm2.dx_inf0 = mp.dm2.dm_spacing*(dx2/pitch2);
    
        if mp.dm2.inf_sign[0] in ['-','n','m']:
            mp.dm2.inf0 = -1*mp.dm2.inf0;
        else:
            #--Leave coefficient as +1
            pass

    mp.dm1.compact = falco.config.Object()
    mp.dm2.compact = falco.config.Object()
    if np.any(mp.dm_ind==1):
        mp.dm1.centering = mp.centering;
        mp.dm1.compact = mp.dm1;
        falco.dms.falco_gen_dm_poke_cube(mp.dm1, mp, mp.P2.full.dx,NOCUBE=True);
        falco.dms.falco_gen_dm_poke_cube(mp.dm1.compact, mp, mp.P2.compact.dx);
    else:
        mp.dm1.compact = mp.dm1;
        falco.dms.falco_gen_dm_poke_cube(mp.dm1, mp, mp.P2.full.dx,NOCUBE=True);
        falco.dms.falco_gen_dm_poke_cube(mp.dm1.compact, mp, mp.P2.compact.dx,NOCUBE=True);

    if np.any(mp.dm_ind==2):
        mp.dm2.centering = mp.centering;
        mp.dm2.compact = mp.dm2;
        mp.dm2.dx = mp.P2.full.dx;
        mp.dm2.compact.dx = mp.P2.compact.dx;
    
        falco.dms.falco_gen_dm_poke_cube(mp.dm2, mp, mp.P2.full.dx, NOCUBE=True);
        falco.dms.falco_gen_dm_poke_cube(mp.dm2.compact, mp, mp.P2.compact.dx);

    else:
        mp.dm2.compact = mp.dm2;
        mp.dm2.dx = mp.P2.full.dx;
        mp.dm2.compact.dx = mp.P2.compact.dx;
    
        falco.dms.falco_gen_dm_poke_cube(mp.dm2, mp, mp.P2.full.dx, NOCUBE=True);
        falco.dms.falco_gen_dm_poke_cube(mp.dm2.compact, mp, mp.P2.compact.dx,NOCUBE=True);

    #--Initial DM voltages
    if not hasattr(mp.dm1,'V'):
        mp.dm1.V = np.zeros((mp.dm1.Nact,mp.dm1.Nact))
    if not hasattr(mp.dm2,'V'): 
        mp.dm2.V = np.zeros((mp.dm2.Nact,mp.dm2.Nact))
    
    ## DM Aperture Masks (moved here because the commands mp.dm2.compact = mp.dm2; and mp.dm1.compact = mp.dm1; otherwise would overwrite the compact model masks)
    
    #SFF NOTE
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
    
    ## #--First delta DM settings are zero (for covariance calculation in Kalman filters or robust controllers)
    mp.dm1.dV = np.zeros((mp.dm1.Nact,mp.dm1.Nact)); # delta voltage on DM1;
    mp.dm2.dV = np.zeros((mp.dm2.Nact,mp.dm2.Nact)); # delta voltage on DM2;
    mp.dm8.dV = np.zeros((mp.dm8.NactTotal,1)); # delta voltage on DM8;
    mp.dm9.dV = np.zeros((mp.dm9.NactTotal,1)); # delta voltage on DM9;
    
    ## Array Sizes for Angular Spectrum Propagation with FFTs
    
    #--Compact Model: Set nominal DM plane array sizes as a power of 2 for angular spectrum propagation with FFTs
    if np.any(mp.dm_ind==1) and np.any(mp.dm_ind==2):
        NdmPad = 2**np.ceil(1 + np.log2(np.max([mp.dm1.compact.NdmPad,mp.dm2.compact.NdmPad])));
    elif np.any(mp.dm_ind==1):
        NdmPad = 2**np.ceil(1 + np.log2(mp.dm1.compact.NdmPad));
    elif np.any(mp.dm_ind==2):
        NdmPad = 2**np.ceil(1 + np.log2(mp.dm2.compact.NdmPad));
    else:
        NdmPad = 2*mp.P1.compact.Nbeam;

    while (NdmPad < np.min(mp.sbp_centers)*np.abs(mp.d_dm1_dm2)/mp.P2.full.dx**2) or (NdmPad < np.min(mp.sbp_centers)*np.abs(mp.d_P2_dm1)/mp.P2.compact.dx**2): 
        #--Double the zero-padding until the angular spectrum sampling requirement is not violated
        NdmPad = 2*NdmPad;
    mp.compact.NdmPad = NdmPad;
    
    #--Full Model: Set nominal DM plane array sizes as a power of 2 for angular spectrum propagation with FFTs
    if np.any(mp.dm_ind==1) and np.any(mp.dm_ind==2):
        NdmPad = 2**np.ceil(1 + np.log2(np.max([mp.dm1.NdmPad,mp.dm2.NdmPad])));
    elif np.any(mp.dm_ind==1):
        NdmPad = 2**np.ceil(1 + np.log2(mp.dm1.NdmPad))
    elif np.any(mp.dm_ind==2):
        NdmPad = 2**np.ceil(1 + np.log2(mp.dm2.NdmPad))
    else:
        NdmPad = 2*mp.P1.full.Nbeam;
    while (NdmPad < np.min(mp.full.lambdas)*np.abs(mp.d_dm1_dm2)/mp.P2.full.dx**2) or (NdmPad < np.min(mp.full.lambdas)*np.abs(mp.d_P2_dm1)/mp.P2.full.dx**2): #--Double the zero-padding until the angular spectrum sampling requirement is not violated
        NdmPad = 2*NdmPad;
    mp.full.NdmPad = NdmPad;

    ## Initial Electric Fields for Star and Exoplanet
    
    if not hasattr(mp.P1.full,'E'):
        mp.P1.full.E  = np.ones((mp.P1.full.Narr,mp.P1.full.Narr,mp.Nwpsbp,mp.Nsbp),dtype=complex); # Input E-field at entrance pupil
    
    mp.Eplanet = mp.P1.full.E; #--Initialize the input E-field for the planet at the entrance pupil. Will apply the phase ramp later
    
    if not hasattr(mp.P1.compact,'E'):
        mp.P1.compact.E = np.ones((mp.P1.compact.Narr,mp.P1.compact.Narr,mp.Nsbp),dtype=complex)
    mp.sumPupil = np.sum(np.sum(np.abs(mp.P1.compact.mask*falco.utils.padOrCropEven(np.mean(mp.P1.compact.E,2),mp.P1.compact.mask.shape[0] ))**2)); #--Throughput is computed with the compact model
    
    ## Off-axis, incoherent point source (exoplanet)
    
    if not mp.flagFiber:
        mp.c_planet = 1; # contrast of exoplanet
        mp.x_planet = 6; # x position of exoplanet in lambda0/D
        mp.y_planet = 0; # y position of exoplanet in lambda0/D
    
    ## Contrast to Normalized Intensity Map Calculation 
    
    ## Get the starlight normalization factor for the compact and full models (to convert images to normalized intensity)
    falco.imaging.falco_get_PSF_norm_factor(mp);

    #--Check that the normalization of the coronagraphic PSF is correct
    
    modvar = falco.config.Object()
    modvar.ttIndex = 1;
    modvar.sbpIndex = mp.si_ref;
    modvar.wpsbpIndex = mp.wi_ref;
    modvar.whichSource = 'star';
    
    #SFF NOTE:  Since we not plotting at the moment I will comment out
    #E0c = model_compact(mp, modvar);
    #I0c = np.abs(E0c)**2;
    if mp.flagPlot:
        #figure(501); imagesc(log10(I0c)); axis xy equal tight; colorbar;
        #title('(Compact Model: Normalization Check Using Starting PSF)');
        #drawnow;
        pass
    #E0f = model_full(mp, modvar);
    #I0f = np.abs(E0f)**2;
    if mp.flagPlot:
        #figure(502); imagesc(log10(I0f)); axis xy equal tight; colorbar;
        #title('(Full Model: Normalization Check Using Starting PSF)'); drawnow;
        pass
    
    ## Intialize delta DM voltages. Needed for Kalman filters.
    ##--Save the delta from the previous command
    if np.any(mp.dm_ind==1):
        mp.dm1.dV = 0
    if np.any(mp.dm_ind==2):  
        mp.dm2.dV = 0
    if np.any(mp.dm_ind==3): 
        mp.dm3.dV = 0
    if np.any(mp.dm_ind==4): 
        mp.dm4.dV = 0
    if np.any(mp.dm_ind==5):  
        mp.dm5.dV = 0
    if np.any(mp.dm_ind==6):
        mp.dm6.dV = 0
    if np.any(mp.dm_ind==7): 
        mp.dm7.dV = 0
    if np.any(mp.dm_ind==8):  
        mp.dm8.dV = 0
    if np.any(mp.dm_ind==9):  
        mp.dm9.dV = 0

    out = falco.config.Object()
    out.dm1 = falco.config.Object()
    out.dm2 = falco.config.Object()
    out.dm8 = falco.config.Object()
    out.dm9 = falco.config.Object()

    ## Storage Arrays for DM Metrics
    #--EFC regularization history
    out.log10regHist = np.zeros((mp.Nitr,1))
    
    #--Peak-to-Valley DM voltages
    out.dm1.Vpv = np.zeros((mp.Nitr,1))
    out.dm2.Vpv = np.zeros((mp.Nitr,1))
    out.dm8.Vpv = np.zeros((mp.Nitr,1))
    out.dm9.Vpv = np.zeros((mp.Nitr,1))
    
    #--Peak-to-Valley DM surfaces
    out.dm1.Spv = np.zeros((mp.Nitr,1))
    out.dm2.Spv = np.zeros((mp.Nitr,1))
    out.dm8.Spv = np.zeros((mp.Nitr,1))
    out.dm9.Spv = np.zeros((mp.Nitr,1))
    
    #--RMS DM surfaces
    out.dm1.Srms = np.zeros((mp.Nitr,1))
    out.dm2.Srms = np.zeros((mp.Nitr,1))
    out.dm8.Srms = np.zeros((mp.Nitr,1))
    out.dm9.Srms = np.zeros((mp.Nitr,1))
    
    #--Zernike sensitivities to 1nm RMS
    if not hasattr(mp.eval,'Rsens'): 
        mp.eval.Rsens = []
    if not hasattr(mp.eval,'indsZnoll'):  
        mp.eval.indsZnoll = [1,2]
    Nannuli = mp.eval.Rsens.shape[0]
    Nzern = len(mp.eval.indsZnoll)
    out.Zsens = np.zeros((Nzern,Nannuli,mp.Nitr));

    #--Store the DM commands at each iteration
    if hasattr(mp,'dm1'): 
        if hasattr(mp.dm1,'V'):  
            out.dm1.Vall = np.zeros((mp.dm1.Nact,mp.dm1.Nact,mp.Nitr+1))
    if hasattr(mp,'dm2'): 
        if hasattr(mp.dm2,'V'):  
            out.dm2.Vall = np.zeros((mp.dm2.Nact,mp.dm2.Nact,mp.Nitr+1))
    if hasattr(mp,'dm8'): 
        if hasattr(mp.dm8,'V'):  
            out.dm8.Vall = np.zeros((mp.dm8.NactTotal,mp.Nitr+1))
    if hasattr(mp,'dm9'): 
        if hasattr(mp.dm9,'V'):  
            out.dm9.Vall = np.zeros((mp.dm9.NactTotal,mp.Nitr+1))
    
    ## 
    print('\nBeginning Trial %d of Series %d.\n'%(mp.TrialNum,mp.SeriesNum))

    #return mp, out
    return out
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

    #mp, out = falco_init_ws(mp, fn_config)
    out = falco_init_ws(mp)
    if not mp.flagSim:  
        mp.bench = bench


    print('AFTER INIT: ', mp)
    print('FlagFiber = ', mp.flagFiber)
    ## Initializations of Arrays for Data Storage 
    
    #--Raw contrast (broadband)
    
    InormHist = np.zeros((mp.Nitr+1,)); # Measured, mean raw contrast in scoring regino of dark hole.
    
    ## Plot the pupil masks
    
    # if(mp.flagPlot); figure(101); imagesc(mp.P1.full.mask);axis image; colorbar; title('pupil');drawnow; end
    # if(mp.flagPlot && (length(mp.P4.full.mask)==length(mp.P1.full.mask))); figure(102); imagesc(mp.P4.full.mask);axis image; colorbar; title('Lyot stop');drawnow; end
    # if(mp.flagPlot && isfield(mp,'P3.full.mask')); figure(103); imagesc(padOrCropEven(mp.P1.full.mask,mp.P3.full.Narr).*mp.P3.full.mask);axis image; colorbar; drawnow; end
    
    ## Take initial broadband image 
    
    Im = falco.imaging.falco_get_summed_image(mp)
    

    ##
    ###########################################################################
    #Begin the Correction Iterations
    ###########################################################################

    mp.flagCullActHist = np.zeros((mp.Nitr+1,),dtype=np.bool)

    for Itr in range(mp.Nitr):
    
        #--Start of new estimation+control iteration
        #print(['Iteration: ' num2str(Itr) '/' num2str(mp.Nitr) '\n' ]);
        print('Iteration: %d / %d\n'%(Itr, mp.Nitr),end='');
        
        #--Re-compute the starlight normalization factor for the compact and full models (to convert images to normalized intensity). No tip/tilt necessary.
        falco.imaging.falco_get_PSF_norm_factor(mp);
           
        ## Updated DM data
        #--Change the selected DMs if using the scheduled EFC controller
        if mp.controller.lower() in ['plannedefc']:
            mp.dm_ind = mp.dm_ind_sched[Itr];
    
        #--Report which DMs are used in this iteration
        print('DMs to be used in this iteration = [',end='')
        for jj in range(len(mp.dm_ind)):
            print(' %d'%(mp.dm_ind[jj]),end='')
        print(' ]')
        
        #--Fill in History of DM commands to Store
        if hasattr(mp,'dm1'): 
            if hasattr(mp.dm1,'V'):  
                out.dm1.Vall[:,:,Itr] = mp.dm1.V  
        if hasattr(mp,'dm2'): 
            if hasattr(mp.dm2,'V'):  
                out.dm2.Vall[:,:,Itr] = mp.dm2.V
        if hasattr(mp,'dm5'): 
            if hasattr(mp.dm5,'V'):  
                out.dm5.Vall[:,:,Itr] = mp.dm5.V
        if hasattr(mp,'dm8'): 
            if hasattr(mp.dm8,'V'):  
                out.dm8.Vallr[:,Itr] = mp.dm8.V[:]
        if hasattr(mp,'dm9'): 
            if hasattr(mp.dm9,'V'):  
                out.dm9.Vall[:,Itr] = mp.dm9.V[:]
    
        #--Compute the DM surfaces
        if np.any(mp.dm_ind==1): 
            DM1surf =  falco.dms.falco_gen_dm_surf(mp.dm1, mp.dm1.compact.dx, mp.dm1.compact.Ndm)
        else: 
            DM1surf = np.zeros((mp.dm1.compact.Ndm,mp.dm1.compact.Ndm))
    
        if np.any(mp.dm_ind==2): 
            DM2surf =  falco.dms.falco_gen_dm_surf(mp.dm2, mp.dm2.compact.dx, mp.dm2.compact.Ndm);  
        else: 
            DM2surf = np.zeros((mp.dm2.compact.Ndm,mp.dm2.compact.Ndm))
    
        ## Updated plot and reporting
        #--Calculate the core throughput (at higher resolution to be more accurate)
        thput,ImSimOffaxis = falco.utils.falco_compute_thput(mp);
        if mp.flagFiber:
            mp.thput_vec[Itr] = np.max(thput)
        else:
            mp.thput_vec[Itr] = thput; #--record keeping
        
        #--Compute the current contrast level
        InormHist[Itr] = np.mean(Im[mp.Fend.corr.maskBool]);
        
        if(any(mp.dm_ind==1)):
            mp.dm1 = falco.dms.falco_enforce_dm_constraints(mp.dm1)
        if(any(mp.dm_ind==2)):
            mp.dm2 = falco.dms.falco_enforce_dm_constraints(mp.dm2)
        
        #--Plotting
        if(mp.flagPlot):
            
#            if(Itr==1):
#                plt.ion()
#                plt.show()
            plt.figure(1)
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
            fig.subplots_adjust(hspace=0.4, wspace=0.0)
            fig.suptitle(mp.coro+': Iteration %d'%Itr)
            
            im1=ax1.imshow(np.log10(Im),cmap='magma',interpolation='none',extent=[np.min(mp.Fend.xisDL),np.max(mp.Fend.xisDL),np.min(mp.Fend.xisDL),np.max(mp.Fend.xisDL)])
            ax1.set_title('Stellar PSF: NI=%.2e'%InormHist[Itr])
            ax1.tick_params(labelbottom=False)
            cbar1 = fig.colorbar(im1, ax = ax1)#,shrink=0.95)

            im3 = ax3.imshow(ImSimOffaxis/np.max(ImSimOffaxis),cmap=plt.cm.get_cmap('Blues'),interpolation='none',extent=[np.min(mp.Fend.xisDL),np.max(mp.Fend.xisDL),np.min(mp.Fend.xisDL),np.max(mp.Fend.xisDL)])            
#            im3 = ax3.imshow(ImSimOffaxis/np.max(ImSimOffaxis),cmap=plt.cm.get_cmap('Blues', 4),interpolation='none',extent=[np.min(mp.Fend.xisDL),np.max(mp.Fend.xisDL),np.min(mp.Fend.xisDL),np.max(mp.Fend.xisDL)])
            ax3.set_title('Off-axis Thput = %.2f%%'%(100*thput))
            cbar3 = fig.colorbar(im3, ax = ax3)
            cbar3.set_ticks(np.array([0.0, 0.5, 1.0]))
            cbar3.set_ticklabels(['0', '0.5', '1'])
            
            im2 = ax2.imshow(1e9*DM1surf,cmap='viridis')
            ax2.set_title('DM1 Surface (nm)')
            ax2.tick_params(labelbottom=False,labelleft=False,bottom=False,left=False)
            cbar2 = fig.colorbar(im2, ax = ax2)
            
            im4 = ax4.imshow(1e9*DM2surf,cmap='viridis')
            ax4.set_title('DM2 Surface (nm)')
            ax4.tick_params(labelbottom=False,labelleft=False,bottom=False,left=False)
            cbar4 = fig.colorbar(im4, ax = ax4)
            
            plt.show()
            plt.pause(0.1)
            
        ## Updated selection of Zernike modes targeted by the controller
        #--Decide with Zernike modes to include in the Jacobian
        if Itr==0:
            mp.jac.zerns0 = mp.jac.zerns;
        
        print('Zernike modes (Noll indexing) used in this Jacobian:\t',end=''); 
        print(mp.jac.zerns)
        
        #--Re-compute the Jacobian weights
        falco.configs.falco_config_jac_weights(mp);
        
        ## Actuator Culling: Initialization of Flag and Which Actuators
        
        #--If new actuators are added, perform a new cull of actuators.
        cvar = falco.config.Object()
        if Itr==0:
            cvar.flagCullAct = True;
        else:
            if hasattr(mp,'dm_ind_sched'):
                cvar.flagCullAct = np.not_equal(mp.dm_ind_sched[Itr], mp.dm_ind_sched[Itr-1]);
            else:
                cvar.flagCullAct = False;
        
        mp.flagCullActHist[Itr] = cvar.flagCullAct;
        
        #--Before performing new cull, include all actuators again
        if cvar.flagCullAct:
            #--Re-include all actuators in the basis set.
            if np.any(mp.dm_ind==1): 
                mp.dm1.act_ele = list(range(mp.dm1.NactTotal)) 
            if np.any(mp.dm_ind==2): 
                mp.dm2.act_ele = list(range(mp.dm2.NactTotal)) 
            if np.any(mp.dm_ind==5):
                mp.dm5.act_ele = list(range(mp.dm5.NactTotal))
            if np.any(mp.dm_ind==8): 
                mp.dm8.act_ele = list(range(mp.dm8.NactTotal))
            if np.any(mp.dm_ind==9): 
                mp.dm9.act_ele = list(range(mp.dm9.NactTotal))
            #--Update the number of elements used per DM
            if np.any(mp.dm_ind==1): 
                mp.dm1.Nele = len(mp.dm1.act_ele)
            else: 
                mp.dm1.Nele = 0; 
            if np.any(mp.dm_ind==2): 
                mp.dm2.Nele = len(mp.dm2.act_ele)
            else: 
                mp.dm2.Nele = 0; 
            if np.any(mp.dm_ind==5): 
                mp.dm5.Nele = len(mp.dm5.act_ele)
            else: 
                mp.dm5.Nele = 0; 
            if np.any(mp.dm_ind==8): 
                mp.dm8.Nele = len(mp.dm8.act_ele)
            else: 
                mp.dm8.Nele = 0; 
            if np.any(mp.dm_ind==9): 
                mp.dm9.Nele = len(mp.dm9.act_ele)
            else: 
                mp.dm9.Nele = 0; 
        
        ## Compute the control Jacobians for each DM
        
        #--Relinearize about the DMs only at the iteration numbers in mp.relinItrVec.
        if np.any(mp.relinItrVec==Itr):
            cvar.flagRelin=True;
        else:
            cvar.flagRelin=False;
        
        if  Itr==0 or cvar.flagRelin:
            jacStruct =  falco.models.model_Jacobian(mp); #--Get structure containing Jacobians
        
        ## Modify jacStruct to cull actuators, but only if(cvar.flagCullAct && cvar.flagRelin)
        falco_ctrl_cull(mp,cvar,jacStruct);
        
        ## Load the improved Jacobian if using the E-M technique
#        if mp.flagUseLearnedJac:
#            jacStructLearned = load('jacStructLearned.mat');
#            if np.any(mp.dm_ind==1):  
#            jacStruct.G1 = jacStructLearned.G1
#            if np.any(mp.dm_ind==1):  
#            jacStruct.G2 = jacStructLearned.G2
        
        ## Wavefront Estimation
        if mp.estimator.lower() in ['perfect']:
            EfieldVec  = falco_est_perfect_Efield_with_Zernikes(mp)
        elif mp.estimator.lower in ['pwp-bp','pwp-kf']:
            if mp.est.flagUseJac: #--Send in the Jacobian if true
                ev = falco_est_pairwise_probing(mp,jacStruct);
            else: #--Otherwise don't pass the Jacobian
                ev = falco_est_pairwise_probing(mp);
        
            EfieldVec = ev.Eest;
            IincoVec = ev.IincoEst;
                
        ## Add spatially-dependent weighting to the control Jacobians
        if np.any(mp.dm_ind==1): 
            jacStruct.G1 = jacStruct.G1*np.moveaxis(np.tile(mp.WspatialVec[:,None],[mp.jac.Nmode,1,mp.dm1.Nele]),0,-1)
        if np.any(mp.dm_ind==2): 
            jacStruct.G2 = jacStruct.G2*np.moveaxis(np.tile(mp.WspatialVec[:,None],[mp.jac.Nmode,1,mp.dm2.Nele]),0,-1)
        if np.any(mp.dm_ind==8): 
            jacStruct.G8 = jacStruct.G8*np.moveaxis(np.tile(mp.WspatialVec[:,None],[mp.jac.Nmode,1,mp.dm8.Nele]),0,-1)
        if np.any(mp.dm_ind==9): 
            jacStruct.G9 = jacStruct.G9*np.moveaxis(np.tile(mp.WspatialVec[:,None],[mp.jac.Nmode,1,mp.dm9.Nele]),0,-1)
            
        #fprintf('Total Jacobian Calcuation Time: #.2f\n',toc);
    
        #--Compute the number of total actuators for all DMs used. 
        cvar.NeleAll = mp.dm1.Nele + mp.dm2.Nele + mp.dm3.Nele + mp.dm4.Nele + mp.dm5.Nele + mp.dm6.Nele + mp.dm7.Nele + mp.dm8.Nele + mp.dm9.Nele; #--Number of total actuators used 
    
        ## Wavefront Control
    
        cvar.Itr = Itr
        cvar.EfieldVec = EfieldVec
        cvar.InormHist = InormHist[Itr]
        falco_ctrl(mp,cvar,jacStruct)
    
        #--Save out regularization used.
        if hasattr(cvar, "log10regUsed"):
            out.log10regHist[Itr] = cvar.log10regUsed;
    
        #-----------------------------------------------------------------------------------------
        
        ## DM Stats
        #--Calculate and report updated P-V DM voltages.
        if np.any(mp.dm_ind==1):
            out.dm1.Vpv[Itr] = np.max(mp.dm1.V) - np.min(mp.dm1.V)
            print(' DM1 P-V in volts: %.3f'%(out.dm1.Vpv[Itr]))
            if(mp.dm1.tied.size>0):  
                print(' DM1 has %d pairs of tied actuators.' % (mp.dm1.tied.shape[0]) )
#            Nrail1 = len(np.where( (mp.dm1.V <= -mp.dm1.maxAbsV) | (mp.dm1.V >= mp.dm1.maxAbsV) )); 
#            print(' DM1 P-V in volts: %.3f\t\t%d/%d (%.2f%%) railed actuators \n'%(out.dm1.Vpv(Itr), Nrail1, mp.dm1.NactTotal, 100*Nrail1/mp.dm1.NactTotal))
#            if mp.dm1.tied.shape[0]>0:  
#                print(' DM1 has %d pairs of tied actuators.\n'%(mp.dm1.tied.shape[0]))
        if np.any(mp.dm_ind==2):
            out.dm2.Vpv[Itr] = np.max(mp.dm2.V) - np.min(mp.dm2.V)
            print(' DM2 P-V in volts: %.3f'%(out.dm2.Vpv[Itr]))
            if(mp.dm2.tied.size>0):  
                print(' DM2 has %d pairs of tied actuators.' % (mp.dm2.tied.shape[0]) )

#            Nrail2 = len(np.where( (mp.dm2.V <= -mp.dm2.maxAbsV) | (mp.dm2.V >= mp.dm2.maxAbsV) )); 
#            print(' DM2 P-V in volts: %.3f\t\t%d/%d (%.2f%%) railed actuators \n'%( out.dm2.Vpv(Itr), Nrail2, mp.dm2.NactTotal, 100*Nrail2/mp.dm2.NactTotal))
#            if mp.dm2.tied.shape[0]>0:  
#                print(' DM2 has %d pairs of tied actuators.\n'%(mp.dm2.tied.shape[0]))
        # if(any(mp.dm_ind==8))
        #     out.dm8.Vpv(Itr) = (max(max(mp.dm8.V))-min(min(mp.dm8.V)));
        #     Nrail8 = length(find( (mp.dm8.V <= mp.dm8.Vmin) | (mp.dm8.V >= mp.dm8.Vmax) ));
        #     fprintf(' DM8 P-V in volts: #.3f\t\t#d/#d (#.2f##) railed actuators \n', out.dm8.Vpv(Itr), Nrail8,mp.dm8.NactTotal,100*Nrail8/mp.dm8.NactTotal); 
        # end
#        if np.any(mp.dm_ind==9):
#            out.dm9.Vpv[Itr] = np.max(np.max(mp.dm9.V))-np.min(np.min(mp.dm9.V))
#            Nrail9 = len(np.where( (mp.dm9.V <= mp.dm9.Vmin) | (mp.dm9.V >= mp.dm9.Vmax) )); 
#            print(' DM9 P-V in volts: %.3f\t\t%d/%d (%.2f%%) railed actuators \n'%(out.dm9.Vpv(Itr), Nrail9,mp.dm9.NactTotal,100*Nrail9/mp.dm9.NactTotal))
        
        #--Calculate and report updated RMS DM surfaces.               
        if(any(mp.dm_ind==1)):
            # Pupil-plane coordinates
            dx_dm = mp.P2.compact.dx/mp.P2.D #--Normalized dx [Units of pupil diameters]
            xs = falco.utils.create_axis(mp.dm1.compact.Ndm, dx_dm, centering=mp.centering)
            RS = falco.utils.radial_grid(xs)
            rmsSurf_ele = np.logical_and(RS>=mp.P1.IDnorm/2., RS<=0.5)
            
            DM1surf = falco.dms.falco_gen_dm_surf(mp.dm1, mp.dm1.compact.dx, mp.dm1.compact.Ndm)
            out.dm1.Spv[Itr] = np.max(DM1surf)-np.min(DM1surf)
            out.dm1.Srms[Itr] = np.sqrt(np.mean(np.abs( (DM1surf[rmsSurf_ele]) )**2))
            print('RMS surface of DM1 = %.1f nm' % (1e9*out.dm1.Srms[Itr]))
        if(any(mp.dm_ind==2)):
            # Pupil-plane coordinates
            dx_dm = mp.P2.compact.dx/mp.P2.D #--Normalized dx [Units of pupil diameters]
            xs = falco.utils.create_axis(mp.dm2.compact.Ndm, dx_dm, centering=mp.centering)
            RS = falco.utils.radial_grid(xs)
            rmsSurf_ele = np.logical_and(RS>=mp.P1.IDnorm/2., RS<=0.5)
            
            DM2surf = falco.dms.falco_gen_dm_surf(mp.dm2, mp.dm2.compact.dx, mp.dm2.compact.Ndm)
            out.dm2.Spv[Itr] = np.max(DM2surf)-np.min(DM2surf)
            out.dm2.Srms[Itr] = np.sqrt(np.mean(np.abs( (DM2surf[rmsSurf_ele]) )**2))
            print('RMS surface of DM2 = %.1f nm' % (1e9*out.dm2.Srms[Itr]))
        
        #--Calculate sensitivities to 1nm RMS of Zernike phase aberrations at entrance pupil.
        if( (mp.eval.Rsens.size>0) and (mp.eval.indsZnoll.size>0) ):
            out.Zsens[:,:,Itr] = falco.zernikes.falco_get_Zernike_sensitivities(mp)
        
        # Take the next image to check the contrast level (in simulation only)
        with falco.utils.TicToc('Getting updated summed image'):
            Im = falco.imaging.falco_get_summed_image(mp);
        
        #--REPORTING NORMALIZED INTENSITY
        print('Itr: ', Itr)
        InormHist[Itr+1] = np.mean(Im[mp.Fend.corr.maskBool]);
        print('Prev and New Measured Contrast (LR):\t\t\t %.2e\t->\t%.2e\t (%.2f x smaller)  \n\n'%(InormHist[Itr], InormHist[Itr+1], InormHist[Itr]/InormHist[Itr+1]))

    
        # --END OF ESTIMATION + CONTROL LOOP
    
    Itr = Itr + 1;
    
    #--Data to store
    if np.any(mp.dm_ind==1): 
        out.dm1.Vall[:,:,Itr] = mp.dm1.V
    if np.any(mp.dm_ind==2): 
        out.dm2.Vall[:,:,Itr] = mp.dm2.V
    # if(any(mp.dm_ind==5)); out.dm5.Vall(:,:,Itr) = mp.dm5.V; end
    if np.any(mp.dm_ind==8): 
        out.dm8.Vall[:,Itr] = mp.dm8.V
    if np.any(mp.dm_ind==9): 
        out.dm9.Vall[:,Itr] = mp.dm9.V
    
    #--Calculate the core throughput (at higher resolution to be more accurate)
    thput,ImSimOffaxis = falco.utils.falco_compute_thput(mp);
    if mp.flagFiber:
        mp.thput_vec[Itr] = np.max(thput);
    else:
        mp.thput_vec[Itr] = thput; #--record keeping
    
    ## Save the final DM commands separately for faster reference
    if hasattr(mp,'dm1'): 
        if hasattr(mp.dm1,'V'): 
            out.DM1V = mp.dm1.V
    if hasattr(mp,'dm2'): 
        if hasattr(mp.dm2,'V'): 
            out.DM2V = mp.dm2.V
    if mp.coro.upper() in ['HLC','EHLC']:
        if hasattr(mp.dm8,'V'): 
            out.DM8V = mp.dm8.V
        if hasattr(mp.dm9,'V'): 
            out.DM9V = mp.dm9.V
    
    ## Save out an abridged workspace
    
    #--Variables to save out:
    # contrast vs iter
    # regularization history
    #  DM1surf,DM1V, DM2surf,DM2V, DM8surf,DM9surf, fpm sampling, base pmgi thickness, base nickel thickness, dm_tilts, aoi, ...
    # to reproduce your basic design results of NI, throughput, etc..
    
    out.thput = mp.thput_vec;
    out.Nitr = mp.Nitr;
    out.InormHist = InormHist;
    
    fnOut = mp.path.config + mp.runLabel + '_snippet.pkl'
    
    print('\nSaving abridged workspace to file:\n\t%s\n'%(fnOut),end='')
    #save(fnOut,'out');
    with open(fnOut, 'wb') as f:
        pickle.dump(out,f)
    print('...done.\n')

    ## Save out the data from the workspace
    if mp.flagSaveWS:
        #clear cvar G* h* jacStruct; # Save a ton of space when storing the workspace
        del cvar; del G; del h; del jacStruct
    
        # Don't bother saving the large 2-D, floating point maps in the workspace (they take up too much space)
        mp.P1.full.mask=1; mp.P1.compact.mask=1;
        mp.P3.full.mask=1; mp.P3.compact.mask=1;
        mp.P4.full.mask=1; mp.P4.compact.mask=1;
        mp.F3.full.mask=1; mp.F3.compact.mask=1;
    
        mp.P1.full.E = 1; mp.P1.compact.E=1; mp.Eplanet=1;
        mp.dm1.full.mask = 1; mp.dm1.compact.mask = 1; mp.dm2.full.mask = 1; mp.dm2.compact.mask = 1;
        mp.complexTransFull = 1; mp.complexTransCompact = 1;
    
        mp.dm1.compact.inf_datacube = 0;
        mp.dm2.compact.inf_datacube = 0;
        mp.dm8.compact.inf_datacube = 0;
        mp.dm9.compact.inf_datacube = 0;
        mp.dm8.inf_datacube = 0;
        mp.dm9.inf_datacube = 0;
    
        fnAll = mp.path.ws + mp.runLabel + '_all.pkl'
        print('Saving entire workspace to file ' + fnAll + '...',end='')
        #save(fnAll);
        with open(fnAll, 'wb') as f:
            pickle.dump(mp,f)

        print('done.\n\n')
    else:
        print('Entire workspace NOT saved because mp.flagSaveWS==false')
    
    #--END OF main FUNCTION
    print('END OF WFSC LOOP: ', mp)

    
def falco_est_perfect_Efield_with_Zernikes(mp):
    """
   Function to return the perfect-knowledge E-field from the full model. Can include 
   Zernike aberrations at the input pupil.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
        
    Returns
    -------
    Emat : numpy ndarray
        2-D array with the vectorized, complex E-field of the dark hole pixels for each 
        mode included in the control Jacobian.
    """  
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    
    Emat = np.zeros((mp.Fend.corr.Npix, mp.jac.Nmode),dtype=complex)
    modvar = falco.config.Object() #--Initialize
    
    for im in range(mp.jac.Nmode):
        modvar.sbpIndex = mp.jac.sbp_inds[im]
        modvar.zernIndex = mp.jac.zern_inds[im]
        modvar.whichSource = 'star'
        
        #--Take the mean over the wavelengths within the sub-bandpass
        EmatSbp = np.zeros((mp.Fend.corr.Npix, mp.Nwpsbp),dtype=complex)
        for wi in range(mp.Nwpsbp):
            modvar.wpsbpIndex = wi
            E2D = falco.models.model_full(mp, modvar)
            EmatSbp[:,wi] = mp.full.lambda_weights[wi]*E2D[mp.Fend.corr.maskBool] #--Actual field in estimation area. Apply spectral weight within the sub-bandpass
        Emat[:,im] = np.sum(EmatSbp,axis=1)
    
    return Emat
    
    #pass
    #return falco.config.Object()

def falco_est_pairwise_probing(mp, **kwargs):
    
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    pass

    return falco.config.Object()


def falco_ctrl(mp,cvar,jacStruct):
    """
    Outermost wrapper function for all the controller functions.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    cvar : ModelParameters
        Structure containing controller variables
    jacStruct : ModelParameters
        Structure containing control Jacobians for each specified DM.

    Returns
    -------
    None
        Changes are made by reference to mp.
    """     
#    if type(mp) is not falco.config.ModelParameters:
#        raise TypeError('Input "mp" must be of type ModelParameters')
#    pass

    #with falco.utils.TicToc('Using the Jacobian to make other matrices'):
    print('Using the Jacobian to make other matrices...',end='')
    
    #--Compute matrices for linear control with regular EFC
    cvar.GstarG_wsum  = np.zeros((cvar.NeleAll,cvar.NeleAll)) 
    cvar.RealGstarEab_wsum = np.zeros((cvar.NeleAll, 1))

    for im in range(mp.jac.Nmode):

        Gmode = np.zeros((mp.Fend.corr.Npix,1)) #--Initialize a row to concatenate onto
        if(any(mp.dm_ind==1)): Gmode = np.hstack((Gmode,np.squeeze(jacStruct.G1[:,:,im])))
        if(any(mp.dm_ind==2)): Gmode = np.hstack((Gmode,np.squeeze(jacStruct.G2[:,:,im])))
        if(any(mp.dm_ind==8)): Gmode = np.hstack((Gmode,np.squeeze(jacStruct.G8[:,:,im])))
        if(any(mp.dm_ind==9)): Gmode = np.hstack((Gmode,np.squeeze(jacStruct.G9[:,:,im])))
        Gmode = Gmode[:,1:] #--Remove the zero column used for initialization
        #Gstack = [jacStruct.G1[:,:,im], jacStruct.G2[:,:,im], jacStruct.G8[:,:,im], jacStruct.G9[:,:,im] ]

        #--Square matrix part stays the same if no re-linearization has occurrred. 
        cvar.GstarG_wsum += mp.jac.weights[im]*np.real(np.conj(Gmode).T @ Gmode) 

        #--The G^*E part changes each iteration because the E-field changes.
        Eweighted = mp.WspatialVec*cvar.EfieldVec[:,im] #--Apply 2-D spatial weighting to E-field in dark hole pixels.
        cvar.RealGstarEab_wsum += mp.jac.weights[im]*np.real(np.conj(Gmode).T @ Eweighted.reshape(mp.Fend.corr.Npix,1)) #--Apply the Jacobian weights and add to the total.
    
    #--Make the regularization matrix. (Define only the diagonal here to save RAM.)
    cvar.EyeGstarGdiag = np.max(np.diag(cvar.GstarG_wsum))*np.ones(cvar.NeleAll)
    cvar.EyeNorm = np.max(np.diag(cvar.GstarG_wsum))
    print('done.') #fprintf(' done. Time: %.3f\n',toc);

    #--Call the Controller Function
    print('Control beginning ...') # tic
    #--Established, conventional controllers
    if(mp.controller.lower()=='plannedefc'): #--EFC regularization is scheduled ahead of time
        dDM = falco_ctrl_planned_EFC(mp,cvar)
    elif(mp.controller.lower()=='gridsearchefc'):
        dDM = falco_ctrl_grid_search_EFC(mp,cvar)

    #--Experimental controllers
    elif(mp.controller.lower()=='plannedefcts'): #--EFC regularization is scheduled ahead of time. total stroke also minimized
        dDM = falco_ctrl_planned_EFC_TS(mp,cvar)   

    elif(mp.controller.lower()=='plannedefccon'): #--Constrained-EFC regularization is scheduled ahead of time
        dDM = falco_ctrl_planned_EFCcon(mp,cvar)          
        
#    print('done.\n') #print(' done. Time: %.3f sec\n',toc);
    
    #--Update the DM commands by adding the delta control signal
    if(any(mp.dm_ind==1)):  mp.dm1.V += dDM.dDM1V
    if(any(mp.dm_ind==2)):  mp.dm2.V += dDM.dDM2V
    if(any(mp.dm_ind==8)):  mp.dm8.V += dDM.dDM8V
    if(any(mp.dm_ind==9)):  mp.dm9.V += dDM.dDM9V

    #--Save the delta from the previous command
    if(any(mp.dm_ind==1)):  mp.dm1.dV = dDM.dDM1V 
    if(any(mp.dm_ind==2)):  mp.dm2.dV = dDM.dDM2V
    if(any(mp.dm_ind==8)):  mp.dm8.dV = dDM.dDM8V
    if(any(mp.dm_ind==9)):  mp.dm9.dV = dDM.dDM9V    
    

def falco_ctrl_cull(mp, cvar, jacStruct):
    """
    Function that removes weak actuators from the controlled set.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    cvar : ModelParameters
        Structure containing controller variables
    jacStruct : ModelParameters
        Structure containing control Jacobians for each specified DM.

    Returns
    -------
    None
        Changes are made by reference to mp and jacStruct.
    """ 
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    
    #--Reduce the number of actuators used based on their relative strength in the Jacobian
    if(cvar.flagCullAct and cvar.flagRelin):
        
        print('Weeding out weak actuators from the control Jacobian...'); 
        if(any(mp.dm_ind==1)):
            G1intNorm = np.sum( np.mean(np.abs(jacStruct.G1)**2,axis=2), axis=0)
            G1intNorm = G1intNorm/np.max(G1intNorm)
            mp.dm1.act_ele = np.nonzero(G1intNorm>=10**(mp.logGmin))[0]
            del G1intNorm
        if(any(mp.dm_ind==2)):
            G2intNorm = np.sum( np.mean(np.abs(jacStruct.G2)**2,axis=2), axis=0)
            G2intNorm = G2intNorm/np.max(G2intNorm)
            mp.dm2.act_ele = np.nonzero(G2intNorm>=10**(mp.logGmin))[0]
            del G2intNorm
        if(any(mp.dm_ind==8)):
            G8intNorm = np.sum( np.mean(np.abs(jacStruct.G8)**2,axis=2), axis=0)
            G8intNorm = G8intNorm/np.max(G8intNorm)
            mp.dm8.act_ele = np.nonzero(G8intNorm>=10**(mp.logGmin))[0]
            del G8intNorm
        if(any(mp.dm_ind==9)):
            G9intNorm = np.sum( np.mean(np.abs(jacStruct.G9)**2,axis=2), axis=0)
            G9intNorm = G9intNorm/np.max(G9intNorm)
            mp.dm9.act_ele = np.nonzero(G9intNorm>=10**(mp.logGmin))[0]
            del G9intNorm

        #--Add back in all actuators that are tied (to make the tied actuator logic easier)
        if(any(mp.dm_ind==1)):
            for ti in range(mp.dm1.tied.shape[0]):
                if not (any(mp.dm1.act_ele==mp.dm1.tied[ti,0])):  mp.dm1.act_ele = np.hstack([mp.dm1.act_ele, mp.dm1.tied[ti,0]])
                if not (any(mp.dm1.act_ele==mp.dm1.tied[ti,1])):  mp.dm1.act_ele = np.hstack([mp.dm1.act_ele, mp.dm1.tied[ti,1]])
            mp.dm1.act_ele = np.sort(mp.dm1.act_ele) #--Need to sort for the logic in model_Jacobian.m

        if(any(mp.dm_ind==2)):
            for ti in range(mp.dm2.tied.shape[0]):
                if not any(mp.dm2.act_ele==mp.dm2.tied[ti,0]):  mp.dm2.act_ele = np.hstack([mp.dm2.act_ele, mp.dm2.tied[ti,0]])
                if not any(mp.dm2.act_ele==mp.dm2.tied[ti,1]):  mp.dm2.act_ele = np.hstack([mp.dm2.act_ele, mp.dm2.tied[ti,1]])
            mp.dm2.act_ele = np.sort(mp.dm2.act_ele) #--Need to sort for the logic in model_Jacobian.m
#            if(any(mp.dm_ind==8))
#                for ti=1:size(mp.dm8.tied,1)
#                    if(any(mp.dm8.act_ele==mp.dm8.tied(ti,1))==false);  mp.dm8.act_ele = [mp.dm8.act_ele; mp.dm8.tied(ti,1)];  end
#                    if(any(mp.dm8.act_ele==mp.dm8.tied(ti,2))==false);  mp.dm8.act_ele = [mp.dm8.act_ele; mp.dm8.tied(ti,2)];  end
#                end
#                mp.dm8.act_ele = sort(mp.dm8.act_ele);
#            end
#            if(any(mp.dm_ind==9))
#                for ti=1:size(mp.dm9.tied,1)
#                    if(any(mp.dm9.act_ele==mp.dm9.tied(ti,1))==false);  mp.dm9.act_ele = [mp.dm9.act_ele; mp.dm9.tied(ti,1)];  end
#                    if(any(mp.dm9.act_ele==mp.dm9.tied(ti,2))==false);  mp.dm9.act_ele = [mp.dm9.act_ele; mp.dm9.tied(ti,2)];  end
#                end
#                mp.dm9.act_ele = sort(mp.dm9.act_ele);
#            end
            
        #--Update the number of elements used per DM
        if(any(mp.dm_ind==1)): mp.dm1.Nele = mp.dm1.act_ele.size
        if(any(mp.dm_ind==2)): mp.dm2.Nele = mp.dm2.act_ele.size
        if(any(mp.dm_ind==8)): mp.dm8.Nele = mp.dm8.act_ele.size
        if(any(mp.dm_ind==9)): mp.dm9.Nele = mp.dm9.act_ele.size

        if(any(mp.dm_ind==1)): print('  DM1: %d/%d (%.2f%%) actuators kept for Jacobian' % (mp.dm1.Nele, mp.dm1.NactTotal,100*mp.dm1.Nele/mp.dm1.NactTotal))
        if(any(mp.dm_ind==2)): print('  DM2: %d/%d (%.2f%%) actuators kept for Jacobian' % (mp.dm2.Nele, mp.dm2.NactTotal,100*mp.dm2.Nele/mp.dm2.NactTotal))
        if(any(mp.dm_ind==8)): print('  DM8: %d/%d (%.2f%%) actuators kept for Jacobian' % (mp.dm8.Nele, mp.dm8.NactTotal,100*mp.dm8.Nele/mp.dm8.NactTotal))
        if(any(mp.dm_ind==9)): print('  DM9: %d/%d (%.2f%%) actuators kept for Jacobian' % (mp.dm9.Nele, mp.dm9.NactTotal,100*mp.dm9.Nele/mp.dm9.NactTotal))
        
        #--Crop out unused actuators from the control Jacobian
        if(any(mp.dm_ind==1)): jacStruct.G1 = jacStruct.G1[:,mp.dm1.act_ele,:]
        if(any(mp.dm_ind==2)): jacStruct.G2 = jacStruct.G2[:,mp.dm2.act_ele,:]
        if(any(mp.dm_ind==8)): jacStruct.G8 = jacStruct.G8[:,mp.dm8.act_ele,:]
        if(any(mp.dm_ind==9)): jacStruct.G9 = jacStruct.G9[:,mp.dm9.act_ele,:]


    #return jacStruct
    #return falco.config.Object()

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


def falco_ctrl_grid_search_EFC(mp,cvar):
    """
    Wrapper controller function that performs a grid search over specified variables.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    cvar : ModelParameters
        Structure containing controller variables

    Returns
    -------
    dDM : ModelParameters
        Structure containing the delta DM commands for each DM
    """    
    #--Initializations    
    vals_list = [(x,y) for y in mp.ctrl.dmfacVec for x in mp.ctrl.log10regVec ] #--Make all combinations of the values
    Nvals = mp.ctrl.log10regVec.size*mp.ctrl.dmfacVec.size
    InormVec = np.zeros(Nvals)

    #--Temporarily store computed DM commands so that the best one does not have to be re-computed
    if(any(mp.dm_ind==1)): dDM1V_store = np.zeros((mp.dm1.Nact,mp.dm1.Nact,Nvals))
    if(any(mp.dm_ind==2)): dDM2V_store = np.zeros((mp.dm2.Nact,mp.dm2.Nact,Nvals))
    if(any(mp.dm_ind==8)): dDM8V_store = np.zeros((mp.dm8.NactTotal,Nvals))
    if(any(mp.dm_ind==9)): dDM9V_store = np.zeros((mp.dm9.NactTotal,Nvals))

    ## Empirically find the regularization value giving the best contrast
    if(mp.flagMultiproc):
        #--Run the controller in parallel
        pool = multiprocessing.Pool(processes=mp.Nthreads)
        results = [pool.apply_async(falco_ctrl_EFC_base, args=(ni,vals_list,mp,cvar)) for ni in np.arange(Nvals,dtype=int) ]
        results_ctrl = [p.get() for p in results] #--All the Jacobians in a list
        pool.close()
        pool.join()
        
        #--Convert from a list to arrays:
        for ni in range(Nvals):
            InormVec[ni] = results_ctrl[ni][0]
            if(any(mp.dm_ind==1)): dDM1V_store[:,:,ni] = results_ctrl[ni][1].dDM1V
            if(any(mp.dm_ind==2)): dDM2V_store[:,:,ni] = results_ctrl[ni][1].dDM2V
    else:
        for ni in range(Nvals):
            [InormVec[ni],dDM_temp] = falco_ctrl_EFC_base(ni,vals_list,mp,cvar) 
            #--delta voltage commands
            if(any(mp.dm_ind==1)): dDM1V_store[:,:,ni] = dDM_temp.dDM1V
            if(any(mp.dm_ind==2)): dDM2V_store[:,:,ni] = dDM_temp.dDM2V
            if(any(mp.dm_ind==8)): dDM8V_store[:,ni] = dDM_temp.dDM8V
            if(any(mp.dm_ind==9)): dDM9V_store[:,ni] = dDM_temp.dDM9V

    #--Print out results to the command line
    print('Scaling factor:\t',end='')
    for ni in range(Nvals): print('%.2f\t\t' % (vals_list[ni][1]),end='')

    print('\nlog10reg:\t',end='')
    for ni in range(Nvals): print('%.1f\t\t' % (vals_list[ni][0]),end='')
    
    print('\nInorm:  \t',end='')
    for ni in range(Nvals):  print('%.2e\t' % (InormVec[ni]),end='')
    print('\n',end='')

    #--Find the best scaling factor and Lagrange multiplier pair based on the best contrast.
    #[cvar.cMin,indBest] = np.min(InormVec)
    indBest = np.argmin(InormVec)
    cvar.cMin = np.min(InormVec)
    dDM = falco.config.Object()
    #--delta voltage commands
    if(any(mp.dm_ind==1)): dDM.dDM1V = np.squeeze(dDM1V_store[:,:,indBest])
    if(any(mp.dm_ind==2)): dDM.dDM2V = np.squeeze(dDM2V_store[:,:,indBest])
    if(any(mp.dm_ind==8)): dDM.dDM8V = np.squeeze(dDM8V_store[:,indBest])
    if(any(mp.dm_ind==9)): dDM.dDM9V = np.squeeze(dDM9V_store[:,indBest])

    cvar.log10regUsed = vals_list[indBest][0]
    dmfacBest = vals_list[indBest][1]
    if(mp.ctrl.flagUseModel):
        print('Model-based grid search gives log10reg, = %.1f,\t dmfac = %.2f,\t %4.2e contrast.' % (cvar.log10regUsed, dmfacBest, cvar.cMin) )
    else:
        print('Empirical grid search gives log10reg, = %.1f,\t dmfac = %.2f,\t %4.2e contrast.' % (cvar.log10regUsed, dmfacBest, cvar.cMin) )
    
    return dDM


def falco_ctrl_EFC_base(ni,vals_list,mp,cvar):
    """
    Function that computes the main EFC equation. Called by a wrapper controller function. 

    Parameters
    ----------
    ni : int
        index for the set of possible combinations of variables to do a grid search over
    vals_list : list
        the set of possible combinations of values to do a grid search over
    mp : ModelParameters
        Structure containing optical model parameters
    cvar : ModelParameters
        Structure containing controller variables

    Returns
    -------
    InormAvg : float
        Normalized intensity averaged over wavelength and over the entire dark hole
    dDM : ModelParameters
        Structure containing the delta DM commands for each DM
    """
    #function [InormAvg,dDM] = falco_ctrl_EFC_base(ni,vals_list,mp,cvar)

    if(any(mp.dm_ind==1)):
        DM1V0 = mp.dm1.V.copy()
#        dDM1V0 = mp.dm1.dV.copy()
    if(any(mp.dm_ind==2)):
        DM2V0 = mp.dm2.V.copy()
        

    #--Initializations
    log10reg = vals_list[ni][0] #--log 10 of regularization value
    dmfac = vals_list[ni][1] #--Scaling factor for entire DM command
    
    #--Save starting point for each delta command to be added to.
    #--Get the indices of each DM's command vector within the single concatenated command vector
    falco_ctrl_setup(mp,cvar) #--Modifies cvar
    
    #--Least-squares solution with regularization:
    duVecNby1 = -dmfac*np.linalg.solve( (10**log10reg*np.diag(cvar.EyeGstarGdiag) + cvar.GstarG_wsum), cvar.RealGstarEab_wsum)
    duVec = duVecNby1.reshape((-1,)) #--Convert to true 1-D array from an Nx1 array
     
    #--Parse the command vector by DM and assign the output commands
    mp,dDM = falco_ctrl_wrapup(mp,cvar,duVec)
    
    #--Take images and compute average intensity in dark hole
    if(mp.ctrl.flagUseModel): #--Perform a model-based grid search using the compact model
        Itotal = falco.imaging.falco_get_expected_summed_image(mp,cvar,dDM)
        InormAvg = np.mean(Itotal[mp.Fend.corr.maskBool])
    else: #--Perform an empirical grid search with actual images from the testbed or full model
        Itotal = falco.imaging.falco_get_summed_image(mp)
        InormAvg = np.mean(Itotal[mp.Fend.corr.maskBool])
        
    #--Reset voltage commands in mp
    if(any(mp.dm_ind==1)): mp.dm1.V = DM1V0
    if(any(mp.dm_ind==2)): mp.dm2.V = DM2V0

    return InormAvg,dDM #dDMtemp


def falco_ctrl_setup(mp,cvar):
    """
    Function to vectorize DM commands and otherwise prepare variables for the controller. 

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    cvar : ModelParameters
        Structure containing controller variables

    Returns
    -------
    None
        Changes are made by reference to structures mp and cvar.
    """
    #--Save starting point for each delta command to be added to.
    if(any(mp.dm_ind==1)): cvar.DM1Vnom = mp.dm1.V
    if(any(mp.dm_ind==2)): cvar.DM2Vnom = mp.dm2.V
    if(any(mp.dm_ind==8)): cvar.DM8Vnom = mp.dm8.V
    if(any(mp.dm_ind==9)): cvar.DM9Vnom = mp.dm9.V
    
    #--Make the vector of total DM commands from before
    u1 = mp.dm1.V.reshape(mp.dm1.NactTotal)[mp.dm1.act_ele] if(any(mp.dm_ind==1)) else np.array([])
    u2 = mp.dm2.V.reshape(mp.dm2.NactTotal)[mp.dm2.act_ele] if(any(mp.dm_ind==2)) else np.array([])
    u8 = mp.dm1.V.reshape(mp.dm8.NactTotal)[mp.dm8.act_ele] if(any(mp.dm_ind==8)) else np.array([])
    u9 = mp.dm1.V.reshape(mp.dm9.NactTotal)[mp.dm9.act_ele] if(any(mp.dm_ind==9)) else np.array([])
    cvar.uVec = np.concatenate((u1,u2,u8,u9))
    cvar.NeleAll = cvar.uVec.size
    
    #--Get the indices of each DM's command within the full command
    u1dummy = 1*np.ones(mp.dm1.Nele,dtype=int) if(any(mp.dm_ind==1)) else np.array([])
    u2dummy = 2*np.ones(mp.dm2.Nele,dtype=int) if(any(mp.dm_ind==2)) else np.array([])
    u8dummy = 8*np.ones(mp.dm8.Nele,dtype=int) if(any(mp.dm_ind==8)) else np.array([])
    u9dummy = 9*np.ones(mp.dm9.Nele,dtype=int) if(any(mp.dm_ind==9)) else np.array([])
    cvar.uLegend = np.concatenate((u1dummy, u2dummy, u8dummy, u9dummy))
    

def falco_ctrl_wrapup(mp,cvar,duVec):
    """
    Function to take the controller commands and apply them to the DMs. 

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    cvar : ModelParameters
        Structure containing controller variables
    duVec : numpy ndarray
        Vector of delta control commands computed by the controller.

    Returns
    -------
    mp : ModelParameters
        Structure containing optical model parameters
    dDM : ModelParameters
        Structure containing the delta DM commands for each DM
    """        
    dDM = falco.config.Object()
    
    #--Parse the command vector by DM
    if(any(mp.dm_ind==1)):
        dDM1Vvec = np.zeros(mp.dm1.NactTotal)
        dDM1Vvec[mp.dm1.act_ele]  = mp.dm1.weight*duVec[cvar.uLegend==1] # Parse the command vector to get component for DM and apply the DM's weight
        dDM.dDM1V = dDM1Vvec.reshape((mp.dm1.Nact,mp.dm1.Nact))
    if(any(mp.dm_ind==2)):  
        dDM2Vvec = np.zeros(mp.dm2.NactTotal)
        dDM2Vvec[mp.dm2.act_ele] = mp.dm2.weight*duVec[cvar.uLegend==2] # Parse the command vector to get component for DM and apply the DM's weight
        dDM.dDM2V = dDM2Vvec.reshape((mp.dm2.Nact,mp.dm2.Nact))
    if(any(mp.dm_ind==8)):  
        dDM.dDM8V = np.zeros(mp.dm8.NactTotal)
        dDM.dDM8V[mp.dm8.act_ele] = mp.dm8.weight*duVec[cvar.uLegend==8] # Parse the command vector to get component for DM and apply the DM's weight
    if(any(mp.dm_ind==9)):  
        dDM.dDM9V = np.zeros(mp.dm9.NactTotal)
        dDM.dDM9V[mp.dm9.act_ele] = mp.dm9.weight*duVec[cvar.uLegend==9] # Parse the command vector to get component for DM and apply the DM's weight
    
    #--Enforce tied actuator pair commands. 
    #   Assign command of first actuator to the second as well.
#    if(any(mp.dm_ind==8)):
#        for ti=1:size(mp.dm8.tied,1)
#            dDM.dDM8V(mp.dm8.tied(ti,2)) = dDM.dDM8V(mp.dm8.tied(ti,1));
#
#    if(any(mp.dm_ind==9)):
#        for ti=1:size(mp.dm9.tied,1)
#            dDM.dDM9V(mp.dm9.tied(ti,2)) = dDM.dDM9V(mp.dm9.tied(ti,1));

    
    #--Combine the delta command with the previous command
    if(any(mp.dm_ind==1)):  mp.dm1.V = cvar.DM1Vnom + dDM.dDM1V
    if(any(mp.dm_ind==2)):  mp.dm2.V = cvar.DM2Vnom + dDM.dDM2V
    if(any(mp.dm_ind==8)):  mp.dm8.V = cvar.DM8Vnom + dDM.dDM8V
    if(any(mp.dm_ind==9)):  mp.dm9.V = cvar.DM9Vnom + dDM.dDM9V


    return mp,dDM
    