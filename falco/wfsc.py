import numpy as np
import falco
import os
import pickle
import math
import scipy

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

    # SFF Stuff:  This stuff probably got defined in the functions above
    if not hasattr(mp.P1.compact, 'Narr'):
        mp.P1.compact.Narr = 252
    if not hasattr(mp.P4.compact, 'mask'):
        mp.P4.compact.mask = np.zeros((mp.P1.compact.Narr,mp.P1.compact.Narr))
    if not hasattr(mp.P1.compact, 'mask'):
        mp.P1.compact.mask = np.zeros((mp.P1.compact.Narr,mp.P1.compact.Narr))

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
                P1andP3 = mp.P1.compact.mask + padOrCropEven(mp.P3.compact.mask,len(mp.P1.compact.mask));
                #figure(302); imagesc(P1andP3); axis xy equal tight; colorbar; set(gca,'Fontsize',20); title('Pupil and Apod Superimposed','Fontsize',16');

    ## DM Initialization
    
    # SFF Stuff
    mp.dm3 = falco.config.EmptyObject()   
    mp.dm4 = falco.config.EmptyObject()   
    mp.dm5 = falco.config.EmptyObject()   
    mp.dm6 = falco.config.EmptyObject()   
    mp.dm7 = falco.config.EmptyObject()   
    mp.dm8 = falco.config.EmptyObject()   
    mp.dm9 = falco.config.EmptyObject()   

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
            mp.compact.Nfpm = max([mp.dm8.compact.NdmPad,mp.dm9.compact.NdmPad]); #--Width of the FPM array in the compact model.
            mp.full.Nfpm = max([mp.dm8.NdmPad,mp.dm9.NdmPad]); #--Width of the FPM array in the full model.
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

    # SFF Stuff
    if not hasattr(mp.F3.compact, 'Nxi'):
        mp.F3.compact.Nxi = 24
    if not hasattr(mp.F3.compact, 'Neta'):
        mp.F3.compact.Neta = 24
    if not hasattr(mp.F3.compact, 'mask'):
        mp.F3.compact.mask = falco.config.EmptyObject() 
    if not hasattr(mp.F3.compact.mask, 'amp'):
        mp.F3.compact.mask.amp = np.zeros((24,24))

    if not hasattr(mp.F3.full, 'Nxi'):
        mp.F3.full.Nxi = 24
    if not hasattr(mp.F3.full, 'Neta'):
        mp.F3.full.Neta = 24
    if not hasattr(mp.F3.full, 'mask'):
        mp.F3.full.mask = falco.config.EmptyObject() 
    if not hasattr(mp.F3.full.mask, 'amp'):
        mp.F3.full.mask.amp = np.zeros((24,24))

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
            #mp.F3.compact.etas = np.arange(-(mp.F3.compact.Neta-1)/2,(mp.F3.compact.Neta-1)/2).'*mp.F3.compact.deta;
        else:
            mp.F3.compact.xis  = np.arange(-mp.F3.compact.Nxi/2, (mp.F3.compact.Nxi/2-1))*mp.F3.compact.dxi;
            #mp.F3.compact.etas = (-mp.F3.compact.Neta/2:(mp.F3.compact.Neta/2-1)).'*mp.F3.compact.deta;

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
    
    #--Set Inputs:  SFF NOTE:  The use of dictionary of maskCorr was done in ModelParameters.py so I went along with it
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
    mp.Fend.corr.mask, mp.Fend.xisDL, mp.Fend.etasDL = falco.masks.falco_gen_SW_mask(**maskCorr);
    mp.Fend.corr.settings = maskCorr; #--Store values for future reference
    #--Size of the output image 
    #--Need the sizes to be the same for the correction and scoring masks
    mp.Fend.Nxi  = mp.Fend.corr.mask.shape[1] #size(mp.Fend.corr.mask,2);
    mp.Fend.Neta = mp.Fend.corr.mask.shape[0] #size(mp.Fend.corr.mask,1);

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
        mp.F5.mask, mp.F5.xisDL, mp.F5.etasDL = falco.masks.falco_gen_SW_mask(**maskF5);
    
        #--Size of the output image in F5
        mp.F5.Nxi = mp.F5.mask.shape[1] #size(mp.F5.mask, 2);
        mp.F5.Neta = mp.F5.mask.shape[0]
    
        ## Set up the fiber mode in F5
    
        V = 2*pi/mp.lambda0*mp.fiber.a*mp.fiber.NA;
        W = 1.1428*V - 0.996;
        U = math.sqrt(V**2 - W**2);
    
        maskFiberCore["pixresFP"] = mp.F5.res;
        maskFiberCore["rhoInner"] = 0;
        maskFiberCore["rhoOuter"] = mp.fiber.a;
        maskFiberCore["angDeg"] = 180;
        maskFiberCore["FOV"] = mp.F5.FOV;
        maskFiberCore["whichSide"] = mp.Fend.sides;
        mp.F5.fiberCore.mask, unused_1, unused_2 = falco.masks.falco_gen_SW_mask(**maskFiberCore);
    
        maskFiberCladding["pixresFP"] = mp.F5.res;
        maskFiberCladding["rhoInner"] = mp.fiber.a;
        maskFiberCladding["rhoOuter"] = 10;
        maskFiberCladding["angDeg"] = 180;
        maskFiberCladding["FOV"] = mp.F5.FOV;
        maskFiberCladding["whichSide"] = mp.Fend.sides;
        mp.F5.fiberCladding.mask, unused_1, unused_2 = falco.masks.falco_gen_SW_mask(**maskFiberCladding);
    
        F5XIS, F5ETAS = np.meshgrid(mp.F5.xisDL, mp.F5.etasDL);
    
        mp.F5.RHOS = math.sqrt((F5XIS - mp.F5.fiberPos[0])**2 + (F5ETAS - mp.F5.fiberPos[1])**2);
        mp.F5.fiberCore.mode = mp.F5.fiberCore.mask*scipy.special.j0(U*mp.F5.RHOS/mp.fiber.a)/scipy.special.j0(0,U);
        mp.F5.fiberCladding.mode = mp.F5.fiberCladding.mask*scipy.special.k0(W*mp.F5.RHOS/mp.fiber.a)/scipy.special.k0(W);
        mp.F5.fiberCladding.mode[np.isnan(mp.F5.fiberCladding.mode)] = 0;
        mp.F5.fiberMode = mp.F5.fiberCore.mode + mp.F5.fiberCladding.mode;
        fiberModeNorm = np.sqrt(np.sum(np.sum(np.abs(mp.F5.fiberMode)**2)));
        mp.F5.fiberMode = mp.F5.fiberMode/fiberModeNorm;

#    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#    
#    #--Evaluation Model for Computing Throughput (same as Compact Model but
#    # with different Fend.resolution)
#    mp.Fend.eval.dummy = 1; #--Initialize the structure if it doesn't exist.
#    if not hasattr(mp.Fend.eval,'res'):  
#        mp.Fend.eval.res = 10
#    maskCorr.pixresFP = mp.Fend.eval.res; #--Assign the resolution
#    mp.Fend.eval.mask, mp.Fend.eval.xisDL, mp.Fend.eval.etasDL = falco.masks.falco_gen_SW_mask(**maskCorr);  #--Generate the mask
#    mp.Fend.eval.Nxi  = mp.Fend.eval.mask.shape[1]
#    mp.Fend.eval.Neta = mp.Fend.eval.mask.shape[0]
#    mp.Fend.eval.dxi = (mp.fl*mp.lambda0/mp.P4.D)/mp.Fend.eval.res; # higher sampling at Fend.for evaulation [meters]
#    mp.Fend.eval.deta = mp.Fend.eval.dxi; # higher sampling at Fend.for evaulation [meters]   
#    
#    # (x,y) location [lambda_c/D] in dark hole at which to evaluate throughput
#    XIS,ETAS = np.meshgrid(mp.Fend.eval.xisDL - mp.thput_eval_x, mp.Fend.eval.etasDL - mp.thput_eval_y);
#    mp.Fend.eval.RHOS = np.sqrt(XIS**2 + ETAS**2);
#    
#    #--Storage array for throughput at each iteration
#    mp.thput_vec = np.zeros(mp.Nitr+1,1);


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
