"""Functions to setup FALCO by filling in all the necessary variables."""
import numpy as np
import os
import types
# import pickle
import psutil  # For checking number of cores available
from astropy.io import fits
import matplotlib.pyplot as plt
import copy

import falco
from falco.util import ceil_even, pad_crop


def flesh_out_workspace(mp):
    """
    Prepare for WFSC by generating masks and storage arrays.

    Parameters
    ----------
    mp : falco.config.ModelParameter
        Object containing all model parameters.

    Returns
    -------
    out : falco.config.Object()
        Object containing arrays to be filled in during WFSC.

    """
    set_optional_variables(mp)  # Optional/hidden variables
    verify_key_values(mp)

    falco_set_spectral_properties(mp)
    falco_set_jacobian_modal_weights(mp)  # Zernike Modes and Subband Weighting

    # Pupil Masks
    compute_entrance_pupil_coordinates(mp)
    compute_apodizer_shape(mp)
    crop_lyot_stop(mp)
    compute_lyot_stop_coordinates(mp)
    plot_superimposed_pupil_masks(mp)  # For visual inspection

    # Focal plane mask
    falco_gen_FPM(mp)
    falco_compute_fpm_coordinates(mp)

    # Final focal plane
    compute_Fend_resolution(mp)
    # Software Mask for Correction (corr) and Scoring (score):
    falco_configure_dark_hole_region(mp)
    falco_set_spatial_weights(mp)  # Spatial weighting for control Jacobian.

    # DM1 and DM2
    falco_configure_dm1_and_dm2(mp)  # Flesh out the dm1 and dm2 structures
    falco_gen_DM_stops(mp)
    falco_set_dm_surface_padding(mp)  # Angular Spectrum Propagation with FFTs

    falco_set_initial_Efields(mp)

    falco.imaging.calc_psf_norm_factor(mp)
    # falco_gen_contrast_over_NI_map(mp)  # Contrast-to-NI Map Calculation

    # Initialize Arrays to Store Performance History
    out = init_storage_arrays(mp)

    print('\nBeginning Trial %d of Series %d.\n' % (mp.TrialNum, mp.SeriesNum))
    print('DM 1-to-2 Fresnel number (using radius) = ' +
          str((mp.P2.D/2)**2/(mp.d_dm1_dm2*mp.lambda0)))

    return out


#######################################################################


def verify_key_values(mp):
    """Verify that important text options are valid."""
    mp.allowedCenterings = frozenset(('pixel', 'interpixel'))
    mp.allowedCoronagraphTypes = frozenset(('VC', 'VORTEX', 'LC', 'APLC',
                                            'FLC', 'SPLC', 'HLC'))
    mp.allowedLayouts = frozenset(('fourier', 'fpm_scale', 'proper',
                                   'roman_phasec_proper',
                                   'wfirst_phaseb_proper'))

    # Check centering
    mp.centering = mp.centering.lower()
    if mp.centering not in mp.allowedCenterings:
        raise ValueError('%s is not an allowed value of mp.centering.',
                         mp.centering)

    # Check coronagraph type
    mp.coro = mp.coro.upper()
    if mp.coro not in mp.allowedCoronagraphTypes:
        raise ValueError('%s is not an allowed value of mp.coro.', mp.coro)

    # Check optical layout
    mp.layout = mp.layout.lower()
    if mp.layout not in mp.allowedLayouts:
        raise ValueError('%s is not an allowed value of mp.layout.', mp.layout)


def set_optional_variables(mp):
    """
    Set values for optional values.

    Parameters
    ----------
    mp : falco.config.ModelParameter
        Object containing all model parameters.

    Returns
    -------
    None

    """
    # Intializations of structures (if they don't exist yet)
    if not hasattr(mp, "compact"):
        mp.compact = falco.config.Object()
    if not hasattr(mp, "full"):
        mp.full = falco.config.Object()
    if not hasattr(mp, "jac"):
        mp.jac = falco.config.Object()
    if not hasattr(mp, "est"):
        mp.est = falco.config.Object()
    if not hasattr(mp, "detector"):
        mp.detector = falco.config.Object()
    if not hasattr(mp, "star"):
        mp.star = falco.config.Object()
    if not hasattr(mp.compact, "star"):
        mp.compact.star = falco.config.Object()
    if not hasattr(mp.jac, "star"):
        mp.jac.star = falco.config.Object()
    if not hasattr(mp, "path"):
        mp.path = falco.config.Object()

    # File Paths for Data Storage (excluded from git)
    localpath = os.path.dirname(os.path.abspath(__file__))
    head, tail = os.path.split(localpath)
    mp.path.falco = head
    # Store minimal data to re-construct the data from the run:
    # the "out" structure after a trial goes here
    if not hasattr(mp.path, 'brief'):
        mp.path.brief = os.path.join(mp.path.falco, 'data', 'brief')
    if not hasattr(mp.path, 'config'):
        mp.path.config = os.path.join(mp.path.falco, 'data', 'config')
    # Store final workspace data here
    if not hasattr(mp.path, 'ws'):
        mp.path.ws = os.path.join(mp.path.falco, 'data', 'ws')

    # multiprocessing
    if not hasattr(mp, "flagMultiproc"):
        mp.flagMultiproc = False
    if not hasattr(mp, "Nthreads"):
        mp.Nthreads = psutil.cpu_count(logical=False)

    # How many stars to use and their positions
    # mp.star is for the full model,
    # and mp.compact.star is for the compact andJacobian models.
    if not hasattr(mp.star, 'count'):
        mp.star.count = 1
    if not hasattr(mp.star, 'xiOffsetVec'):
        mp.star.xiOffsetVec = 0
    if not hasattr(mp.star, 'etaOffsetVec'):
        mp.star.etaOffsetVec = 0
    if not hasattr(mp.star, 'weights'):
        mp.star.weights = 1
    if not hasattr(mp.compact.star, 'count'):
        mp.compact.star.count = 1
    if not hasattr(mp.compact.star, 'xiOffsetVec'):
        mp.compact.star.xiOffsetVec = 0
    if not hasattr(mp.compact.star, 'etaOffsetVec'):
        mp.compact.star.etaOffsetVec = 0
    if not hasattr(mp.compact.star, 'weights'):
        mp.compact.star.weights = 1
    # Spatial weighting in the Jacobian by star:
    if not hasattr(mp.jac.star, 'weights'):
        mp.jac.star.weights = np.ones(mp.compact.star.count)

    # Saving data
    if not hasattr(mp, 'flagSaveWS'):
        mp.flagSaveWS = False
        # Whehter to save out the entire workspace at the end of the trial.
    if not hasattr(mp, 'flagSaveEachItr'):
        mp.flagSaveEachItr = False  # Whether to save out the performance at each iteration. Useful for long trials in case it crashes or is stopped early.
    if not hasattr(mp, 'flagSVD'):
        mp.flagSVD = False    # Whether to compute and save the singular mode spectrum of the control Jacobian (each iteration)
    # Jacobian or controller related
    if not hasattr(mp, 'flagTrainModel'):
        mp.flagTrainModel = False  # Whether to call the Expectation-Maximization (E-M) algorithm to improve the linearized model. 
    if not hasattr(mp, 'flagUseLearnedJac'):
        mp.flagUseLearnedJac = False  # Whether to load and use an improved Jacobian from the Expectation-Maximization (E-M) algorithm 
    if not hasattr(mp.est, 'flagUseJac'):
        mp.est.flagUseJac = False   # Whether to use the Jacobian or not for estimation. (If not using Jacobian, model is called and differenced.)
    if not hasattr(mp.est, 'ItrStartKF'):
        mp.est.ItrStartKF = 2  # Which iteration to start the Kalman filter at
    if not hasattr(mp.ctrl, 'flagUseModel'):
        mp.ctrl.flagUseModel = False  # Whether to perform a model-based (vs empirical) grid search for the controller

    # Model options (Very specialized cases--not for the average user)
    if not hasattr(mp, 'flagFiber'):
        mp.flagFiber = False  # Whether to couple the final image through lenslets and a single mode fiber.
    if not hasattr(mp, 'flagLenslet'):
        mp.flagLenslet = False    # Whether to propagate through a lenslet array placed in Fend before coupling light into fibers
    if not hasattr(mp, 'flagDMwfe'):
        mp.flagDMwfe = False  # Temporary for BMC quilting study

    # Detector properties for adding noise to images
    # Default values are for the Andor Neo sCMOS detector and testbed flux
    if not hasattr(mp, 'flagImageNoise'):
        mp.flagImageNoise = False  # whether to include noise in the images
    if not hasattr(mp.detector, 'gain'):
        mp.detector.gain = 1.0  # [e-/count]
    if not hasattr(mp.detector, 'darkCurrentRate'):
        mp.detector.darkCurrentRate = 0.015  # [e-/pixel/second]
    if not hasattr(mp.detector, 'readNoiseStd'):
        mp.detector.readNoiseStd = 1.7  # [e-/count]
    if not hasattr(mp.detector, 'wellDepth'):
        mp.detector.wellDepth = 3e4  # [e-]
    if not hasattr(mp.detector, 'peakFluxVec'):
        mp.detector.peakFluxVec = 1e8 * np.ones(mp.Nsbp)  # [counts/pixel/second]
    if not hasattr(mp.detector, 'tExpVec'):
        mp.detector.tExpVec = 1.0 * np.ones(mp.Nsbp)  # [seconds]
    if not hasattr(mp.detector, 'Nexp'):
        mp.detector.Nexp = 1  # number of exposures to stack

    # Optical model/layout:
    # Whether to use a full model written in PROPER.
    if not hasattr(mp.full, 'flagPROPER'):
        mp.full.flagPROPER = False
    # Whether to have the E-field rotate 180 degrees from one pupil to the next
    # Does not apply to PROPER full models.
    if not hasattr(mp, 'flagRotation'):
        mp.flagRotation = True

    # Optional/Hidden variables
    if not hasattr(mp.full, 'pol_conds'):
        mp.full.pol_conds = np.array([0])  # Vector of which polarization state(s) to use when creating images from the full model. Currently only used with PROPER full models from John Krist.
    if not hasattr(mp, 'propMethodPTP'):
        mp.propMethodPTP = 'fft'  # Propagation method for postage stamps around the influence functions. 'mft' or 'fft'

    # Sensitivities to Zernike-Mode Perturbations
    if not hasattr(mp.full, 'ZrmsVal'):
        mp.full.ZrmsVal = 1e-9  # Amount of RMS Zernike mode used to calculate aberration sensitivities [meters]. WFIRST CGI uses 1e-9, and LUVOIR and HabEx use 1e-10. 
    if not hasattr(mp.full, 'Rsens'):
        mp.full.Rsens = np.array([])
    if not hasattr(mp.full, 'indsZnoll'):
        mp.full.indsZnoll = np.array([2, 3])

    # DM Initialization
    if not hasattr(mp, 'dm1'):
        mp.dm1 = falco.config.Object()
    if not hasattr(mp, 'dm2'):
        mp.dm2 = falco.config.Object()
    if not hasattr(mp, 'dm3'):
        mp.dm3 = falco.config.Object()
    if not hasattr(mp, 'dm4'):
        mp.dm4 = falco.config.Object()
    if not hasattr(mp, 'dm5'):
        mp.dm5 = falco.config.Object()
    if not hasattr(mp, 'dm6'):
        mp.dm6 = falco.config.Object()
    if not hasattr(mp, 'dm7'):
        mp.dm7 = falco.config.Object()
    if not hasattr(mp, 'dm8'):
        mp.dm8 = falco.config.Object()
    if not hasattr(mp, 'dm9'):
        mp.dm9 = falco.config.Object()

    # Initialize the number of actuators (NactTotal) and actuators used (Nele).
    mp.dm1.NactTotal=0; mp.dm2.NactTotal=0; mp.dm3.NactTotal=0; mp.dm4.NactTotal=0; mp.dm5.NactTotal=0; mp.dm6.NactTotal=0; mp.dm7.NactTotal=0; mp.dm8.NactTotal=0; mp.dm9.NactTotal=0  # Initialize for bookkeeping later.
    mp.dm1.Nele=0; mp.dm2.Nele=0; mp.dm3.Nele=0; mp.dm4.Nele=0; mp.dm5.Nele=0; mp.dm6.Nele=0; mp.dm7.Nele=0; mp.dm8.Nele=0; mp.dm9.Nele=0  # Initialize for Jacobian calculations later. 

    # Deformable mirror settings
    # DM1
    if not hasattr(mp.dm1,'orientation'):
        mp.dm1.orientation = 'rot0'  # Change to mp.dm1.V orientation before generating DM surface. Options: rot0, rot90, rot180, rot270, flipxrot0, flipxrot90, flipxrot180, flipxrot270
    if not hasattr(mp.dm1, 'Vmin'):
        mp.dm1.Vmin = -1000.  # Min allowed voltage command
    if not hasattr(mp.dm1, 'Vmax'):
        mp.dm1.Vmax = 1000.  # Max allowed voltage command
    if not hasattr(mp.dm1, 'pinned'):
        mp.dm1.pinned = np.array([])  # Indices of pinned actuators
    if not hasattr(mp.dm1, 'Vpinned'):
        mp.dm1.Vpinned = np.array([])  # (Fixed) voltage commands of pinned actuators
    if not hasattr(mp.dm1, 'tied'):
        mp.dm1.tied = np.zeros((0, 2))  # Indices of paired actuators. Two indices per row
    if not hasattr(mp.dm1, 'flagNbrRule'):
        mp.dm1.flagNbrRule = False  # Whether to set constraints on neighboring actuator voltage differences. If set to true, need to define mp.dm1.dVnbr
    # DM2
    if not hasattr(mp.dm2,'orientation'):
        mp.dm2.orientation = 'rot0'  # Change to mp.dm1.V orientation before generating DM surface. Options: rot0, rot90, rot180, rot270, flipxrot0, flipxrot90, flipxrot180, flipxrot270
    if not hasattr(mp.dm2, 'Vmin'):
        mp.dm2.Vmin = -1000.  # Min allowed voltage command
    if not hasattr(mp.dm2, 'Vmax'):
        mp.dm2.Vmax = 1000.  # Max allowed voltage command
    if not hasattr(mp.dm2, 'pinned'):
        mp.dm2.pinned = np.array([])  # Indices of pinned actuators
    if not hasattr(mp.dm2, 'Vpinned'):
        mp.dm2.Vpinned = np.array([])  # (Fixed) voltage commands of pinned actuators
    if not hasattr(mp.dm2, 'tied'):
        mp.dm2.tied = np.zeros((0, 2))  # Indices of paired actuators. Two indices per row
    if not hasattr(mp.dm2, 'flagNbrRule'):
        mp.dm2.flagNbrRule = False  # Whether to set constraints on neighboring actuator voltage differences. If set to true, need to define mp.dm2.dVnbr

    # Loading previous DM commands as the starting point
    # Stash DM8 and DM9 starting commands if they are given in the main script
    if hasattr(mp, 'dm8'):
        if hasattr(mp.dm8, 'V'):
            mp.DM8V0 = mp.dm8.V
        if hasattr(mp.dm9, 'V'):
            mp.DM9V0 = mp.dm9.V

    # Intialize delta DM voltages. Needed for Kalman filters.
    # Save the delta from the previous command
    if np.any(mp.dm_ind == 1):
        mp.dm1.dV = 0
    if np.any(mp.dm_ind == 2):
        mp.dm2.dV = 0
    if np.any(mp.dm_ind == 3):
        mp.dm3.dV = 0
    if np.any(mp.dm_ind == 4):
        mp.dm4.dV = 0
    if np.any(mp.dm_ind == 5):
        mp.dm5.dV = 0
    if np.any(mp.dm_ind == 6):
        mp.dm6.dV = 0
    if np.any(mp.dm_ind == 7):
        mp.dm7.dV = 0
    if np.any(mp.dm_ind == 8):
        mp.dm8.dV = 0
    if np.any(mp.dm_ind == 9):
        mp.dm9.dV = 0

#    # First delta DM settings are zero (for covariance calculation in Kalman filters)
#    mp.dm1.dV = np.zeros((mp.dm1.Nact, mp.dm1.Nact))  # delta voltage on DM1;
#    mp.dm2.dV = np.zeros((mp.dm2.Nact, mp.dm2.Nact))  # delta voltage on DM2;
#    mp.dm8.dV = np.zeros((mp.dm8.NactTotal,1))  # delta voltage on DM8;
#    mp.dm9.dV = np.zeros((mp.dm9.NactTotal,1))  # delta voltage on DM9;

    # Control
    if not hasattr(mp, 'WspatialDef'):
        mp.WspatialDef = np.array([])  # spatial weight matrix for the Jacobian

    # Performance Evaluation
    # Conversion factor: milliarcseconds (mas) to lambda0/D
    mp.mas2lam0D = 1/(mp.lambda0/mp.P1.D*180/np.pi*3600*1000)
    if not hasattr(mp.Fend, 'eval'):  # Initialize the structure if it doesn't exist.
        mp.Fend.eval = falco.config.Object()
    if not hasattr(mp.Fend.eval, 'res'):
        mp.Fend.eval.res = 10

    # Pupil ID, needed for computing RMS DM commands
    if not hasattr(mp.P1, 'IDnorm'):
        mp.P1.IDnorm = 0.0

    pass


def falco_set_spectral_properties(mp):
    """Set bandwidth and wavelength specifications."""
    # Center-ish wavelength indices (ref = reference)(Only the center if
    #  an odd number of wavelengths is used.)
    mp.si_ref = np.floor(mp.Nsbp/2).astype(int)

    # Wavelengths used for Compact Model (and Jacobian Model)
    mp.sbp_weights = np.ones((mp.Nsbp, 1))
    if mp.Nwpsbp == 1 and mp.flagSim:  # Set ctrl wvls evenly between endpoints (inclusive) of total bandpass.
        if mp.Nsbp == 1:
            mp.sbp_centers = np.array([mp.lambda0])
        else:
            mp.sbp_centers = mp.lambda0*np.linspace(1-mp.fracBW/2, 1+mp.fracBW/2, mp.Nsbp)
            mp.sbp_weights[0] = 1/2
            mp.sbp_weights[-1] = 1/2
    else:  # For cases with multiple sub-bands: Choose wavelengths to be at subbandpass centers since the wavelength samples will span to the full extent of the sub-bands.
        # Bandwidth between centers of endpoint subbandpasses.
        mp.fracBWcent2cent = mp.fracBW*(1 - 1/mp.Nsbp)
        # Space evenly at the centers of the subbandpasses.
        mp.sbp_centers = mp.lambda0*np.linspace(1-mp.fracBWcent2cent/2,
                                                1+mp.fracBWcent2cent/2,
                                                mp.Nsbp)
    # Normalize the sum of the weights
    mp.sbp_weights = mp.sbp_weights / np.sum(mp.sbp_weights)

    print(' Using %d discrete wavelength(s) in each of %d sub-bandpasses over '
          'a %.1f%% total bandpass \n' % (mp.Nwpsbp, mp.Nsbp, 100*mp.fracBW))
    print('Sub-bandpasses are centered at wavelengths [nm]:\t ', end='')
    print(1e9*mp.sbp_centers)

    # Bandwidth and Wavelength Specs: Full Model

    # Center(-ish) wavelength indices (ref = reference). (Only the center if an odd number of wavelengths is used.)
    mp.wi_ref = np.floor(mp.Nwpsbp/2).astype(int)

    # Wavelength factors/weights within sub-bandpasses in the full model
    mp.full.lambda_weights = np.ones((mp.Nwpsbp,1))  # Initialize as all ones. Weights within a single sub-bandpass
    if mp.Nwpsbp == 1:
        mp.full.dlam = 0  # Delta lambda between every wavelength in the sub-band in the full model
    else:
        # Spectral weighting in image
        mp.full.lambda_weights[0] = 1/2  # Include end wavelengths with half weights
        mp.full.lambda_weights[-1] = 1/2  # Include end wavelengths with half weights
        mp.fracBWsbp = mp.fracBW/mp.Nsbp  # Bandwidth per sub-bandpass
        # Indexing of wavelengths in each sub-bandpass
        sbp_facs = np.linspace(1-mp.fracBWsbp/2, 1+mp.fracBWsbp/2, mp.Nwpsbp)  # Factor applied to lambda0 only
        mp.full.dlam = (sbp_facs[1] - sbp_facs[0])*mp.lambda0  # Delta lambda between every wavelength in the full model 

    mp.full.lambda_weights = mp.full.lambda_weights/np.sum(mp.full.lambda_weights)  # Normalize sum of the weights (within the sub-bandpass)

    # Make vector of all wavelengths and weights used in the full model
    lambdas = np.zeros((mp.Nsbp*mp.Nwpsbp,))
    lambda_weights_all = np.zeros((mp.Nsbp*mp.Nwpsbp,))
    mp.full.lambdasMat = np.zeros((mp.Nsbp, mp.Nwpsbp))
    mp.full.indsLambdaMat = np.zeros((mp.Nsbp*mp.Nwpsbp, 2), dtype=int)
    counter = 0
    for si in range(mp.Nsbp):
        if(mp.Nwpsbp == 1):
            mp.full.lambdasMat[si, 0] = mp.sbp_centers[si]
        else:
            mp.full.lambdasMat[si, :] = np.arange(-(mp.Nwpsbp-1)/2,
                                                 (mp.Nwpsbp+1)/2)*mp.full.dlam + mp.sbp_centers[si]
        np.arange(-(mp.Nwpsbp-1)/2, (mp.Nwpsbp-1)/2)*mp.full.dlam
        for wi in range(mp.Nwpsbp):
            lambdas[counter] = mp.full.lambdasMat[si, wi];
            lambda_weights_all[counter] = mp.sbp_weights[si]*mp.full.lambda_weights[wi]
            mp.full.indsLambdaMat[counter, :] = [si, wi]
            counter = counter+1;

    # Get rid of redundant wavelengths in the complete list, and sum weights for repeated wavelengths
    unused_1, inds_unique = np.unique(np.round(1e12*lambdas), return_index=True)  # Check equality at picometer level
    mp.full.indsLambdaUnique = inds_unique
    duplicate_inds = np.setdiff1d(np.arange(len(lambdas), dtype=int), inds_unique)
    # duplicate_values = lambda_weights_all[duplicate_inds] # duplicate weight values

    # Shorten the vectors to contain only unique values. Combine weights for repeated wavelengths.
    mp.full.lambdas = lambdas[inds_unique]
    mp.full.lambda_weights_all = lambda_weights_all[inds_unique]
    for idup in range(len(duplicate_inds)):
        wvl = lambdas[duplicate_inds[idup]]
        weight = lambda_weights_all[duplicate_inds[idup]]
        ind = np.where(np.abs(mp.full.lambdas-wvl) <= 1e-11)
        mp.full.lambda_weights_all[ind] = mp.full.lambda_weights_all[ind] + weight
    mp.full.NlamUnique = len(inds_unique)
    pass


def falco_set_jacobian_modal_weights(mp):
    """
    Set the relative weights in Jacobian based on wavelength and Zernike mode.

    Function to set the relative weights for the Jacobian modes. The weights
    are formulated first in a 2-D array with rows for wavelengths and columns
    for Zernike modes. The weights are then normalized in each column. The
    weight matrix is then vectorized, with all zero weights removed.

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

    # Initialize mp.jac if it doesn't exist
    if not hasattr(mp, 'jac'):
        mp.jac = falco.config.EmptyClass()

    # Which Zernike modes to include in Jacobian. A vector of Noll indices.
    # 1 is the on-axis piston mode.
    if not hasattr(mp.jac, 'zerns'):
        mp.jac.zerns = np.array([1])
        mp.jac.Zcoef = np.array([1])
    else:
        mp.jac.zerns = np.atleast_1d(mp.jac.zerns)
        mp.jac.Zcoef = np.atleast_1d(mp.jac.Zcoef)

    mp.jac.Nzern = np.size(mp.jac.zerns)
    # Reset coefficient for piston term to 1
    mp.jac.Zcoef[mp.jac.zerns == 1] = 1

    # Initialize weighting matrix of each Zernike-wavelength mode for the controller
    mp.jac.weightMat = np.zeros((mp.Nsbp, mp.jac.Nzern))
    for izern in range(0, mp.jac.Nzern):
        whichZern = mp.jac.zerns[izern]
        if whichZern == 1:  # Include all wavelengths for piston Zernike mode
            mp.jac.weightMat[:, 0] = np.ones(mp.Nsbp)
        else:  # Include just middle and end wavelengths for Zernike mode 2 and up
            mp.jac.weightMat[0, izern] = 1
            mp.jac.weightMat[mp.si_ref, izern] = 1
            mp.jac.weightMat[mp.Nsbp-1, izern] = 1

    # Half-weighting if endpoint wavelengths are used.
    # For design or modeling without estimation: Choose ctrl wvls evenly
    # between endpoints of the total bandpass
    if mp.estimator.lower() == 'perfect':
        mp.jac.weightMat[0, :] = 0.5*mp.jac.weightMat[0, :]
        mp.jac.weightMat[mp.Nsbp-1, :] = 0.5*mp.jac.weightMat[mp.Nsbp-1, :]

    # Normalize the summed weights of each column separately
    for izern in range(mp.jac.Nzern):
        colSum = np.double(sum(mp.jac.weightMat[:, izern]))
        mp.jac.weightMat[:, izern] = mp.jac.weightMat[:, izern]/colSum

    # Zero out columns for which the RMS Zernike value is zero
    for izern in range(mp.jac.Nzern):
        if mp.jac.Zcoef[izern] == 0:
            mp.jac.weightMat[:, izern] = 0*mp.jac.weightMat[:, izern]

    # Indices of the non-zero control Jacobian modes in the weighting matrix
    mp.jac.weightMat_ele = np.nonzero(mp.jac.weightMat > 0)
    # Vector of control Jacobian mode weights
    mp.jac.weights = mp.jac.weightMat[mp.jac.weightMat_ele]
    # Number of (Zernike-wavelength pair) modes in the control Jacobian
    mp.jac.Nmode = np.size(mp.jac.weights)

    # Get the wavelength indices for the nonzero values in the weight matrix.
    tempMat = np.tile(np.arange(mp.Nsbp).reshape((mp.Nsbp, 1)), (1, mp.jac.Nzern))
    mp.jac.sbp_inds = tempMat[mp.jac.weightMat_ele]

    # Get the Zernike indices for the nonzero elements in the weight matrix.
    tempMat = np.tile(mp.jac.zerns, (mp.Nsbp, 1))
    mp.jac.zern_inds = tempMat[mp.jac.weightMat_ele]

    pass


def compute_entrance_pupil_coordinates(mp):
    """
    Compute the resolution and coordinates at the entrance pupil (plane P1).

    Values also are true at P2 in FALCO models.

    Parameters
    ----------
    mp : falco.config.ModelParameter
        Object containing all model parameters.

    Returns
    -------
    mp : falco.config.ModelParameter
        Object containing all model parameters.
    """
    # Resolution at input pupil, pupil P2, DM1, and DM2
    if not hasattr(mp.P2, 'full'):
        mp.P2.full = falco.config.Object()
    mp.P2.full.dx = mp.P2.D / mp.P1.full.Nbeam

    if not hasattr(mp.P2, 'compact'):
        mp.P2.compact = falco.config.Object()
    mp.P2.compact.dx = mp.P2.D / mp.P1.compact.Nbeam

    # Same at apodizer plane (P3)
    mp.P3.full.dx = mp.P2.full.dx
    mp.P3.compact.dx = mp.P2.compact.dx

    # Compact model: Make sure mask is square
    mp.P1.compact.mask = falco.util.pad_to_even_square(mp.P1.compact.mask)

    # Compact model coordinates normalized to pupil diameter
    # Used to make the tip/tilted input wavefront within the compact model.
    mp.P1.compact.Narr = mp.P1.compact.mask.shape[0]
    if mp.centering == 'pixel':
        mp.P2.compact.xsDL = (np.linspace(-mp.P1.compact.Narr/2,
                                          mp.P1.compact.Narr/2 - 1,
                                          mp.P1.compact.Narr) *
                              mp.P2.compact.dx / mp.P2.D)
    elif mp.centering == 'interpixel':
        mp.P2.compact.xsDL = (np.linspace(-(mp.P1.compact.Narr-1)/2,
                                          (mp.P1.compact.Narr-1)/2,
                                          mp.P1.compact.Narr) *
                              mp.P2.compact.dx / mp.P2.D)

    [mp.P2.compact.XsDL, mp.P2.compact.YsDL] = np.meshgrid(mp.P2.compact.xsDL,
                                                           mp.P2.compact.xsDL)

    # Full model: Number of points across array
    if mp.full.flagPROPER:
        if mp.centering == 'pixel':
            mp.P1.full.Narr = ceil_even(mp.P1.full.Nbeam + 1)
        elif mp.centering == 'interpixel':
            mp.P1.full.Narr = ceil_even(mp.P1.full.Nbeam)
    else:
        mp.P1.full.mask = falco.util.pad_to_even_square(mp.P1.full.mask)
        mp.P1.full.Narr = mp.P1.full.mask.shape[0]

    # Full model coordinates
    if mp.centering == 'pixel':
        mp.P2.full.xsDL = np.linspace(-mp.P1.full.Narr/2,
                                      mp.P1.full.Narr/2 - 1,
                                      mp.P1.full.Narr)*mp.P2.full.dx/mp.P2.D
    elif mp.centering.lower() == ('interpixel'):
        mp.P2.full.xsDL = np.linspace(-(mp.P1.full.Narr-1)/2,
                                      (mp.P1.full.Narr-1)/2,
                                      mp.P1.full.Narr)*mp.P2.full.dx/mp.P2.D

    [mp.P2.full.XsDL, mp.P2.full.YsDL] = np.meshgrid(mp.P2.full.xsDL,
                                                     mp.P2.full.xsDL)


def compute_apodizer_shape(mp):
    """
    Make the apodizer array square and store its shape.

    Parameters
    ----------
    mp : falco.config.ModelParameter
        Object containing all model parameters.

    Returns
    -------
    mp : falco.config.ModelParameter
        Object containing all model parameters.
    """
    if hasattr(mp.P3.compact, 'mask'):
        mp.P3.compact.mask = falco.util.pad_to_even_square(mp.P3.compact.mask)
        mp.P3.compact.Narr = mp.P3.compact.mask.shape[0]

    if hasattr(mp.P3.full, 'mask'):
        mp.P3.full.mask = falco.util.pad_to_even_square(mp.P3.full.mask)
        mp.P3.full.Narr = mp.P3.full.mask.shape[0]


def crop_lyot_stop(mp):
    """
    Crop extra zero padding around the Lyot stop to speed up MFT propagation.

    Parameters
    ----------
    mp : falco.config.ModelParameter
        Object containing all model parameters.

    Returns
    -------
    mp : falco.config.ModelParameter
        Object containing all model parameters.
    """
    # Full model
    if not mp.full.flagPROPER:

        lyotSum = np.sum(mp.P4.full.mask)
        lyotDiff = 0
        counter = 2
        while np.abs(lyotDiff) <= 1e-7:
            mp.P4.full.Narr = len(mp.P4.full.mask)-counter
            # Subtract an extra 2 to negate the extra step that overshoots:
            lyotDiff = lyotSum - np.sum(pad_crop(mp.P4.full.mask,
                                               mp.P4.full.Narr-2))
            counter += 2
        mp.P4.full.croppedMask = pad_crop(mp.P4.full.mask, mp.P4.full.Narr)

    # Compact model
    lyotSum = np.sum(mp.P4.compact.mask)
    lyotDiff = 0
    counter = 2
    while np.abs(lyotDiff) <= 1e-7:
        # Number of points across the cropped-down Lyot stop
        mp.P4.compact.Narr = len(mp.P4.compact.mask) - counter
        # Subtract an extra 2 to negate the extra step that overshoots.
        lyotDiff = lyotSum - np.sum(falco.util.pad_crop(mp.P4.compact.mask,
                                                        mp.P4.compact.Narr-2))
        counter += 2

    mp.P4.compact.croppedMask = pad_crop(mp.P4.compact.mask,
                                         mp.P4.compact.Narr)


def compute_lyot_stop_coordinates(mp):
    """
    Crop extra zero padding around the Lyot stop to speed up MFT propagation.

    Parameters
    ----------
    mp : falco.config.ModelParameter
        Object containing all model parameters.

    Returns
    -------
    mp : falco.config.ModelParameter
        Object containing all model parameters.
    """
    # Full model
    if not mp.full.flagPROPER:
        mp.P4.full.dx = mp.P4.D / mp.P4.full.Nbeam  # [meters per pixel]

    # Compact model
    mp.P4.compact.dx = mp.P4.D/mp.P4.compact.Nbeam  # [meters per pixel]
    if mp.centering == 'pixel':
        mp.P4.compact.xs = np.linspace(-mp.P4.compact.Narr/2,
                                       (mp.P4.compact.Narr/2-1),
                                       mp.P4.compact.Narr) * mp.P4.compact.dx
    elif mp.centering == 'interpixel':
        mp.P4.compact.xs = np.linspace(-(mp.P4.compact.Narr-1)/2,
                                       (mp.P4.compact.Narr-1)/2,
                                       mp.P4.compact.Narr) * mp.P4.compact.dx

    mp.P4.compact.ys = np.transpose(mp.P4.compact.xs)


def plot_superimposed_pupil_masks(mp):
    """Plot the pupil and Lyot stop on top of each other."""
    if mp.flagPlot:

        # Apodizer check
        if mp.flagApod:
            P3mask = pad_crop(mp.P3.compact.mask, mp.P1.compact.mask.shape)
            if mp.flagRotation:
                P3mask = falco.prop.relay(P3mask,
                                                mp.Nrelay1to2 + mp.Nrelay2to3)

            plt.figure(300)
            plt.imshow(P3mask - mp.P1.compact.mask)
            plt.clim(-1, 1)
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.title('Apodizer - Entrance Pupil')
            plt.pause(0.1)

            plt.figure(302)
            plt.imshow(mp.P1.compact.mask + P3mask)
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.title('Superimposed Pupil and Apodizer')
            plt.pause(0.1)

        # Lyot stop check
        if mp.P1.compact.Nbeam == mp.P4.compact.Nbeam:

            P4mask = pad_crop(mp.P4.compact.mask, mp.P1.compact.Narr)
            if mp.flagRotation:
                P4mask = falco.prop.relay(
                    P4mask, mp.Nrelay1to2 + mp.Nrelay2to3 + mp.Nrelay3to4)

            P1andP4 = mp.P1.compact.mask + P4mask

            plt.figure(301)
            plt.imshow(P1andP4)
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.title('Superimposed Pupil and Lyot Stop')
            plt.pause(0.1)


def falco_gen_FPM(mp):
    """Generate the FPM (for the HLC only)."""
    if mp.layout.lower() == 'fourier':

        if mp.coro == 'HLC':

            # Stash DM8 and DM9 starting commands
            # if they are pre-defined
            if hasattr(mp, 'dm8'):
                if hasattr(mp.dm8, 'V'):
                    mp.DM8V0 = copy.deepcopy(mp.dm8.V)
                if hasattr(mp.dm9, 'V'):
                    mp.DM9V0 = copy.deepcopy(mp.dm9.V)

            if mp.dm9.inf0name.upper() in ('COS', 'COSINE'):
                falco.hlc.setup_fpm_cosine(mp)
            elif mp.dm9.inf0name.upper() == '3foldZern':
                # falco.hlc.setup_fpm_3foldZern(mp)
                pass
            else:
                falco.hlc.setup_fpm(mp)
            falco.hlc.gen_fpm(mp)

            # Pre-compute the complex transmission of the allowed Ni+PMGI FPMs.
            if mp.coro in ('HLC',):
                [mp.complexTransCompact, mp.complexTransFull] =\
                    falco.thinfilm.gen_complex_trans_table(mp)


def falco_compute_fpm_coordinates(mp):
    """Generate coordinates in the FPM's plane."""
    if mp.coro.upper() in ['VORTEX', 'VC']:
        pass  # Not needed

    else:
        fLamD = mp.fl * mp.lambda0 / mp.P2.D

        # COMPACT MODEL
        if not hasattr(mp.F3, 'compact'):
            mp.F3.compact = falco.config.Object()

        if not hasattr(mp.F3.compact, 'Nxi'):
            mp.F3.compact.Nxi = mp.F3.compact.mask.shape[1]
        if not hasattr(mp.F3.compact, 'Neta'):
            mp.F3.compact.Neta = mp.F3.compact.mask.shape[0]

        # Resolution in compact model
        mp.F3.compact.dxi = fLamD / mp.F3.compact.res  # [meters/pixel]
        mp.F3.compact.deta = mp.F3.compact.dxi  # [meters/pixel]

        # FPM coordinates in the compact model [meters]
        # horizontal axis, xis
        if mp.centering == 'interpixel' or (mp.F3.compact.Nxi % 2) == 1:
            mp.F3.compact.xis = (np.linspace(-(mp.F3.compact.Nxi-1)/2,
                                            (mp.F3.compact.Nxi-1)/2,
                                            mp.F3.compact.Nxi) *
                                 mp.F3.compact.dxi)
        elif mp.centering == 'pixel':
            mp.F3.compact.xis = (np.arange(-mp.F3.compact.Nxi/2,
                                          mp.F3.compact.Nxi/2) *
                                 mp.F3.compact.dxi)

        # vertical axis, etas
        if mp.centering == 'interpixel' or (mp.F3.compact.Neta % 2) == 1:
            mp.F3.compact.etas = (np.linspace(-(mp.F3.compact.Neta-1)/2,
                                              (mp.F3.compact.Neta-1)/2,
                                              mp.F3.compact.Neta) *
                                  mp.F3.compact.deta)
        elif mp.centering == 'pixel':
            mp.F3.compact.etas = (np.arange(-mp.F3.compact.Neta/2,
                                            mp.F3.compact.Neta/2) *
                                  mp.F3.compact.deta)

        # Dimensionless FPM Coordinates in compact model
        mp.F3.compact.xisDL = mp.F3.compact.xis / fLamD
        mp.F3.compact.etasDL = mp.F3.compact.etas / fLamD

        # FULL MODEL
        if mp.layout in ('roman_phasec_proper',
                         'wfirst_phaseb_proper',
                         'proper'):
            pass  # Coordinates not used by the PROPER model

        else:

            if not hasattr(mp.F3, 'full'):
                mp.F3.full = falco.config.Object()

            if not hasattr(mp.F3.full, 'Nxi'):
                mp.F3.full.Nxi = mp.F3.full.mask.shape[1]
            if not hasattr(mp.F3.full, 'Neta'):
                mp.F3.full.Neta = mp.F3.full.mask.shape[0]

            # Resolution
            mp.F3.full.dxi = fLamD/mp.F3.full.res  # [meters/pixel]
            mp.F3.full.deta = mp.F3.full.dxi  # [meters/pixel]

            # Coordinates (dimensionless [DL]) for the FPMs in the full model
            if mp.centering == 'interpixel' or (mp.F3.full.Nxi % 2) == 1:
                mp.F3.full.xisDL = (np.linspace(-(mp.F3.full.Nxi-1)/2,
                                                (mp.F3.full.Nxi-1)/2,
                                                mp.F3.full.Nxi) /
                                    mp.F3.full.res)
                mp.F3.full.etasDL = (np.linspace(-(mp.F3.full.Neta-1)/2,
                                                 (mp.F3.full.Neta-1)/2,
                                                 mp.F3.full.Neta) /
                                     mp.F3.full.res)
            elif mp.centering == 'pixel':
                mp.F3.full.xisDL = (np.arange(-mp.F3.full.Nxi/2,
                                             (mp.F3.full.Nxi/2)) /
                                    mp.F3.full.res)
                mp.F3.full.etasDL = (np.arange(-mp.F3.full.Neta/2,
                                              (mp.F3.full.Neta/2)) /
                                     mp.F3.full.res)


def compute_Fend_resolution(mp):
    """Define the resolution at the final plane."""
    # Sampling/Resolution and Scoring/Correction Masks for Final Focal Plane
    fLamD = mp.fl * mp.lambda0 / mp.P4.D

    # sampling at Fend [meters]
    mp.Fend.dxi = fLamD / mp.Fend.res
    mp.Fend.deta = mp.Fend.dxi

    if mp.flagFiber:
        mp.Fend.lenslet.D = 2*mp.Fend.res*mp.Fend.lensletWavRad*mp.Fend.dxi
        mp.Fend.x_lenslet_phys = mp.Fend.dxi*mp.Fend.res*mp.Fend.x_lenslet
        mp.Fend.y_lenslet_phys = mp.Fend.deta*mp.Fend.res*mp.Fend.y_lenslet

        mp.F5.dxi = mp.lensletFL*mp.lambda0/mp.Fend.lenslet.D/mp.F5.res
        mp.F5.deta = mp.F5.dxi
    pass

    # Compact evaluation model at higher resolution
    if not hasattr(mp.Fend, 'eval'):
        mp.Fend.eval = falco.config.Object()
    mp.Fend.eval.dxi = fLamD / mp.Fend.eval.res  # [meters/pixel]
    mp.Fend.eval.deta = mp.Fend.eval.dxi  # [meters/pixel]


def falco_configure_dark_hole_region(mp):
    """Generate the software mask indicating the dark hole pixels."""
    # Convert needed values to iterables if they are scalars
    mp.Fend.corr.Rin = np.atleast_1d(mp.Fend.corr.Rin)
    mp.Fend.corr.Rout = np.atleast_1d(mp.Fend.corr.Rout)
    mp.Fend.corr.ang = np.atleast_1d(mp.Fend.corr.ang)
    sides = np.atleast_1d(mp.Fend.sides)  # array of strings
    Nzones = mp.Fend.corr.Rin.size

    # Correction Region
    CORR = {}
    CORR["pixresFP"] = mp.Fend.res
    CORR["centering"] = mp.centering
    if hasattr(mp.Fend, 'FOV'):
        CORR["FOV"] = mp.Fend.FOV
    if hasattr(mp.Fend, 'xiFOV'):
        CORR["xiFOV"] = mp.Fend.xiFOV
    if hasattr(mp.Fend, 'etaFOV'):
        CORR["etaFOV"] = mp.Fend.etaFOV
    if hasattr(mp.Fend, 'Nxi'):
        CORR["Nxi"] = mp.Fend.Nxi
    if hasattr(mp.Fend, 'etaFOV'):
        CORR["Neta"] = mp.Fend.Neta

    if not hasattr(mp.Fend, 'shape'):
        mp.Fend.shape = []
        for ii in range(Nzones):
            mp.Fend.shape.append('circle')  # Default to circular dark hole
    shapes = np.atleast_1d(mp.Fend.shape)  # array of strings

    maskCorr = np.zeros((1, 1))  # initialize
    for iZone in range(Nzones):
        CORR["rhoInner"] = mp.Fend.corr.Rin[iZone]  # lambda0/D
        CORR["rhoOuter"] = mp.Fend.corr.Rout[iZone]  # lambda0/D
        CORR["angDeg"] = mp.Fend.corr.ang[iZone]  # degrees
        CORR["whichSide"] = sides[iZone]
        CORR["shape"] = shapes[iZone]
        if hasattr(mp.Fend, 'clockAngDeg'):
            mp.Fend.clockAngDeg = np.atleast_1d(mp.Fend.clockAngDeg)
            CORR["clockAngDeg"] = mp.Fend.clockAngDeg[iZone]
        if hasattr(mp.Fend, 'xiOffset'):
            mp.Fend.xiOffset = np.atleast_1d(mp.Fend.xiOffset)
            CORR["xiOffset"] = mp.Fend.xiOffset[iZone]
        if hasattr(mp.Fend, 'etaOffset'):
            mp.Fend.etaOffset = np.atleast_1d(mp.Fend.etaOffset)
            CORR["etaOffset"] = mp.Fend.etaOffset[iZone]

        # Combine multiple zones. Use the largest array size
        [maskTemp, _, _] = falco.mask.falco_gen_SW_mask(CORR)
        Nrow = int(np.max(np.array([maskTemp.shape[0], maskCorr.shape[0]])))
        Ncol = int(np.max(np.array([maskTemp.shape[1], maskCorr.shape[1]])))
        maskCorr = (falco.util.pad_crop(maskCorr, [Nrow, Ncol]) +
                    falco.util.pad_crop(maskTemp, [Nrow, Ncol]))

    mp.Fend.corr.maskBool = np.array(maskCorr, dtype=bool)

    CORR["Nxi"] = maskCorr.shape[1]
    CORR["Neta"] = maskCorr.shape[0]
    [_, mp.Fend.xisDL, mp.Fend.etasDL] = falco.mask.falco_gen_SW_mask(CORR)

    # Size of the output image
    mp.Fend.Nxi = mp.Fend.corr.maskBool.shape[1]
    mp.Fend.Neta = mp.Fend.corr.maskBool.shape[0]

    [XIS, ETAS] = np.meshgrid(mp.Fend.xisDL, mp.Fend.etasDL)
    mp.Fend.RHOS = np.sqrt(XIS**2 + ETAS**2)

    # %% Evaluation Model for Computing Throughput
    # (just need size and coordinates, not mask)
    if not hasattr(mp.Fend, 'eval'):
        mp.Fend.eval = falco.config.Object()
    CORR["pixresFP"] = mp.Fend.eval.res  # Assign the resolution
    CORR["Nxi"] = ceil_even(mp.Fend.eval.res / mp.Fend.res*mp.Fend.Nxi)
    CORR["Neta"] = ceil_even(mp.Fend.eval.res / mp.Fend.res*mp.Fend.Neta)
    mp.Fend.eval.Nxi = CORR["Nxi"]
    mp.Fend.eval.Neta = CORR["Neta"]
    [_, mp.Fend.eval.xisDL, mp.Fend.eval.etasDL] = \
        falco.mask.falco_gen_SW_mask(CORR)

    # (x,y) location [lambda_c/D] in dark hole at which to evaluate throughput
    [XIS, ETAS] = np.meshgrid(mp.Fend.eval.xisDL - mp.thput_eval_x,
                              mp.Fend.eval.etasDL - mp.thput_eval_y)
    mp.Fend.eval.RHOS = np.sqrt(XIS**2 + ETAS**2)

    # %% Scoring Region
    mp.Fend.score.Rin = np.atleast_1d(mp.Fend.score.Rin)
    mp.Fend.score.Rout = np.atleast_1d(mp.Fend.score.Rout)
    mp.Fend.score.ang = np.atleast_1d(mp.Fend.score.ang)
    Nzones = mp.Fend.score.Rin.size

    # These are same as for correction region
    SCORE = {}
    SCORE["Nxi"] = mp.Fend.Nxi
    SCORE["Neta"] = mp.Fend.Neta
    SCORE["pixresFP"] = mp.Fend.res
    SCORE["centering"] = mp.centering

    maskScore = 0
    for iZone in range(Nzones):
        SCORE["rhoInner"] = mp.Fend.score.Rin[iZone]  # lambda0/D
        SCORE["rhoOuter"] = mp.Fend.score.Rout[iZone]  # lambda0/D
        SCORE["angDeg"] = mp.Fend.score.ang[iZone]  # degrees
        SCORE["whichSide"] = sides[iZone]
        SCORE["shape"] = shapes[iZone]
        if hasattr(mp.Fend, 'clockAngDeg'):
            mp.Fend.clockAngDeg = np.atleast_1d(mp.Fend.clockAngDeg)
            SCORE["clockAngDeg"] = mp.Fend.clockAngDeg[iZone]
        if hasattr(mp.Fend, 'xiOffset'):
            mp.Fend.xiOffset = np.atleast_1d(mp.Fend.xiOffset)
            SCORE["xiOffset"] = mp.Fend.xiOffset[iZone]
        if hasattr(mp.Fend, 'etaOffset'):
            mp.Fend.etaOffset = np.atleast_1d(mp.Fend.xiOffset)
            SCORE["etaOffset"] = mp.Fend.etaOffset[iZone]

        [maskTemp, _, _] = falco.mask.falco_gen_SW_mask(SCORE)
        maskScore += maskTemp

    mp.Fend.score.maskBool = np.array(maskScore, dtype=bool)

    # Number of pixels used in the dark hole
    mp.Fend.corr.Npix = np.sum(mp.Fend.corr.maskBool)
    mp.Fend.score.Npix = np.sum(mp.Fend.score.maskBool)

    # vector indicating which pixels in vectorized correction region
    # are also in the scoring region
    mp.Fend.scoreInCorr = mp.Fend.score.maskBool[mp.Fend.corr.maskBool]


def falco_set_spatial_weights(mp):
    """
    Set up spatially-based weighting of the dark hole intensity.

    Set up spatially-based weighting of the dark hole intensity in annular
    zones centered on the star. Zones are specified with rows of three values:
    zone inner radius [l/D], zone outer radius [l/D], and intensity weight.
    As many rows can be used as desired.

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

    # Define 2-D coordinate grid
    [XISLAMD, ETASLAMD] = np.meshgrid(mp.Fend.xisDL, mp.Fend.etasDL)
    RHOS = np.sqrt(XISLAMD**2+ETASLAMD**2)
    mp.Wspatial = mp.Fend.corr.maskBool.astype(float)
    if hasattr(mp, 'WspatialDef'):
        if(np.size(mp.WspatialDef) > 0):
            for kk in range(0, mp.WspatialDef.shape[0]):
                Wannulus = 1. + (np.sqrt(mp.WspatialDef[kk, 2])-1.) *\
                    ((RHOS >= mp.WspatialDef[kk, 0]) &
                     (RHOS < mp.WspatialDef[kk, 1]))
                mp.Wspatial = mp.Wspatial*Wannulus

    mp.WspatialVec = mp.Wspatial[mp.Fend.corr.maskBool]
    if(mp.flagFiber and mp.flagLenslet):
        mp.WspatialVec = np.ones((mp.Fend.Nlens,))

    pass


def falco_configure_dm1_and_dm2(mp):
    """Flesh out the dm1 and dm2 objects."""
    if hasattr(mp, 'dm1'):
        # Read the influence function header data from the FITS file
        dx1 = None
        pitch1 = None
        mp.dm1.inf0 = None
        mp.dm1.dx_inf0 = None
        with fits.open(mp.dm1.inf_fn) as hdul:
            PrimaryData = hdul[0].header
            dx1 = PrimaryData['P2PDX_M']  # pixel width of influence function IN THE FILE [meters]
            pitch1 = PrimaryData['C2CDX_M']  # actuator spacing x (m)

            mp.dm1.inf0 = np.squeeze(hdul[0].data)
        mp.dm1.dx_inf0 = mp.dm1.dm_spacing*(dx1/pitch1)

        if mp.dm1.inf_sign[0] in ['-', 'n', 'm']:
            mp.dm1.inf0 = -1*mp.dm1.inf0
        elif mp.dm1.inf_sign[0] in ['+', 'p']:
            pass
        else:
            raise ValueError('Sign of influence function not recognized')

    if hasattr(mp, 'dm2'):
        # Read the influence function header data from the FITS file
        dx2 = None
        pitch2 = None
        mp.dm2.inf0 = None
        mp.dm2.dx_inf0 = None
        with fits.open(mp.dm2.inf_fn) as hdul:
            PrimaryData = hdul[0].header
            dx2 = PrimaryData['P2PDX_M']  # pixel width of influence function IN THE FILE [meters]
            pitch2 = PrimaryData['C2CDX_M']  # actuator spacing x (m)

            mp.dm2.inf0 = np.squeeze(hdul[0].data)
        mp.dm2.dx_inf0 = mp.dm2.dm_spacing*(dx2/pitch2)

        if mp.dm2.inf_sign[0] in ['-', 'n', 'm']:
            mp.dm2.inf0 = -1*mp.dm2.inf0
        elif mp.dm2.inf_sign[0] in ['+', 'p']:
            pass
        else:
            raise ValueError('Sign of influence function not recognized')

    # DM1
    mp.dm1.centering = mp.centering
    mp.dm1.compact = falco.config.Object()
    mp.dm1.compact = copy.deepcopy(mp.dm1)
    mp.dm1.dx = mp.P2.full.dx
    mp.dm1.compact.dx = mp.P2.compact.dx
    falco.dm.gen_poke_cube(mp.dm1, mp, mp.P2.full.dx, NOCUBE=True)
    if np.any(mp.dm_ind == 1):
        falco.dm.gen_poke_cube(mp.dm1.compact, mp, mp.P2.compact.dx)
    else:
        falco.dm.gen_poke_cube(mp.dm1.compact, mp, mp.P2.compact.dx, NOCUBE=True)

    # DM2
    mp.dm2.centering = mp.centering
    mp.dm2.compact = falco.config.Object()
    mp.dm2.compact = copy.deepcopy(mp.dm2)
    mp.dm2.dx = mp.P2.full.dx
    mp.dm2.compact.dx = mp.P2.compact.dx
    falco.dm.gen_poke_cube(mp.dm2, mp, mp.P2.full.dx, NOCUBE=True)
    if np.any(mp.dm_ind == 2):
        falco.dm.gen_poke_cube(mp.dm2.compact, mp, mp.P2.compact.dx)
    else:
        falco.dm.gen_poke_cube(mp.dm2.compact, mp, mp.P2.compact.dx, NOCUBE=True)

    # Initial DM voltages
    if not hasattr(mp.dm1, 'V'):
        mp.dm1.V = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
    if not hasattr(mp.dm2, 'V'):
        mp.dm2.V = np.zeros((mp.dm2.Nact, mp.dm2.Nact))
    pass


def falco_gen_DM_stops(mp):
    """Generate circular stops for the DMs."""
    if not hasattr(mp.dm2, 'full'):
        mp.dm2.full = falco.config.Object()
    if not hasattr(mp.dm2, 'compact'):
        mp.dm2.compact = falco.config.Object()

    if mp.flagDM1stop:
        mp.dm1.full.mask = falco.mask.falco_gen_DM_stop(mp.P2.full.dx, mp.dm1.Dstop, mp.centering)
        mp.dm1.compact.mask = falco.mask.falco_gen_DM_stop(mp.P2.compact.dx, mp.dm1.Dstop, mp.centering)
    if mp.flagDM2stop:
        mp.dm2.full.mask = falco.mask.falco_gen_DM_stop(mp.P2.full.dx, mp.dm2.Dstop, mp.centering)
        mp.dm2.compact.mask = falco.mask.falco_gen_DM_stop(mp.P2.compact.dx, mp.dm2.Dstop, mp.centering)
    pass


def falco_set_dm_surface_padding(mp):
    """Set how much the DM surface arrays get padded prior to propagation."""
    #% DM Surface Array Sizes for Angular Spectrum Propagation with FFTs
    # Array Sizes for Angular Spectrum Propagation with FFTs
    
    # Compact Model: Set nominal DM plane array sizes as a power of 2 for angular spectrum propagation with FFTs
    if np.any(mp.dm_ind == 1) and np.any(mp.dm_ind == 2):
        NdmPad = 2**np.ceil(1 + np.log2(np.max([mp.dm1.compact.NdmPad, mp.dm2.compact.NdmPad])))
    elif np.any(mp.dm_ind == 1):
        NdmPad = 2**np.ceil(1 + np.log2(mp.dm1.compact.NdmPad))
    elif np.any(mp.dm_ind == 2):
        NdmPad = 2**np.ceil(1 + np.log2(mp.dm2.compact.NdmPad))
    else:
        NdmPad = 2*mp.P1.compact.Nbeam;

    while (NdmPad < np.min(mp.sbp_centers)*np.abs(mp.d_dm1_dm2)/mp.P2.full.dx**2) or (NdmPad < np.min(mp.sbp_centers)*np.abs(mp.d_P2_dm1)/mp.P2.compact.dx**2): 
        # Double the zero-padding until the angular spectrum sampling requirement is not violated
        NdmPad = 2*NdmPad;
    mp.compact.NdmPad = NdmPad;
    
    # Full Model: Set nominal DM plane array sizes as a power of 2 for angular spectrum propagation with FFTs
    if np.any(mp.dm_ind == 1) and np.any(mp.dm_ind == 2):
        NdmPad = 2**np.ceil(1 + np.log2(np.max([mp.dm1.NdmPad, mp.dm2.NdmPad])))
    elif np.any(mp.dm_ind == 1):
        NdmPad = 2**np.ceil(1 + np.log2(mp.dm1.NdmPad))
    elif np.any(mp.dm_ind == 2):
        NdmPad = 2**np.ceil(1 + np.log2(mp.dm2.NdmPad))
    else:
        NdmPad = 2*mp.P1.full.Nbeam
    # Double the zero-padding until the angular spectrum sampling requirement is not violated
    while (NdmPad < np.min(mp.full.lambdas)*np.abs(mp.d_dm1_dm2)/mp.P2.full.dx**2) or \
        (NdmPad < np.min(mp.full.lambdas)*np.abs(mp.d_P2_dm1)/mp.P2.full.dx**2): 
        NdmPad = 2*NdmPad
    mp.full.NdmPad = NdmPad
    pass


def falco_set_initial_Efields(mp):
    """Define star and optional planet E-fields at the input pupil."""
    # Initial Electric Fields for Star and Exoplanet

    if not hasattr(mp.P1.full, 'E'):  # Input E-field at entrance pupil
        mp.P1.full.E = np.ones((mp.P1.full.Narr, mp.P1.full.Narr, mp.Nwpsbp,
                                mp.Nsbp), dtype=complex)

    # Initialize the input E-field for the planet at the entrance pupil.
    # Will apply the phase ramp later
    mp.Eplanet = mp.P1.full.E

    if not hasattr(mp.P1.compact, 'E'):
        mp.P1.compact.E = np.ones((mp.P1.compact.Narr, mp.P1.compact.Narr,
                                   mp.Nsbp), dtype=complex)
    else:
        if mp.P1.compact.E.shape[0] != mp.P1.compact.Narr:
            EcubeTemp = copy.deepcopy(mp.P1.compact.E)
            mp.P1.compact.E = np.ones((mp.P1.compact.Narr, mp.P1.compact.Narr,
                                       mp.Nsbp), dtype=complex)
            for si in range(mp.Nsbp):
                mp.P1.compact.E[:, :, si] = pad_crop(EcubeTemp[:, :, si],
                                                     mp.P1.compact.Narr)

    # Throughput is computed with the compact model
    mp.sumPupil = np.sum(np.sum(np.abs(mp.P1.compact.mask*falco.util.pad_crop(
        np.mean(mp.P1.compact.E, 2), mp.P1.compact.mask.shape[0]))**2))


def init_storage_arrays(mp):
    """
    Initialize arrays that store performance history.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        custom object of model parameters

    Returns
    -------
    out : types.SimpleNamespace()
        namespace object of performance history arrays.

    """
    # Initialize objects
    out = types.SimpleNamespace()
    out.dm1 = types.SimpleNamespace()
    out.dm2 = types.SimpleNamespace()
    out.dm8 = types.SimpleNamespace()
    out.dm9 = types.SimpleNamespace()
    out.Fend = types.SimpleNamespace()
    out.Fend.corr = types.SimpleNamespace()
    out.Fend.score = types.SimpleNamespace()

    # Storage Arrays for DM Metrics
    # EFC regularization history
    out.Nitr = mp.Nitr
    out.log10regHist = np.zeros(mp.Nitr)

    # Peak-to-Valley DM voltages
    out.dm1.Vpv = np.zeros(mp.Nitr)
    out.dm2.Vpv = np.zeros(mp.Nitr)
    out.dm8.Vpv = np.zeros(mp.Nitr)
    out.dm9.Vpv = np.zeros(mp.Nitr)

    # Peak-to-Valley DM surfaces
    out.dm1.Spv = np.zeros(mp.Nitr)
    out.dm2.Spv = np.zeros(mp.Nitr)
    out.dm8.Spv = np.zeros(mp.Nitr)
    out.dm9.Spv = np.zeros(mp.Nitr)

    # RMS DM surfaces
    out.dm1.Srms = np.zeros(mp.Nitr)
    out.dm2.Srms = np.zeros(mp.Nitr)
    out.dm8.Srms = np.zeros(mp.Nitr)
    out.dm9.Srms = np.zeros(mp.Nitr)

    # Sensitivities Zernike-Mode Perturbations
    if not hasattr(mp.eval, 'Rsens'):
        mp.eval.Rsens = []
    if not hasattr(mp.eval, 'indsZnoll'):
        mp.eval.indsZnoll = [1, 2]
    Nannuli = mp.eval.Rsens.shape[0]
    Nzern = len(mp.eval.indsZnoll)
    out.Zsens = np.zeros((Nzern, Nannuli, mp.Nitr))

    # Store the DM commands at each iteration
    if hasattr(mp, 'dm1'):
        if hasattr(mp.dm1, 'V'):
            out.dm1.Vall = np.zeros((mp.dm1.Nact, mp.dm1.Nact, mp.Nitr+1))
    if hasattr(mp, 'dm2'):
        if hasattr(mp.dm2, 'V'):
            out.dm2.Vall = np.zeros((mp.dm2.Nact, mp.dm2.Nact, mp.Nitr+1))
    if hasattr(mp, 'dm8'):
        if hasattr(mp.dm8, 'V'):
            out.dm8.Vall = np.zeros((mp.dm8.NactTotal, mp.Nitr+1))
    if hasattr(mp, 'dm9'):
        if hasattr(mp.dm9, 'V'):
            out.dm9.Vall = np.zeros((mp.dm9.NactTotal, mp.Nitr+1))

    # Delta electric field performance metrics
    out.complexProjection = np.zeros((mp.Nitr-1, mp.Nsbp))  # Metric to compare magnitude of the correction step taken to the expected one
    out.complexCorrelation = np.zeros((mp.Nitr-1, mp.Nsbp))  # Metric to compare the morphology of the delta E-field estimated vs expected in the model

    # Intensity history at each iteration
    out.InormHist = np.zeros(mp.Nitr + 1)  # Measured, mean raw NI in correction region of dark hole.
    out.IrawCorrHist = np.zeros(mp.Nitr + 1)  # Measured, mean raw NI in correction region of dark hole.
    out.IrawScoreHist = np.zeros(mp.Nitr + 1)  # Measured, mean raw NI in scoring region of dark hole.
    out.IestCorrHist = np.zeros(mp.Nitr)  # Mean estimated coherent NI in correction region of dark hole.
    out.IestScoreHist = np.zeros(mp.Nitr)  # Mean estimated coherent NI in scoring region of dark hole.
    out.IincoCorrHist = np.zeros(mp.Nitr)  # Mean estimated incoherent NI in correction region of dark hole.
    out.IincoScoreHist = np.zeros(mp.Nitr)  # Mean estimated incoherent NI in scoring region of dark hole.

    out.normIntMeasCorr = np.zeros((mp.Nitr, mp.Nsbp))  # Measured raw NI in correction region of dark hole.
    out.normIntMeasScore = np.zeros((mp.Nitr, mp.Nsbp))  # Measured raw NI in scoring region of dark hole.
    out.normIntModCorr = np.zeros((mp.Nitr, mp.Nsbp*mp.compact.star.count))  # Estimated modulated NI in correction region of dark hole.
    out.normIntModScore = np.zeros((mp.Nitr, mp.Nsbp*mp.compact.star.count))  # Estimated modulated NI in scoring region of dark hole.
    out.normIntUnmodCorr = np.zeros((mp.Nitr, mp.Nsbp*mp.compact.star.count))  # Estimated unmodulated NI in correction region of dark hole.
    out.normIntUnmodScore = np.zeros((mp.Nitr, mp.Nsbp*mp.compact.star.count))  # Estimated unmodulated NI in correction region of dark hole.

    # Storage array for throughput at each iteration
    out.thput = np.zeros(mp.Nitr + 1)

    # Variables related to final image
    out.Fend.res = mp.Fend.res
    out.Fend.xisDL = mp.Fend.xisDL
    out.Fend.etasDL = mp.Fend.etasDL
    out.Fend.scoreInCorr = mp.Fend.scoreInCorr
    out.Fend.corr.maskBool = mp.Fend.corr.maskBool
    out.Fend.score.maskBool = mp.Fend.score.maskBool

    out.serialDate = np.zeros(mp.Nitr)  # start time of each iteration as float

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
    
    # Make or read in focal plane mask (FPM) amplitude for the full model
    FPMgenInputs = {}
    FPMgenInputs["pixresFPM"] = mp.F3.full.res  # pixels per lambda_c/D
    FPMgenInputs["rhoInner"] = mp.F3.Rin  # radius of inner FPM amplitude spot (in lambda_c/D)
    FPMgenInputs["rhoOuter"] = mp.F3.Rout  # radius of outer opaque FPM ring (in lambda_c/D)
    if hasattr(mp, 'FPMampFac'):
        FPMgenInputs["FPMampFac"] = mp.FPMampFac  # amplitude transmission of inner FPM spot
    else:
        FPMgenInputs["FPMampFac"] = 0.0
    FPMgenInputs["centering"] = mp.centering
    
    if not hasattr(mp.F3.full, 'mask'):
        mp.F3.full.mask = falco.config.Object()

    mp.F3.full.mask = falco.mask.falco_gen_annular_FPM(FPMgenInputs)

    mp.F3.full.Nxi = mp.F3.full.mask.shape[1]
    mp.F3.full.Neta = mp.F3.full.mask.shape[0]

    # Number of points across the FPM in the compact model
    if np.isinf(mp.F3.Rout):
        if mp.centering == 'pixel':
            mp.F3.compact.Nxi = ceil_even((2*(mp.F3.Rin*mp.F3.compact.res + 1/2)))
        else:
            mp.F3.compact.Nxi = ceil_even((2*mp.F3.Rin*mp.F3.compact.res))

    else:
        if mp.centering == 'pixel':
            mp.F3.compact.Nxi = ceil_even((2*(mp.F3.Rout*mp.F3.compact.res + 1/2)))
        else:  # case 'interpixel'
            mp.F3.compact.Nxi = ceil_even((2*mp.F3.Rout*mp.F3.compact.res))

    mp.F3.compact.Neta = mp.F3.compact.Nxi
    
    # Make or read in focal plane mask (FPM) amplitude for the compact model
    FPMgenInputs["pixresFPM"] = mp.F3.compact.res  # pixels per lambda_c/D
    
    if not hasattr(mp.F3.compact, 'mask'):
        mp.F3.compact.mask = falco.config.Object()
        
    mp.F3.compact.mask = falco.mask.falco_gen_annular_FPM(FPMgenInputs)

def falco_gen_FPM_SPLC(mp):
    """Generate the FPM for an SPLC."""
    if not hasattr(mp.F3, 'ang'):
        mp.F3.ang = 180
    
    if(mp.full.flagGenFPM):
        # Generate the FPM amplitude for the full model
        inputs = {}
        inputs["rhoInner"] = mp.F3.Rin  # radius of inner FPM amplitude spot (in lambda_c/D)
        inputs["rhoOuter"] = mp.F3.Rout  # radius of outer opaque FPM ring (in lambda_c/D)
        inputs["ang"] = mp.F3.ang  # [degrees]
        inputs["centering"] = mp.centering;
        inputs["pixresFPM"] = mp.F3.full.res  # pixels per lambda_c/D
        mp.F3.full.mask = falco.mask.falco_gen_bowtie_FPM(inputs)
    
    if(mp.compact.flagGenFPM):
        # Generate the FPM amplitude for the compact model
        inputs = {}
        inputs["rhoInner"] = mp.F3.Rin  # radius of inner FPM amplitude spot (in lambda_c/D)
        inputs["rhoOuter"] = mp.F3.Rout  # radius of outer opaque FPM ring (in lambda_c/D)
        inputs["ang"] = mp.F3.ang  # [degrees]
        inputs["centering"] = mp.centering
        inputs["pixresFPM"] = mp.F3.compact.res
        mp.F3.compact.mask = falco.mask.falco_gen_bowtie_FPM(inputs)
    
    if not mp.full.flagPROPER:
        mp.F3.full.Nxi = mp.F3.full.mask.shape[1]
        mp.F3.full.Neta = mp.F3.full.mask.shape[0]
    
    mp.F3.compact.Nxi = mp.F3.compact.mask.shape[1]
    mp.F3.compact.Neta = mp.F3.compact.mask.shape[0]
    pass
