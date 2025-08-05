"""Script to define configuration values for FALCO."""
import numpy as np
import astropy
import falco

mp = falco.config.ModelParameters()
mp.path = falco.config.Object()

# Record Keeping
mp.SeriesNum = 1
mp.TrialNum = 1

# Special Computational Settings
mp.flagParallel = False
mp.flagPlot = True

# General
mp.centering = 'pixel'

# Method of computing core throughput:
# - 'HMI' for energy within half-max isophote divided by energy at telescope pupil
# - 'EE' for encircled energy within a radius (mp.thput_radius) divided by energy at telescope pupil
mp.thput_metric = 'EE'
mp.thput_radius = 0.7  # photometric aperture radius [lambda_c/D]. Used ONLY for 'EE' method.
mp.thput_eval_x = 7  # x location [lambda_c/D] in dark hole at which to evaluate throughput
mp.thput_eval_y = 0  # y location [lambda_c/D] in dark hole at which to evaluate throughput

# Where to shift the source to compute the intensity normalization value.
mp.source_x_offset_norm = 7  # x location [lambda_c/D] in dark hole at which to compute intensity normalization
mp.source_y_offset_norm = 0  # y location [lambda_c/D] in dark hole at which to compute intensity normalization

# Bandwidth and Wavelength Specs
mp.lambda0 = 550e-9  # Central wavelength of the whole spectral bandpass [meters]
mp.fracBW = 0.02  # fractional bandwidth of the whole bandpass (Delta lambda / lambda0)
mp.Nsbp = 1  # Number of sub-bandpasses to divide the whole bandpass into for estimation and control
mp.Nwpsbp = 1  # Number of wavelengths to used to approximate an image in each sub-bandpass


# %% Wavefront Estimation

mp.dm_ind_static = [];  # --DMs ONLY holding dark zone shape, not injecting drift or part of control

# -- Variables for ekf maintenance estimation
mp.estimator = 'ekf_maintenance'

mp.est = falco.config.Object()
mp.est.probe = falco.config.Probe()
mp.est.probe.Npairs = 1
mp.est.probe.whichDM = 2  # --Which DM is used for dither/control
mp.est.dither = 9.5e-5  # --std dev of dither command for random dither [V/sqtr(iter)]
mp.est.flagUseJac = True  # EKF needs the jacobian for estimation
mp.est.read_noise = 1  # --Read noise of detector [e-]
mp.est.dark_current = 0.01  # --Dark current of detector [e-/s]
mp.est.itr_ol = [0]  # --"open-loop" iterations where an image is taken with initial DM command + drift command
mp.est.itr_reset = np.nan
mp.est.dither_cycle_iters = 500  # --Number of unique dither commands used

# %% Wavefront Control: General

mp.ctrl = falco.config.Object()
mp.ctrl.flagUseModel = False  # Whether to perform a model-based (vs empirical) grid search for the controller

# Threshold for culling weak actuators from the Jacobian:
mp.logGmin = -6  # 10^(mp.logGmin) used on the intensity of DM1 and DM2 Jacobians to weed out the weakest actuators

# Zernikes to suppress with controller
mp.jac = falco.config.Object()
mp.jac.zerns = [1, ]  # Which Noll Zernike modes to include in Jacobian. Always include the value "1" for the on-axis piston mode.
mp.jac.Zcoef = 1e-9*np.ones(np.size(mp.jac.zerns))  # meters RMS of Zernike aberrations. (piston value is reset to 1 later)

# Zernikes to compute sensitivities for
mp.eval = falco.config.Object()
mp.eval.indsZnoll = [2, 3]  # Noll indices of Zernikes to compute values for [1-D ndarray]

# Annuli to compute 1nm RMS Zernike sensitivities over. Columns are [inner radius, outer radius]. One row per annulus.
mp.eval.Rsens = np.array([[2., 3.], [3., 4.], [4., 5.]])  # [2-D ndarray]


# Spatial pixel weighting
mp.WspatialDef = []  # [3, 4.5, 3]; #--spatial control Jacobian weighting by annulus: [Inner radius, outer radius, intensity weight; (as many rows as desired)] [ndarray]

# DM weighting
mp.dm1.weight = 1.
mp.dm2.weight = 1.

# %% Wavefront Control: Controller Specific
# Controller options:
#  - 'gridsearchEFC' for EFC as an empirical grid search over tuning parameters
#  - 'plannedEFC' for EFC with an automated regularization schedule
mp.controller = 'gridsearchEFC'
# Grid- or Line-Search Settings
mp.ctrl.log10regVec = np.array([-6,-5])  # log10 of the regularization exponents (often called Beta values)
mp.ctrl.dmfacVec = np.array([1.])  # Proportional gain term applied to the total DM delta command. Usually in range [0.5,1]. [1-D ndarray]

# # GRID SEARCH EFC DEFAULTS
# WFSC Iterations and Control Matrix Relinearization
mp.Nitr = 500  # Number of estimation+control iterations to perform
mp.relinItrVec = np.array([0])  # Which correction iterations at which to re-compute the control Jacobian [1-D ndarray]
mp.dm_ind = np.array([1, 2]) # Which DMs to use [1-D ndarray]

# # # PLANNED SEARCH EFC DEFAULTS
# mp.dm_ind = [1 2 ]; # vector of DMs used in controller at ANY time (not necessarily all at once or all the time).
# mp.ctrl.dmfacVec = 1;
# #--CONTROL SCHEDULE. Columns of mp.ctrl.sched_mat are:
#     # Column 1: # of iterations,
#     # Column 2: log10(regularization),
#     # Column 3: which DMs to use (12, 128, 129, or 1289) for control
#     # Column 4: flag (0 = False, 1 = True), whether to re-linearize
#     #   at that iteration.
#     # Column 5: flag (0 = False, 1 = True), whether to perform an
#     #   EFC parameter grid search to find the set giving the best
#     #   contrast .
#     # The imaginary part of the log10(regularization) in column 2 is
#     #  replaced for that iteration with the optimal log10(regularization)
#     # A row starting with [0, 0, 0, 1...] is for relinearizing only at that time
#
# mp.ctrl.sched_mat = [...
#     repmat([1,1j,12,1,1],[4,1]);...
#     repmat([1,1j-1,12,1,1],[25,1]);...
#     repmat([1,1j,12,1,1],[1,1]);...
#     ];
# [mp.Nitr, mp.relinItrVec, mp.gridSearchItrVec, mp.ctrl.log10regSchedIn, mp.dm_ind_sched] = falco_ctrl_EFC_schedule_generator(mp.ctrl.sched_mat);

### Drift Injection
mp.drift = falco.config.Drift()
# TODO: update elsewhere in code to use drift.WhichDM
mp.dm_drift_ind = np.array([1])  # --which DM is drifting
mp.drift.WhichDM = np.array([1])  # --which DM is drifting
mp.drift.type = 'rand_walk'  # --what type of drift is happening
mp.drift.magnitude = 0.001  # --std dev of random walk [V/sqrt(iter)]
mp.drift.presumed_dm_std = mp.drift.magnitude  # --std dev of random walk provided to estimator, change this to account for the uncertainty of the drift magnitude


# %% Deformable Mirrors: Influence Functions

# Influence Function Options:
# - falco.INFLUENCE_XINETICS uses the file 'influence_dm5v2.fits' for one type of Xinetics DM
# - INFLUENCE_BMC_2K uses the file 'influence_BMC_2kDM_400micron_res10.fits' for BMC 2k DM
# - INFLUENCE_BMC_KILO uses the file 'influence_BMC_kiloDM_300micron_res10_spline.fits' for BMC kiloDM
mp.dm1.inf_fn = falco.INFLUENCE_BMC_2K
mp.dm2.inf_fn = falco.INFLUENCE_BMC_2K

mp.dm1.dm_spacing = 400e-6 #--User defined actuator pitch
mp.dm2.dm_spacing = 400e-6 #--User defined actuator pitch

mp.dm1.inf_sign = '+'
mp.dm2.inf_sign = '+'

# %% Deformable Mirrors: Optical Layout Parameters

# DM1 parameters
mp.dm1.Nact = 32               # of actuators across DM array
mp.dm1.VtoH = 1e-9*np.ones((32, 32))  # gains of all actuators [nm/V of free stroke]
mp.dm1.xtilt = 0               # for foreshortening. angle of rotation about x-axis [degrees]
mp.dm1.ytilt = 0               # for foreshortening. angle of rotation about y-axis [degrees]
mp.dm1.zrot = 0                # clocking of DM surface [degrees]
mp.dm1.xc = (32/2 - 1/2)       # x-center location of DM surface [actuator widths]
mp.dm1.yc = (32/2 - 1/2)       # y-center location of DM surface [actuator widths]
mp.dm1.edgeBuffer = 1          # max radius (in actuator spacings) outside of beam on DM surface to compute influence functions for. [actuator widths]

# DM2 parameters
mp.dm2.Nact = 32               # # of actuators across DM array
mp.dm2.VtoH = 1e-9*np.ones((32, 32))  # gains of all actuators [nm/V of free stroke]
mp.dm2.xtilt = 0               # for foreshortening. angle of rotation about x-axis [degrees]
mp.dm2.ytilt = 0                # for foreshortening. angle of rotation about y-axis [degrees]
mp.dm2.zrot = 0                 # clocking of DM surface [degrees]
mp.dm2.xc = (32/2 - 1/2)        # x-center location of DM surface [actuator widths]
mp.dm2.yc = (32/2 - 1/2)        # y-center location of DM surface [actuator widths]
mp.dm2.edgeBuffer = 1          # max radius (in actuator spacings) outside of beam on DM surface to compute influence functions for. [actuator widths]

#  Aperture stops at DMs
mp.flagDM1stop = False   # Whether to apply an iris or not
mp.dm1.Dstop = 50e-3  # Diameter of iris [meters]
mp.flagDM2stop = True  # Whether to apply an iris or not
mp.dm2.Dstop = 50e-3  # Diameter of iris [meters]

# DM separations
mp.d_P2_dm1 = 0  # distance (along +z axis) from P2 pupil to DM1 [meters]
mp.d_dm1_dm2 = 0.20  # distance between DM1 and DM2 [meters]


# %% Optical Layout: All models

# Key Optical Layout Choices
mp.flagSim = True  # Simulation or not
mp.layout = 'Fourier'  # Which optical layout to use
mp.coro = 'vortex'

mp.Fend = falco.config.Object()

# Final Focal Plane Properties
mp.Fend.res = 3  # Sampling [ pixels per lambda0/D]
mp.Fend.FOV = 15.  # half-width of the field of view in both dimensions [lambda0/D]

# Correction and scoring region definition
mp.Fend.corr = falco.config.Object()
mp.Fend.corr.Rin = 2.0   # inner radius of dark hole correction region [lambda0/D]
mp.Fend.corr.Rout = 15  # outer radius of dark hole correction region [lambda0/D]
mp.Fend.corr.ang = 180  # angular opening of dark hole correction region [degrees]
#
mp.Fend.score = falco.config.Object()
mp.Fend.score.Rin = 2.0  # inner radius of dark hole scoring region [lambda0/D]
mp.Fend.score.Rout = 15  # outer radius of dark hole scoring region [lambda0/D]
mp.Fend.score.ang = 180  # angular opening of dark hole scoring region [degrees]

mp.Fend.sides = 'leftright'  # Which side(s) for correction: 'left', 'right', 'top', 'up', 'bottom', 'down', 'lr', 'rl', 'leftright', 'rightleft', 'tb', 'bt', 'ud', 'du', 'topbottom', 'bottomtop', 'updown', 'downup'

# %% Optical Layout: Compact Model (and Jacobian Model)

# Focal Lengths
mp.fl = 1.  # [meters] Focal length value used for all FTs in the compact model. Don't need different values since this is a Fourier model.

# Pupil Plane Diameters
mp.P2.D = mp.dm1.Nact*mp.dm1.dm_spacing
mp.P3.D = mp.P2.D
mp.P4.D = mp.P2.D

# Pupil Plane Resolutions
mp.P1.compact.Nbeam = 513
# mp.P2.compact.Nbeam = mp.P1.compact.Nbeam
# mp.P3.compact.Nbeam = mp.P1.compact.Nbeam
mp.P4.compact.Nbeam = mp.P1.compact.Nbeam  # P4 size must be the same as P1 for Vortex.

# Number of re-imaging relays between pupil planesin compact model. Needed
# to keep track of 180-degree rotations compared to the full model, which
# in general can have probably has extra collimated beams compared to the
# compact model.
mp.flagRotation = True  # Whether to rotate 180 degrees between conjugate planes in the compact model
mp.Nrelay1to2 = 1
mp.Nrelay2to3 = 1
mp.Nrelay3to4 = 1
mp.NrelayFend = 0  # How many times to rotate the final image by 180 degrees


# %% Optical Layout: Full Model

# # Focal Lengths
# mp.fl = 1

# Pupil Plane Resolutions
mp.P1.full.Nbeam = 513
# mp.P2.full.Nbeam = 250
# mp.P3.full.Nbeam = 250
mp.P4.full.Nbeam = 513  # P4 size must be the same as P1 for Vortex.

# Mask Definitions
mp.full = falco.config.Object()
mp.compact = falco.config.Object()


# %% Entrance Pupil (P1) Definition and Generation

mp.whichPupil = 'HWO_EAC1'  # Used only for run label
mp.P1.IDnorm = 0.00  # ID of the central obscuration [diameter]. Used only for computing the RMS DM surface from the ID to the OD of the pupil. OD is assumed to be 1.
mp.P1.ODnorm = 1.00  # Outer diameter of the telescope [diameter]
mp.P1.D = 7.2  # circumscribed telescope diameter [meters]. Used only for converting milliarcseconds to lambda0/D or vice-versa.

# Generate the entrance pupil aperture
inputs = {"centering": mp.centering}

# Full model:
inputs["Nbeam"] = mp.P1.full.Nbeam
mp.P1.full.mask = astropy.io.fits.open(r'C:\Users\sredmond\Documents\gitub_repos\falco-python\data\HWO\EAC1/pupil_stopped_82.7_513_gen_avc.fits')[0].data

# Compact model
inputs["Nbeam"] = mp.P1.compact.Nbeam
mp.P1.compact.mask = falco.util.pad_crop(
    falco.mask.falco_gen_pupil_LUVOIR_B(inputs),
    2**(falco.util.nextpow2(inputs["Nbeam"])))


# %% "Apodizer" (P3) Definition and Generation

mp.flagApod = True  # Whether to use an apodizer or not

# Inputs common to both the compact and full models
inputs = {"OD": 0.84}

# Full model only
inputs["Nbeam"] = mp.P1.full.Nbeam
inputs["Npad"] = 2**(falco.util.nextpow2(mp.P1.full.Nbeam))
mp.P3.full.mask = astropy.io.fits.open(r'C:\Users\sredmond\Documents\gitub_repos\falco-python\data\HWO\EAC1/ap_eac1b_aavc_fast_optimize_cds_513_15_ls0.81_ps0.827_dz48_ch6_375_ch2iter1999.fits')[0].data


# Compact model only
inputs["Nbeam"] = mp.P1.compact.Nbeam
inputs["Npad"] = 2**(falco.util.nextpow2(mp.P1.compact.Nbeam))
mp.P3.compact.mask = astropy.io.fits.open(r'C:\Users\sredmond\Documents\gitub_repos\falco-python\data\HWO\EAC1/ap_eac1b_aavc_fast_optimize_cds_513_15_ls0.81_ps0.827_dz48_ch6_375_ch2iter1999.fits')[0].data


# %% Lyot stop (P4) Definition and Generation

mp.P4.IDnorm = 0  # Lyot stop ID [Dtelescope]
mp.P4.ODnorm = 0.81  # Lyot stop OD [Dtelescope]

# Inputs common to both the compact and full models
inputs = {}
inputs["ID"] = mp.P4.IDnorm
inputs["OD"] = mp.P4.ODnorm

# Full model
inputs["Nbeam"] = mp.P4.full.Nbeam
inputs["Npad"] = 2**(falco.util.nextpow2(mp.P4.full.Nbeam))
mp.P4.full.mask = astropy.io.fits.open(r'C:\Users\sredmond\Documents\gitub_repos\falco-python\data\HWO\EAC1/lyot_stop_81_513_gen_avc.fits')[0].data


# Compact model
inputs["Nbeam"] = mp.P4.compact.Nbeam
inputs["Npad"] = 2**(falco.util.nextpow2(mp.P4.compact.Nbeam))
mp.P4.compact.mask = astropy.io.fits.open(r'C:\Users\sredmond\Documents\gitub_repos\falco-python\data\HWO\EAC1/lyot_stop_81_513_gen_avc.fits')[0].data



# %% Vortex Specific Values

mp.F3.VortexCharge = 6  # Charge of the vortex mask

mp.F3.compact.res = 6  # FPM sampling in compact model [pixels per lambda0/D]
mp.F3.full.res = 6  # FPM sampling in full model [pixels per lambda0/D]