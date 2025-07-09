"""Script to define configuration values for FALCO."""
import numpy as np

import falco

mp = falco.config.ModelParameters()
mp.path = falco.config.Object()

# Record Keeping
mp.SeriesNum = 1
mp.TrialNum = 1

# Special Computational Settings
mp.flagParallel = False
mp.flagPlot = False

# General
mp.centering = 'pixel'

# Method of computing core throughput:
# - 'HMI' for energy within half-max isophote divided by energy at telescope pupil
# - 'EE' for encircled energy within a radius (mp.thput_radius) divided by energy at telescope pupil
mp.thput_metric = 'HMI'
mp.thput_radius = 0.7  # photometric aperture radius [lambda_c/D]. Used ONLY for 'EE' method.
mp.thput_eval_x = 7  # x location [lambda_c/D] in dark hole at which to evaluate throughput
mp.thput_eval_y = 0  # y location [lambda_c/D] in dark hole at which to evaluate throughput

# Where to shift the source to compute the intensity normalization value.
mp.source_x_offset_norm = 7  # x location [lambda_c/D] in dark hole at which to compute intensity normalization
mp.source_y_offset_norm = 0  # y location [lambda_c/D] in dark hole at which to compute intensity normalization

# Bandwidth and Wavelength Specs
mp.lambda0 = 575e-9  # Central wavelength of the whole spectral bandpass [meters]
mp.fracBW = 0.01  # fractional bandwidth of the whole bandpass (Delta lambda / lambda0)
mp.Nsbp = 1  # Number of sub-bandpasses to divide the whole bandpass into for estimation and control
mp.Nwpsbp = 1  # Number of wavelengths to used to approximate an image in each sub-bandpass


# %% Wavefront Estimation

# Estimator Options:
# - 'perfect' for exact numerical answer from full model
# - 'pwp-bp' for pairwise probing with batch process estimation
mp.estimator = 'perfect'

# Pairwise probing:
mp.est = falco.config.Object()
mp.est.probe = falco.config.Probe()
mp.est.probe.Npairs = 3  # Number of pair-wise probe PAIRS to use.
mp.est.probe.whichDM = 1  # Which DM # to use for probing. 1 or 2. Default is 1
mp.est.probe.radius = 12  # Max x/y extent of probed region [lambda/D].
mp.est.probe.xOffset = 0  # offset of probe center in x [lambda/D]. Use to avoid central obscurations.
mp.est.probe.yOffset = 14  # offset of probe center in y [lambda/D]. Use to avoid central obscurations.
mp.est.probe.axis = 'alternate'  # which axis to have the phase discontinuity along [x or y or xy/alt/alternate]
mp.est.probe.gainFudge = 1  # empirical fudge factor to make average probe amplitude match desired value.


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
mp.eval.Rsens = np.array([[3., 4.], [4., 8.]])  # [2-D ndarray]

# Grid- or Line-Search Settings
mp.ctrl.log10regVec = np.arange(-6, -2+0.5, 1)  # log10 of the regularization exponents (often called Beta values)
mp.ctrl.dmfacVec = np.array([1.])  # Proportional gain term applied to the total DM delta command. Usually in range [0.5,1]. [1-D ndarray]

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

# # GRID SEARCH EFC DEFAULTS
# WFSC Iterations and Control Matrix Relinearization
mp.Nitr = 3  # Number of estimation+control iterations to perform
mp.relinItrVec = np.arange(mp.Nitr)  # Which correction iterations at which to re-compute the control Jacobian [1-D ndarray]
mp.dm_ind = [1, ]  # Which DMs to use [1-D array_like]

# # PLANNED SEARCH EFC DEFAULTS
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


# %% Deformable Mirrors: Influence Functions

# Influence Function Options:
# - falco.INFLUENCE_XINETICS uses the file 'influence_dm5v2.fits' for one type of Xinetics DM
# - INFLUENCE_BMC_2K uses the file 'influence_BMC_2kDM_400micron_res10.fits' for BMC 2k DM
# - INFLUENCE_BMC_KILO uses the file 'influence_BMC_kiloDM_300micron_res10_spline.fits' for BMC kiloDM
# mp.dm1.inf_fn = falco.INFLUENCE_BMC_2K
# mp.dm2.inf_fn = falco.INFLUENCE_BMC_2K
mp.dm1.inf_fn = falco.INFLUENCE_XINETICS
mp.dm2.inf_fn = falco.INFLUENCE_XINETICS

mp.dm1.dm_spacing = 0.9906e-3  # actuator pitch
mp.dm2.dm_spacing = 0.9906e-3  # actuator pitch

mp.dm1.inf_sign = '+'
mp.dm2.inf_sign = '+'

# %% Deformable Mirrors: Optical Layout Parameters

# DM1 parameters
mp.dm1.Nact = 32               # of actuators across DM array
mp.dm1.VtoH = 1e-9*np.ones((32, 32))  # gains of all actuators [nm/V of free stroke]
mp.dm1.xtilt = 0               # for foreshortening. angle of rotation about x-axis [degrees]
mp.dm1.ytilt = 5.83            # for foreshortening. angle of rotation about y-axis [degrees]
mp.dm1.zrot = 0                # clocking of DM surface [degrees]
mp.dm1.xc = (32/2 - 1/2)       # x-center location of DM surface [actuator widths]
mp.dm1.yc = (32/2 - 1/2)       # y-center location of DM surface [actuator widths]
mp.dm1.edgeBuffer = 1          # max radius (in actuator spacings) outside of beam on DM surface to compute influence functions for. [actuator widths]

# DM2 parameters
mp.dm2.Nact = 32               # # of actuators across DM array
mp.dm2.VtoH = 1e-9*np.ones((32, 32))  # gains of all actuators [nm/V of free stroke]
mp.dm2.xtilt = 0               # for foreshortening. angle of rotation about x-axis [degrees]
mp.dm2.ytilt = 5.55             # for foreshortening. angle of rotation about y-axis [degrees]
mp.dm2.zrot = 0                 # clocking of DM surface [degrees]
mp.dm2.xc = (32/2 - 1/2)        # x-center location of DM surface [actuator widths]
mp.dm2.yc = (32/2 - 1/2)        # y-center location of DM surface [actuator widths]
mp.dm2.edgeBuffer = 1          # max radius (in actuator spacings) outside of beam on DM surface to compute influence functions for. [actuator widths]

#  Aperture stops at DMs
mp.flagDM1stop = False   # Whether to apply an iris or not
mp.dm1.Dstop = 50e-3  # Diameter of iris [meters]
mp.flagDM2stop = False  # Whether to apply an iris or not
mp.dm2.Dstop = 50e-3  # Diameter of iris [meters]

# DM separations
mp.d_P2_dm1 = 0  # distance (along +z axis) from P2 pupil to DM1 [meters]
mp.d_dm1_dm2 = 0  # distance between DM1 and DM2 [meters]


# %% Optical Layout: All models

# Key Optical Layout Choices
mp.flagSim = True  # Simulation or not
mp.layout = 'Fourier'  # Which optical layout to use
mp.coro = 'LC'

mp.Fend = falco.config.Object()

# Final Focal Plane Properties
mp.Fend.res = 2.5  # Sampling [ pixels per lambda0/D]
mp.Fend.FOV = 11.  # half-width of the field of view in both dimensions [lambda0/D]

# Correction and scoring region definition
mp.Fend.corr = falco.config.Object()
mp.Fend.corr.Rin = 1.5   # inner radius of dark hole correction region [lambda0/D]
mp.Fend.corr.Rout = 9.7  # outer radius of dark hole correction region [lambda0/D]
mp.Fend.corr.ang = 180  # angular opening of dark hole correction region [degrees]
#
mp.Fend.score = falco.config.Object()
mp.Fend.score.Rin = 3.0  # inner radius of dark hole scoring region [lambda0/D]
mp.Fend.score.Rout = 9  # outer radius of dark hole scoring region [lambda0/D]
mp.Fend.score.ang = 180  # angular opening of dark hole scoring region [degrees]

mp.Fend.sides = 'right'  # Which side(s) for correction: 'left', 'right', 'top', 'up', 'bottom', 'down', 'lr', 'rl', 'leftright', 'rightleft', 'tb', 'bt', 'ud', 'du', 'topbottom', 'bottomtop', 'updown', 'downup'

# %% Optical Layout: Compact Model (and Jacobian Model)

# Focal Lengths
mp.fl = 1.  # [meters] Focal length value used for all FTs in the compact model. Don't need different values since this is a Fourier model.

# Pupil Plane Diameters
mp.P2.D = 30e-3  # [meters]
mp.P3.D = mp.P2.D  # [meters]
mp.P4.D = mp.P2.D  # [meters]

# Pupil Plane Resolutions
mp.P1.compact.Nbeam = 100
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

mp.F3.compact.res = 3  # sampling of FPM for compact model [pixels per lambda0/D]

# %% Optical Layout: Full Model

# # Focal Lengths
# mp.fl = 1

# Pupil Plane Resolutions
mp.P1.full.Nbeam = mp.P1.compact.Nbeam
# mp.P2.full.Nbeam = 250
# mp.P3.full.Nbeam = 250
mp.P4.full.Nbeam = mp.P1.full.Nbeam  # P4 size must be the same as P1 for Vortex.

# Mask Definitions
mp.full = falco.config.Object()
mp.compact = falco.config.Object()

mp.F3.full.res = 4  # sampling of FPM for full model [pixels per lambda0/D]


# %% Entrance Pupil (P1) Definition and Generation

mp.whichPupil = 'ROMAN'  # Used only for run label
mp.P1.IDnorm = 0.303  # ID of the central obscuration [diameter]. Used only for computing the RMS DM surface from the ID to the OD of the pupil. OD is assumed to be 1.
mp.P1.ODnorm = 1.00  # Outer diameter of the telescope [diameter]
mp.P1.D = 2.3631  # circumscribed telescope diameter [meters]. Used only for converting milliarcseconds to lambda0/D or vice-versa.

mp.P1.full.mask = falco.mask.falco_gen_pupil_Roman_CGI_20200513(mp.P1.full.Nbeam, mp.centering)
mp.P1.compact.mask = falco.mask.falco_gen_pupil_Roman_CGI_20200513(mp.P1.compact.Nbeam, mp.centering)


# %% "Apodizer" (P3) Definition and Generation

mp.flagApod = False  # Whether to use an apodizer or not


# %% Lyot stop (P4) Definition and Generation

changes = {}
changes['flagLyot'] = True
changes['ID'] = 0.50
changes['OD'] = 0.80
changes['wStrut'] = 3.6/100  # nominal pupil's value is 76mm = 3.216%
changes['flagRot180'] = True

mp.P4.full.mask = falco.mask.falco_gen_pupil_Roman_CGI_20200513(mp.P4.full.Nbeam, mp.centering, changes=changes)
mp.P4.compact.mask = falco.mask.falco_gen_pupil_Roman_CGI_20200513(mp.P4.compact.Nbeam, mp.centering, changes=changes)
mp.P4.compact.maskAtP1res = falco.mask.falco_gen_pupil_Roman_CGI_20200513(mp.P1.compact.Nbeam, mp.centering, changes=changes)


# %% FPM (F3) Definition and Generation

mp.F3.Rin = 2.7  # maximum radius of inner part of the focal plane mask [lambda0/D]
mp.F3.RinA = 2.7  # inner hard-edge radius of the focal plane mask [lambda0/D]. Needs to be <= mp.F3.Rin 
mp.F3.Rout = np.inf  # radius of outer opaque edge of FPM [lambda0/D]
mp.F3.ang = 180  # on each side, opening angle [degrees]
mp.FPMampFac = 0  # amplitude transmission of the FPM

# Both models
fpmDict = {}
fpmDict['rhoInner'] = mp.F3.Rin  # radius of inner FPM amplitude spot (in lambda_c/D)
fpmDict['rhoOuter'] = mp.F3.Rout  # radius of outer opaque FPM ring (in lambda_c/D)
fpmDict['centering'] = mp.centering
fpmDict['FPMampFac'] = mp.FPMampFac  # amplitude transmission of inner FPM spot
# Full model
fpmDict['pixresFPM'] = mp.F3.full.res
mp.F3.full.mask = falco.mask.gen_annular_fpm(fpmDict)
# Compact model
fpmDict['pixresFPM'] = mp.F3.compact.res
mp.F3.compact.mask = falco.mask.gen_annular_fpm(fpmDict)
