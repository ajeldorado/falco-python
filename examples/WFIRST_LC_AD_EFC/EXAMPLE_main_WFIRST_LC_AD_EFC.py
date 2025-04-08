# import sys
# sys.path.insert(0,"../")
from copy import deepcopy
import os
import time

import numpy as np

import falco

import EXAMPLE_config_WFIRST_LC_AD_EFC as CONFIG

# %% Load/run config script
mp = deepcopy(CONFIG.mp)


# %% Set Output Data Directories
mp.path = falco.config.Object()
# mp.path.config = './'  # Location of config files and minimal output files. Default is [mainPath filesep 'data' filesep 'brief' filesep]
# mp.path.ws = './'  # (Mostly) complete workspace from end of trial. Default is [mainPath filesep 'data' filesep 'ws' filesep];

# %% Overwrite default values as desired

# Special Computational Settings
mp.flagPlot = True
mp.flagParallel = False   # whether to use multiprocessing to parallelize some large computations
mp.Nthreads = 4  # Number of threads to use when using multiprocessing. If undefined, it is set to the max number of cores

# Record Keeping
mp.TrialNum = 1
mp.SeriesNum = 1

# Use just 1 wavelength for initial debugging of code
mp.fracBW = 0.01  # fractional bandwidth of the whole bandpass (Delta lambda / lambda0)
mp.Nsbp = 1  # Number of sub-bandpasses to divide the whole bandpass into for estimation and control
mp.Nwpsbp = 1

# # Use least-squares surface fitting instead of back-propagation model.
# mp.dm1.useDifferentiableModel = False
# mp.dm2.useDifferentiableModel = False
# mp.dm1.surfFitMethod = 'lsq'
# mp.dm2.surfFitMethod = 'lsq'

mp.ctrl.sched_mat = np.array([
    [1, -2, 12, 1, 0],
    [1, -2, 12, 1, 0],
    [1, -4, 12, 1, 0],
    [1, -4, 12, 1, 0],
    [1, -2, 12, 1, 0],
    ])
mp.Nitr, mp.relinItrVec, mp.gridSearchItrVec, mp.ctrl.log10regSchedIn, mp.dm_ind_sched = falco.ctrl.efc_schedule_generator(mp.ctrl.sched_mat)


# %% Set up the workspace

mp.runLabel = ('ADEFC_Series%04d_Trial%04d_%s' %
               (mp.SeriesNum, mp.TrialNum, mp.coro))

out = falco.setup.flesh_out_workspace(mp)


# %% Compute the scaling factor for the actuator commands in the AD-EFC cost function

# Check a subset of actuators to see what the max actuator effect is in the pupil plane
if np.any(mp.dm_ind == 1):
    mp.ctrl.ad.dm1_act_mask_for_jac_norm = np.eye(mp.dm1.Nact, dtype=bool).flatten()
if np.any(mp.dm_ind == 2):
    mp.ctrl.ad.dm2_act_mask_for_jac_norm = np.eye(mp.dm2.Nact, dtype=bool).flatten()

falco.ctrl.set_utu_scale_fac(mp)


# %% Calculate and use the Jacobian just upfront to weed out weak actuators

cvar = falco.config.Object()
cvar.Itr = 0
cvar.flagRelin = True

falco.setup.falco_set_jacobian_modal_weights(mp)

# Compute the control Jacobians for each DM
jacStruct = falco.model.jacobian(mp)

falco.ctrl.cull_weak_actuators(mp, cvar, jacStruct)
falco.ctrl.init(mp, cvar)


# %% Perform the Wavefront Sensing and Control
tStart = time.time()
falco.wfsc.loop(mp, out)
tStop = time.time()

tDiff = tStop-tStart
print(tDiff)

# %% Plot the output

falco.plot.plot_trial_output(out)

fnPickle = os.path.join(mp.path.brief, f'{mp.runLabel}_snippet.pkl')
falco.plot.plot_trial_output_from_pickle(fnPickle)
