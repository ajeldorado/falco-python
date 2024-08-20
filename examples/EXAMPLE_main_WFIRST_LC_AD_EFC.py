# import sys
# sys.path.insert(0,"../")
from copy import deepcopy
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


# %% Set up the workspace

mp.runLabel = ('Series%04d_Trial%04d_%s' %
               (mp.SeriesNum, mp.TrialNum, mp.coro))

out = falco.setup.flesh_out_workspace(mp)


# %% Compute the scaling factor for the actuator commands in the AD-EFC cost function

# Check a subset of actuators to see what the max actuator effect is in the pupil plane
if np.any(mp.dm_ind == 1):
    mp.ctrl.ad.dm1_act_mask_for_jac_norm = np.eye(mp.dm1.Nact, dtype=bool).flatten()
if np.any(mp.dm_ind == 2):
    mp.ctrl.ad.dm2_act_mask_for_jac_norm = np.eye(mp.dm2.Nact, dtype=bool).flatten()

falco.ctrl.set_utu_scale_fac(mp)


# %% Perform the Wavefront Sensing and Control

falco.wfsc.loop(mp, out)


# %% Plot the output

falco.plot.plot_trial_output(out)

fnPickle = mp.runLabel + '_snippet.pkl'
falco.plot.plot_trial_output_from_pickle(fnPickle)
