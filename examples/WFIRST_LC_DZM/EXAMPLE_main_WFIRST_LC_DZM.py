# import sys
# sys.path.insert(0,"../")
from copy import deepcopy
import os
import numpy as np
import pickle


import falco

import EXAMPLE_config_WFIRST_LC_ADDZM as CONFIG

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
mp.SeriesNum = 3

# Use just 1 wavelength for initial debugging of code
mp.fracBW = 0.01  # fractional bandwidth of the whole bandpass (Delta lambda / lambda0)
mp.Nsbp = 1  # Number of sub-bandpasses to divide the whole bandpass into for estimation and control
mp.Nwpsbp = 1

# Estimator stuff:
mp.est.probe.whichDM = 2 #--Which DM is used for dither/control
mp.est.dither = 0.04 #--std dev of dither command for random dither [V/sqtr(iter)]
mp.est.itr_ol = np.arange(0, mp.Nitr) #--"open-loop" iterations where an image is taken with initial DM command + drift command
mp.est.itr_reset = [mp.Nitr+1]
mp.est.flagUseJacAlgDiff = True

#-- DM settings
# mp.dm1.V_dz = mp.dm1.V #--DM command that generates the initial dark zone
# mp.dm2.V_dz = mp.dm2.V

# TODO: Do these nede to be lists?
mp.dm1.V_drift = np.zeros((mp.dm1.Nact, mp.dm1.Nact)) #--Drift injected, initialize to 0
mp.dm2.V_drift = np.zeros((mp.dm2.Nact, mp.dm2.Nact))

mp.dm1.V_shift = np.zeros((mp.dm1.Nact, mp.dm1.Nact)) #--DM shift command for estimator reset to avoid linearization / phase wrapping errors, initialize to zero
mp.dm2.V_shift = np.zeros((mp.dm2.Nact, mp.dm2.Nact))

# %% Perform the Wavefront Sensing and Control

mp.runLabel = ('DZM_Series%04d_Trial%04d_%s' %
               (mp.SeriesNum, mp.TrialNum, mp.coro))

# TODO: add OL to this?
out = falco.setup.flesh_out_workspace(mp)

##---- Initial state
startSoln_TrialNum = 1
startSoln_SeriesNum = 1
startSoln_coro = 'LC'
startSoln_runLabel = ('Series%04d_Trial%04d_%s' %
                      (startSoln_SeriesNum, startSoln_TrialNum, startSoln_coro))

fnPickle = os.path.join(mp.path.brief, f'{startSoln_runLabel}_snippet.pkl')
with open(fnPickle, 'rb') as pickle_file:
    startSoln_out = pickle.load(pickle_file)

mp.dm1.V_dz = startSoln_out.dm1.Vall[:, :, -1] #np.zeros((mp.dm1.Nact, mp.dm1.Nact)) #--Drift injected, initialize to 0
mp.dm2.V_dz = startSoln_out.dm2.Vall[:, :, -1] #np.zeros((mp.dm2.Nact, mp.dm2.Nact))

##----



falco.wfsc.loop(mp, out)


# %% Plot the output

falco.plot.plot_trial_output(out)

fnPickle = os.path.join(mp.path.brief, f'{mp.runLabel}_snippet.pkl')
falco.plot.plot_trial_output_from_pickle(fnPickle)
