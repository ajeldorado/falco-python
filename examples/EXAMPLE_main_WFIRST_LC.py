# import sys
# sys.path.insert(0,"../")

import numpy as np

import falco

import EXAMPLE_defaults_WFIRST_LC as DEFAULTS

# %% Load default values from config script file
mp = DEFAULTS.mp

# %% Overwrite default values as desired

mp.path = falco.config.Object()

mp.path.falco = './'  #--Location of FALCO
mp.path.proper = './' #--Location of the MATLAB PROPER library

# Output Data Directories (Comment these lines out to use defaults within falco-matlab/data/ directory.)
mp.path.config = './' #--Location of config files and minimal output files. Default is [mainPath filesep 'data' filesep 'brief' filesep]
mp.path.ws = './' # (Mostly) complete workspace from end of trial. Default is [mainPath filesep 'data' filesep 'ws' filesep];

# Special Computational Settings
mp.flagPlot = True;
mp.flagParallel = False; #--whether to use multiprocessing to parallelize some large computations
mp.Nthreads = 4         #--Number of threads to use when using multiprocessing. If undefined, it is set to the max number of cores

# Record Keeping
mp.TrialNum = 1;
mp.SeriesNum = 1;

# Use just 1 wavelength for initial debugging of code
mp.fracBW = 0.01  # fractional bandwidth of the whole bandpass (Delta lambda / lambda0)
mp.Nsbp = 1  # Number of sub-bandpasses to divide the whole bandpass into for estimation and control
mp.Nwpsbp = 1

# %% Generate the label associated with this trial
mp.runLabel = ('Series%04d_Trial%04d_%s' %
               (mp.SeriesNum, mp.TrialNum, mp.coro))
print(mp.runLabel)


# %% Step 5: Perform the Wavefront Sensing and Control

out = falco.setup.flesh_out_workspace(mp)

falco.wfsc.loop(mp, out)

# %% Plot the output
falco.plot.plot_trial_output(out)

fnPickle = mp.runLabel + '_snippet.pkl'
falco.plot.plot_trial_output_from_pickle(fnPickle)
