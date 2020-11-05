import sys
sys.path.insert(0,"../")

import numpy as np

import falco

import EXAMPLE_defaults_DST_LC_design as DEFAULTS
mp = DEFAULTS.mp


mp.path = falco.config.Object()

mp.path.falco = './'  # Location of FALCO
mp.path.proper = './'  # Location of the MATLAB PROPER library

## Output Data Directories (Comment these lines out to use defaults within falco-matlab/data/ directory.)
mp.path.config = './'  # Location of config files and minimal output files. Default is [mainPath filesep 'data' filesep 'brief' filesep]
mp.path.ws = './'  # (Mostly) complete workspace from end of trial. Default is [mainPath filesep 'data' filesep 'ws' filesep];


## Step 3: Overwrite default values as desired

## Special Computational Settings
mp.flagPlot = True
mp.flagMultiproc = False  # whether to use multiprocessing to parallelize some large computations
mp.Nthreads = 4  # Number of threads to use when using multiprocessing. If undefined, it is set to the max number of cores

# mp.propMethodPTP = 'mft';

# Record Keeping
mp.TrialNum = 1
mp.SeriesNum = 1

# Use just 1 wavelength for initial debugging of code
mp.fracBW = 0.01  # fractional bandwidth of the whole bandpass (Delta lambda / lambda0)
mp.Nsbp = 1  # Number of sub-bandpasses to divide the whole bandpass into for estimation and control
mp.Nwpsbp = 1

mp.estimator = 'perfect'

## Step 4: Generate the label associated with this trial

mp.runLabel = ('Series' + ('%04d'%(mp.SeriesNum)) + '_Trial' +
('%04d_'%(mp.TrialNum)) + mp.coro + '_' + mp.whichPupil + '_' +
str(np.size(mp.dm_ind)) + 'DM' + str(mp.dm1.Nact) + '_z' + str(mp.d_dm1_dm2) +
'_IWA' + str(mp.Fend.corr.Rin) + '_OWA' + str(mp.Fend.corr.Rout) + '_' +
str(mp.Nsbp) + 'lams' + str(round(1e9*mp.lambda0)) + 'nm_BW' +
str(mp.fracBW*100) + '_' + mp.controller)
print(mp.runLabel)


## Step 5: Perform the Wavefront Sensing and Control

out = falco.setup.flesh_out_workspace(mp)
falco.wfsc.loop(mp, out)
