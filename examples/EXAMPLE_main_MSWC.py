"""Simple functional example used to verify that FALCO runs correctly."""
from copy import deepcopy
import numpy as np

import falco

import EXAMPLE_config_MSWC as CONFIG

# %% Load the config file (a script)

mp = deepcopy(CONFIG.mp)


# %% Define directories for data output
# # Location of config files and minimal output files.
# # Default is mp.path.falco + 'data/brief/'
# mp.path.config = os.path.join(mp.path.falco, 'data', 'brief')
# # (Mostly) complete workspace from end of trial.
# # Default is mp.path.falco + 'data/ws/'
# mp.path.ws = os.path.join(mp.path.falco, 'data', 'ws')


# %% Overwrite values from config file if desired

# ## Special Computational Settings
mp.flagPlot = True
mp.flagParallel = False  # whether to use multiprocessing to parallelize some large computations
# mp.Nthreads = 2  # Number of threads to use when using multiprocessing.

# Record Keeping
mp.TrialNum = 1
mp.SeriesNum = 1

# Use just 1 wavelength for initial testing of code
mp.fracBW = 0.01       # fractional bandwidth of the whole bandpass (Delta lambda / lambda0)
mp.Nsbp = 1            # Number of sub-bandpasses to divide the whole bandpass into for estimation and control
mp.Nwpsbp = 1          # Number of wavelengths to used to approximate an image in each sub-bandpass

mp.Nitr = 5 #3  # Number of wavefront control iterations


# %% Perform the Wavefront Sensing and Control

mp.runLabel = ('Series%04d_Trial%04d_%s' %
               (mp.SeriesNum, mp.TrialNum, mp.coro))

out = falco.setup.flesh_out_workspace(mp)

# Apply a grid of spots to the input pupil to allow SNWC
block0 = np.ones((5, 5))
block0[2, 2] = 0.7
dotGrid = np.tile(block0, [51, 51])
dotGrid = falco.util.pad_crop(dotGrid, mp.P1.compact.Narr, extrapval=1)
# figure; imagesc(dotGrid); axis xy equal tight; colorbar;
for si in range(mp.Nsbp):
    wvl = mp.sbp_centers[si]
    mp.P1.compact.E[:, :, si] = dotGrid * np.ones((mp.P1.compact.Narr,
                                                   mp.P1.compact.Narr))
    for wi in range(mp.Nwpsbp):
        mp.P1.full.E[:, :, wi, si] = dotGrid * np.ones((mp.P1.full.Narr,
                                                        mp.P1.full.Narr))

falco.wfsc.loop(mp, out)


# # %% Plot the output

falco.plot.plot_trial_output(out)

# fnPickle = mp.runLabel + '_snippet.pkl'
# falco.plot.plot_trial_output_from_pickle(fnPickle)
