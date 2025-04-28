"""Simple functional example used to verify that FALCO runs correctly."""
import os

import numpy as np

import falco

HERE = os.path.dirname(os.path.abspath(__file__))
mp = falco.config.ModelParameters.from_yaml_file(os.path.join(HERE, 'hd_config.yaml'))

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

mp.Nitr = 3  # Number of wavefront control iterations

mp.controller = 'AD-EFC'
mp.ctrl.ad = falco.config.Object()
# mp.ctrl.ad.dv_max = 20  # max delta voltage step per iteration. must be positive
mp.ctrl.ad.maxiter = 30
mp.ctrl.ad.iprint = 1
mp.ctrl.ad.maxfun = 1000000


mp.ctrl.log10regVec = [-5.5,]  # [-12]  # np.array([-6, ])

# mp.controller = 'gridsearchEFC'
# mp.ctrl.log10regVec = np.arange(-6, -2+0.5, 0.5)


# Use least-squares surface fitting instead of back-propagation model.
mp.dm1.useDifferentiableModel = True
mp.dm2.useDifferentiableModel = True
# mp.dm1.surfFitMethod = 'lsq'
# mp.dm2.surfFitMethod = 'lsq'
# mp.dm1.surfFitMethod = 'lsq'
# mp.dm2.surfFitMethod = 'lsq'

# %% Perform the Wavefront Sensing and Control

mp.runLabel = ('Series%04d_Trial%04d_%s' %
               (mp.SeriesNum, mp.TrialNum, mp.coro))

out = falco.setup.flesh_out_workspace(mp)


# mp.dm1.V = 3*np.random.rand(mp.dm1.Nact, mp.dm1.Nact)
# mp.dm2.V = 3*np.random.rand(mp.dm1.Nact, mp.dm1.Nact)


# %% Compute the scaling factor for the actuator commands in the AD-EFC cost function

# Check a subset of actuators to see what the max actuator effect is in the pupil plane
if np.any(mp.dm_ind == 1):
    mp.ctrl.ad.dm1_act_mask_for_jac_norm = np.eye(mp.dm1.Nact, dtype=bool).flatten()
if np.any(mp.dm_ind == 2):
    mp.ctrl.ad.dm2_act_mask_for_jac_norm = np.eye(mp.dm2.Nact, dtype=bool).flatten()

falco.ctrl.set_utu_scale_fac(mp)

falco.wfsc.loop(mp, out)


# %% Plot the output

falco.plot.plot_trial_output(out)

fnPickle = os.path.join(mp.path.brief, f'{mp.runLabel}_snippet.pkl')
falco.plot.plot_trial_output_from_pickle(fnPickle)
