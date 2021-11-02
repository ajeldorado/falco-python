"""
Script to run the high-order WFSC with the Roman CGI's SPC-Spec mode.

Requires the Roman CGI Phase C model and data from
https://sourceforge.net/projects/cgisim/
"""
import numpy as np
import copy
from astropy.io import fits
import matplotlib.pyplot as plt

import falco
import proper

import EXAMPLE_config_Roman_CGI_SPC_Spec_Band3 as CONFIG

# %% Load the config file (a script)
mp = CONFIG.mp


# %% Define directories for data output
# #
# # Location of config files and minimal output files.
# mp.path.config =   # Default is mp.path.falco + 'data/brief/'
# #
# # (Mostly) complete workspace from end of trial.
# mp.path.ws =  # Default is mp.path.falco + 'data/ws/'


# %% Overwrite values from config file if desired

# ## Special Computational Settings
mp.flagPlot = True
mp.flagMultiproc = False  # whether to use multiprocessing to parallelize some large computations
# mp.Nthreads = 2  # Number of threads to use when using multiprocessing.

# Record Keeping
mp.TrialNum = 1
mp.SeriesNum = 1

# QUICK CHECK WITH JUST 1 WAVELENGTH AND POLARIZATION
mp.fracBW = 0.01  # fractional bandwidth of the whole bandpass (Delta lambda / lambda0)
mp.Nsbp = 1  # Number of sub-bandpasses to divide the whole bandpass into for estimation and control
mp.Nwpsbp = 1  # Number of wavelengths to used to approximate an image in each sub-bandpass
mp.full.pol_conds = [10, ]
mp.Nitr = 3  # Number of wavefront control iterations


# %% Perform an idealized phase retrieval (get the E-field directly)

optval = copy.copy(mp.full)
optval.source_x_offset = 0
optval.use_dm1 = True
optval.dm1_m = mp.full.dm1.flatmap
optval.use_dm2 = True
optval.dm2_m = mp.full.dm2.flatmap
optval.end_at_fpm_exit_pupil = True
optval.use_fpm = False
nout = 1024
optval.output_dim = 1024
optval.use_pupil_mask = False  # No SPM for getting initial phase

if mp.Nsbp == 1:
    lambdaFacs = np.array([1.])
else:
    lambdaFacs = np.linspace(1-mp.fracBW/2, 1+mp.fracBW/2, mp.Nsbp)

# Get the Input Pupil's E-field
nCompact = falco.util.ceil_even(mp.P1.compact.Nbeam + 1)
mp.P1.compact.E = np.ones((nCompact, nCompact, mp.Nsbp), dtype=complex)
for iSubband in range(mp.Nsbp):

    lambda_um = 1e6*mp.lambda0*lambdaFacs[iSubband]
    fieldFull, sampling = proper.prop_run('roman_phasec', lambda_um, nout,  QUIET=True, PASSVALUE=optval.__dict__)
    if mp.flagPlot:
        plt.figure(1); plt.imshow(np.angle(fieldFull)); plt.colorbar(); plt.hsv(); plt.pause(1e-2)
        plt.figure(2); plt.imshow(np.abs(fieldFull)); plt.colorbar(); plt.magma(); plt.pause(0.5)

    # phIn = np.angle(fieldFull);
    # [phOut, _] = falco.zern.removeZernikes(phIn, [0 1 1], [0 1 -1], falco.util.ampthresh(fieldFull))
    # fieldFull = np.abs(fieldFull) * np.exp(1j*phOut)

    fieldCompactReal = falco.mask.rotate_shift_downsample_pupil_mask(
        np.real(fieldFull), mp.P1.full.Nbeam, mp.P1.compact.Nbeam, 0, 0, 0)
    fieldCompactImag = falco.mask.rotate_shift_downsample_pupil_mask(
        np.imag(fieldFull), mp.P1.full.Nbeam, mp.P1.compact.Nbeam, 0, 0, 0)
    fieldCompact = fieldCompactReal + 1j*fieldCompactImag

    fieldCompact = falco.util.pad_crop(fieldCompact, (nCompact, nCompact))
    mp.P1.compact.E[:, :, iSubband] = falco.prop.relay(fieldCompact, 1, centering=mp.centering)

    if mp.flagPlot:
        plt.figure(11); plt.imshow(np.angle(fieldCompact)); plt.colorbar(); plt.hsv(); plt.pause(1e-2)
        plt.figure(12); plt.imshow(np.abs(fieldCompact)); plt.colorbar(); plt.magma(); plt.pause(0.5)

# Don't double count the pupil amplitude with the phase retrieval and a model-based mask
mp.P1.compact.mask = np.ones_like(mp.P1.compact.mask)


# %% Generate the label associated with this trial
mp.runLabel = ('Series%04d_Trial%04d_%s' %
               (mp.SeriesNum, mp.TrialNum, mp.coro))
print(mp.runLabel)


# %% Perform the Wavefront Sensing and Control
out = falco.setup.flesh_out_workspace(mp)
falco.wfsc.loop(mp, out)


# %% Plot the output
falco.plot.plot_trial_output(out)

# fnPickle = mp.runLabel + '_snippet.pkl'
# falco.plot.plot_trial_output_from_pickle(fnPickle)
