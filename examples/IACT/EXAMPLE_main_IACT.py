# import sys
# sys.path.insert(0,"../")
from copy import deepcopy
import os
# import numpy as np

import falco

import falco_iact_config_BMC50F_vortex_20220210 as CONFIG

# %% Load/run config script
mp = deepcopy(CONFIG.mp)


# %% Set Output Data Directories
mp.path = falco.config.Object()
# mp.path.config = './'  # Location of config files and minimal output files. Default is [mainPath filesep 'data' filesep 'brief' filesep]
# mp.path.ws = './'  # (Mostly) complete workspace from end of trial. Default is [mainPath filesep 'data' filesep 'ws' filesep];


# TODO: actually set up tb
# %% Set up testbed object
# if(~exist('tb','var'))
#     tb = tb_config();
#     setUpAll(tb)



# %% Overwrite default values as desired

mp.testbed = 'iact'; # Name of the testbed
mp.flagSim = True;
if 'tb' in locals():
    tb.info.source = 'nkt';#'laser' or 'nkt';
    tb.info.darkMethod = 'FSblock';
    tb.DM.suppressRangeWarnings=True;
    tb.sciCam.writeRawFiles = False; # Turns off the auto-saving feature

    tb.info.do_StabilityTest = False; # Do the post-EFC stability test
    tb.info.do_LDFC = False; # Launch LDFC after EFC
    tb.info.do_DHM = False; # Launch dark hole maintainance


# %%--Special Computational Settings
mp.flagParfor = false; %- Uses parfor for jacobian calculation when true
mp.flagPlot = false;% Show FALCO plots

# %--Record Keeping
mp.SeriesNum = 8;
mp.TrialNum = 65;

# %%-- Use a previous DM solution
loadPreviousDMsoln = true;  % Set true to use a previous DM solution
prevSoln.SeriesNum = mp.SeriesNum; % Series number of previous DM solution
prevSoln.TrialNum = 63;%mp.TrialNum-1; % Trial number of previous DM solution
prevSoln.itNum = np.nan; % Iteration number for previous DM solution


# %--Modify the wavelength info
mp.lambda0 = 635e-9; #760e-9;    %--Central wavelength of the whole spectral bandpass [meters]
mp.fracBW = 0.1;       %--fractional bandwidth of the whole bandpass (Delta lambda / lambda0)
mp.Nsbp = 1;            %--Number of sub-bandpasses to divide the whole bandpass into for estimation and control
mp.Nwpsbp = 1;

# % Set the number of pixels per lambda/D
pixelPerLamOverD = 8
subwindowsize = 40 * pixelPerLamOverD
if 'tb' in locals():
    tb.info.pixelPerLamOverD = pixelPerLamOverD;   % #SFR TODO CHANGED THIS
    tb.info.lambda0 = 637e-9; % <- TODO FIX THIS
    tb.sciCam.pixelPerLamOverD = tb.info.pixelPerLamOverD*mp.lambda0/tb.info.lambda0;
    tb.sciCam.subwindowsize = subwindowsize; %1000; % Sub-window size [pixels]
    tb.info.OUT_DATA_DIR = data_dir;
    tb.FPM.whichMask = 'A';
    tb.FPM.maskProperties.A.charge = 6;

mp.Fend.res = pixelPerLamOverD; %--Sampling [ pixels per lambda0/D]
mp.Fend.eval.res = mp.Fend.res ; % SFR TO SAVE MEMORY
mp.Fend.FOV = (subwindowsize-2)/(2*mp.Fend.res); %--half-width of the field of view in both dimensions [lambda0/D]

# %% Perform the Wavefront Sensing and Control

mp.runLabel = ('Series%04d_Trial%04d_%s' %
               (mp.SeriesNum, mp.TrialNum, mp.coro))

out = falco.setup.flesh_out_workspace(mp)

falco.wfsc.loop(mp, out)


# %% Plot the output

falco.plot.plot_trial_output(out)

fnPickle = os.path.join(mp.path.brief, f'{mp.runLabel}_snippet.pkl')
falco.plot.plot_trial_output_from_pickle(fnPickle)
