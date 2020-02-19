import sys
sys.path.insert(0,"../")
sys.path.insert(0,"../tests/")
import falco
import proper
import numpy as np
import testing_defaults_WFIRST_LC as DEFAULTS
import matplotlib.pyplot as plt

mp = DEFAULTS.mp

flagPlotDebug = False

mp.path = falco.config.Object()
mp.path.falco = './'  #--Location of FALCO
mp.path.proper = './' #--Location of the MATLAB PROPER library

##--Output Data Directories (Comment these lines out to use defaults within falco-matlab/data/ directory.)
mp.path.config = './' #--Location of config files and minimal output files. Default is [mainPath filesep 'data' filesep 'brief' filesep]
mp.path.ws = './' # (Mostly) complete workspace from end of trial. Default is [mainPath filesep 'data' filesep 'ws' filesep];


## Step 3: Overwrite default values as desired

mp.dm1.xtilt = 45
mp.dm1.ytilt = 20
mp.dm1.zrot = 30

mp.dm1.xc = mp.dm1.Nact/2 - 1/2 + 1;
mp.dm1.yc = mp.dm1.Nact/2 - 1/2 -1;


## Step 4: Initialize the rest of the workspace

out = falco.setup.flesh_out_workspace(mp)


# %% Generate a DM surface and try to re-create the actuator commands

normFac = 1;
mp.dm1.V = normFac*np.random.rand(mp.dm1.Nact,mp.dm1.Nact)
DM1Surf =  falco.dms.falco_gen_dm_surf(mp.dm1, mp.dm1.compact.dx, mp.dm1.compact.Ndm)

if(flagPlotDebug):
    plt.figure(1); plt.imshow(mp.dm1.V); plt.colorbar(); plt.pause(0.1);
    plt.figure(2); plt.imshow(DM1Surf); plt.colorbar(); plt.pause(0.1);

#--Fit the surface
# DMSurf = padOrCropEven(DMSurf,500);
Vout = falco.dms.falco_fit_dm_surf(mp.dm1,DM1Surf)/mp.dm1.VtoH
Verror = mp.dm1.V - Vout;
rmsVError = np.sqrt(np.mean(Verror.flatten()**2))/normFac;
print('RMS fitting error to voltage map is %.2f%%.\n'%(rmsVError*100))

if(flagPlotDebug):
    plt.figure(3); plt.imshow(Vout); plt.colorbar(); plt.pause(0.1);
    plt.figure(4); plt.imshow(Verror); plt.colorbar(); plt.pause(0.1);
