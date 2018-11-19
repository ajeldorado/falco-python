# 2018-08-09, D. Marx
# simple hack script to unit test model_jacobian_HLC.py and model_jacobian_HLC_pool.py
# load saved inputs from the matlab version
# call model_jacobian_HLC
# compare outputs to saved outputs from the matlab version

import sys, os
import numpy as np
import pdb

import scipy.io as sio

sys.path.append('/home/dmarx/src/Falco-jpl/FALCO-python/falco')
import config.init_from_mat
#from model_jacobian_HLC_pool import model_Jacobian_HLC
from model_jacobian_HLC import model_Jacobian_HLC

#Sin = config.init_from_mat.loadmat('main_HLC_5layerDisk_WFIRST_model_Jacobian_HLC_inputs_DM1.mat')
#Sin = config.init_from_mat.loadmat('main_HLC_5layerDisk_WFIRST_model_Jacobian_HLC_inputs_DM2.mat')
Sin = config.init_from_mat.loadmat(
    'main_HLC_5layerDisk_WFIRST_model_Jacobian_HLC_inputs_DM9_iter21.mat')

# some things need to be converted from scalars to arrays of size 1
Sin['mp']['ttx'] = np.array([Sin['mp']['ttx']])
Sin['mp']['tty'] = np.array([Sin['mp']['tty']])

# some things need to be converted from lists to np arrays
# Sin['mp']['P1']['compact']['E'] = np.array(Sin['mp']['P1']['compact']['E'])
# Sin['mp']['P2']['compact']['XsDL'] = np.array(Sin['mp']['P2']['compact']['XsDL'])
# Sin['mp']['P2']['compact']['YsDL'] = np.array(Sin['mp']['P2']['compact']['YsDL'])
# Sin['mp']['P1']['compact']['mask'] = np.array(Sin['mp']['P1']['compact']['mask'])

#model_Jacobian_HLC(Sin['mp'], Sin['DM'], tsi, Sin['whichDM'])
retGdm = pdb.runcall(model_Jacobian_HLC, Sin['mp'], Sin['DM'], Sin['tsi'], Sin['whichDM']) 

sio.savemat('deleteme.mat',{'retGdm':retGdm})

