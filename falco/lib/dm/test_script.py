# load input data from .mat file
# call falco_gen_dm_poke_cube
# compare output data to saved output data .mat file

import sys, os
import pdb
#import scipy.io as sio

#sys.path.append('y:/src/Falco/falco-python/falco/')
sys.path.append('/home/dmarx/src/Falco/falco-python/falco/')
import config.init_from_mat

from falco_gen_dm_poke_cube import * # falco_gen_dm_poke_cube

#Sin = sio.loadmat('inputs_TEMPLATE_main_LC_single_trial.mat',squeeze_me=True,struct_as_record=True)
Sin = config.init_from_mat.loadmat('inputs_TEMPLATE_main_LC_single_trial.mat')

listVars = Sin.keys()
# 'dm', 'mp', 'dx_dm', 'varargin'

# def RecordToDict(rec):
#     names = rec.dtype.names # returns a tuple
#     adict = dict()

#     #[adict.update({name:rec[name]}) for name in names]
#     # atmp = [(rec[name].astype('float64')).dtype for name in names]
#     # print(atmp)
#     for name in names:
#         atmp = rec[name]
#         print(name, ': ', atmp.dtype)
        
#     return adict

# dictDM = RecordToDict(Sin['dm'])
# dictMP = RecordToDict(Sin['mp'])

dictDM = Sin['dm']
dictMP = Sin['mp']

dx_dm  = Sin['dx_dm'] # float
flagGenCube = True if len(Sin['varargin']) == 0 else not Sin['varargin'].lower()=='nocube'

# more corrections to make useful
#dictDM['inf0'] = dictDM['inf0'].flatten()[0]
dictDM['inf0'] = np.array(dictDM['inf0'])

ret = pdb.runcall(falco_gen_dm_poke_cube, dictDM, dictMP, dx_dm, flagGenCube=flagGenCube)
#ret = falco_gen_dm_poke_cube(dictDM, dictMP, dx_dm, flagGenCube=flagGenCube)

# compare function return to Matlab outputs
Sout = config.init_from_mat.loadmat('outputs_TEMPLATE_main_LC_single_trial.mat')
Sout['dm']['inf_datacube'] = np.array(Sout['dm']['inf_datacube']) # make numpy array like ret['inf_datacube']

[key for key in Sout['dm'].keys() if key not in ret.keys()]
#
# ['xy_cent_act_inPix',
#  'xy_cent_act_box',
#  'xy_cent_act_box_inM',
#  'xy_box_lowerLeft',
#  'x_box0',
#  'Xbox0',
#  'Ybox0',
#  'inf_datacube',
#  'Xbox',
#  'Ybox'
# ]

#[np.all(ret[key]) == Sout['dm'][key] for key in ret.keys()]

def eDiff(a,b):
    return np.abs(a-b)/(0.5*(a+b))
    
for key in ret.keys():
    if type(ret[key]) is int or type(ret[key]) is np.int32 or type(ret[key]) is bool or type(ret[key]) is str:
        print(key, ret[key] == Sout['dm'][key])

    elif type(ret[key]) is list:
        print(key, ' is list')

    elif type(ret[key]) is float or type(ret[key]) is np.float64:
        print(key, ' %.3e'%(eDiff(ret[key],Sout['dm'][key])))

    elif type(ret[key]) is np.ndarray:
        print(key, ' %.3e'%(np.max(eDiff(ret[key],Sout['dm'][key]))))

    else:
        print(key, 'type: ', type(ret[key]))


        
# VtoH  is list
# NactTotal False
# x_pupPad  nan
# y_pupPad  nan
# act_ele  is list
# inf_datacube  2.000e+00
