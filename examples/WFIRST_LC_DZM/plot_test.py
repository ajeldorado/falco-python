import matplotlib.pyplot as plt

from copy import deepcopy
import os
import numpy as np
import pickle

import matplotlib as mpl
# mpl.use('Qt5Agg')  # interactive mode works with this, pick one
mpl.use('TkAgg')

import falco


##---- Initial state
startSoln_TrialNum = 1
startSoln_SeriesNum = 2
startSoln_coro = 'LC'
startSoln_runLabel = ('DZM_Series%04d_Trial%04d_%s' %
                      (startSoln_SeriesNum, startSoln_TrialNum, startSoln_coro))
path_brief = r'C:\Users\sredmond\Documents\github_repos\falco-python\data\brief'


fnPickle = os.path.join(path_brief, f'{startSoln_runLabel}_snippet.pkl')
with open(fnPickle, 'rb') as pickle_file:
    out = pickle.load(pickle_file)


plt.figure()

plt.semilogy(np.arange(0, out.Nitr+1, 1), out.IrawScoreHist, label='closed loop')
plt.semilogy(np.arange(0, out.Nitr, 1), out.IOLScoreHist, label='open loop')
plt.legend()

plt.show()
