#   Copyright 2019 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#  Written by J. Krist - 19 April 2019

import proper
import numpy as np
import os
import pyfftw
import pickle

def prop_load_fftw_wisdom( gridsize, nthreads ):

    wisdompath = os.path.join(proper.lib_dir, '.{}pix'.format(str(gridsize)) + '{}threads'.format(str(nthreads)) + '_wisdomfile')

    if os.path.exists(wisdompath):
        pyfftw.forget_wisdom()
        with open(wisdompath, 'rb') as infile:
            wisdom = pickle.load(infile)
            pyfftw.import_wisdom(wisdom)
        proper.fftw_use_wisdom = True
    else:
        proper.fftw_use_wisdom = False

    return
