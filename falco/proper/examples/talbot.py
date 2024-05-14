#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np


def talbot(wavelength, gridsize, PASSVALUE = {'period': 0., 'diam': 0., 'dist': 0.}):
    talbot_length = 2. * PASSVALUE['period']**2 / wavelength
    
    wfo = proper.prop_begin(PASSVALUE['diam'], wavelength, gridsize)
    
    # create 1-D grating pattern
    m = 0.2
    x = (np.arange(gridsize, dtype = np.float64) - gridsize/2)  \
                * proper.prop_get_sampling(wfo)
                
    grating = 0.5 * (1 + m * np.cos(2*np.pi*x/PASSVALUE['period']))
    
    # create 2-D amplitude grating pattern
    grating = np.dot(grating.reshape(gridsize,1), np.ones([1,gridsize], dtype = np.float64))
    
    proper.prop_multiply(wfo, grating)
    
    proper.prop_define_entrance(wfo)
    
    proper.prop_propagate(wfo, PASSVALUE['dist'], TO_PLANE = True)
    
    (wfo, sampling) = proper.prop_end(wfo, NOABS = True)
    
    return (wfo, sampling)
