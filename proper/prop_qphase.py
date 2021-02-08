#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#  Modified 18 April 2019 by J. Krist - switch to *=


import proper
import numpy as np


def prop_qphase(wf, c):
    """Apply a quadratic phase factor to the current wavefront. 
    
    This routine applies a radially-dependent phase alteration caused by a 
    wavefront curvature. It is used by lens, stw, and wts functions.
    
    
    Parameters
    ----------
    wf : obj
        WaveFront class object
        
    c : float
        Phase curvature radius
        
        
    Returns
    -------
        None 
        Modifies wavefront.
    """    
    ngrid = proper.prop_get_gridsize(wf)
    sampl = proper.prop_get_sampling(wf)
    
    if c == 0.0:
        return
        
    i = complex(0., 1.)
    
    dx = sampl
    
    xsqr = np.tile(((np.arange(ngrid, dtype = np.float64) - ngrid/2.) * dx)**2, (ngrid, 1))
    rsqr = xsqr + np.transpose(xsqr)
    rsqr[:,:] = np.roll(np.roll(rsqr, int(-ngrid/2), 0), int(-ngrid/2), 1)
    xsqr = 0.
    
    wf.wfarr *= np.exp(i*np.pi/(wf.lamda*c) * rsqr)
    
    return
