#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#   Modified by J. Krist on 19 April 2019: changed to /=



import proper
import numpy as np


def prop_divide(wf, value):
    """Divide the current wavefront amplitude by a user-provided value or array.
    
    Parameters
    ----------
    wf : obj
        The current WaveFront class object
        
    value : numpy ndarray
        Either a scalar or a 2-D array containing the amplitude map by which
        the current wavefront will be divided.  The map is assumed to be 
        centered at pixel (n/2,n/2).
    
    Returns
    -------
        None
    """
    if type(value) != np.ndarray and type(value) != list:
        value = float(value)
        wf.wfarr /= complex(value, 0.)
    else:
        value = np.asarray(value)
        wf.wfarr /= proper.prop_shift_center(value)
        
    return
