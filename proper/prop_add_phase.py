#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
# Modified 18 April 2019 by J. Krist (switched to *=)


import proper
import numpy as np


def prop_add_phase(wf, phase_error):
    """Add a phase error map or value to the current wavefront array. 
    
    The phase error array is assumed to be at the same sampling as the current
    wavefront. Note that this is wavefront, not surface error.
    
    Parameters
    ----------
    wf : object
        WaveFront class object
        
    phase_error : numpy ndarray
        A scalar or 2D image containing the phase error in meters
        
    Returns
    -------
        None
    """
    i = complex(0., 1.)
    
    if type(phase_error) != np.ndarray and type(phase_error) != list:
        phase_error = float(phase_error)
        wf.wfarr *= np.exp(2*np.pi*i/wf.lamda*phase_error)
    else:
        phase_error = np.asarray(phase_error)
        wf.wfarr *= np.exp(2*np.pi*i/wf.lamda*proper.prop_shift_center(phase_error))

    return
