#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np


def prop_get_phase(wf):
    """Function returns the phase of the current wavefront in radians.
    
    Parameters
    ----------
    wf : obj
        Wavefront class object
        
    Returns
    -------
    numpy ndarray
        A 2D image corresponding to the phase of the current wavefront
    """
    return proper.prop_shift_center(np.arctan2(wf.wfarr.imag, wf.wfarr.real))
