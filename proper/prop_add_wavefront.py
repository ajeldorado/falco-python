#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np


def prop_add_wavefront(wfo, wf):
    """Add a wavefront the current wavefront array. 
    
    The wavefront array is assumed to be at the same sampling as the current 
    wavefront. 
              
    Parameters
    ----------
    wfo : object
        WaveFront class object
        
    wf : numpy ndarray
        A scalar or 2D image containing the value or wavefront to add. 
        NOTE: All responsibility is on the user to ensure that the two fields 
        have the same sampling and reference phase curvature. NO CHECKING is 
        done by this routine.
        
    Returns
    -------
        None
    """  
    if type(wf) != np.ndarray and type(wf) != list:
        wfo.wfarr += wf
    else:
        wfo.wfarr += proper.prop_shift_center(wf)
     
    return
