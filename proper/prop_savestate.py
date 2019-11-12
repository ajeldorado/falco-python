#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np


def prop_savestate(wf):
    """Write out the current wavefront state to a file for the current wavelength.
    
    Parameters
    ----------
    wf : obj
        WaveFront class object
        
    Returns
    -------
        None
    """
    if proper.save_state == False:
        return
        
    statefile_lam = str(int(wf.lamda*1.e15)).strip() + proper.statefile
    
    np.savez(statefile_lam, total_original_pupil = proper.total_original_pupil, wf = wf)
        
    proper.save_state_lam.append(wf.lamda)
    
    return
    
