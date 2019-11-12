#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np


def prop_state(wf):
    """Save the current state for the current wavelength, if one doesn't already 
    exist. 
    
    If one does, read it in and use it to define the current wavefront. The 
    current contents of the wavefront array structure and some other info 
    necessary to represent the current state of propagation are saved to a file.
    
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
        
    # If the state for the current wavelength does not exist, write it out
    if not proper.prop_is_statesaved(wf):
        proper.prop_savestate(wf)
        return
        
    # If it does exist, read it in
    statefile_lam = str(int(wf.lamda*1.e15)).strip() + proper.statefile + '.npz'
    
    proper.total_original_pupil = 0.
    proper.n = int(0)
    proper.first_pass = np.fix(0)
    
    statedata = np.load(statefile_lam)
    proper.total_original_pupil = statedata["total_original_pupil"]
    wf = statedata["wf"].tolist()
    proper.n = wf.wfarr.shape[0]
        
    return
