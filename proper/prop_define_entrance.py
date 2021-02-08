#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np


def prop_define_entrance(wf):
    """Establish entrance aperture function for later use by pyPROPER routines.
    
    The input image describes the entrance aperture amplitude (0 to 1). This
    routine then normalizes the wavefront to have a total intensity of one.
    
    Parameters
    ----------
    wf : obj
        WaveFront class object
        
    Returns
    -------
        None
    """
    total_original_pupil = np.sum(np.abs(wf.wfarr)**2)
    proper.total_original_pupil = total_original_pupil

    wf.wfarr /= np.sqrt(total_original_pupil)

    return
