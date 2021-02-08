#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri

import proper

def prop_get_z(wf):
    """Return the distance from the initialization of the wavefront to the 
    current surface in meters.
    
    Parameters
    ----------
    wf : obj
        WaveFront class object
        
    Returns
    -------
    float
        Distance from intialization of the wavefront to the current surface
    
    """
    return wf.z
