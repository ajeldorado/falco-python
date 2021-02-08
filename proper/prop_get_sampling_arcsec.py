#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np


def prop_get_sampling_arcsec(wf):
    """Function determines current wavefront sampling in arcseconds/pixel. 
    
    This is only valid when the current wavefront is at focus.
    
    Parameters
    ----------
    wf : obj
        Wavefront class object
        
    Returns
    -------
    float
        The current wavefront sampling in arcseconds/pixel    
    """
    fl = proper.prop_get_fratio(wf) * wf.diam
    return proper.prop_get_sampling(wf) * 360. * 3600. / (2. * np.pi * fl)
