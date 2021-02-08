#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri


import proper

def prop_get_distancetofocus(wf):
    """Function to determine distance to focus (in meters) from current location.
    
    Parameters
    ----------
    wf : obj
        Wavefront class object
        
    Returns
    -------
    distance to focus : float
        Distance to focus in meters
    """
    return wf.z_w0 - wf.z
