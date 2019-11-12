#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper


def prop_get_sampling_radians(wf):
    """Function determines current wavefront sampling in radians/pixel.
    
    This funtion is only valid when the current wavefront is at focus.
    
    Parameters
    ----------
    wf : obj
       Wavefront class object
       
    Returns
    -------
    float
       Current wavefront sampling in radians/pixel 
    """
    fl = proper.prop_get_fratio(wf) * wf.diam
    return proper.prop_get_sampling(wf) / fl
