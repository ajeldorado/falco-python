#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri


import proper

def prop_get_fratio(wf):
    """Function to compute the current beam's focal ratio by dividing the 
    current distance to focus by the current beam diameter.
    
    Parameters
    ----------
    wf : obj
      Wavefront class object
      
    Returns
    -------
    float
        Current beam's focal ratio
    """
    return wf.current_fratio
