#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri


import proper

def prop_get_gridsize(wf):
    """Function returns dimension of wavefront grid
    
    Parameter
    ---------
    wf : obj
        Wavefront class object
        
    Returns
    -------
    float
        Wavefront grid size
    """
    return wf.ngrid
