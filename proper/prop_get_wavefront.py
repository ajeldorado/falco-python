#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper


def prop_get_wavefront(wf):
    """Function returns current complex-values wavefront array.
    
    Parameters
    ----------
    wf : obj
        Wavefront class object
        
    Returns
    -------
    numpy ndarray
        A 2D, complex valued wavefront array centered in the array
    """
    return proper.prop_shift_center(wf.wfarr)
