#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper


def prop_rectangular_obscuration(wf, width, height, xc = 0.0, yc = 0.0, **kwargs):
    """Multiply the current wavefront by a rectangular obscuration.
    
    Parameters
    ----------
    wf : obj
        WaveFront class object
        
    width, height : float
        X and Y widths of the obscuration in meters (unless the norm switch is  
        specified, in which case these are normalized to the beam diameter) 
    
    xc, yc : float
        Center of obscuration relative to the center of the beam in meters 
        (unless norm is specified, in which case they are normalized relative 
        to the beam radius; the default is (0,0)
        
        
    Returns
    -------
        None
        Multiplies the wavefront array in "wf" object by a dark rectangular mask.

    
    Other Parameters
    ----------------
    NORM : bool
        Indicates that obscuration halfwidths and center are specified relative 
        to the beam radius.
        
    ROTATION : float
        Degrees counter-clockwise to rotate obscuration about its center.
    """

    norm = proper.switch_set("NORM",**kwargs)
        
    if "ROTATION" in kwargs:
        rotation = kwargs["ROTATION"]
    else:
        rotation = 0.0
        
    wf.wfarr *= proper.prop_shift_center(proper.prop_rectangle(wf, width, height, xc, yc, NORM = norm, ROTATION = rotation, DARK = True))
    
    return
