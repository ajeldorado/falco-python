#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper


def prop_rectangular_aperture(wf, width, height, xc =0., yc = 0., **kwargs):
    """Multiply the current wavefront by a rectangular aperture.
    
    Parameters
    ----------
    wf : obj
        WaveFront class object
        
    width, height : float
        X and Y widths of aperture in meters, unless the norm switch is 
        specified, in which case the are normalized to the beam diameter.
        
    xc, yc : float
        Center of aperture relative to the center of the beam in meters (unless 
        norm is specified); the default is (0,0).
    
    
    Returns
    -------
        None
        Multiplies the wavefront array in "wf" object by a rectangular mask.

    
    Other Parameters
    ----------------
    NORM : bool
        Indicates that the given aperture sizes and the centers are normalized 
        to the beam radius.
        
    ROTATION : float
        Angle degrees counter-clockwise to rotate rectangular aperture about 
        its center. Default is 0 degrees.        
    """

    norm = proper.switch_set("NORM",**kwargs)
        
    if "ROTATION" in kwargs:
        rotation = kwargs["ROTATION"]
    else:
        rotation = 0.0
    
    wf.wfarr *= proper.prop_shift_center(proper.prop_rectangle(wf, width, height, xc, yc, NORM = norm, ROTATION = rotation))
    
    return    
