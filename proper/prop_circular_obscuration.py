#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper


def prop_circular_obscuration(wf, radius, xc = 0.0, yc = 0.0, **kwargs):
    """Multiply the wavefront by a circular obscuration.
    
    Parameters
    ----------
    wf : obj
        WaveFront class object
    
    radius : float
       Radius of aperture in meters, unless norm is specified
      
    xc : float
        X-center of aperture relative to center of wavefront. Default is 0.0
        
    yc : float
        Y-center of aperture relative to center of wavefront. Default is 0.0
        
        
    Returns
    -------
    Multiplies current wavefront in "wf" by a circular obscuration (0 inside, 
    1 outside).    
        
        
    Other Parameters
    ----------------
    NORM : bool
        If set to True, the specified radius and xc, yc aperure centers are 
        assumed to be normalized to the current beam radius (e.g. radius is 1.0
        means the aperture is the same size as  the current beam). xc, yc = 0,0 
        is the center of the wavefront. Default is False.
    """

    norm = proper.switch_set("NORM",**kwargs)
    wf.wfarr *= proper.prop_shift_center(proper.prop_ellipse(wf, radius, radius, xc, yc, NORM = norm, DARK = True))
        
    return
