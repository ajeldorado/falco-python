#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper


def prop_elliptical_aperture(wf, xradius, yradius, xc = 0.0, yc = 0.0, **kwargs):
    """Multiply the wavefront by an elliptical clear aperture.
    
    Parameters
    ----------
    wf : obj
        WaveFront class object
    
    xradius : float
       X ellipse radius in meters, unless norm is specified
       
    yradius : float
       Y ellipse radius in meters, unless norm is specified
      
    xc : float
        X-center of aperture relative to center of wavefront. Default is 0.0
        
    yc : float
        Y-center of aperture relative to center of wavefront. Default is 0.0
    
    
    Returns
    -------
        None
        Multiplies current wavefront in wf object by an elliptical aperture
    
    
    Other Parameters
    ----------------
    NORM : bool
        If set to True, the specified radii and aperure center are assumed to 
        be normalized to the current beam radius. Default is False.

    ROTATION : float
        Angle in degrees to rotate the ellipse about its center.
    """

    norm = proper.switch_set("NORM",**kwargs)

    if "ROTATION" in kwargs:
        rotation = kwargs["ROTATION"]
    else:
        rotation = 0.0
    
    wf.wfarr *= proper.prop_shift_center(proper.prop_ellipse(wf, xradius, yradius, xc, yc, ROTATION=rotation, NORM=norm))
        
    return
