#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri

import proper


def prop_get_refradius(wf):
    """Function returns the radius of the reference sphere to which the current
    wavefront's phase is reference. 
    
    The reference radius is defined to be the distance from the pilot beam waist 
    to the current position. If the reference surface is planar (near field), 
    then the radius will be 0.0. Assuming that forward propagation occurs from 
    left to right, a negative radius indicates that the center of the reference 
    surface is to the right of the current position (e.g. the beam is converging).
    
    Parameters
    ----------
    wf : obj
        Wavefront class object
      
    Returns
    -------  
    float
        Radius of curvature of sphere to which the current wavefront's phase
        is referenced.
    """
    return wf.z - wf.z_w0
