#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np


def prop_radius(wf, **kwargs):
    """Return a 2D array in which the value of each element corresponds to the 
    distance of that element from the center of the current wavefront. 
    
    By default, the distance is in meters, unless the /NORM switch is set, in
    which case it is normalize to the current radius of the beam. The center 
    of the wavefront is set to be at the center of the array.
    
    
    Parameters
    ----------
    wf : obj
        WaveFront class object
        
    Other Parameters
    ----------------
    NORM : bool
        Indicates that the returned array contains the distances divided by
        the beam radius.  This assumes the radius of the pilot tracer beam
        accurately reflects the size of the actual beam in the wavefront
        array, which will not be true in the case of significant aberrations.
    
    Returns
    -------
    r : numpy ndarray
        A 2D array.
    """
    ngrid = proper.prop_get_gridsize(wf)
    sampling = proper.prop_get_sampling(wf)
    beamradius = proper.prop_get_beamradius(wf)
    
    
    r = np.zeros([ngrid, ngrid], dtype = np.float64)
    x2 = (np.arange(float(ngrid)) - int(ngrid/2))**2
    
    # Using map and lambda
    #f = lambda j: np.sqrt(x2 + float(j - ngrid/2.)**2)
    #r[:,:] = map(f, range(ngrid))
    
    # Using list comprehension - which seems to be faster
    #r[:,:] = [np.sqrt(x2 + float(j - ngrid/2.)**2) for j in range(ngrid)]
    
    # Using conventional for loop
    for j in range(ngrid):
        r[j,:] = np.sqrt(x2 + float(j - int(ngrid/2))**2)
        
    if proper.switch_set("NORM",**kwargs):
        r *= (sampling/beamradius)
    else:
        r *= sampling
    
    return r
