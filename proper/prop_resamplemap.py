#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#   Revised 5 March 2018 - John Krist - Added GRID=True to call to prop_cubic_conv


import proper
import numpy as np

if not proper.use_cubic_conv:
    from scipy.ndimage.interpolation import map_coordinates


def prop_resamplemap(wf, dmap, pixscale, xc, yc, xshift = 0., yshift = 0.):
    """Interpolate input map using cubic convolution onto grid with same size 
    and sampling as the current wavefront array. 
    
    Optionally shift the map. The new, resampled map replaces the old one.
    

    Returns
    -------
    dmap : numpy ndarray
        "map" is replaced with a new version of itself that has equal sampling
        as the current wavefront array

        
    Parameters
    ----------
    wf : obj
        WaveFront class object
        
    dmap : numpy ndarray
        Aberration map to be resampled
        
    pixscale : float
        Spacing of "map" in meters
        
    xc, yc : float
        Pixel coordinates of map center (0,0 is center of 1st pixel)
        
    xshift, yshift : float
        Amount of shift map in meters
        
        
    Notes
    -----
    Intended for internal use only.
    """
    n = proper.prop_get_gridsize(wf)
    
    x = np.arange(n, dtype = np.float64) - int(n/2)
    x = x * proper.prop_get_sampling(wf) / pixscale
    x += xc - xshift/pixscale
    
    y = np.arange(n, dtype = np.float64) - int(n/2)
    y = y * proper.prop_get_sampling(wf) / pixscale
    y += yc - xshift/pixscale

    if proper.use_cubic_conv:
        dmap = proper.prop_cubic_conv(dmap.T, x, y, GRID=True)
    else:
        xygrid = np.meshgrid(x,y)        
        dmap = map_coordinates(dmap.T, xygrid, order = 3, mode = "nearest")

    return dmap
