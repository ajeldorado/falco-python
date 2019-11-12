#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np


def prop_hex_aperture(wf, nrings, hexrad, hexsep, xc = 0.0, yc = 0.0, **kwargs):
    """
    Return an image containing a hexagonal mask consisting of multiple hexagons.
    This is useful for modeling systems with multisegmented mirrors, such as
    the Keck or JWST telescopes. The hexagons have antialiased edges. This 
    routines does not modify the wavefront.
    
    Parameters
    ----------
    wf : object
        WaveFront class object
        
    nrings : int
        Number of rings of hexagons in aperture (e.g. 1 = a central hexagon 
        surrounded by a ring of hexagons)
        
    hexrad : float
        The distance in meters from the center of a hexagon segment to a vertex.
        
    hexsep : float
        The distance between the centers of adjacent hexagons.
        
    xc, yc : float
        The offset in meters of the aperture from the center of the wavefront.  
        By default, the aperture is centered within the wavefront.
        
    Optional Keywords
    -----------------
    DARK : boolean
        If set, the central hexagonal segment will be set to 0.0.
    
    ROTATION : float
        The counterclockwise rotation in degrees of the aperture about its center.
        
    Returns
    -------
        numpy ndarray
        A hexagonal mask
    """
    # ngrid
    ngrid = wf.ngrid
    
    image = np.zeros([ngrid, ngrid], dtype = np.float64)
    
    if "ROTATION" in kwargs:
        angle = kwargs["ROTATION"]
        angle_rad = angle * np.pi/180.
    else:
        angle = 0.0
        angle_rad = 0.0
        
    for iring in range(0, nrings+1):
        x = hexsep * np.cos(30 * np.pi/180.) * iring
        y = -nrings * hexsep + iring * hexsep * 0.5
        for iseg in range(0, 2*nrings-iring+1):
            xhex = x * np.cos(angle_rad) - y * np.sin(angle_rad) + xc
            yhex = x * np.sin(angle_rad) + y * np.cos(angle_rad) + yc
            if (iring != 0 or not (iseg == nrings and "DARK" in kwargs)):
                image = image + proper.prop_polygon(wf, 6, hexrad, xhex, yhex, rotation = angle)
            
            if (iring != 0):
                xhex = -x * np.cos(angle_rad) - y * np.sin(angle_rad) + xc
                yhex = -x * np.sin(angle_rad) + y * np.cos(angle_rad) + yc
                image = image + proper.prop_polygon(wf, 6, hexrad, xhex, yhex, rotation = angle)
                
            y += hexsep
            
    return image
