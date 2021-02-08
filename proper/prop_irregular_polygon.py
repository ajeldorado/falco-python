#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#   Modified 14 Oct 2019 - JEK - fixed check when drawing polygon that extends
#   beyond maximum Y edge of array.


import proper
import numpy as np


def prop_irregular_polygon(wf, xvert, yvert, **kwargs):
    """Return an image containing a filled (interior value is 1.0) convex polygon 
    with antialiased edges. If the dark switch is set, then the interior is set 
    to 0.0.

        
    Parameters
    ----------
    wf : obj
       WaveFront class object
       
    xvert, yvert : numpy ndarray
       Vector arrays of the same dimension containing the coordinates of the 
       convex polygon verticies in meters (unless norm is set). The center of 
       the wavefront array is designated to have the coordinates (0,0). The 
       verticies must be in sequential order, either clockwise or 
       counter-clockwise. The coordinates of the first and last verticies 
       should be the same (ie. close the polygon).
       
    
    Returns
    -------
    image : numpy ndarray
        Returns an image containing the polygon.
    
    
    Other Parameters
    ----------------
    DARK : bool
        The interior of the polygon will be set to 0 and the exterior to 1.
        
    NORM : bool
        Indicates that the radius and polygon center values are relative to the 
        current beam radius.
    
    
    Notes
    -----
    This routine only works on convex polygons and will produce incorrect
    results given anything else.
    """
    # grid size, beam radius and sampling
    ngrid = proper.prop_get_gridsize(wf)
    beamradius = proper.prop_get_beamradius(wf)
    sampling = proper.prop_get_sampling(wf)
    
    # beam radius in pixels
    pr = beamradius/sampling
    
    # subsampling factor at edges; must be odd
    mag = proper.antialias_subsampling
    
    # drawing algorithm needs vertices to be clockwise-ordered.
    # Test for clockwise-ness of vertices; for a convex polygon, if the cross 
    # product of adjacent edges is positive, it is counter-clockwise.
    cp = (xvert[1] - xvert[0]) * (yvert[2] - yvert[1]) - (yvert[1] - yvert[0]) * (xvert[2] - xvert[1])
    if cp > 0:
        xp = xvert[::-1]
        yp = yvert[::-1]
    else:
        xp = xvert
        yp = yvert
        
    xcpix = int(ngrid/2)
    ycpix = int(ngrid/2)
    
    if proper.switch_set("NORM",**kwargs):
        xp = xp * pr + xcpix
        yp = yp * pr + ycpix
    else:
        xp = xp / sampling + xcpix
        yp = yp / sampling + ycpix        
    
    image = np.zeros([ngrid, ngrid], dtype = np.float64)
    nvert = len(xvert)
    
    left = np.where(yp == np.min(yp))
    left = left[0][np.where(xp[left] == np.min(xp[left]))[0]]
    left = left[0]
    
    if left != nvert-1:
        leftnext = left + 1
    else:
        leftnext = 0
        
    right = left
    
    if right != 0:
        rightnext = right - 1
    else:
        rightnext = nvert-1
        
        
    if int(np.round(np.min(yp))) < 0:
        imin = 0
    else:
        imin = int(np.round(np.min(yp)))
       
    imax = int(np.round(np.max(yp))) + 1
    if ( imax > ngrid ):
        imax = ngrid

    for ypix in range(imin, imax):
        for ysub in range(0, mag):
            y = ypix - 0.5 + (0.5 + ysub)/mag
            
            if y < yp[left]:
                continue
            if y > np.max(yp):
                break
            
            if y >= yp[leftnext]:
                left = leftnext
                if left != nvert-1:
                    leftnext = left + 1
                else:
                    leftnext = 0
                    
            if y >= yp[rightnext]:
                right = rightnext
                if right != 0:
                    rightnext = right - 1
                else:
                    rightnext = nvert - 1
                    
            leftdy = yp[leftnext] - yp[left]
            if leftdy != 0:
                leftdx = xp[leftnext] - xp[left]
                xleft = leftdx/leftdy * (y-yp[left]) + xp[left]
            else:
                xleft = xp[left]
                
            rightdy = yp[rightnext] - yp[right]
            if rightdy != 0:
                rightdx = xp[rightnext] - xp[right]
                xright = rightdx/rightdy * (y - yp[right]) + xp[right]
            else:
                xright = xp[right]
            
            xleftpix = int(np.round(xleft))
            xrightpix = int(np.round(xright))
            
            if xleftpix != xrightpix:
                if xleftpix >= 0:
                    image[ypix,xleftpix] = image[ypix,xleftpix] + mag * ((xleftpix + 0.5) - xleft)
                if xrightpix < ngrid:
                    image[ypix,xrightpix] = image[ypix,xrightpix] + mag * (xright - (xrightpix - 0.5))
                if (xrightpix - xleftpix) > 1:
                    if xleftpix+1 < 0:
                        imin = 0
                    else:
                        imin = xleftpix+1
                        
                    if xrightpix >= ngrid:
                        imax = ngrid - 1 
                    else:
                        imax = xrightpix
                    image[ypix,imin:imax] = image[ypix,imin:imax] + mag
            else:
                if xleftpix >= 0 and xleftpix < ngrid:
                    image[ypix,xleftpix] = image[ypix,xleftpix] + mag * (xright - xleft)
    
    image = image / float(mag)**2
    
    if proper.switch_set("DARK",**kwargs):
        image = 1.0 - image
        
    return image        
