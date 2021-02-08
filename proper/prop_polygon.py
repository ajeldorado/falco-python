#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np


def prop_polygon(wf, nvert, rad, xc = np.nan, yc = np.nan, **kwargs):
    """Return an image containing a filled, symmetric (verticies are the same 
    distance from the center) polygon with antialiased edges. 
    
    A clear polygon is drawn (one inside, zero outside), unless the dark switch 
    is set. Unless the rotation keyword is set, the polygon is drawn with one 
    vertex positioned along the +X axis relative to the polygon center.
    
    Parameters
    ----------
    wf : obj
        WaveFront class object
            
    nvert : int
        Number of vertices (e.g. 4=square, 6=hexagon); must be 3 or greater.
            
    radius : float
        Distance from any vertex to the center of the polygon (same for all
        vertices). By default this is in meters. If the norm switch is set, 
        then this is relative to the current beam radius.
            
    xc, yc : float
        Center of the polygon relative to the wavefront center. These are in 
        meters unless norm is specified, in which case they are relative to the 
        current beam radius. The default is for the polygon to be centered 
        in the array.
        
    
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
        
    ROTATION : float
        Specifies the counter-clockwise angle in degrees to rotate the polygon.  
        By default, one vertex lies along the +X axis relative to the polygon 
        center.
    """
    # grid size, beam radius and sampling
    ngrid = proper.prop_get_gridsize(wf)
    beamradius = proper.prop_get_beamradius(wf)
    sampling = proper.prop_get_sampling(wf)
    
    # beam radius in pixels
    pr = beamradius/sampling
    
    # get keyword argument values
    dark = proper.switch_set("DARK",**kwargs)
    norm = proper.switch_set("NORM",**kwargs)
    
    if "ROTATION" in kwargs:
        rotation = kwargs["ROTATION"]
    else:
        rotation = 0.0
    
    
    if np.isnan(xc):
        xcpix = ngrid/2.
    else:
        if norm:
            xcpix = xc*pr + ngrid/2.
        else:
            xcpix = xc / sampling + ngrid/2.
            
    if np.isnan(yc):
        ycpix = ngrid/2.
    else:
        if norm:
            ycpix = yc*pr + ngrid/2.
        else:
            ycpix = yc / sampling + ngrid/2.
            
    if not norm:
        radpix = rad/sampling
    else:
        radpix = rad*pr
        
    if rotation != 0.:
        angle_rad = rotation * np.pi/180.
    else:
        angle_rad = 0.
        
    # subsampling factor at edges,; must be odd
    mag = proper.antialias_subsampling
    
    t = -(np.arange(float(nvert)) / nvert) * 2 * np.pi  # clockwise vertex list
    xp0 = np.cos(t)
    yp0 = np.sin(t)
    
    xp = xp0 * np.cos(angle_rad) - yp0 * np.sin(angle_rad)
    yp = xp0 * np.sin(angle_rad) + yp0 * np.cos(angle_rad)
    
    xp = xp * radpix + xcpix 
    yp = yp * radpix + ycpix
    
    image = np.zeros([ngrid, ngrid], dtype = np.float64)
    
    left = np.where(yp == np.min(yp))
    left = left[0][np.where(xp[left] == np.min(xp[left]))[0]]
    left = left[0]
    
    if left != nvert - 1:
        leftnext = left + 1
    else:
        leftnext = 0
        
    right = left
    
    if right != 0:
        rightnext = right - 1
    else:
        rightnext = nvert - 1
        
    if int(np.round(np.min(yp))) < 0:
        imin = 0
    else:
        imin = int(np.round(np.min(yp)))
        
    if int(np.round(np.max(yp))) > ngrid:
        imax = ngrid            
    else:
        imax = int(np.round(np.max(yp))) + 1   
    
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
    
    if dark:
        image = 1.0 - image
        
    return image
