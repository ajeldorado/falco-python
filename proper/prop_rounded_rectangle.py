#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper


def prop_rounded_rectangle(wf, corner_radius, width, height, xc = 0.0, yc = 0.0):
    """Return a 2-D array containing a rectangular mask (1 inside, 0 outside)
    with rounded corners. 
    
    This routine was created to allow modeling of the TPF-C primary mirror. 
    
    Parameters
    ----------
    wf : obj
        WaveFront class object
        
    corner_radius : float
        Radius (meters) of the rounded corners (which are 90 degree sections of a circle)
        
    width, height : float
        Width and height of the mask in meters
        
    xc, yc : float
        Center of rectangle in array in meters as offset from the array center.
        The default is a mask centered in the array.

    Returns
    -------
    mask : numpy ndarray
        Returns an 2-D array containing the mask.    
    """
    # Get sampling and gridsize
    dx = proper.prop_get_sampling(wf)
    n = proper.prop_get_gridsize(wf)
    
    # x,y offset of the circle defining rounded corner (meters)
    rad_pix = corner_radius/dx
    xoff = width/2. - corner_radius
    yoff = height/2. - corner_radius
    
    mask = proper.prop_rectangle(wf, width, height, xc, yc)
    
    circle = proper.prop_ellipse(wf, corner_radius, corner_radius, xoff+xc, yoff+yc)
    xoff_pix = (xoff+xc) / dx + n//2
    yoff_pix = (yoff+yc) / dx + n//2
    
    if xoff_pix < 0:
        x1 = 0
    else:
        x1 = xoff_pix

    if xoff_pix + rad_pix + 3 > n:
        x2 = n
    else:
        x2 = xoff_pix + rad_pix + 3
                        
    if yoff_pix < 0:
        y1 = 0
    else:
        y1 = yoff_pix
    
    if yoff_pix + rad_pix + 3 > n:
        y2 = n
    else:
        y2 = yoff_pix + rad_pix + 3
    
    mask[y1:y2,x1:x2] = 0
    mask[y1:y2,x1:x2] = circle[y1:y2,x1:x2]
    
    circle = proper.prop_ellipse(wf, corner_radius, corner_radius, xoff+xc, -yoff+yc)
    xoff_pix = (xoff+xc)/dx + n//2
    yoff_pix = (-yoff+yc)/dx + n//2
    
    if xoff_pix < 0:
        x1 = 0
    else:
        x1 = xoff_pix

    if xoff_pix + rad_pix + 3 > n:
        x2 = n
    else:
        x2 = xoff_pix + rad_pix + 3
        
    if yoff_pix - rad_pix - 2 < 0:
        y1 = 0
    else:
        y1 = yoff_pix - rad_pix - 2
        
    if yoff_pix > n:
        y2 = n
    else:
        y2 = yoff_pix + 1
            
    mask[y1:y2,x1:x2] = 0
    mask[y1:y2,x1:x2] = circle[y1:y2,x1:x2]
    
    circle = proper.prop_ellipse(wf, corner_radius, corner_radius, -xoff+xc, yoff+yc)
    xoff_pix = (-xoff+xc) / dx + n//2
    yoff_pix = (yoff+yc) / dx + n//2
    
    if xoff_pix - rad_pix - 2 < 0:
        x1 = 0
    else:
        x1 = xoff_pix - rad_pix - 2
        
    if xoff_pix > n:
        x2 = n
    else:
        x2 = xoff_pix + 1
        
    if yoff_pix < 0:
        y1 = 0
    else:
        y1 = yoff_pix
        
    if yoff_pix + rad_pix + 3 > n:
        y2 = n
    else:
        y2 = yoff_pix + rad_pix + 3

    mask[y1:y2,x1:x2] = 0
    mask[y1:y2,x1:x2] = circle[y1:y2,x1:x2] 

    circle = proper.prop_ellipse(wf, corner_radius, corner_radius, -xoff+xc, -yoff+yc)
    xoff_pix = (-xoff+xc) / dx + n//2
    yoff_pix = (-yoff+yc) / dx + n//2
    
    if xoff_pix - rad_pix - 2 < 0:
        x1 = 0
    else:
        x1 = xoff_pix - rad_pix - 2
        
    if xoff_pix > n:
        x2 = n
    else:
        x2 = xoff_pix + 1
        
    if yoff_pix - rad_pix - 2 < 0:
        y1 = 0
    else:
        y1 = yoff_pix - rad_pix - 2
        
    if yoff_pix > n:
        y2 = n
    else:
        y2 = yoff_pix + 1

    mask[y1:y2,x1:x2] = 0
    mask[y1:y2,x1:x2] = circle[y1:y2,x1:x2] 
    
    return mask
