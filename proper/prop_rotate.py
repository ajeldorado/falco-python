#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np

if not proper.use_cubic_conv:
    from scipy.ndimage.interpolation import rotate


def prop_rotate(old_image, theta, **kwargs):
    """Rotate and shift an image via interpolation (bilinear by default)
    
    Parameters
    ----------
    old_image : numpy ndarray
        Image to be rotated
        
    theta : float
        Angle to rotate image in degrees counter-clockwise
        
    
    Returns
    -------
    new_image : numpy ndarray
        Returns rotated & shifted image with the same dimensions as the input image

    
    Other Parameteres
    -----------------
    XC, YC : float
        Center of rotation in image pixels; (0,0) is at center of first pixel;
        if not specified, the center of the image is assumed to be the center
        of rotation
        
    XSHIFT, YSHIFT : float
        Amount to shift rotated image in pixels
        
    MISSING : float
        Value to set extrapolated pixels.        
    """
    if old_image.dtype == np.dtype("complex128") or old_image.dtype == np.dtype("complex64"):
        is_complex = 1
    else:
        is_complex = 0
        
    new_image = np.copy(old_image)
    
    if proper.use_cubic_conv:
        n = old_image.shape[0]
    
        if not "XC" in kwargs:
            XC = int(n / 2)
        
        if not "YC" in kwargs:
            YC = int(n / 2)
        
        if not "XSHIFT" in kwargs:
            xshift = 0.
    
        if not "YSHIFT" in kwargs:
            yshift = 0.        
        
        if not "MISSING" in kwargs:
            missing = 0.    
        
        t = -theta * np.pi / 180.
    
        x0 = np.arange(n, dtype = np.float64) - XC - xshift
    
        for j in range(n):
            y0 = j - YC - yshift
            xi = x0 * np.cos(t) - y0 * np.sin(t) + YC
            yi = x0 * np.sin(t) + y0 * np.cos(t) + XC
        
            new_image[j,:] = proper.prop_cubic_conv(old_image, xi, yi, GRID = False)
    else:
        theta = -1. * theta
        if is_complex:
            new_image.real = rotate(old_image.real, theta, reshape = False, prefilter = True) 
            new_image.imag = rotate(old_image.imag, theta, reshape = False, prefilter = True)
        else:
            new_image = rotate(old_image, theta, reshape = False, prefilter = False)
    
    return new_image
