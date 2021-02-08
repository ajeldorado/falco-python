#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import numpy as np
import proper

def prop_shift_center(image):
    """Shift a n by n image by (x,y)=(n/2,n/2), either shifting from the image
    origin to the center or vice-verse
    
    Parameters
    ----------
    image : numpy ndarray
        2D image to be shifted
    
    Returns
    -------
    shifted image : numpy ndarray
         Shifted image       
    """
    image = np.asarray(image)
    
    if image.ndim != 2:
        raise ValueError("Only 2D images can be shifted. Stopping.")
        
    s = image.shape
    
    return np.roll(np.roll(image, int(s[0]/2), 0), int(s[1]/2), 1)
