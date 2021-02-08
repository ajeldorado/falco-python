#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np
from numpy.fft import fft2, ifft2


def prop_pixellate(image_in, sampling_in, sampling_out, n_out = 0):
    """Integrate a sampled PSF onto detector pixels. 
    
    This routine takes as input a sampled PSF and integrates it over pixels of a 
    specified size. This is done by convolving the Fourier transform of the input 
    PSF with a sinc function representing the transfer function of an idealized 
    square pixel and transforming back. This result then represents the PSF 
    integrated onto detector-sized pixels with the same sampling as the PSF. 
    The result is interpolated to get detector-sized pixels at detector-pixel 
    spacing.
    
    Parameters
    ----------
    image_in : numpy ndarray
        2D floating image containing PSF
        
    sampling_in : float
        Sampling of image_in in meters/pixel
        
    sampling_out : float
        Size(=sampling) of detector pixels
        
    n_out : int
        Output image dimension (n_out by n_out)
        
    Returns
    -------
    new : numpy ndarray
        Returns image integrated over square detector pixels.
    """
    n_in = image_in.shape[0]
    
    # Compute pixel transfer function (MTF)
    psize = 0.5 * (sampling_out / sampling_in)
    constant = psize * np.pi
    mag = sampling_in / sampling_out
    
    arr = np.arange(n_in, dtype = np.float64) - int(n_in/2)
    x = np.roll(arr, -int(n_in/2), 0) / int(n_in/2)
    t = x * constant
    y = np.zeros(n_in, dtype = np.float64)
    y[1:] = np.sin(t[1:]) / t[1:]
    y[0] = 1.
    
    #pixel_mtf = np.tile(np.vstack(y), (1, n_in)) * y
    pixel_mtf = np.dot(y[:,np.newaxis], y[np.newaxis,:])
    
    # Convolve image with detector pixel
    image_mtf = fft2(np.roll(np.roll(image_in, -int(n_in/2), 0), -int(n_in/2), 1)) / image_in.size
    image_mtf *= pixel_mtf
    
    convolved_image = np.abs(ifft2(image_mtf) * image_mtf.size)
    image_mtf = 0
    convolved_image = np.roll(np.roll(convolved_image/mag**2, int(n_in/2), 0), int(n_in/2), 1)
    
    # Image is integrated over pixels but has original sampling; now, resample
    # pixel sampling
    if n_out != 0:
        n_out = int(np.fix(n_in * mag))
        
    new = proper.prop_magnify(convolved_image, mag, n_out)
    
    return new  
