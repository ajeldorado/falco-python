#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#  Modified 18 April 2019 by J. Krist - Switched to /= and *=, new in-place FFT calls
#  for FFTW/FFTI


import proper
import numpy as np
from numpy.fft import fft2, ifft2


def prop_wts(wf, dz):
    """Propagate from a planar reference surface that is inside the Rayleigh 
    distance from focus to a spherical reference surface that is outside. 
    
    Used by propagate function.
    
    Parameters
    ----------
    wf : obj
        WaveFront class object
        
    dz : float
        Distance in meters to propagate
        
    Returns
    -------
        None 
        Wavefront is modified.
    """
    if proper.verbose:
        print("  WTS: dz = ", dz)
    
    wf.reference_surface = "SPHERI"
    
    if dz == 0.0:
        return
   
    wf.z = wf.z + dz
             
    direct = dz >= 0.0
    
    proper.prop_qphase(wf, dz)
    
    if direct == 1:    # forward transform
        if proper.use_fftw:
            proper.prop_fftw(wf.wfarr, directionFFTW = 'FFTW_FORWARD') 
            wf.wfarr /= np.size(wf.wfarr)
        else:
            wf.wfarr[:,:] = fft2(wf.wfarr) / np.size(wf.wfarr)
        wf.wfarr *= wf.ngrid
    else:             # inverse transform
        if proper.use_fftw:
            proper.prop_fftw(wf.wfarr,directionFFTW = 'FFTW_BACKWARD') 
            wf.wfarr *= np.size(wf.wfarr)
        else:
            wf.wfarr[:,:] = ifft2(wf.wfarr) * np.size(wf.wfarr)
        wf.wfarr /= wf.ngrid
        
    if proper.phase_offset:
        wf.wfarr *= np.exp(complex(0.,1.) * 2*np.pi*dz/wf.lamda)
        
    wf.dx = wf.lamda * np.abs(dz) / (wf.ngrid * wf.dx)
    
    if proper.verbose:
        print("  WTS: z = ", wf.z, "  dx = ", wf.dx)
    
    return
