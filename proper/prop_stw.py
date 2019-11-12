#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#  Modified 18 April 2019 by J. Krist - switched to *= and /= and use in-place
#  calls for FFTW/FFTI


import proper
import numpy as np
from numpy.fft import fft2, ifft2


def prop_stw(wf, dz = 0.0):
    """Propagate from a spherical reference surface that is outside the Rayleigh 
    limit from focus to a planar one that is inside. Used by propagate function.
    
    Parameters
    ----------
    wf : obj
        WaveFront class object
        
    dz : float
        Distance in meters to propagate
        
    Returns
    -------
        None 
        Modifies the wavefront.
    """
    ngrid = wf.ngrid
    
    if proper.verbose:
        print("  STW: dz = %3.6f" %(dz))
    
    if wf.reference_surface != "SPHERI":
        if proper.verbose: 
            print("  STW: Input reference surface not spherical. Using PTP")
        proper.prop_ptp(wf, dz)
        return
        
    if dz == 0.0:
        dz = wf.z_w0 - wf.z
        
    wf.z = wf.z + dz
    wf.dx = wf.lamda * np.abs(dz) / (ngrid * wf.dx)
    
    direct = dz >= 0.0
    
    if direct:                 # forward transform
        if proper.use_fftw:
            proper.prop_fftw(wf.wfarr,directionFFTW = 'FFTW_FORWARD') 
            wf.wfarr /= np.size(wf.wfarr)
        else:
            wf.wfarr[:,:] = fft2(wf.wfarr) / np.size(wf.wfarr)
        wf.wfarr *= ngrid
    else:
        if proper.use_fftw:
            proper.prop_fftw(wf.wfarr, directionFFTW = 'FFTW_BACKWARD') 
            wf.wfarr *= np.size(wf.wfarr)
        else:
            wf.wfarr[:,:] = ifft2(wf.wfarr) * np.size(wf.wfarr)
        wf.wfarr /= ngrid

    proper.prop_qphase(wf, dz)
    
    if proper.phase_offset:
        wf.wfarr *= np.exp(complex(0.,1.) * 2*np.pi*dz/wf.lamda)
    
    if proper.verbose:
        print("  STW: z = %4.6f    dx = %.6e" %(wf.z, wf.dx))
    
    wf.reference_surface = "PLANAR"
    
    return
