#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import numpy as np
import proper

def prop_get_beamradius(wf):
    """This function returns the half-width-at-half-max of the Gaussian beam
    that is used to keep track of the beam size.  
    
    The Gaussian beam starts off with a beam diameter equal to the entrance 
    pupil diameter. This ignores widening of the beam due to aberrations.
    
    Parameters
    ----------
    wf : obj
        Wavefront class object
        
    Returns
    -------
    beam radius : float
        Beam radius in meters
    """
    return wf.w0 * np.sqrt( 1.0 + (wf.lamda * (wf.z - wf.z_w0)/(np.pi * wf.w0**2))**2 )
