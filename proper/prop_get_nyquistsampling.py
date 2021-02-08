#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri


import proper

def prop_get_nyquistsampling(wf, lamx = 0.0):
    """Funtion determines the Nyquist sampling interval for the current beam,
    which is focal_ratio * wavelength / 2.
    
    Parameters
    ----------
    wf : obj
        Wavefront class object
        
    lamx : float
        Wavelength to use for computing sampling. By default, the current
        wavefront's wavelength is used. This parameter can be used when you 
        want to know the Nyquist sampling for a wavelength other than for the
        current wavefront.
        
    Returns
    -------
    float
        Nyquist sampling interval corresponding to the current wavefront
    """
    if lamx != 0.:
        return wf.current_fratio * lamx / 2.
    else:
        return wf.current_fratio * wf.lamda / 2.
