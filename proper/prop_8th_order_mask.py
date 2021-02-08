#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np


def prop_8th_order_mask(wf, hwhm, min_transmission = 0.0, max_transmission = 1.0, **kwargs):
    """Multiply the current wavefront by an 8th-order occulter. 
    
    An 8th-order transmission mask is generated. The mask can be linear 
    (the default), circular, or elliptical in shape. The mask width is specified 
    as the radius at which the transmission is 0.5 (along the X-axis for linear 
    (unless /Y_AXIS is specified) and elliptical apertures).
    
    
    Parameters
    ----------
    wf : object
      WaveFront class object
      
    hwhm : float
      The radius which the mask transmission is 0.5 (along the direction of the
      X axis from elliptical apertures). By default, this is in units of 
      lambda/D. If METERS is specified, this is in meters.
    
    min_transmission : float
      Sets the mask minimum intensity transmission to be "min". Default is 0.0.
      
    max_transmission : float
      Sets the mask maximum intensity transmission to be "max". Default is 1.0.  
            
            
    Returns
    -------
    mask : numpy ndarray
        Returns the amplitude (not intensity) mask image array.
        
        
    Other Parameters
    ----------------
    METERS : bool
      Indicates that value in "hwhm" is in meters.
      
    CIRCULAR : bool
      Switch that if set creates a circular occulter (linear is default)
      
    Elliptical : float
      If specified, creates an elliptical occulter with axis ratio of 
      ratio=y_width/x_width.
      
    Y_AXIS : bool
      Specifies that the linear occulter is to be drawn with transmission 
      varying along the Y axis rather than the X axis (the default); only valid 
      for the linear occulter  
    """

    fratio = proper.prop_get_fratio(wf)
    wavelength = proper.prop_get_wavelength(wf)
    sampling = proper.prop_get_sampling(wf)
    
    if proper.switch_set("METERS",**kwargs):
        # convert from meters to lamdba/D units
        hwhm_nlamd = hwhm / (fratio * wavelength)
        e = 1.788 / hwhm_nlamd
    else:
        e = 1.788 / hwhm
        
    if proper.switch_set("CIRCULAR",**kwargs) or "ELLIPTICAL" in kwargs:
        linear = False
    else:
        linear = True
        
    n = proper.prop_get_gridsize(wf)
    ll = 3.
    mm = 1.
    
    # Compute beam spacing in lamdba/D units
    c = sampling / (fratio * wavelength)
    
    isqr = (np.arange(n, dtype = np.float64) - n/2)**2
    
    if linear:
        x = (np.arange(n, dtype = np.float64) - n/2) * c    
        mask = (ll-mm)/ll - proper.prop_sinc(np.pi * x * e/ll)**ll + (mm/ll) * proper.prop_sinc(np.pi*x*e/mm)**mm  # amplitude
    elif proper.switch_set("CIRCULAR",**kwargs):
        mask = np.zeros([n,n], dtype = np.float64)
        for j in range(n):
            r = np.sqrt(isqr + float(j-n/2)**2) * c
            mask[j,:] = (ll-mm)/ll - proper.prop_sinc(r*(np.pi*e/ll))**ll + mm/ll * proper.prop_sinc(r*(np.pi*e/mm))**mm # amplitude
    else:
        ELLIPTICAL = kwargs["ELLIPTICAL"]
        mask = np.zeros([n,n], dtype = np.float64)
        for j in range(n):
            r = np.sqrt(isqr + (float(j-n/2)/ELLIPTICAL)**2) * c
            mask[j,:] = (ll-mm)/ll - proper.prop_sinc(r*(np.pi*e/ll))**ll + mm/ll * proper.prop_sinc(r*(np.pi*e/mm))**mm # amplitude

    mask = mask**2               # intensity
    mask -= np.min(mask)
    mask /= np.max(mask)                
    mask = mask * (max_transmission - min_transmission) + min_transmission
    mask = np.sqrt(mask)         # back to amplitude
    
    if linear:
        if proper.switch_set("Y_AXIS",**kwargs):
            mask = np.tile(mask, (n,1)).T
        else:
            mask = np.tile(mask, (n,1))           
            
    wf.wfarr *=  proper.prop_shift_center(mask)
    
    return mask
