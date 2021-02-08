#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#   Modified 13 Feb 2020 - J. Krist - If the map was rotated, it was not
#   shifted back to the origin of the wavefront array (Python indentation error)


import proper
import numpy as np


def prop_errormap(wf, filename, xshift = 0., yshift = 0., **kwargs):
    """Read in a surface, wavefront, or amplitude error map from a FITS file. 
    
    Map is assumed to be in meters of surface error. One (and only one) of the 
    MIRROR_SURFACE, WAVEFRONT, or AMPLITUDE switches must be specified in order 
    to properly apply the map to the wavefront.  For surface or wavefront error 
    maps, the map values are assumed to be in meters, unless the NM or MICRONS 
    switches are used to specify the units. The amplitude map must range 
    from 0 to 1.  The map will be interpolated to match the current wavefront 
    sampling if necessary.
    
    Parameters
    ----------
    wf : obj
        WaveFront class object
        
    filename : str
        Name of FITS file containing map
        
    xshify, yshift : float
        Amount to shift map (meters) in X,Y directions
    
    Returns
    -------
    DMAP : numpy ndarray
        Returns map (after any necessary resampling) if set.
    
    
    Other Parameters
    ----------------
    XC_MAP, YC_MAP : float
        Pixel coordinates of map center (Assumed n/2,n/2)
        
    SAMPLING : float
        Sampling of map in meters
        
    ROTATEMAP : float
        Degrees counter-clockwise to rotate map, after any resampling and 
        shifting
        
    MULTIPLY : float
        Multiplies the map by the specified factor
        
    MAGNIFY : float
        Spatially magnify the map by a factor of "constant" from its default
        size; do not use if SAMPLING is specified
        
    MIRROR_SURFACE : bool
        Indicates file contains a mirror surface height error map; It assumes a 
        positive value indicates a surface point higher than the mean surface.  
        The map will be multiplied by -2 to convert it to a wavefront map to 
        account for reflection and wavefront delay (a low region on the surface 
        causes a positive increase in the phase relative to the mean)
        
    WAVEFRONT : bool
        Indicates file contains a wavefront error map
        
    AMPLITUDE : bool
        Indicates file contains an amplitude error map
        
    NM or MICRONS : bool
        Indicates map values are in nanometers or microns. For surface or 
        wavefront maps only
        
    Raises
    ------
    SystemExit:
        If AMPLITUDE and (NM or MICRONS) parameters are input.
        
    SystemExit:
        If NM and MICRONS parameteres are input together. 
        
    ValueError:
        If map type is MIRROR_SURFACE, WAVEFRONT, or AMPLITUDE.
    """

    if proper.switch_set("AMPLITUDE",**kwargs) and (proper.switch_set("NM",**kwargs) or proper_switch_set("MICRONS",**kwargs)):
        raise SystemExit("ERRORMAP: Cannot specify NM or MICRON for an amplitude map")
    
    if proper.switch_set("NM",**kwargs) and proper.switch_set("MICRONS",**kwargs):
        raise SystemExit("ERRORMAP: Cannot specify both NM and MICRONS")

    if ("XC_MAP" in kwargs or "YC_MAP" in kwargs or "SAMPLING" in kwargs):
        if ("XC_MAP" in kwargs and "YC_MAP" in kwargs and "SAMPLING" in kwargs):
            dmap = proper.prop_readmap(wf, filename, xshift, yshift, XC_MAP = kwargs["XC_MAP"], 
                YC_MAP = kwargs["YC_MAP"], SAMPLING = kwargs["SAMPLING"])
        elif ("XC_MAP" in kwargs and "YC_MAP" in kwargs):
            dmap = proper.prop_readmap(wf, filename, xshift, yshift, XC_MAP = kwargs['XC_MAP'], 
                YC_MAP = kwargs["YC_MAP"])
        elif "SAMPLING" in kwargs:
            dmap = proper.prop_readmap(wf, filename, xshift, yshift, SAMPLING = kwargs['SAMPLING'])
    else:
        dmap = proper.prop_readmap(wf, filename, xshift, yshift)    
    
    
    if "ROTATEMAP" in kwargs or "MAGNIFY" in kwargs:
        # readmap stores map with center at (0,0), so shift 
        # before and after rotation
        dmap = proper.prop_shift_center(dmap)
        if "ROTATEMAP" in kwargs:
            dmap = proper.prop_rotate(dmap, kwargs["ROTATEMAP"], CUBIC=-0.5, MISSING=0.0)
        if "MAGNIFY" in kwargs:
            dmap = proper.prop_magnify(dmap, kwargs["MAGNIFY"], dmap.shape[0])
        dmap = proper.prop_shift_center(dmap)
            
    if proper.switch_set("MICRONS",**kwargs):
        dmap *= 1.e-6
        
    if proper.switch_set("NM",**kwargs):
        dmap *= 1.e-9
        
    if "MULTIPLY" in kwargs:
        dmap *= kwargs["MULTIPLY"]
        
    i = complex(0.,1.)
    
    if proper.switch_set("MIRROR_SURFACE",**kwargs):
        wf.wfarr *= np.exp(-4*np.pi*i/wf.lamda * dmap)
    elif proper.switch_set("WAVEFRONT",**kwargs):
        wf.wfarr *= np.exp(2*np.pi*i/wf.lamda * dmap)
    elif proper.switch_set("AMPLITUDE",**kwargs):
        wf.wfarr *= dmap
    else:
        raise ValueError("PROP_ERRORMAP: Unspecified map type: Use MIRROR_SURFACE, WAVEFRONT, or AMPLITUDE")
    
    dmap = proper.prop_shift_center(dmap)
    
    return dmap

