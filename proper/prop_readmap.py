#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper


def prop_readmap(wf, filename, xshift = 0, yshift = 0, **kwargs):
    """Read in a surface, wavefront, or amplitude error map, scaling if necessary.
    
    Parameters
    ----------
    wf : obj
        WaveFront class object
        
    filename : str
        Name of FITS file containing map
        
    xshift, yshift : float
        Amount to shift map in meters in X and Y directions
        
        
    Returns
    -------
    dmap : numpy ndarray
        Returns the error map in a 2D numpy array
    
    
    Other Parameters
    ----------------
    XC_MAP, YC_MAP : float
        Specifies center pixel of map (default is n/2)
        
    SAMPLING : float
        Specifies sampling of map in meters (will override any sampling 
        specified in the file header; must be specified if header does not 
        specify sampling using the PIXSIZE value)
        
    
    Raises
    ------
    ValueError:
        if pixsize keyword does not exist in FITS image header
    
    
    Notes
    -----
    (a) The sampling is not returned in this variable 
    (b) if the header value RADPIX is specified (the radius of the beam in the 
        map in pixels), then this will override any other sampling specifiers, 
        either in the header or using SAMPLING.
    (c) Intended for internal use by PROP routines. Users should call either
        prop_errormap or prop_psd_errormap.
    """
    # Read input FITS image
    dmap, header = proper.prop_fits_read(filename, header = True)
    
    # if the radius of the beam (in pixels) in the map is specified in the 
    # header, this will override any other sampling specifications, either from 
    # the header (PIXSIZE value) or procedure call (SAMPLING keyword)
    if "radpix" in header:
        pixsize = proper.prop_get_beamradius(wf) / header["radpix"]
    elif not "SAMPLING" in kwargs:
        if not "pixsize" in header:
            raise ValueError("READMAP: No pixel scale (PIXSIZE) specified in %s header. Stopping" %(filename))
        else:
            pixsize = header["pixsize"]
    else:
        pixsize = kwargs["SAMPLING"]
    
    if not "XC_MAP" in kwargs:
        s = dmap.shape
        xc = s[0]//2
        yc = s[1]//2
    else:
        xc = kwargs["XC_MAP"]
        yc = kwargs["YC_MAP"]
        
    # resample map to current wavefront grid spacing
    dmap = proper.prop_resamplemap(wf, dmap, pixsize, xc, yc, xshift, yshift)
        
    dmap = proper.prop_shift_center(dmap)
    
    return dmap    
