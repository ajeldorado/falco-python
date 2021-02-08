#   Copyright 2016, 2017, 2020 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri

import proper
import numpy as np

def prop_ellipse(wf, xradius, yradius, xc = 0.0, yc = 0.0, **kwargs):
    """Creates an image containing an antialiased filled ellipse

    Parameters
    ----------
    wf : obj
        Wavefront class object

    xradius : float
        Ellipse X radius (in meters unless NORM is set)

    yradius : float
        Ellipse Y radius (in meters unless NORM is set)

    Returns
    -------
    mask : numpy ndarray
        Image array containing antialiased, filled ellipse

    Other Parameters
    ----------------
    xc : float
        X center of ellipse relative to wavefront center (in meters unless norm
        is specified). Default is centered at wavefront center.

    yc : float
        Y center of ellipse relative to wavefront center (in meters unless norm
        is specified). Default is centered at wavefront center.

    DARK : bool
        Draw a dark ellipse (0 inside, 1 outside) (default is opposite way)

    NORM : bool
       Indicates radii and center coordinates are normalized to beam radius.

    ROTATION : float
        Angle in degrees to rotate ellipse.

    Modified by JEK - Jan 2020 - added rotation & subsample parameters.
    """

    nsub = proper.antialias_subsampling
    
    if "ROTATION" in kwargs:
        rotation = kwargs["ROTATION"]
    else:
        rotation = 0.0

    norm = proper.switch_set("NORM",**kwargs)

    n = proper.prop_get_gridsize(wf)
    dx = proper.prop_get_sampling(wf)
    beamrad_pix = proper.prop_get_beamradius(wf) / dx

    xcenter_pix = n // 2
    ycenter_pix = n // 2

    if norm:
        xcenter_pix = xcenter_pix + xc * beamrad_pix 
        ycenter_pix = ycenter_pix + yc * beamrad_pix
        xrad_pix = xradius * beamrad_pix
        yrad_pix = yradius * beamrad_pix
    else:
        xcenter_pix = xcenter_pix + xc / dx
        ycenter_pix = ycenter_pix + yc / dx
        xrad_pix = xradius / dx 
        yrad_pix = yradius / dx

    t = rotation * np.pi / 180

    # rotate coordinates defining box containing unrotated ellipse

    sint = np.sin(t)
    cost = np.cos(t)
    xp = np.array( [-xrad_pix,xrad_pix,xrad_pix,-xrad_pix] )
    yp = np.array( [-yrad_pix,-yrad_pix,yrad_pix,yrad_pix] )
    xbox = xp * cost - yp * sint + xcenter_pix
    ybox = xp * sint + yp * cost + ycenter_pix

    minx_pix = int(np.clip( np.round(np.min(xbox))-1, 0, n-1))
    maxx_pix = int(np.clip( np.round(np.max(xbox))+1, 0, n-1))
    nx = maxx_pix - minx_pix + 1
    miny_pix = int(np.clip( np.round(np.min(ybox))-1, 0, n-1))
    maxy_pix = int(np.clip( np.round(np.max(ybox))+1, 0, n-1))
    ny = maxy_pix - miny_pix + 1

    # create & rotate coordinate arrays 
  
    y_array, x_array = np.mgrid[ 0:ny, 0:nx ] 
    x = x_array + (minx_pix - xcenter_pix)
    y = y_array + (miny_pix - ycenter_pix)

    xr = (x * cost - y * sint) / xrad_pix
    yr = (x * sint + y * cost) / yrad_pix
    r = np.sqrt(xr*xr + yr*yr)
    drx = np.abs(r[0,1] - r[0,0])
    dry = np.abs(r[1,0] - r[0,0])

    delx = 1.0 / xrad_pix
    dely = 1.0 / yrad_pix
    drx = delx * cost - dely * sint
    dry = delx * sint + dely * cost
    dr = max( abs(drx), abs(dry) )

    mask = np.full( (ny,nx), -1, dtype=np.float64 ) 
    mask *= (1 - (r > (1+dr)))
    m = (r <= (1-dr))
    mask = mask * (1 - m) + m
    (y_edge, x_edge) = np.where(mask == -1)    # find pixels along edge of ellipse to subpixellate
    npix = len(x_edge)

    # for each pixel along the edge, subpixellate it to compute fractional coverage (anti-aliasing)

    nsubpix = float(nsub * nsub)
    y_array, x_array = np.mgrid[ 0:nsub, 0:nsub ]
    subpix_x_array = (x_array - nsub//2) / float(nsub) + (minx_pix - xcenter_pix)
    subpix_y_array = (y_array - nsub//2) / float(nsub) + (miny_pix - ycenter_pix)

    xs = np.zeros( (nsub,nsub) )
    ys = np.zeros( (nsub,nsub) )
    x = np.zeros( (nsub,nsub) )
    y = np.zeros( (nsub,nsub) )

    limit = 1 + 1e-10

    for i in range(npix):
        xs[:,:] = subpix_x_array + x_edge[i]
        ys[:,:] = subpix_y_array + y_edge[i]
        x[:,:] = (xs * cost - ys * sint) / xrad_pix
        y[:,:] = (xs * sint + ys * cost) / yrad_pix
        mask[y_edge[i],x_edge[i]] = np.sum((x*x+y*y) <= limit) / nsubpix

    if proper.switch_set("DARK",**kwargs):
        image = np.ones( (n,n), dtype=np.float64 )
        image[miny_pix:miny_pix+ny,minx_pix:minx_pix+nx] = 1 - mask
    else:
        image = np.zeros( (n,n), dtype=np.float64 )
        image[miny_pix:miny_pix+ny,minx_pix:minx_pix+nx] = mask
   
    return image 

