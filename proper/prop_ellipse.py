#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np


def prop_ellipse(wf, xradius, yradius, xc = np.nan, yc = np.nan, **kwargs):
    """Creates an image containing an antialiased filled ellipse

    Parameters
    ----------
    wf : obj
        Wavefront class object

    xradius : float
        Ellipse X radius

    yradius : float
        Ellipse Y radius

    Returns
    -------
    mask : numpy ndarray
        Image array containing antialiased, filled ellipse

    Other Parameters
    ----------------
    xc : float
        X center of ellpise relative to wavefront center (in meters unless norm
        is specified). Default is centered at wavefront center.

    yc : float
        Y center of ellipse relative to wavefront center (in meters unless norm
        is specified). Default is centered at wavefront center.

    DARK : bool
        Draw a dark ellipse (0 inside, 1 outside) (default is opposite way)

    NORM : bool
       Indicates radii and center coordinates are normalized to beam radius.
    """
    # grid size, beam radius and sampling
    ngrid = proper.prop_get_gridsize(wf)
    beamradius = proper.prop_get_beamradius(wf)
    sampling = proper.prop_get_sampling(wf)

    # beam radius in pixels
    pr = beamradius / sampling

    # get keyword argument values
    if ("DARK" in kwargs and kwargs["DARK"]):
        dark = kwargs["DARK"]
    else:
        dark = False

    if ("NORM" in kwargs and kwargs["NORM"]):
        norm = kwargs["NORM"]
    else:
        norm = False

    # Set xcenter and ycenter values
    if np.isnan(xc):
        xcenter = int(ngrid / 2)
    else:
        if norm:
            xcenter = xc * pr + int(ngrid/2)
        else:
            xcenter = xc / sampling + int(ngrid/2)

    if np.isnan(yc):
        ycenter = int(ngrid / 2)
    else:
        if norm:
            ycenter = yc * pr + int(ngrid/2)
        else:
            ycenter = yc / sampling + int(ngrid/2)

    # Set xradius and yradius
    if norm:
        xrad = xradius * pr
        yrad = yradius * pr
    else:
        xrad = xradius / sampling
        yrad = yradius / sampling

    # Define a circle
    t = (np.arange(1000, dtype = np.float64)/999) * 2 * np.pi
    xel = xrad * np.cos(t)
    yel = yrad * np.sin(t)
    xdiff = np.diff(xel)
    ydiff = np.diff(yel)
    dt = t[1] / np.max(np.sqrt(xdiff**2 + ydiff**2))
    dt = dt / 100

    nt = int(2 * np.pi/dt)
    t = np.arange(nt, dtype = np.float64)/(nt-1) * 2 * np.pi
    xel = xrad * np.cos(t)
    yel = yrad * np.sin(t)

    xel += xcenter
    yel += ycenter

    w = np.where(xel - xel.astype(np.int) == 0.5)
    nw = w[0].shape[0]
    if nw != 0:
        xel[w] += -0.0000001

    w = np.where(xel - xel.astype(np.int) == -0.5)
    nw = w[0].shape[0]
    if nw != 0:
        xel[w] += 0.0000001

    w = np.where(yel - yel.astype(np.int) == 0.5)
    nw = w[0].shape[0]
    if nw != 0:
        yel[w] += -0.0000001

    w = np.where(yel - yel.astype(np.int) == -0.5)
    nw = w[0].shape[0]
    if nw != 0:
        yel[w] += 0.0000001

    xel = np.round(xel)
    yel = np.round(yel)

    cond = (xel >= 0) & (xel < ngrid) & (yel >= 0) & (yel < ngrid)
    w = np.where(cond)
    totout = w[0].shape[0]
    if totout == 0:
        return np.ones([ngrid, ngrid], dtype = np.float64)

    xel = xel[w]
    yel = yel[w]

    miny = int(np.min(yel))
    maxy = int(np.max(yel))

    mask = np.zeros([ngrid, ngrid], dtype = np.float64)
    for i in range(np.size(xel)):
        mask[int(yel[i]), int(xel[i])] = 1.0

    w = np.where(mask == 1.0)
    nw = len(w[0])
    mask[:,:] = 0.

    nsub = 11
    x0 =  (np.tile(np.arange(nsub, dtype = np.float64), nsub).reshape(nsub,nsub) - nsub//2)/float(nsub)
    y0 = np.transpose(x0)

    for i in range(0, nw):
        ypix = w[0][i]
        xpix = w[1][i]
        xsub = x0 + xpix
        ysub = y0 + ypix
        pix = ((xsub - xcenter)**2/xrad**2 + (ysub - ycenter)**2/yrad**2) <= 1.0
        tmp = np.sum(pix) / float(nsub)**2
        if tmp > 1.e-8:
            mask[ypix,xpix] = tmp
        else:
            mask[ypix,xpix] = 1.e-8

    x =  (np.arange(ngrid, dtype = np.float64) - xcenter)/xrad

    if miny < 0:
        i1 = 0
    else:
        i1 = miny

    if maxy > ngrid-1:
        i2 = ngrid
    else:
        i2 = maxy + 1

    for j in range(i1, i2):
        y = (j - ycenter) / yrad
        r = np.sqrt(x**2 + y**2)
        w = np.where((mask[j,:] == 0) & (r <= 1.))
        nw = len(w[0])
        if (nw != 0):
            mask[j, w] = 1.

    if dark:
        mask = 1. - mask

    mask[mask < 0.] = 0.
    mask[mask > 1.] = 1.

    return mask
