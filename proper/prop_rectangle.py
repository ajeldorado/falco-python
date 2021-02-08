#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#   Modified 12 Feb 2020 by J. Krist - fixed bugs when rectangle goes off array


import proper
import numpy as np


def prop_rectangle(wf, xsize, ysize, xc = np.nan, yc = np.nan, **kwargs):
    """Return an image containing a filled rectangle with antialiased edges.

    By default, a clear rectangle (one inside, zero outside) is assumed.
    Typically used to draw apertures or obscurations (like spiders). The
    rectangles are drawn antiliased, so that the edge values correspond to the
    fraction of the pixel area covered by the rectangle.

    Parameters
    ----------
    wf : obj
       WaveFront class object

    xsize : float
       X width of the rectangle in meters (unless norm is specified, in which
       case this is the size relative to the beam diameter).

    ysize : float
       Y width of the rectangle in meters (unless norm is specified, in which
       case this is the size relative to the beam diameter).

    xc : float
        X-center of the rectangle relative to the center of the beam in the
        wavefront array in meters (unless norm is specified, in which case
        this is the default center of the rectangle.

    yc : float
        Y-center of the rectangle relative to the center of the beam in the
        wavefront array in meters (unless norm is specified, in which case
        this is the default center of the rectangle. (0,0) is the center of
        a pixel.


    Returns
    -------
    image : numpy ndarray
        Returns an image array containing an antialiased rectangular mask with
        the same dimensions as the wavefront array.


    Other Parameters
    ----------------
    DARK : bool
        Specifies that the rectangle is filled with zeros and is 1.0 outside,
        rather than the default of zero outside and 1.0 inside.

    NORM : bool
        Specifies that the rectangle dimensions and position are specified
        relative to the beam radius.

    ROTATION : float
        Specifies the angle degrees counter-clockwise to rotate the rectangle
        about its center.
    """
    # grid size, beam radius and sampling
    ngrid = proper.prop_get_gridsize(wf)
    beamradius = proper.prop_get_beamradius(wf)
    sampling = proper.prop_get_sampling(wf)

    mag = proper.antialias_subsampling

    # beam radius in pixels
    pr = beamradius / sampling

    # get keyword argument values
    dark = proper.switch_set("DARK",**kwargs)
    norm = proper.switch_set("NORM",**kwargs)

    if "ROTATION" in kwargs:
        rotation = kwargs["ROTATION"]
    else:
        rotation = 0.0

    # Set xcpix and ycpix values
    if np.isnan(xc):
        xcpix = ngrid // 2
    else:
        if norm:
            xcpix = xc * pr + ngrid // 2
        else:
            xcpix = xc/sampling + ngrid // 2

    if np.isnan(yc):
        ycpix = ngrid // 2
    else:
        if norm:
            ycpix = yc * pr + ngrid // 2
        else:
            ycpix = yc/sampling + ngrid // 2

    # Set xradpix and yradpix
    if norm:
        xradpix = 0.5 * xsize * pr
        yradpix = 0.5 * ysize * pr
    else:
        xradpix = 0.5 * xsize / sampling
        yradpix = 0.5 * ysize / sampling

    # Rotation angle in radians
    angle_rad = rotation * np.pi / 180.

    xp0 = np.array([-xradpix, -xradpix, xradpix, xradpix])
    yp0 = np.array([-yradpix, yradpix, yradpix, -yradpix])
    nvert = 4

    xp = xp0 * np.cos(angle_rad) - yp0 * np.sin(angle_rad) + xcpix
    yp = xp0 * np.sin(angle_rad) + yp0 * np.cos(angle_rad) + ycpix

    image = np.zeros([ngrid, ngrid], dtype = np.float64)

    left = np.where(yp == np.min(yp))
    left = left[0][np.where(xp[left] == np.min(xp[left]))[0]]
    left = left[0]

    if left != nvert -1:
        leftnext = left + 1
    else:
        leftnext = 0

    right = left
    if right != 0:
        rightnext = right - 1
    else:
        rightnext = nvert - 1

    if int(np.round(np.min(yp))) < 0:
        imin = 0
    else:
        imin = int(np.round(np.min(yp)))

    if int(np.round(np.max(yp)))+1 >= ngrid:
        imax = ngrid 
    else:
        imax = int(np.round(np.max(yp))) + 1 

    for ypix in range(imin, imax):
        for ysub in range(0, mag):
            y = ypix - 0.5 + (0.5 + ysub)/mag

            if y < yp[left]:
                continue
            if y > np.max(yp):
                break

            if y >= yp[leftnext]:
                left = leftnext
                if left != nvert-1:
                    leftnext = left + 1
                else:
                    leftnext = 0

            if y >= yp[rightnext]:
                right = rightnext
                if right != 0:
                    rightnext = right - 1
                else:
                    rightnext = nvert - 1

            leftdy = yp[leftnext] - yp[left]
            if leftdy != 0:
                leftdx = xp[leftnext] - xp[left]
                xleft = leftdx/leftdy * (y-yp[left]) + xp[left]
            else:
                xleft = xp[left]

            rightdy = yp[rightnext] - yp[right]
            if rightdy != 0:
                rightdx = xp[rightnext] - xp[right]
                xright = rightdx/rightdy * (y - yp[right]) + xp[right]
            else:
                xright = xp[right]

            xleftpix = int(np.round(xleft))
            xrightpix = int(np.round(xright))

            if xleftpix != xrightpix:
                if (xleftpix >= 0 and xleftpix < ngrid):
                    image[ypix,xleftpix] = image[ypix,xleftpix] + mag * ((xleftpix + 0.5) - xleft)
                if (xrightpix >= 0 and xrightpix < ngrid):
                    image[ypix,xrightpix] = image[ypix,xrightpix] + mag * (xright - (xrightpix - 0.5))
                if (xrightpix - xleftpix > 1 and xleftpix + 1 < ngrid and xrightpix > 0):
                    if xleftpix+1 < 0:
                        imin = 0
                    else:
                        imin = xleftpix+1

                    if xrightpix > ngrid:
                        imax = ngrid
                    else:
                        imax = xrightpix
                    image[ypix,imin:imax] = image[ypix,imin:imax] + mag
            else:
                if xleftpix >= 0 and xleftpix < ngrid:
                    image[ypix,xleftpix] = image[ypix,xleftpix] + mag * (xright - xleft)

    image = image / float(mag)**2

    if dark:
        image = 1.0 - image


    return image
