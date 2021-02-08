#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#   Revised 5 March 2018 - John Krist - Fixed call to prop_cubic_conv by
#   getting rid of the flattening of the coordinate arrays.
#   Revised 28 May 2019 - John Krist - fixed bug caused by assumed integer
#   divides actually being float divides, causing an erroneous 1 pixels offset of
#   the DM surface 
#   Revised 28 Oct 2019 - John Krist - fixed checks of flags (True/False); previously,
#   if a flag keyword was provided, it was treated as True regardless of its value

import os
import proper
import numpy as np
from math import sin, cos
from . import lib_dir
import scipy.signal as ss

if not proper.use_cubic_conv:
    from scipy.ndimage.interpolation import map_coordinates


def prop_dm(wf, dm_z0, dm_xc, dm_yc, spacing = 0., **kwargs):
    """Simulate a deformable mirror of specified actuator spacing, including the
    effects of the DM influence function.

    Parameters
    ----------
    wf : obj
        WaveFront class object

    dm_z0 : str or numpy ndarray
        Either a 2D numpy array containing the surface piston of each DM
        actuator in meters or the name of a 2D FITS image file containing the
        above

    dm_xc, dm_yc : list or numpy ndarray
        The location of the optical axis (center of the wavefront) on the DM in
        actuator units (0 ro num_actuator-1). The center of the first actuator
        is (0.0, 0.0)

    spacing : float
        Defines the spacing in meters between actuators; must not be used when
        n_act_across_pupil is specified.


    Returns
    -------
    dmap : numpy ndarray
        Returns DM surface (not wavefront) map in meters


    Other Parameters
    ----------------
    FIT : bool
       Switch that tells routine that the values in "dm_z" are the desired
       surface heights rather than commanded actuator heights, and so the
       routine should fit this map, accounting for actuator influence functions,
       to determine the necessary actuator heights. An iterative error-minimizing
       loop is used for the fit.

    NO_APPLY : bool
        If set, the DM pattern is not added to the wavefront. Useful if the DM
        surface map is needed but should not be applied to the wavefront

    N_ACT_ACROSS_PUPIL : int
        Specifies the number of actuators that span the X-axis beam diameter. If
        it is a whole number, the left edge of the left pixel is aligned with
        the left edge of the beam, and the right edge of the right pixel with
        the right edge of the beam. This determines the spacing and size of the
        actuators. Should not be used when "spacing" value is specified.

    XTILT, YTILT, ZTILT : float
        Specify the rotation of the DM surface with respect to the wavefront plane
        in degrees about the X, Y, Z axes, respectively, with the origin at the
        center of the wavefront. The DM surface is interpolated and orthographically
        projected onto the wavefront grid. The coordinate system assumes that
        the wavefront and initial DM surface are in the X,Y plane with a lower
        left origin with Z towards the observer. The rotations are left handed.
        The default rotation order is X, Y, then Z unless the /ZYX switch is set.

    XYZ or ZYX : bool
        Specifies the rotation order if two or more of XTILT, YTILT, or ZTILT
        are specified. The default is /XYZ for X, Y, then Z rotations.


    Raises
    ------
    ValueError:
        User cannot specify both actuator spacing and N_ACT_ACROSS_PUPIL

    ValueError:
        User must specify either actuator spacing or N_ACT_ACROSS_PUPIL
    """
    if proper.switch_set("ZYX",**kwargs) and proper.switch_set("XYZ",**kwargs):
        raise ValueError('PROP_DM: Error: Cannot specify both XYZ and ZYX rotation orders. Stopping')
    elif not "ZYX" in kwargs and not 'XYZ' in kwargs:
        XYZ = 1    # default is rotation around X, then Y, then Z
        ZYX = 0
    elif proper.switch_set("ZYX",**kwargs):
        ZYX = 1
        XYZ = 0
    elif proper.switch_set("XYZ",**kwargs):
        XYZ = 1
        ZYX = 0

    if "XTILT" in kwargs:
        xtilt = kwargs["XTILT"]
    else:
        xtilt = 0.

    if "YTILT" in kwargs:
        ytilt = kwargs["YTILT"]
    else:
        ytilt = 0.

    if "ZTILT" in kwargs:
        ztilt = kwargs["ZTILT"]
    else:
        ztilt = 0.

    if type(dm_z0) == str:
        dm_z = proper.prop_fits_read(dm_z0) # Read DM setting from FITS file
    else:
        dm_z = dm_z0

    n = proper.prop_get_gridsize(wf)
    dx_surf = proper.prop_get_sampling(wf)  # sampling of current surface in meters
    beamradius = proper.prop_get_beamradius(wf)

    # influence function sampling is 0.1 mm, peak at (x,y)=(45,45)
    # Influence function has shape = 1x91x91. Saving it as a 2D array
    # before continuing with processing
    inf = proper.prop_fits_read(os.path.join(lib_dir, "influence_dm5v2.fits"))
    inf = inf[0,:,:]

    s = inf.shape
    nx_inf = s[1]
    ny_inf = s[0]
    xc_inf = nx_inf // 2
    yc_inf = ny_inf // 2
    dx_inf = 0.1e-3            # influence function spacing in meters
    dx_dm_inf = 1.e-3          # spacing between DM actuators in meters assumed by influence function
    inf_mag = 10

    if spacing != 0 and "N_ACT_ACROSS_PUPIL" in kwargs:
        raise ValueError("PROP_DM: User cannot specify both actuator spacing and N_ACT_ACROSS_PUPIL. Stopping.")

    if spacing == 0 and not "N_ACT_ACROSS_PUPIL" in kwargs:
        raise ValueError("PROP_DM: User must specify either actuator spacing or N_ACT_ACROSS_PUPIL. Stopping.")

    if "N_ACT_ACROSS_PUPIL" in kwargs:
        dx_dm = 2. * beamradius / int(kwargs["N_ACT_ACROSS_PUPIL"])
    else:
        dx_dm = spacing

    dx_inf = dx_inf * dx_dm / dx_dm_inf   # Influence function sampling scaled
                                          # to specified DM actuator spacing

    if proper.switch_set("FIT",**kwargs):
        x = (np.arange(5, dtype = np.float64) - 2) * dx_dm

        if proper.use_cubic_conv:
            inf_kernel = proper.prop_cubic_conv(inf.T, x/dx_inf+xc_inf, x/dx_inf+yc_inf, GRID=True)
        else:
            xygrid = np.meshgrid(x/dx_inf+xc_inf, x/dx_inf+yc_inf)
            inf_kernel = map_coordinates(inf.T, xygrid, order = 3, mode = "nearest")

        (dm_z_commanded, dms) = proper.prop_fit_dm(dm_z, inf_kernel)
    else:
        dm_z_commanded = dm_z

    s = dm_z.shape
    nx_dm = s[1]
    ny_dm = s[0]

    # Create subsampled DM grid
    margin = 9 * inf_mag
    nx_grid = nx_dm * inf_mag + 2 * margin
    ny_grid = ny_dm * inf_mag + 2 * margin
    xoff_grid = margin + inf_mag/2           # pixel location of 1st actuator center in subsampled grid
    yoff_grid = xoff_grid
    dm_grid = np.zeros([ny_grid, nx_grid], dtype = np.float64)

    x = np.arange(nx_dm, dtype = np.int16) * int(inf_mag) + int(xoff_grid)
    y = np.arange(ny_dm, dtype = np.int16) * int(inf_mag) + int(yoff_grid)
    dm_grid[np.tile(np.vstack(y), (nx_dm,)), np.tile(x, (ny_dm,1))] = dm_z_commanded

    newinf = np.zeros( inf.shape, dtype=np.float64 )  # fix for Python >=3.8, force inf to be numpy array
    newinf[:,:] = inf
    dm_grid = ss.fftconvolve(dm_grid, newinf, mode = 'same')

    # 3D rotate DM grid and project orthogonally onto wavefront
    xdim = int(np.round(np.sqrt(2) * nx_grid * dx_inf / dx_surf)) # grid dimensions (pix) projected onto wavefront
    ydim = int(np.round(np.sqrt(2) * ny_grid * dx_inf / dx_surf))

    if xdim > n: xdim = n

    if ydim > n: ydim = n

    x = np.ones((ydim,1), dtype = np.int) * ((np.arange(xdim) - xdim // 2) * dx_surf)
    y = (np.ones((xdim,1), dtype = np.int) * ((np.arange(ydim) - ydim // 2) * dx_surf)).T

    a = xtilt * np.pi / 180
    b = ytilt * np.pi / 180
    g = ztilt * np.pi /180

    if XYZ:
        m = np.array([ [cos(b)*cos(g), -cos(b)*sin(g), sin(b), 0],
            [cos(a)*sin(g) + sin(a)*sin(b)*cos(g), cos(a)*cos(g)-sin(a)*sin(b)*sin(g), -sin(a)*cos(b), 0],
            [sin(a)*sin(g)-cos(a)*sin(b)*cos(g), sin(a)*cos(g)+cos(a)*sin(b)*sin(g), cos(a)*cos(b), 0],
            [0, 0, 0, 1] ])
    else:
        m = np.array([[cos(b)*cos(g), cos(g)*sin(a)*sin(b)-cos(a)*sin(g), cos(a)*cos(g)*sin(b)+sin(a)*sin(g), 0],
            [cos(b)*sin(g), cos(a)*cos(g)+sin(a)*sin(b)*sin(g), -cos(g)*sin(a)+cos(a)*sin(b)*sin(g), 0],
            [-sin(b), cos(b)*sin(a), cos(a)*cos(b), 0],
            [0, 0, 0, 1] ])

    # Forward project a square
    edge = np.array([[-1.0,-1.0,0.0,0.0], [1.0,-1.0,0.0,0.0], [1.0,1.0,0.0,0.0], [-1.0,1.0,0.0,0.0]])
    new_xyz = np.dot(edge, m)

    # determine backward projection for screen-raster-to-DM-surce computation
    dx_dxs = (new_xyz[0,0] - new_xyz[1,0]) / (edge[0,0] - edge[1,0])
    dx_dys = (new_xyz[1,0] - new_xyz[2,0]) / (edge[1,1] - edge[2,1])
    dy_dxs = (new_xyz[0,1] - new_xyz[1,1]) / (edge[0,0] - edge[1,0])
    dy_dys = (new_xyz[1,1] - new_xyz[2,1]) / (edge[1,1] - edge[2,1])

    xs = ( x/dx_dxs - y*dx_dys/(dx_dxs*dy_dys) ) / ( 1 - dy_dxs*dx_dys/(dx_dxs*dy_dys) )
    ys = ( y/dy_dys - x*dy_dxs/(dx_dxs*dy_dys) ) / ( 1 - dx_dys*dy_dxs/(dx_dxs*dy_dys) )

    xdm = (xs + dm_xc * dx_dm) / dx_inf + xoff_grid
    ydm = (ys + dm_yc * dx_dm) / dx_inf + yoff_grid

    if proper.use_cubic_conv:
        grid = proper.prop_cubic_conv(dm_grid.T, xdm, ydm, GRID = False)
        grid = grid.reshape([xdm.shape[1], xdm.shape[0]])
    else:
        grid = map_coordinates(dm_grid.T, [xdm, ydm], order = 3, mode = "nearest", prefilter = True)

    dmap = np.zeros([n,n], dtype = np.float64)
    nx_grid, ny_grid = grid.shape
    xmin, xmax = n // 2 - xdim //2, n // 2 - xdim // 2 + nx_grid
    ymin, ymax =  n // 2 - ydim // 2, n // 2 - ydim // 2 + ny_grid
    dmap[ymin:ymax, xmin:xmax] = grid

    if not proper.switch_set("NO_APPLY",**kwargs):
        proper.prop_add_phase(wf, 2 * dmap)            # x2 to convert surface to wavefront error

    return dmap
