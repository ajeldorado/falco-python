"""Module for DM-related functions."""
import os
from math import sin, cos, radians

from astropy.io import fits
import numpy as np

import scipy.signal as ss
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.ndimage import convolve
import scipy.sparse

import falco
from falco import check, proper, diff_dm

if not proper.use_cubic_conv:
    try:
        from scipy.ndimage import map_coordinates
    except:
        from scipy.ndimage.interpolation import map_coordinates


def gen_surf_from_act(dm, dx, Nout):
    """
    Compute the surface shape of a deformable mirror using PROPER.

    Parameters
    ----------
    dm : ModelParameters
        Structure containing parameter values for the DM
    dx : float
        Pixel width [meters] at the DM plane
    Nout : int
        Number of points across the array to return at the DM plane

    Returns
    -------
    DMsurf : array_like
        2-D surface map of the DM

    """
    check.real_positive_scalar(dx, 'dx', TypeError)
    check.positive_scalar_integer(Nout, 'Nout', TypeError)
    # if dm.NdmPad % 2 != 0:
    #     raise ValueError('dm.NdmPad must be even')

    # Set the order of operations
    flagXYZ = True
    if hasattr(dm, 'flagZYX'):
        if dm.flagZYX:
            flagXYZ = False

    # Adjust the centering of the output DM surface. The shift needs to be in
    # units of actuators, not meters, for prop_dm.m.
    Darray = Nout*dx  # dm.NdmPad*dx
    Narray = Nout  # dm.NdmPad
    if dm.centering == 'interpixel':
        cshift = -Darray/2./Narray/dm.dm_spacing
    elif dm.centering == 'pixel':
        cshift = 0

    pupil_ratio = 1  # beam diameter fraction
    wl_dummy = 1e-6  # dummy value needed to initialize PROPER (meters)

    bm = proper.prop_begin(Narray*dx, wl_dummy, Narray, pupil_ratio)

    # Apply various constraints to DM commands
    dm = enforce_constraints(dm)

    # Quantization of DM actuation steps based on least significant bit of the
    # DAC (digital-analog converter). In height, so called HminStep
    # If HminStep (minimum step in H) is defined, then quantize the DM voltages
    if hasattr(dm, 'HminStep'):
        if not hasattr(dm, 'HminStepMethod'):
            dm.HminStepMethod = 'round'
        # Discretize/Quantize the DM voltages (creates dm.Vquantized)
        dm = discretize_surf(dm, dm.HminStepMethod)
        heightMap = dm.VtoH*dm.Vquantized
    else:  # Quantization not desired; send raw, continuous voltages
        heightMap = dm.VtoH * dm.V

    if hasattr(dm, 'orientation'):
        if dm.orientation.lower() == 'rot0':
            pass  # no change
        elif dm.orientation.lower() == 'rot90':
            heightMap = np.rot90(heightMap, 1)
        elif dm.orientation.lower() == 'rot180':
            heightMap = np.rot90(heightMap, 2)
        elif dm.orientation.lower() == 'rot270':
            heightMap = np.rot90(heightMap, 3)
        elif dm.orientation.lower() == 'flipxrot0':
            heightMap = np.fliplr(heightMap)
        elif dm.orientation.lower() == 'flipxrot90':
            heightMap = np.rot90(np.fliplr(heightMap), 1)
        elif dm.orientation.lower() == 'flipxrot180':
            heightMap = np.rot90(np.fliplr(heightMap), 2)
        elif dm.orientation.lower() == 'flipxrot270':
            heightMap = np.rot90(np.fliplr(heightMap), 3)
        else:
            raise ValueError('invalid value of dm.orientation')

    if dm.useDifferentiableModel:
        if not hasattr(dm, "differentiableModel"):  # or dm.differentiableModel.Nout!=Narray:
            # Initialize the model object if this is the first time
            print("Initializing differentiable DM model.")
            if flagXYZ:
                dm.differentiableModel = diff_dm.dm_init_falco_wrapper(
                    dm, dx, Nout, heightMap, dm.xc-cshift, dm.yc-cshift, 
                    spacing=dm.dm_spacing, XTILT=dm.xtilt, YTILT=dm.ytilt, ZTILT=dm.zrot,
                    XYZ=True, inf_sign=dm.inf_sign, inf_fn=dm.inf_fn,
                )
            else:
                dm.differentiableModel = diff_dm.dm_init_falco_wrapper(
                    dm, dx, Nout, heightMap, dm.xc-cshift, dm.yc-cshift, 
                    spacing=dm.dm_spacing, XTILT=dm.xtilt, YTILT=dm.ytilt, ZTILT=dm.zrot,
                    ZYX=True, inf_sign=dm.inf_sign, inf_fn=dm.inf_fn,
                )
        else:
            dm.differentiableModel.update(heightMap)
            
        DMsurf = dm.differentiableModel.render(Nout=Nout,wfe=False)  # returns surface rather than wfe
        if (DMsurf.shape[0] != Nout) or (DMsurf.shape[1] != Nout):
            raise RuntimeError("Differentiable DM Model output size does not match the requested array size!")
        
        # # Convert surface to WFE like at the end of propcustom_dm??
        # proper.prop_add_phase(bm, 2 * DMsurf)

    else:
        if flagXYZ:
            DMsurf = falco.dm.propcustom_dm(
                bm, heightMap, dm.xc-cshift, dm.yc-cshift, dm.dm_spacing,
                XTILT=dm.xtilt, YTILT=dm.ytilt, ZTILT=dm.zrot, XYZ=True,
                inf_sign=dm.inf_sign, inf_fn=dm.inf_fn,
            )
        else:
            DMsurf = falco.dm.propcustom_dm(
                bm, heightMap, dm.xc-cshift, dm.yc-cshift, dm.dm_spacing,
                XTILT=dm.xtilt, YTILT=dm.ytilt, ZTILT=dm.zrot, ZYX=True,
                inf_sign=dm.inf_sign, inf_fn=dm.inf_fn,
            )

    # DMsurf = falco.util.pad_crop(DMsurf, Nout)

    return DMsurf


def discretize_surf(dm, HminStepMethod):
    """
    Discretize the DM commands used to make the DM surface map.

    Parameters
    ----------
    dm : TYPE
        DESCRIPTION.
    HminStepMethod : TYPE
        DESCRIPTION.

    Raises
    ------
    TypeError
        DESCRIPTION.
    ValueError
        DESCRIPTION.

    Returns
    -------
    dm : TYPE
        DESCRIPTION.

    """
    if not isinstance(HminStepMethod, str):
        raise TypeError('HminStepMethod must be a str')

    # Use HminStep for dm model
    HminStep = dm.HminStep

    # Calculate surface heights to maximum precision
    h_cont = dm.VtoH*dm.V

    # Calculate number of (fractional) steps required to represent the surface
    nSteps = h_cont/HminStep

    # Discretize by removing fractional step
    if 'round' in HminStepMethod.lower():
        nSteps = np.round(nSteps)
    elif 'floor' in HminStepMethod.lower():
        nSteps = np.floor(nSteps)
    elif 'ceil' in HminStepMethod.lower():
        nSteps = np.ceil(nSteps)
    elif 'fix' in HminStepMethod.lower():
        nSteps = np.fix(nSteps)
    else:
        raise ValueError('Method for rounding must be one of: round, floor,' +
                         ' ceil, or fix')

    # Calculate discretized surface
    dm.Vquantized = nSteps*HminStep/dm.VtoH

    return dm


def propcustom_dm(wf, dm_z0, dm_xc, dm_yc, spacing=0., **kwargs):
    """
    Generate a deformable mirror surface almost exactly like PROPER.

    Simulate a deformable mirror of specified actuator spacing, including the
    effects of the DM influence function. Has two more optional keywords
    compared to  proper.prop_dm

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
    inf_fn : string
        specify a new influence function as a FITS file with the same header keywords as 
        PROPER's default influence function. Needs these values in info.PrimaryData.Keywords:
            'P2PDX_M' % pixel width x (m)
            'P2PDY_M' % pixel width y (m)
            'C2CDX_M' % actuator pitch x (m)
            'C2CDY_M' % actuator pitch y (m)
    inf_sign : {+,-}
        specifies the sign (+/-) of the influence function. Given as an option because 
        the default influence function file is positive, but positive DM actuator 
        commands make a negative deformation for Xinetics and BMC DMs.

    Raises
    ------
    ValueError:
        User cannot specify both actuator spacing and N_ACT_ACROSS_PUPIL
    ValueError:
        User must specify either actuator spacing or N_ACT_ACROSS_PUPIL
    """
    if "ZYX" in kwargs and "XYZ" in kwargs:
        raise ValueError('PROP_DM: Error: Cannot specify both XYZ and ZYX ' +
                         'rotation orders. Stopping')
    elif "ZYX" not in kwargs and 'XYZ' not in kwargs:
        XYZ = 1  # default is rotation around X, then Y, then Z
        # ZYX = 0
    elif "ZYX" in kwargs:
        # ZYX = 1
        XYZ = 0
    elif "XYZ" in kwargs:
        XYZ = 1
        # ZYX = 0

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

    if isinstance(dm_z0, str):
        dm_z = proper.prop_fits_read(dm_z0)  # Read DM setting from FITS file
    else:
        dm_z = dm_z0

    if "inf_fn" in kwargs:
        inf_fn = kwargs["inf_fn"]
    else:
        inf_fn = "influence_dm5v2.fits"

    if "inf_sign" in kwargs:
        if kwargs["inf_sign"] == '+':
            sign_factor = 1.
        elif kwargs["inf_sign"] == '-':
            sign_factor = -1.
    else:
        sign_factor = 1.

    n = proper.prop_get_gridsize(wf)
    dx_surf = proper.prop_get_sampling(wf)  # sampling of surface in meters
    beamradius = proper.prop_get_beamradius(wf)

    # Default influence function sampling is 0.1 mm, peak at (x,y)=(45,45)
    # Default influence function has shape = 1x91x91. Saving it as a 2D array
    # before continuing with processing
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "data")
    inf = proper.prop_fits_read(os.path.join(dir_path, inf_fn))
    inf = sign_factor*np.squeeze(inf)

    s = inf.shape
    nx_inf = s[1]
    ny_inf = s[0]
    xc_inf = nx_inf // 2
    yc_inf = ny_inf // 2
    # dx_inf = 0.1e-3  # influence function spacing in meters
    # dx_dm_inf = 1.0e-3  # nominal spacing between DM actuators in meters
    # inf_mag = 10
    header = fits.getheader(inf_fn)
    dx_inf = header["P2PDX_M"]  # pixel width in meters
    dx_dm_inf = header["C2CDX_M"]  # center2cen dist of actuators in meters
    inf_mag = round(dx_dm_inf/dx_inf)
    if np.abs(inf_mag - dx_dm_inf/dx_inf) > 1e-8:
        raise ValueError('%s must have an integer number of pixels per actuator' % (inf_fn))

    if spacing != 0 and "N_ACT_ACROSS_PUPIL" in kwargs:
        raise ValueError("PROP_DM: User cannot specify both actuator spacing" +
                         "and N_ACT_ACROSS_PUPIL. Stopping.")

    if spacing == 0 and "N_ACT_ACROSS_PUPIL" not in kwargs:
        raise ValueError("PROP_DM: User must specify either actuator spacing" +
                         " or N_ACT_ACROSS_PUPIL. Stopping.")

    if "N_ACT_ACROSS_PUPIL" in kwargs:
        dx_dm = 2. * beamradius / int(kwargs["N_ACT_ACROSS_PUPIL"])
    else:
        dx_dm = spacing

    # Influence function sampling scaled to specified DM actuator spacing
    dx_inf = dx_inf * dx_dm / dx_dm_inf  

    if "FIT" in kwargs:
        x = (np.arange(5, dtype=np.float64) - 2) * dx_dm

        if proper.use_cubic_conv:
            inf_kernel = proper.prop_cubic_conv(inf.T, x/dx_inf+xc_inf,
                                                x/dx_inf+yc_inf, GRID=True)
        else:
            xygrid = np.meshgrid(x/dx_inf+xc_inf, x/dx_inf+yc_inf)
            inf_kernel = map_coordinates(inf.T, xygrid, order=3,
                                         mode="nearest")

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
    xoff_grid = margin + inf_mag//2  # pixel location of 1st actuator center in subsampled grid
    yoff_grid = xoff_grid
    dm_grid = np.zeros([ny_grid, nx_grid], dtype=float)

    x = np.arange(nx_dm, dtype=int) * int(inf_mag) + int(xoff_grid)
    y = np.arange(ny_dm, dtype=int) * int(inf_mag) + int(yoff_grid)
    dm_grid[np.tile(np.vstack(y), (nx_dm,)),
            np.tile(x, (ny_dm, 1))] = dm_z_commanded
    # hdu = fits.PrimaryHDU(dm_grid)
    # hdu.writeto('/Users/ajriggs/Downloads/dm_grid_every4.fits', overwrite=True)
    dm_grid = ss.fftconvolve(dm_grid, inf, mode='same')
    # hdu = fits.PrimaryHDU(dm_grid)
    # hdu.writeto('/Users/ajriggs/Downloads/dm_grid_every4_after_convolving.fits', overwrite=True)

    # 3D rotate DM grid and project orthogonally onto wavefront
    xdim = int(np.round(np.sqrt(2) * nx_grid * dx_inf / dx_surf))  # grid dimensions (pix) projected onto wavefront
    ydim = int(np.round(np.sqrt(2) * ny_grid * dx_inf / dx_surf))

    if xdim > n:
        xdim = n

    if ydim > n:
        ydim = n

    x = np.ones((ydim, 1)) * ((np.arange(xdim) - xdim // 2) * dx_surf)
    y = (np.ones((xdim, 1)) * ((np.arange(ydim) - ydim // 2) * dx_surf)).T
    # x = np.ones((ydim, 1), dtype=int) * ((np.arange(xdim) - xdim // 2) * dx_surf)
    # y = (np.ones((xdim, 1), dtype=int) * ((np.arange(ydim) - ydim // 2) * dx_surf)).T

    a = xtilt * np.pi / 180
    b = ytilt * np.pi / 180
    g = ztilt * np.pi / 180

    if XYZ:
        m = np.array(
            [[cos(b)*cos(g), -cos(b)*sin(g), sin(b), 0],
             [cos(a)*sin(g) + sin(a)*sin(b)*cos(g), cos(a)*cos(g)-sin(a)*sin(b)*sin(g), -sin(a)*cos(b), 0],
             [sin(a)*sin(g)-cos(a)*sin(b)*cos(g), sin(a)*cos(g)+cos(a)*sin(b)*sin(g), cos(a)*cos(b), 0],
             [0, 0, 0, 1],
             ])
    else:
        m = np.array(
            [[cos(b)*cos(g), cos(g)*sin(a)*sin(b)-cos(a)*sin(g), cos(a)*cos(g)*sin(b)+sin(a)*sin(g), 0],
             [cos(b)*sin(g), cos(a)*cos(g)+sin(a)*sin(b)*sin(g), -cos(g)*sin(a)+cos(a)*sin(b)*sin(g), 0],
             [-sin(b), cos(b)*sin(a), cos(a)*cos(b), 0],
             [0, 0, 0, 1]
             ])

    # Forward project a square
    edge = np.array([[-1.0, -1.0, 0.0, 0.0],
                     [1.0, -1.0, 0.0, 0.0],
                     [1.0, 1.0, 0.0, 0.0],
                     [-1.0, 1.0, 0.0, 0.0]])
    new_xyz = np.dot(edge, m)

    # determine backward projection for screen-raster-to-DM-surce computation
    dx_dxs = (new_xyz[0, 0] - new_xyz[1, 0]) / (edge[0, 0] - edge[1, 0])
    dx_dys = (new_xyz[1, 0] - new_xyz[2, 0]) / (edge[1, 1] - edge[2, 1])
    dy_dxs = (new_xyz[0, 1] - new_xyz[1, 1]) / (edge[0, 0] - edge[1, 0])
    dy_dys = (new_xyz[1, 1] - new_xyz[2, 1]) / (edge[1, 1] - edge[2, 1])

    xs = (x/dx_dxs - y*dx_dys/(dx_dxs*dy_dys)) / \
        (1 - dy_dxs*dx_dys/(dx_dxs*dy_dys))
    ys = (y/dy_dys - x*dy_dxs/(dx_dxs*dy_dys)) / \
        (1 - dx_dys*dy_dxs/(dx_dxs*dy_dys))

    xdm = (xs + dm_xc * dx_dm) / dx_inf + xoff_grid
    ydm = (ys + dm_yc * dx_dm) / dx_inf + yoff_grid

    if proper.use_cubic_conv:
        grid = proper.prop_cubic_conv(dm_grid.T, xdm, ydm, GRID=False)
        grid = grid.reshape([xdm.shape[1], xdm.shape[0]])
    else:
        grid = map_coordinates(dm_grid.T, [xdm, ydm], order=3,
                               mode="nearest", prefilter=True)

    dmap = np.zeros([n, n], dtype=np.float64)
    nx_grid, ny_grid = grid.shape
    xmin, xmax = n // 2 - xdim // 2, n // 2 - xdim // 2 + nx_grid
    ymin, ymax = n // 2 - ydim // 2, n // 2 - ydim // 2 + ny_grid
    dmap[ymin:ymax, xmin:xmax] = grid

    if "NO_APPLY" not in kwargs:
        proper.prop_add_phase(wf, 2 * dmap)  # convert surface to WFE

    return dmap


def gen_poke_cube(dm, mp, dx_dm, **kwargs):
    """
    Compute the datacube of each influence function.

    Influence functions are cropped down or padded up
    to the best size for angular spectrum propagation.

    Parameters
    ----------
    dm : ModelParameters
        Structure containing parameter values for the DM
    mp: falco.config.ModelParameters
        Structure of model parameters
    dx_dm : float
        Pixel width [meters] at the DM plane

    Other Parameters
    ----------------
    NOCUBE : bool
       Switch that tells function not to compute the datacube of influence
       functions.

    Returns
    -------
    None
        modifies structure "dm" by reference

    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    check.real_positive_scalar(dx_dm, 'dx_dm', TypeError)

    if "NOCUBE" in kwargs and kwargs["NOCUBE"]:
        flagGenCube = False
    else:
        flagGenCube = True

    # Define this flag if it doesn't exist in the older code for square actuator arrays only
    if not hasattr(dm, 'flag_hex_array'):
        dm.flag_hex_array = False

    # Set the order of operations
    XYZ = True
    if hasattr(dm, 'flagZYX'):
        if dm.flagZYX:
            XYZ = False

    # Compute sampling of the pupil. Assume that it is square.
    dm.dx_dm = dx_dm
    dm.dx = dx_dm

    # Default to being centered on a pixel if not specified
    if not hasattr(dm, 'centering'):
        dm.centering = 'pixel'

    # Compute coordinates of original influence function
    Ninf0 = dm.inf0.shape[0]  # Number of points across inf func at native res
    x_inf0 = np.linspace(-(Ninf0-1)/2., (Ninf0-1)/2., Ninf0)*dm.dx_inf0
    # True for even- or odd-sized influence function maps as long as they are
    # centered on the array.
    [Xinf0, Yinf0] = np.meshgrid(x_inf0, x_inf0)

    # Number of points across the DM surface at native inf func resolution
    Ndm0 = falco.util.ceil_even(Ninf0 + (dm.Nact - 1)*(dm.dm_spacing/dm.dx_inf0))
    # Number of points across the (un-rotated) DM surface at new, desired res.
    dm.NdmMin = falco.util.ceil_even(Ndm0*(dm.dx_inf0/dm.dx))+2.
    # Number of points across the array to fully contain the DM surface at new
    # desired resolution and z-rotation angle.
    dm.Ndm = int(falco.util.ceil_even(
        (abs(np.array([np.sqrt(2.)*cos(radians(45.-dm.zrot)),
                       np.sqrt(2.)*sin(radians(45.-dm.zrot))])).max())*Ndm0*(dm.dx_inf0/dm.dx))+2)

    # Compute list of initial actuator center coordinates (in actutor widths).
    if dm.flag_hex_array:  # Hexagonal, hex-packed grid
        raise ValueError('flag_hex_array option not implemented yet.')
#     Nrings = dm.Nrings;
#     x_vec = [];
#     y_vec = [];
#     % row number (rowNum) is 1 for the center row and 2 is above it, etc.
#     % Nacross is the total number of segments across that row
#     for rowNum = 1:Nrings
#         Nacross = 2*Nrings - rowNum; % Number of actuators across at that row (for hex tiling in a hex shape)
#         yval = sqrt(3)/2*(rowNum-1);
#         bx = Nrings - (rowNum+1)/2; % x offset from origin
# 
#         xs = (0:Nacross-1).' - bx; % x values are 1 apart
#         ys = yval*ones(Nacross,1); % same y-value for the entire row
# 
#         if(rowNum==1)
#             x_vec = [x_vec;xs];
#             y_vec = [y_vec;ys]; 
#         else
#             x_vec = [x_vec;xs;xs];
#             y_vec = [y_vec;ys;-ys]; % rows +/-n have +/- y coordinates
#         end
#     end
    else:  # Square grid [actuator widths]
        [dm.Xact, dm.Yact] = np.meshgrid(np.arange(dm.Nact) - 
                                         dm.xc, np.arange(dm.Nact)-dm.yc)
#        # Use order='F' to compare the final datacube to Matlab's output.
#        #  Otherwise, use C ordering for Python FALCO.
#        x_vec = dm.Xact.reshape(dm.Nact*dm.Nact,order='F')
#        y_vec = dm.Yact.reshape(dm.Nact*dm.Nact,order='F')
        x_vec = dm.Xact.reshape(dm.Nact*dm.Nact)
        y_vec = dm.Yact.reshape(dm.Nact*dm.Nact)

    dm.NactTotal = x_vec.shape[0]  # Total number of actuators in the 2-D array
    dm.xy_cent_act = np.zeros((2, dm.NactTotal))  # Initialize

    # Compute the rotation matrix to apply to the influence function and
    #  actuator center locations
    tlt = np.zeros(3)
    tlt[0] = radians(dm.xtilt)
    tlt[1] = radians(dm.ytilt)
    tlt[2] = radians(-dm.zrot)

    sa = sin(tlt[0])
    ca = cos(tlt[0])
    sb = sin(tlt[1])
    cb = cos(tlt[1])
    sg = sin(tlt[2])
    cg = cos(tlt[2])

    if XYZ:
        Mrot = np.array(
            [[cb * cg, sa * sb * cg - ca * sg, ca * sb * cg + sa * sg, 0.0],
             [cb * sg, sa * sb * sg + ca * cg, ca * sb * sg - sa * cg, 0.0],
             [-sb, sa * cb, ca * cb, 0.0],
             [0.0, 0.0, 0.0, 1.0]])
    else:
        Mrot = np.array(
            [[cb * cg, -cb * sg, sb, 0.0],
             [ca * sg + sa * sb * cg, ca * cg - sa * sb * sg, -sa * cb, 0.0],
             [sa * sg - ca * sb * cg, sa * cg + ca * sb * sg, ca * cb, 0.0],
             [0.0, 0.0, 0.0, 1.0]])

    # # Compute the actuator center coordinates in units of actuator spacings
    # for iact in range(dm.NactTotal):
    #     xyzVals = np.array([x_vec[iact], y_vec[iact], 0., 1.])
    #     xyzValsRot = Mrot @ xyzVals
    #     dm.xy_cent_act[0, iact] = xyzValsRot[0].copy()
    #     dm.xy_cent_act[1, iact] = xyzValsRot[1].copy()

    actIndMat = np.arange(dm.Nact**2, dtype=int).reshape((dm.Nact, dm.Nact))
    if hasattr(dm, 'orientation'):
        if dm.orientation.lower() == 'rot0':
            pass  # no change
        elif dm.orientation.lower() == 'rot90':
            actIndMat = np.rot90(actIndMat, -1)
        elif dm.orientation.lower() == 'rot180':
            actIndMat = np.rot90(actIndMat, -2)
        elif dm.orientation.lower() == 'rot270':
            actIndMat = np.rot90(actIndMat, -3)
        elif dm.orientation.lower() == 'flipxrot0':
            actIndMat = np.fliplr(actIndMat)
        elif dm.orientation.lower() == 'flipxrot90':
            actIndMat = np.rot90(np.fliplr(actIndMat), 1)
        elif dm.orientation.lower() == 'flipxrot180':
            actIndMat = np.rot90(np.fliplr(actIndMat), 2)
        elif dm.orientation.lower() == 'flipxrot270':
            actIndMat = np.rot90(np.fliplr(actIndMat), 3)
        else:
            raise ValueError('invalid value of dm.orientation')

    # Compute the actuator center coordinates in units of actuator spacings
    for iact, iIndex in enumerate(actIndMat.flatten()):
        xyzVals = np.array([x_vec[iIndex], y_vec[iIndex], 0., 1.])
        xyzValsRot = Mrot @ xyzVals
        dm.xy_cent_act[0, iact] = xyzValsRot[0].copy()
        dm.xy_cent_act[1, iact] = xyzValsRot[1].copy()

    N0 = dm.inf0.shape[0]
    Npad = falco.util.ceil_odd(np.sqrt(2.)*N0)
    inf0pad = np.zeros((Npad, Npad))
    inf0pad[int(np.ceil(Npad/2.)-np.floor(N0/2.)-1):int(np.ceil(Npad/2.)+np.floor(N0/2.)),
            int(np.ceil(Npad/2.)-np.floor(N0/2.)-1):int(np.ceil(Npad/2.)+np.floor(N0/2.))] = dm.inf0

    ydim = inf0pad.shape[0]
    xdim = inf0pad.shape[1]

    xd2 = np.fix(xdim / 2.) + 1
    yd2 = np.fix(ydim / 2.) + 1
    cx = np.arange(xdim) + 1. - xd2
    cy = np.arange(ydim) + 1. - yd2
    [Xs0, Ys0] = np.meshgrid(cx, cy)

    xsNewVec = np.zeros(xdim*xdim)
    ysNewVec = np.zeros(ydim*ydim)
    Xs0Vec = Xs0.reshape(xdim*xdim)
    Ys0Vec = Ys0.reshape(ydim*ydim)

    for ii in range(Xs0.size):
        xyzVals = np.array([Xs0Vec[ii], Ys0Vec[ii], 0., 1.])
        xyzValsRot = Mrot @ xyzVals
        xsNewVec[ii] = xyzValsRot[0]
        ysNewVec[ii] = xyzValsRot[1]

    # Calculate the interpolated DM grid at the new resolution
    # (set extrapolated values to 0.0)
    dm.infMaster = griddata((xsNewVec, ysNewVec), inf0pad.reshape(Npad*Npad),
                            (Xs0, Ys0), method='cubic', fill_value=0.)

    # Crop down the influence function until it has no zero padding left
    infSum = np.sum(dm.infMaster)
    infDiff = 0.
    counter = 0
    while abs(infDiff) <= 1e-7:
        counter = counter + 2
        infDiff = infSum - np.sum(abs(dm.infMaster[int(counter/2):int(-counter/2),
                                                   int(counter/2):int(-counter/2)]))

    # Subtract an extra 2 to negate the extra step that overshoots.
    counter = counter - 2
    Ninf0pad = dm.infMaster.shape[0]-counter
    if counter == 0:
        infMaster2 = dm.infMaster.copy()
    else:
        # The cropped-down influence function
        infMaster2 = dm.infMaster[int(counter/2):int(-counter/2),
                                  int(counter/2):int(-counter/2)].copy()
        dm.infMaster = infMaster2

    Npad = Ninf0pad

    # True for even- or odd-sized influence function maps as long as they are
    # centered on the array.
    x_inf0 = np.linspace(-(Npad-1)/2, (Npad-1)/2., Npad)*dm.dx_inf0
    [Xinf0, Yinf0] = np.meshgrid(x_inf0, x_inf0)

    # Translate and resample the master influence function to be at each 
    # actuator's location in the pixel grid

    # Compute the size of the postage stamps.
    # Number of points across the influence function array at the DM plane's
    # resolution. Want as even
    Nbox = falco.util.ceil_even(Ninf0pad*dm.dx_inf0/dx_dm)
    dm.Nbox = Nbox
    # Also compute their padded sizes for the angular spectrum (AS) propagation
    # between P2 and DM1 or between DM1 and DM2
    # Minimum number of points across for accurate angular spectrum propagation
    Nmin = falco.util.ceil_even(np.max(mp.sbp_centers)*np.max(np.abs(np.array(
        [mp.d_P2_dm1, mp.d_dm1_dm2, (mp.d_P2_dm1+mp.d_dm1_dm2)])))/dx_dm**2)
    # Use a larger array if the max sampling criterion for angular spectrum
    # propagation is violated
    dm.NboxAS = np.max(np.array([Nbox, Nmin]))

    # Pad the pupil to at least the size of the DM(s) surface(s) to allow all
    # actuators to be located outside the pupil.
    # (Same for both DMs)

    # Find actuator farthest from center:
    dm.r_cent_act = np.sqrt(dm.xy_cent_act[0, :]**2 + dm.xy_cent_act[1, :]**2)
    dm.rmax = np.max(np.abs(dm.r_cent_act))
    NpixPerAct = dm.dm_spacing/dx_dm
    if dm.flag_hex_array:
        # padded 2 actuators past the last actuator center to avoid trying to
        # index outside the array
        dm.NdmPad = falco.util.ceil_even((2.*(dm.rmax+2))*NpixPerAct + 1)
    else:
        # DM surface array padded by the width of the padded influence function
        # to prevent indexing outside the array.
        # The 1/2 term is because the farthest actuator center is still half an
        # actuator away from the nominal array edge.
        dm.NdmPad = falco.util.ceil_even(
            (dm.NboxAS + 2.0*(1 + (np.max(np.abs(dm.xy_cent_act.reshape(2*dm.NactTotal)))+0.5)*NpixPerAct)))

    # Compute coordinates (in meters) of the full DM array
    if dm.centering == 'pixel':
        # meters, coords for the full DM arrays. Origin is centered on a pixel
        dm.x_pupPad = np.linspace(-dm.NdmPad/2., (dm.NdmPad/2. - 1),
                                  dm.NdmPad)*dx_dm
    else:
        # meters, coords for the full DM arrays. Origin is interpixel centered
        dm.x_pupPad = np.linspace(-(dm.NdmPad-1)/2., (dm.NdmPad-1)/2.,
                                  dm.NdmPad)*dx_dm

    dm.y_pupPad = dm.x_pupPad

    dm.act_ele = np.arange(dm.NactTotal)  # Include all actuators

    # Make NboxPad-sized postage stamps for each actuator's influence function
    if flagGenCube:
        if not dm.flag_hex_array:
            print("  Influence function padded from %d to %d points for A.S. propagation." % (Nbox, dm.NboxAS))

        print('Computing datacube of DM influence functions... ', end='')

        # Find the locations of the postage stamps arrays in the larger pupilPad array
        dm.xy_cent_act_inPix = dm.xy_cent_act*(dm.dm_spacing/dx_dm)  # Convert units to pupil-plane pixels
        dm.xy_cent_act_inPix = dm.xy_cent_act_inPix + 0.5  # For the half-pixel offset if pixel centered.
        dm.xy_cent_act_box = np.round(dm.xy_cent_act_inPix)  # Center locations of the postage stamps (in between pixels), in actuator widths
        dm.xy_cent_act_box_inM = dm.xy_cent_act_box*dx_dm  # now in meters
        dm.xy_box_lowerLeft = dm.xy_cent_act_box + (dm.NdmPad-Nbox)/2 - 0  # index of pixel in lower left of the postage stamp within the whole pupilPad array. +0 for Python, +1 for Matlab

        # Starting coordinates (in actuator widths) for updated master influence function.
        # This is interpixel centered, so do not translate!
        dm.x_box0 = np.linspace(-(Nbox-1)/2., (Nbox-1)/2., Nbox)*dx_dm
        [dm.Xbox0, dm.Ybox0] = np.meshgrid(dm.x_box0, dm.x_box0)  # meters, interpixel-centered coordinates for the master influence function

        # (Allow for later) Limit the actuators used to those within 1 actuator width of the pupil
        # r_cent_act_box_inM = np.sqrt(dm.xy_cent_act_box_inM[0, :]**2 + dm.xy_cent_act_box_inM[1, :]**2)
        # Compute and store all the influence functions:
        dm.inf_datacube = np.zeros((Nbox, Nbox, dm.NactTotal))  # initialize array of influence function "postage stamps"
        inf_datacube = np.zeros((dm.NactTotal, Nbox, Nbox))

        interp_spline = RectBivariateSpline(x_inf0, x_inf0, dm.infMaster)  # RectBivariateSpline is faster in 2-D than interp2d
        # Refer to https://scipython.com/book/chapter-8-scipy/examples/two-dimensional-interpolation-with-scipyinterpolaterectbivariatespline/

        for iact in range(dm.NactTotal):
            xbox = dm.x_box0 - (dm.xy_cent_act_inPix[0, iact]-dm.xy_cent_act_box[0, iact])*dx_dm  # X = X0 -(x_true_center-x_box_center)
            ybox = dm.x_box0 - (dm.xy_cent_act_inPix[1, iact]-dm.xy_cent_act_box[1, iact])*dx_dm  # Y = Y0 -(y_true_center-y_box_center)
            dm.inf_datacube[:, :, iact] = interp_spline(ybox, xbox)
            inf_datacube[iact, :, :] = interp_spline(ybox, xbox)

        print('done.')

    else:
        dm.act_ele = np.arange(dm.NactTotal)


def apply_neighbor_rule(Vin, Vlim, Nact):
    """
    Apply the neighbor rule to DM commands.

    Find neighboring actuators that exceed a specified difference
    in voltage and to scale down those voltages until the rule is met.

    Parameters
    ----------
    Vin : numpy ndarray
        2-D array of DM voltage commands
    Vlim : float
        maximum difference in command values between neighboring actuators
    Nact : int
        Number of actuators across the DM

    Returns
    -------
    Vout : numpy ndarray
        2-D array of DM voltage commands
    indPair : numpy ndarray
        [nPairs x 2] array of tied actuator linear indices

    """
    check.twoD_array(Vin, 'Vin', TypeError)
    check.real_scalar(Vlim, 'Vlim', TypeError)
    check.positive_scalar_integer(Nact, 'Nact', TypeError)

    Vout = Vin  # Initialize output voltage map
    indPair = np.zeros((0, 2))  # Initialize the paired indices list. [nPairs x 2]

    kx1 = np.array([[0, 1], [1, 1], [1, 0]])              # R1-C1
    kx2 = np.array([[0, 1], [1, 1], [1, 0], [1, -1]])         # R1, C2 - C47
    kx3 = np.array([[1, 0], [1, -1]])                       # R1, C48
    kx4 = np.array([[-1, 1], [0, 1], [1, 1], [1, 0]])         # R2-R47, C1
    kx5 = np.array([[-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]])  # R2-47, C2-47
    kx6 = np.array([[1, 0], [1, -1]])                       # R2-47, C8
    kx7 = np.array([[-1, 1], [0, 1]])                       # R48, C1 - C47
    kx8 = np.array([[-1, -1]])                             # R48, C48

    for jj in range(Nact):            # Row
        for ii in range(Nact):        # Col

            if jj == 0:
                if ii == 0:
                    kx = kx1
                elif ii < Nact-1:
                    kx = kx2
                else:
                    kx = kx3
            elif jj < Nact-1:
                if ii == 0:
                    kx = kx4
                elif ii < Nact-1:
                    kx = kx5
                else:
                    kx = kx6
            else:
                if ii < Nact-1:
                    kx = kx7
                else:
                    kx = kx8

            kr = jj + kx[:, 0]
            kc = ii + kx[:, 1]
            nNbr = kr.size  # length(kr); # Number of neighbors

            if nNbr >= 1:
                for iNbr in range(nNbr):

                    a1 = Vout[jj, ii] - Vout[kr[iNbr], kc[iNbr]]  # Compute the delta voltage

                    if (np.abs(a1) > Vlim):  # If neighbor rule is violated

                        indLinCtr = (ii-1)*Nact + jj  # linear index of center actuator
                        indLinNbr = (kc[iNbr]-1)*Nact + kr[iNbr]  # linear index of neigboring actuator
                        indPair = np.array([indPair, np.array([indLinCtr, indLinNbr]).reshape(1, 2)])
                        indPair = np.vstack([indPair, np.array([indLinCtr, indLinNbr]).reshape(1, 2)])

                        fx = (np.abs(a1) - Vlim) / 2.
                        Vout[jj, ii] = Vout[jj, ii] - np.sign(a1)*fx
                        Vout[kr[iNbr], kc[iNbr]] = Vout[kr[iNbr], kc[iNbr]] + np.sign(a1)*fx

    return Vout, indPair


def enforce_constraints(dm):
    """
    Enforce various constraints on DM actuator commands.

    1) Apply min/max bounds.
    2) Set commands for pinned, railed, or dead actuators.
    3) Determine which actuators violate the neighbor rule.
    4) Set voltages for tied actuators

    Parameters
    ----------
    dm : ModelParameters
        Structure containing parameter values for the DM

    Returns
    -------
    None
        The dm object is changed by reference.

    """
    # 1) Find actuators that exceed min and max values. Any actuators reaching
    # those limits are added to the pinned actuator list.

    # Create dead actuator map if it doesn't exist already
    if not hasattr(dm, 'dead_map'):
        dm.dead_map = np.zeros((dm.Nact, dm.Nact), dtype=bool)  # initialize
        for ii in dm.dead:
            dm.dead_map.ravel()[ii] = True

    # Apply dead actuator map
    dm.V = dm.V * ~dm.dead_map

    # Min voltage limit
    Vtotal = dm.V + dm.biasMap
    new_inds = np.nonzero(Vtotal.flatten() < dm.Vmin)[0]  # linear indices of new actuators breaking their bounds
    new_vals = dm.Vmin*np.ones(new_inds.size)
    dm.pinned = np.hstack([dm.pinned, new_inds])  # Augment the vector of pinned actuator linear indices
    dm.Vpinned = np.hstack([dm.Vpinned, new_vals])  # Augment the vector of pinned actuator values

    # Max voltage limit
    new_inds = np.nonzero(Vtotal.flatten() > dm.Vmax)[0]  # linear indices of new actuators breaking their bounds
    new_vals = dm.Vmax*np.ones(new_inds.size)
    dm.pinned = np.hstack([dm.pinned, new_inds])     # Augment the vector of pinned actuator linear indices
    dm.Vpinned = np.hstack([dm.Vpinned, new_vals])  # Augment the vector of pinned actuator values

    # 2) Enforce pinned (or railed or dead) actuator values
    if dm.pinned.size > 0:
        Vflat = dm.V.flatten()
        Vflat[dm.pinned.astype(int)] = dm.Vpinned
        dm.V = Vflat.reshape(dm.V.shape)

    # # 3) Find which actuators violate the DM neighbor rule. (This restricts 
    # # the maximum voltage between an actuator and each of its 8 neighbors.) 
    # # Add those actuator pairs to the list of tied actuators.
    # if(dm.flagNbrRule):
    #     dm.V, indPair1 = apply_neighbor_rule(dm.V, dm.dVnbr, dm.Nact);
    #     dm.tied = np.vstack([dm.tied, indPair1])  # Tie together actuators violating the neighbor rule

    # 4) Enforce tied actuator pairs
    # In each pair of tied actuators, assign the command for the first actuator to that of the 2nd actuator
    if (dm.tied.size > 0):
        Vflat = dm.V.flatten()
        Vflat[dm.tied[:, 1]] = Vflat[dm.tied[:, 0]]
        dm.V = Vflat.reshape(dm.V.shape)

    return dm


def fit_surf_to_act(dm, surfaceToFit):
    """
    Compute the deformable mirror (DM) commands to best fit a given surface.

    Parameters
    ----------
    dm : ModelParameters
        Structure containing parameter values for the DM
    surfaceToFit : numpy ndarray
        2-D array of the surface heights for the DM to fit

    Returns
    -------
    Vout : numpy ndarray
        2-D array of DM voltage commands
    """
    check.twoD_array(surfaceToFit, 'surfaceToFit', TypeError)

    [mSurface, nSurface] = surfaceToFit.shape

    # Starting influence function (must be square)
    inf1 = dm.inf0
    N1 = inf1.shape[0]
    actres1 = dm.dm_spacing/dm.dx_inf0
    x = np.linspace(-(N1-1.)/2., (N1-1.)/2., N1)/actres1
    [X, Y] = np.meshgrid(x, x)

    # Influence function resampled to actuator map resolution
    actres2 = 1.  # pixels per actuator width
    N2 = falco.util.ceil_even(N1*actres2/actres1)+1  # Make odd to have peak of 1
    xq = np.linspace(-(N2-1)/2, (N2-1)/2, N2)/actres2  # pixel-centered
    # [Xq,Yq] = np.meshgrid(xq)
    # inf2 = interp2(X,Y,inf1,Xq,Yq,'cubic',0); # MATLAB way
    interp_spline = RectBivariateSpline(x, x, inf1)  # RectBivariateSpline is faster in 2-D than interp2d
    infFuncAtActRes = interp_spline(xq, xq)

    # Set the order of operations
    flagXYZ = True
    if hasattr(dm, 'flagZYX'):
        if dm.flagZYX:
            flagXYZ = False

    # Perform the fit
    if nSurface == dm.Nact:
        gridDerotAtActRes = surfaceToFit

    elif nSurface > dm.Nact:
        # Adjust the centering of the output DM surface. The shift needs to be
        # in units of actuators, not meters
        wArray = nSurface*dm.dx
        cshift = -wArray/2./nSurface/dm.dm_spacing if(dm.centering == 'interpixel') else 0.
        if flagXYZ:
            gridDerotAtActRes = derotate_resize_surface(
                surfaceToFit, dm.dx, dm.Nact, dm.xc-cshift, dm.yc-cshift,
                dm.dm_spacing, XTILT=dm.xtilt, YTILT=dm.ytilt, ZTILT=dm.zrot,
                XYZ=True, inf_sign=dm.inf_sign, inf_fn=dm.inf_fn)
        else:
            gridDerotAtActRes = derotate_resize_surface(
                surfaceToFit, dm.dx, dm.Nact, dm.xc-cshift, dm.yc-cshift,
                dm.dm_spacing, XTILT=dm.xtilt, YTILT=dm.ytilt, ZTILT=dm.zrot,
                ZYX=True, inf_sign=dm.inf_sign, inf_fn=dm.inf_fn)

    elif nSurface < dm.Nact:
        raise ValueError('surfaceToFit cannot be smaller than [Nact x Nact].')

    if dm.surfFitMethod.lower() == 'lsq':
        heightAtActRes = fit_surf_with_dm_lsq_wrapper(dm, gridDerotAtActRes)  # Creates the prefilter if it doesn't exist
        # heightAtActRes = fit_surf_with_dm_lsq(gridDerotAtActRes, dm.act_prefilter)  # Assumes prefilter exists already
    elif dm.surfFitMethod.lower() == 'proper':
        [heightAtActRes, surfaceOut] = proper.prop_fit_dm(gridDerotAtActRes, infFuncAtActRes)
    else:
        raise ValueError('Invalid or missing value of dm.surfFitMethod')

    Vout = heightAtActRes/dm.VtoH
    Vout[np.isinf(Vout)] = 0

    return Vout


def gen_surf_from_poke_cube(dm, model_type):
    """
    Produce a DM surface by superposing actuators from a datacube.

    Parameters
    ----------
    dm : numpy ndarray
        2-D array of DM voltage commands
    model_type : {'compact', 'full'}
        String telling whether to make the surface based on the compact
        model or full model.

    Returns
    -------
    dmSurf : numpy ndarray
        2-D array of DM surface map
    """
    if model_type not in ('compact', 'full'):
        raise ValueError("model_type must be 'compact' or 'full'.")

    if model_type == 'compact':
        NdmPad = dm.compact.NdmPad
        Nbox = dm.compact.Nbox
        inf_datacube = dm.compact.inf_datacube
        xy_box_lowerLeft = dm.compact.xy_box_lowerLeft
    elif model_type == 'full':
        NdmPad = dm.NdmPad
        Nbox = dm.Nbox
        inf_datacube = dm.inf_datacube
        xy_box_lowerLeft = dm.xy_box_lowerLeft

    dmSurf = np.zeros((NdmPad, NdmPad))
    for iact in range(dm.NactTotal):
        if np.sum(np.abs(inf_datacube[:, :, iact])) > 1e-12:
            x_box_ind = np.arange(xy_box_lowerLeft[0, iact],
                                  xy_box_lowerLeft[0, iact] + Nbox, dtype=int)
            y_box_ind = np.arange(xy_box_lowerLeft[1, iact],
                                  xy_box_lowerLeft[1, iact] + Nbox, dtype=int)
            inds = np.ix_(y_box_ind, x_box_ind)
            V = dm.V[np.unravel_index(iact, dm.V.shape)]
            gain = dm.VtoH[np.unravel_index(iact, dm.VtoH.shape)]
            dmSurf[inds] += V*gain*inf_datacube[:, :, iact]

    return dmSurf


def derotate_resize_surface(surfaceToFit, dx, Nact, dm_xc, dm_yc, spacing,
                            **kwargs):
    """
    Derotate and resize a DM surface to size and alignment of actuator grid.

    Does the order of operations in the reverse order of PROPER's prop_dm.


    Parameters
    ----------
    surfaceToFit : numpy ndarray
        2-D DM surface map to be fitted
    dx : float
        width of a pixel in meters
    Nact : int
        number of actuators across the DM array
    dm_xc, dm_yc : list or numpy ndarray
        The location of the optical axis (center of the wavefront) on the DM in
        actuator units (0 ro num_actuator-1). The center of the first actuator
        is (0.0, 0.0)
    spacing : float
        Spacing in meters between actuator centers (aka the pitch).

    Returns
    -------
    gridDerotAtActRes : numpy ndarray
        Returns DM surface at same alignment and resolution as the DM actuator
        array.

    Other Parameters
    ----------------

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

   inf_fn : string
        specify a new influence function as a FITS file with the same header keywords as
        PROPER's default influence function. Needs these values in info.PrimaryData.Keywords:
            'P2PDX_M' % pixel width x (m)
            'P2PDY_M' % pixel width y (m)
            'C2CDX_M' % actuator pitch x (m)
            'C2CDY_M' % actuator pitch y (m)

    inf_sign : {+,-}
        specifies the sign (+/-) of the influence function. Given as an option because
        the default influence function file is positive, but positive DM actuator
        commands make a negative deformation for Xinetics and BMC DMs.

    Raises
    ------
    ValueError:
        User cannot specify both ZYX and XYZ rotations.

    """
    check.twoD_array(surfaceToFit, 'surfaceToFit', TypeError)
    check.real_positive_scalar(dx, 'dx', TypeError)
    check.real_scalar(dm_xc, 'dm_xc', TypeError)
    check.real_scalar(dm_yc, 'dm_yc', TypeError)
    check.real_positive_scalar(spacing, 'spacing', TypeError)

    if "ZYX" in kwargs and "XYZ" in kwargs:
        raise ValueError('Error: Cannot specify both XYZ and ZYX rotation' +
                         ' orders. Stopping')
    elif "ZYX" not in kwargs and 'XYZ' not in kwargs:
        XYZ = 1    # default is rotation around X, then Y, then Z
        # ZYX = 0
    elif "ZYX" in kwargs:
        # ZYX = 1
        XYZ = 0
    elif "XYZ" in kwargs:
        XYZ = 1
        # ZYX = 0

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

    dm_z = np.eye(Nact)

    if "inf_fn" in kwargs:
        inf_fn = kwargs["inf_fn"]
    else:
        inf_fn = "influence_dm5v2.fits"

    if "inf_sign" in kwargs:
        if kwargs["inf_sign"] == '+':
            sign_factor = 1.
        elif kwargs["inf_sign"] == '-':
            sign_factor = -1.
    else:
        sign_factor = 1.

    n = surfaceToFit.shape[0]
    dx_surf = dx  # sampling of current surface in meters

    # Default influence function sampling is 0.1 mm, peak at (x,y)=(45,45)
    # Default influence function has shape = 1x91x91. Saving it as a 2D array
    # before continuing with processing
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "data")
    inf = proper.prop_fits_read(os.path.join(dir_path, inf_fn))
    inf = sign_factor*np.squeeze(inf)

    s = inf.shape
    nx_inf = s[1]
    ny_inf = s[0]
    xc_inf = nx_inf // 2
    yc_inf = ny_inf // 2
    # dx_inf = 0.1e-3  # influence function spacing in meters
    # dx_dm_inf = 1.e-3  # spacing between DM actuators in meters assumed by influence function
    # inf_mag = 10

    header = fits.getheader(inf_fn)
    dx_inf = header["P2PDX_M"]  # pixel width in meters
    dx_dm_inf = header["C2CDX_M"]  # center2cen dist of actuators in meters
    inf_mag = round(dx_dm_inf/dx_inf)
    if np.abs(inf_mag - dx_dm_inf/dx_inf) > 1e-8:
        raise ValueError('%s must have an integer number of pixels per actuator' % (inf_fn))

    dx_dm = spacing

    # Influence function sampling scaled to specified DM actuator spacing
    dx_inf = dx_inf * dx_dm / dx_dm_inf

    dm_z_commanded = dm_z

    s = dm_z.shape
    nx_dm = s[1]
    ny_dm = s[0]

    # Create subsampled DM grid
    margin = 9 * inf_mag
    nx_grid = nx_dm * inf_mag + 2 * margin
    ny_grid = ny_dm * inf_mag + 2 * margin
    xoff_grid = margin + inf_mag//2  # pixel location of 1st actuator center in subsampled grid
    yoff_grid = xoff_grid
    dm_grid = np.zeros([ny_grid, nx_grid], dtype=float)

    x = np.arange(nx_dm, dtype=int) * int(inf_mag) + int(xoff_grid)
    y = np.arange(ny_dm, dtype=int) * int(inf_mag) + int(yoff_grid)
    dm_grid[np.tile(np.vstack(y), (nx_dm,)), np.tile(x, (ny_dm, 1))] = dm_z_commanded
    dm_grid = ss.fftconvolve(dm_grid, inf, mode='same')

    # 3D rotate DM grid and project orthogonally onto wavefront
    xdim = int(np.round(np.sqrt(2) * nx_grid * dx_inf / dx_surf))  # grid dimensions (pix) projected onto wavefront
    ydim = int(np.round(np.sqrt(2) * ny_grid * dx_inf / dx_surf))

    if xdim > n:
        xdim = n

    if ydim > n:
        ydim = n

    x = np.ones((ydim, 1), dtype=int) * ((np.arange(xdim) - xdim // 2) * dx_surf)
    y = (np.ones((xdim ,1), dtype=int) * ((np.arange(ydim) - ydim // 2) * dx_surf)).T

    a = xtilt * np.pi / 180
    b = ytilt * np.pi / 180
    g = ztilt * np.pi / 180

    if XYZ:
        m = np.array(
            [[cos(b)*cos(g), -cos(b)*sin(g), sin(b), 0],
             [cos(a)*sin(g) + sin(a)*sin(b)*cos(g), cos(a)*cos(g)-sin(a)*sin(b)*sin(g), -sin(a)*cos(b), 0],
             [sin(a)*sin(g)-cos(a)*sin(b)*cos(g), sin(a)*cos(g)+cos(a)*sin(b)*sin(g), cos(a)*cos(b), 0],
             [0, 0, 0, 1],
             ])
    else:
        m = np.array(
            [[cos(b)*cos(g), cos(g)*sin(a)*sin(b)-cos(a)*sin(g), cos(a)*cos(g)*sin(b)+sin(a)*sin(g), 0],
             [cos(b)*sin(g), cos(a)*cos(g)+sin(a)*sin(b)*sin(g), -cos(g)*sin(a)+cos(a)*sin(b)*sin(g), 0],
             [-sin(b), cos(b)*sin(a), cos(a)*cos(b), 0],
             [0, 0, 0, 1],
             ])

    # Compute xdm0 and ydm0 for use in de-rotating the DM surface
    edge = np.array([[-1.0, -1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]])
    new_xyz0 = edge  # np.dot(edge, m)

    # determine backward projection for screen-raster-to-DM-surce computation
    dx_dxs0 = (new_xyz0[0, 0] - new_xyz0[1, 0]) / (edge[0, 0] - edge[1, 0])
    dx_dys0 = (new_xyz0[1, 0] - new_xyz0[2, 0]) / (edge[1, 1] - edge[2, 1])
    dy_dxs0 = (new_xyz0[0, 1] - new_xyz0[1, 1]) / (edge[0, 0] - edge[1, 0])
    dy_dys0 = (new_xyz0[1, 1] - new_xyz0[2, 1]) / (edge[1, 1] - edge[2, 1])

    xs0 = (x/dx_dxs0 - y*dx_dys0/(dx_dxs0*dy_dys0)) / (1 - dy_dxs0*dx_dys0/(dx_dxs0*dy_dys0))
    ys0 = (y/dy_dys0 - x*dy_dxs0/(dx_dxs0*dy_dys0)) / (1 - dx_dys0*dy_dxs0/(dx_dxs0*dy_dys0))

#     xdm0 = (xs0 + dm_xc * dx_dm) / dx_inf + xoff_grid  # WRONG
#     ydm0 = (ys0 + dm_yc * dx_dm) / dx_inf + yoff_grid  # WRONG
    xdm0 = (xs0 + (Nact-1)/2 * dx_dm) / dx_inf + xoff_grid
    ydm0 = (ys0 + (Nact-1)/2 * dx_dm) / dx_inf + yoff_grid
    ######

    # Forward project a square
    # edge = np.array([[-1.0,-1.0,0.0,0.0], [1.0,-1.0,0.0,0.0],
    #                  [1.0,1.0,0.0,0.0], [-1.0,1.0,0.0,0.0]])
    new_xyz = np.dot(edge, m)

    # determine backward projection for screen-raster-to-DM-surce computation
    dx_dxs = (new_xyz[0, 0] - new_xyz[1, 0]) / (edge[0, 0] - edge[1, 0])
    dx_dys = (new_xyz[1, 0] - new_xyz[2, 0]) / (edge[1, 1] - edge[2, 1])
    dy_dxs = (new_xyz[0, 1] - new_xyz[1, 1]) / (edge[0, 0] - edge[1, 0])
    dy_dys = (new_xyz[1, 1] - new_xyz[2, 1]) / (edge[1, 1] - edge[2, 1])

    xs = (x/dx_dxs - y*dx_dys/(dx_dxs*dy_dys)) / \
        (1 - dy_dxs*dx_dys/(dx_dxs*dy_dys))
    ys = (y/dy_dys - x*dy_dxs/(dx_dxs*dy_dys)) / \
        (1 - dx_dys*dy_dxs/(dx_dxs*dy_dys))

    xdm = (xs + dm_xc * dx_dm) / dx_inf + xoff_grid
    ydm = (ys + dm_yc * dx_dm) / dx_inf + yoff_grid

    # if proper.use_cubic_conv:
    #     grid = proper.prop_cubic_conv(dm_grid.T, xdm, ydm, GRID = False)
    #     grid = grid.reshape([xdm.shape[1], xdm.shape[0]])
    # else:
    #     grid = map_coordinates(dm_grid.T, [xdm, ydm], order=3,
    #                             mode="nearest", prefilter = True)
    # dm_grid = falco.util.pad_crop(surfaceToFit, xdm.shape[0])

    # Derotate the DM surface
    dm_grid = falco.util.pad_crop(surfaceToFit, xdm.shape[0])
    gridDerot = griddata((xdm.flatten(), ydm.flatten()), dm_grid.flatten(),
                         (xdm0, ydm0), method='cubic', fill_value=0.)
    # gridDerot(isnan(gridDerot)) = 0

    # Resize and decimate the DM surface to get it at the same size as the DM
    # actuator command array.
    #  The result will be fed to fit_surf_to_act() for deconvolution with the
    #  influence function.
    xOffsetInAct = ((Nact/2 - 1/2) - dm_xc)
    yOffsetInAct = ((Nact/2 - 1/2) - dm_yc)

    multipleOfCommandGrid = int(falco.util.ceil_odd(spacing/dx))
    N1 = Nact*multipleOfCommandGrid
    N2 = dm_grid.shape[0]
    xs1 = np.linspace(-(N1-1)/2, (N1-1)/2, N1)/N1  # interpixel centered
    if N2 % 2 == 0:
        xs2 = np.linspace(-N2/2, (N2/2)-1, N2)/N2*(N2*dx/(Nact*spacing))
    else:
        xs2 = np.linspace(-(N2-1)/2, (N2-1)/2, N2)/N2*(N2*dx/(Nact*spacing))

    interp_spline = RectBivariateSpline(xs2, xs2, gridDerot)
#     gridDerotResize = interp_spline(xs1-xOffsetInAct/Nact,
#                                     xs1-yOffsetInAct/Nact)  # WRONG!!!
    gridDerotResize = interp_spline(xs1, xs1)

    xyOffset = int(np.floor(multipleOfCommandGrid/2.))
    gridDerotAtActRes = gridDerotResize[xyOffset::multipleOfCommandGrid,
                                        xyOffset::multipleOfCommandGrid]                         		                            

    return gridDerotAtActRes


def make_dm_prefilter_attribute(dm):
    """
    Make the prefilter for least-squares surface fitting if it doesn't exit yet.

    Assumes the surface being fitted is dm.Nact x dm.Nact/

    Parameters
    ----------
    dm : ModelParameters
        Object containing parameter values for the DM

    Returns
    -------
    None
        The dm object is modified by reference.

    """
    if not hasattr(dm, 'act_prefilter'):
        print('Building prefilter for surface fitting with the DM...', end='')
        inf_func = dm.inf0
        ppa_in = dm.ppact
        nrow = dm.Nact
        ncol = dm.Nact
        dm.act_prefilter = build_prefilter(nrow, ncol, inf_func, ppa_in)
        print('done.')

    return None


def fit_surf_with_dm_lsq_wrapper(dm, surfaceToFit):
    """
    Perform a least-squares fit of the surface with the DM.

    This wrapper computes the stored prefilter if it doesn't exist yet,
    or it uses the existing one if it does.

    The prefilter must be Nact by Nact in this implementation.

    Parameters
    ----------
    dm : ModelParameters
        Structure containing parameter values for the DM

    surfaceToFit : numpy ndarray
        2-D array of the surface heights for the DM to fit

    Returns
    -------
    nrow x ncol ndarray of DM poke heights, same units as input surface

    """
    check.twoD_array(surfaceToFit, 'surfaceToFit', TypeError)
    [nrow, ncol] = surfaceToFit.shape
    if nrow != dm.Nact or ncol != dm.Nact:
        raise ValueError('The shape of surfaceToFit must by dm.Nact x dm.Nact.')

    make_dm_prefilter_attribute(dm)

    return fit_surf_with_dm_lsq(surfaceToFit, dm.act_prefilter)


def fit_surf_with_dm_lsq(surf, act_effect):
    """
    Determine DM commands that best match a surface at actuator resolution.

    Given a pre-computed mapping of the effect of each actuator at the location
    of each other A, and a target surf shape b, solve Ax=b to find the DM
    setting x which best reproduces the shape.

    Parameters
    ----------
    surf : numpy ndarray
        should be a nrow x ncol array of surface heights in meters

    act_effect : CSR-type sparse matrix
        should be of size nrow*ncol x nrow*ncol; the output of
        build_prefilter() is a suitable input here

    Returns
    -------
    nrow x ncol ndarray of DM poke heights, same units as input surface

    """
    # Check inputs
    check.twoD_array(surf, 'surf', TypeError)
    if not scipy.sparse.issparse(act_effect):
        raise TypeError('act_effect must be a sparse matrix')
    sr, sc = surf.shape
    ar, ac = act_effect.shape
    if not (ar, ac) == (sr*sc, sr*sc):
        raise TypeError('surf and act_effect must be sized to the same DM')

    # solve and re-square
    x = scipy.sparse.linalg.spsolve(act_effect, surf.ravel())
    return np.reshape(x, surf.shape)


def build_prefilter(nrow, ncol, inf_func, ppa_in):
    """
    Build a prefilter.

    The influence function of a DM actuator has a finite extent, and so we can
    map the effect of each actuator on the others by brute force.  For an NxN
    matrix, we can assemble an N^2xN^2 sparse matrix which has the effect of
    poking each actuator on all others in the row.  (Same principle for
    rectangular arrays.)

    This approach is identical to the prefiltering step used when fitting to
    higher-order B-splines for interpolation, although the exact shape of the
    B-spline can be used to make that much faster than the step here.

    Parameters
    ----------
    nrow : int
        Number of rows along one edge of the DM

    ncol : int
        Number of columns along one edge of the DM

    inf_func : numpy ndarray
        2D array with nominal influence function

    ppa_in : float
        Pixels per actuator for inf_func, must be > 0


    Returns
    -------
    CSR-type sparse matrix of size nrow*ncol x nrow*ncol

    """
    check.positive_scalar_integer(nrow, 'nrow', TypeError)
    check.positive_scalar_integer(ncol, 'ncol', TypeError)
    check.twoD_array(inf_func, 'inf_func', TypeError)
    check.real_positive_scalar(ppa_in, 'ppa_in', TypeError)

    # lil_matrix is a good sparse format for incremental build; switch to
    # CSR for operations
    act_effect = scipy.sparse.lil_matrix((nrow*ncol, nrow*ncol))

    # Influence function resampled to actuator map resolution
    ppa_out = 1.  # pixels per actuator; by def'n DM map is 1 pixel/actuator
    inf_func_actres = resample_inf_func(inf_func, ppa_in, ppa_out)

    single_poke = np.zeros((nrow, ncol))

    for j in range(nrow*ncol):
        single_poke.ravel()[j] = 1
        dm_surface = convolve(single_poke, inf_func_actres, mode='constant',
                              cval=0.0)
        single_poke.ravel()[j] = 0  # prep for next
        act_effect[j, :] = dm_surface.ravel()
        pass

    return act_effect.tocsr()  # Want CSR for fast matrix solve later


def resample_inf_func(inf_func, ppa_in, ppa_out):
    """
    Resample an influence function at a new pixels-per-actuator sampling.

    Uses spline interpolation to do the job.

    Parameters
    ----------
    inf_func : numpy ndarray
        2D array with nominal influence function

    ppa_in : float
        Pixels per actuator for inf_func, must be > 0

    ppa_out : float
        Target pixels per actuator in resampled influence function, must be > 0


    Returns
    -------
    2D array with resampled influence function

    """
    check.twoD_array(inf_func, 'inf_func', TypeError)
    check.real_positive_scalar(ppa_in, 'ppa_in', TypeError)
    check.real_positive_scalar(ppa_out, 'ppa_out', TypeError)

    if not ppa_in == ppa_out:
        # Get coords for pixels centers along rows/cols
        nr0, nc0 = inf_func.shape
        r0 = np.linspace(-(nr0-1.)/2., (nr0-1.)/2., nr0)/ppa_in
        c0 = np.linspace(-(nc0-1.)/2., (nc0-1.)/2., nc0)/ppa_in

        # Make output coords, possibly undersized
        # Make odd to have peak of 1
        nr1 = int(2*np.floor(0.5*nr0*ppa_out/ppa_in)+1)
        nc1 = int(2*np.floor(0.5*nc0*ppa_out/ppa_in)+1)

        r1 = np.linspace(-(nr1-1)/2., (nr1-1)/2., nr1)/ppa_out
        c1 = np.linspace(-(nc1-1)/2., (nc1-1)/2., nc1)/ppa_out
        interp_spline = RectBivariateSpline(r0, c0, inf_func)
        inf_func_actres = interp_spline(r1, c1)
    else:
        inf_func_actres = inf_func

    return inf_func_actres

def update_gain_map(dm):
    """
    Update the deformable mirror (DM) gain map based on the specified fit type.

    Parameters:
    -----------
    dm : object
        A deformable mirror object with various attributes

    Returns:
    --------
    dm : object
        The updated deformable mirror object
    """

    if dm.fitType.lower() in ['linear', 'poly1']:
        # No change to dm.VtoH
        pass

    elif dm.fitType.lower() in ['quadratic', 'poly2']:
        if not hasattr(dm, 'p1') or not hasattr(dm, 'p2') or not hasattr(dm, 'p3'):
            error_msg = ("The fields p1, p2, and p3 must exist when dm.fitType == 'quadratic'.\n"
                         "Those fields satisfy the formula:\n"
                         "height = p1*V*V + p2*V + p3")
            raise ValueError(error_msg)

        Vtotal = dm.V + dm.biasMap
        dm.VtoH = 2 * dm.p1 * Vtotal + dm.p2

    elif dm.fitType.lower() == 'fourier2':
        if (not hasattr(dm, 'a0') or not hasattr(dm, 'a1') or not hasattr(dm, 'a2') or
                not hasattr(dm, 'b1') or not hasattr(dm, 'b2') or not hasattr(dm, 'w')):
            error_msg = ("The fields a0, a1, a2, b1, b2, and w must exist when dm.fitType == 'fourier2'.\n"
                         "Those fields satisfy the formula:\n"
                         "height = a0 + a1*cos(V*w) + b1*sin(V*w) + a2*cos(2*V*w) + b2*sin(2*V*w)")
            raise ValueError(error_msg)

        Vtotal = dm.V + dm.biasMap
        dm.VtoH = dm.w * (-dm.a1 * np.sin(Vtotal * dm.w) + dm.b1 * np.cos(Vtotal * dm.w) +
                          -2 * dm.a2 * np.sin(2 * Vtotal * dm.w) + 2 * dm.b2 * np.cos(2 * Vtotal * dm.w))

    else:
        raise ValueError('Value of dm.fitType not recognized.')

    return dm

