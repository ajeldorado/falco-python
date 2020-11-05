import os
import numpy as np
from math import sin, cos, radians
import scipy.signal as ss
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline

import proper
import falco
from falco import check

if not proper.use_cubic_conv:
    from scipy.ndimage.interpolation import map_coordinates
    
# import matplotlib.pyplot as plt
# from astropy.io import fits
    
def gen_surf_from_act(dm, dx, N):
    """
    Function to compute the surface shape of a deformable mirror. Uses PROPER.

    Parameters
    ----------
    dm : ModelParameters
        Structure containing parameter values for the DM
    dx : float
        Pixel width [meters] at the DM plane
    N : int
        Number of points across the array to return at the DM plane

    Returns
    -------
    DMsurf : array_like
        2-D surface map of the DM

    """
    check.real_positive_scalar(dx, 'dx', TypeError)
    check.positive_scalar_integer(N, 'N', TypeError)
    # if type(dm) is not falco.config.Object:
    #     raise TypeError('Input "dm" must be of type falco.config.Object')
        
    # Set the order of operations
    flagXYZ = True
    if(hasattr(dm, 'flagZYX')):
        if(dm.flagZYX):
            flagXYZ = False

    # Adjust the centering of the output DM surface. The shift needs to be in
    # units of actuators, not meters, for prop_dm.m.
    Darray = dm.NdmPad*dm.dx
    Narray = dm.NdmPad
    if dm.centering == 'interpixel':
        cshift = -Darray/2./Narray/dm.dm_spacing
    elif dm.centering == 'pixel':
        cshift = 0

    pupil_ratio = 1  # beam diameter fraction
    wl_dummy = 1e-6  # dummy value needed to initialize PROPER (meters)

    bm = proper.prop_begin(N*dx, wl_dummy, N, pupil_ratio)

    # Apply various constraints to DM commands
    dm = enforce_constraints(dm)

    # Quantization of DM actuation steps based on least significant bit of the
    # DAC (digital-analog converter). In height, so called HminStep
    # If HminStep (minimum step in H) is defined, then quantize the DM voltages
    if(hasattr(dm, 'HminStep')):
        if not(hasattr(dm, 'HminStepMethod')):
            dm.HminStepMethod = 'round'
        # Discretize/Quantize the DM voltages (creates dm.Vquantized)
        dm = discretize_surf(dm, dm.HminStepMethod)
        H = dm.VtoH*dm.Vquantized
    else:  # Quantization not desired; send raw, continuous voltages
        H = dm.VtoH*dm.V
    
    # Generate the DM surface
    DMsurf = falco.dm.propcustom_dm(bm, H, dm.xc-cshift, dm.yc-cshift,
    dm.dm_spacing, XTILT=dm.xtilt, YTILT=dm.ytilt, ZTILT=dm.zrot, XYZ=flagXYZ,
    inf_sign=dm.inf_sign, inf_fn=dm.inf_fn)
        
    return DMsurf


def discretize_surf(dm, HminStepMethod):

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

    if type(dm_z0) == str:
        dm_z = proper.prop_fits_read(dm_z0)  # Read DM setting from FITS file
    else:
        dm_z = dm_z0
    
    if "inf_fn" in kwargs:
        inf_fn = kwargs["inf_fn"]
    else:
        inf_fn = "influence_dm5v2.fits"
        
    if "inf_sign" in kwargs:
        if(kwargs["inf_sign"] == '+'):
            sign_factor = 1.
        elif(kwargs["inf_sign"] == '-'):
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
    dx_inf = 0.1e-3  # influence function spacing in meters
    dx_dm_inf = 1.0e-3  # nominal spacing between DM actuators in meters
    inf_mag = 10

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

    dx_inf = dx_inf * dx_dm / dx_dm_inf  # Influence function sampling scaled
                                         # to specified DM actuator spacing

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
    xoff_grid = margin + inf_mag/2           # pixel location of 1st actuator center in subsampled grid
    yoff_grid = xoff_grid
    dm_grid = np.zeros([ny_grid, nx_grid], dtype = np.float64)

    x = np.arange(nx_dm, dtype=np.int16) * int(inf_mag) + int(xoff_grid)
    y = np.arange(ny_dm, dtype=np.int16) * int(inf_mag) + int(yoff_grid)
    dm_grid[np.tile(np.vstack(y), (nx_dm,)),
            np.tile(x, (ny_dm, 1))] = dm_z_commanded
    dm_grid = ss.fftconvolve(dm_grid, inf, mode='same')
    
    # 3D rotate DM grid and project orthogonally onto wavefront
    xdim = int(np.round(np.sqrt(2) * nx_grid * dx_inf / dx_surf)) # grid dimensions (pix) projected onto wavefront
    ydim = int(np.round(np.sqrt(2) * ny_grid * dx_inf / dx_surf))

    if xdim > n: xdim = n

    if ydim > n: ydim = n

    x = np.ones((ydim, 1), dtype=np.int) * ((np.arange(xdim) - xdim // 2) * dx_surf)
    y = (np.ones((xdim, 1), dtype=np.int) * ((np.arange(ydim) - ydim // 2) * dx_surf)).T

    a = xtilt * np.pi / 180
    b = ytilt * np.pi / 180
    g = ztilt * np.pi /180

    if XYZ:
        m = np.array([[cos(b)*cos(g), -cos(b)*sin(g), sin(b), 0],
            [cos(a)*sin(g) + sin(a)*sin(b)*cos(g), cos(a)*cos(g)-sin(a)*sin(b)*sin(g), -sin(a)*cos(b), 0],
            [sin(a)*sin(g)-cos(a)*sin(b)*cos(g), sin(a)*cos(g)+cos(a)*sin(b)*sin(g), cos(a)*cos(b), 0],
            [0, 0, 0, 1] ])
    else:
        m = np.array([	[cos(b)*cos(g), cos(g)*sin(a)*sin(b)-cos(a)*sin(g), cos(a)*cos(g)*sin(b)+sin(a)*sin(g), 0],
            [cos(b)*sin(g), cos(a)*cos(g)+sin(a)*sin(b)*sin(g), -cos(g)*sin(a)+cos(a)*sin(b)*sin(g), 0],
            [-sin(b), cos(b)*sin(a), cos(a)*cos(b), 0],
            [0, 0, 0, 1] ])

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
        grid = proper.prop_cubic_conv(dm_grid.T, xdm, ydm, GRID = False)
        grid = grid.reshape([xdm.shape[1], xdm.shape[0]])
    else:
        grid = map_coordinates(dm_grid.T, [xdm, ydm], order=3,
                               mode="nearest", prefilter = True)

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
    if(hasattr(dm, 'flagZYX')):
        if(dm.flagZYX):
            XYZ = False

    # Compute sampling of the pupil. Assume that it is square.
    dm.dx_dm = dx_dm
    dm.dx = dx_dm

    # Default to being centered on a pixel if not specified
    if not(hasattr(dm, 'centering')):
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
    dm.Ndm = int(falco.util.ceil_even((abs(np.array([np.sqrt(2.)*cos(radians(45.-dm.zrot)),
            np.sqrt(2.)*sin(radians(45.-dm.zrot))])).max())*Ndm0*(dm.dx_inf0/dm.dx))+2)
    
    # Compute list of initial actuator center coordinates (in actutor widths).
    if(dm.flag_hex_array):  # Hexagonal, hex-packed grid
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
        Mrot = np.array([[cb * cg, sa * sb * cg - ca * sg, ca * sb * cg + sa * sg, 0.0],
                 [cb * sg, sa * sb * sg + ca * cg, ca * sb * sg - sa * cg, 0.0],
                [-sb,      sa * cb,                ca * cb,                0.0],
                     [0.0,                    0.0,                    0.0, 1.0]])
    else:
        Mrot = np.array([               [cb * cg,               -cb * sg,       sb, 0.0],
                [ca * sg + sa * sb * cg, ca * cg - sa * sb * sg, -sa * cb, 0.0],
                [sa * sg - ca * sb * cg, sa * cg + ca * sb * sg,  ca * cb, 0.0],
                                   [0.0,                    0.0,      0.0, 1.0]])
    
    # Compute the actuator center coordinates in units of actuator spacings
    for iact in range(dm.NactTotal):
        xyzVals = np.array([x_vec[iact], y_vec[iact], 0., 1.])
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
    while(abs(infDiff) <= 1e-7):
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
    if(dm.flag_hex_array):
        # padded 2 actuators past the last actuator center to avoid trying to
        # index outside the array
        dm.NdmPad = falco.util.ceil_even((2.*(dm.rmax+2))*NpixPerAct + 1)
    else:
        # DM surface array padded by the width of the padded influence function
        # to prevent indexing outside the array. 
        # The 1/2 term is because the farthest actuator center is still half an
        # actuator away from the nominal array edge. 
        dm.NdmPad = falco.util.ceil_even((dm.NboxAS + 2.0*(1 + (np.max(
        np.abs(dm.xy_cent_act.reshape(2*dm.NactTotal)))+0.5)*NpixPerAct)))

    # Compute coordinates (in meters) of the full DM array
    if(dm.centering == 'pixel'):
        # meters, coords for the full DM arrays. Origin is centered on a pixel
        dm.x_pupPad = np.linspace(-dm.NdmPad/2., (dm.NdmPad/2. - 1),
                                  dm.NdmPad)*dx_dm
    else:
        # meters, coords for the full DM arrays. Origin is interpixel centered
        dm.x_pupPad = np.linspace(-(dm.NdmPad-1)/2., (dm.NdmPad-1)/2.,
                                  dm.NdmPad)*dx_dm

    dm.y_pupPad = dm.x_pupPad

    # Make NboxPad-sized postage stamps for each actuator's influence function
    if(flagGenCube):
        if not dm.flag_hex_array:
            print("  Influence function padded from %d to %d points for A.S. propagation." % (Nbox,dm.NboxAS))

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
        r_cent_act_box_inM = np.sqrt(dm.xy_cent_act_box_inM[0, :]**2 + dm.xy_cent_act_box_inM[1, :]**2)
        # Compute and store all the influence functions:
        dm.inf_datacube = np.zeros((Nbox, Nbox, dm.NactTotal))  # initialize array of influence function "postage stamps"
        dm.act_ele = np.arange(dm.NactTotal)  # Initialize as including all actuators

        inf_datacube = np.zeros((dm.NactTotal, Nbox, Nbox))

        interp_spline = RectBivariateSpline(x_inf0, x_inf0, dm.infMaster)  # RectBivariateSpline is faster in 2-D than interp2d
        # Refer to https://scipython.com/book/chapter-8-scipy/examples/two-dimensional-interpolation-with-scipyinterpolaterectbivariatespline/

        for iact in range(dm.NactTotal):
           xbox = dm.x_box0 - (dm.xy_cent_act_inPix[0, iact]-dm.xy_cent_act_box[0, iact])*dx_dm # X = X0 -(x_true_center-x_box_center)
           ybox = dm.x_box0 - (dm.xy_cent_act_inPix[1, iact]-dm.xy_cent_act_box[1, iact])*dx_dm # Y = Y0 -(y_true_center-y_box_center)
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
    indPair = np.zeros((0,2))  # Initialize the paired indices list. [nPairs x 2]
    
    kx1 = np.array([[0, 1], [1, 1], [1, 0]])              # R1-C1
    kx2 = np.array([[0,1], [1,1], [1,0], [1,-1]])         # R1, C2 - C47
    kx3 = np.array([[1,0], [1,-1]])                       # R1, C48
    kx4 = np.array([[-1,1], [0,1], [1,1], [1,0]])         # R2-R47, C1
    kx5 = np.array([[-1,1], [0,1], [1,1], [1,0], [1,-1]]) # R2-47, C2-47
    kx6 = np.array([[1,0], [1,-1]])                       # R2-47, C8
    kx7 = np.array([[-1,1], [0,1]])                       # R48, C1 - C47
    kx8 = np.array([[-1,-1]])                             # R48, C48
    
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
                
            kr = jj + kx[:,0]
            kc = ii + kx[:,1]
            nNbr = kr.size  # length(kr); # Number of neighbors
                    
            if nNbr >= 1:
                for iNbr in range(nNbr):
                    
                    a1 = Vout[jj, ii] - Vout[kr[iNbr],kc[iNbr]] # Compute the delta voltage
                    
                    if (np.abs(a1) > Vlim):  # If neighbor rule is violated
                        
                        indLinCtr = (ii-1)*Nact + jj  # linear index of center actuator
                        indLinNbr = (kc[iNbr]-1)*Nact + kr[iNbr]  # linear index of neigboring actuator
                        indPair = np.array([indPair, np.array([indLinCtr, indLinNbr]).reshape(1, 2)])
                        indPair = np.vstack([indPair, np.array([indLinCtr, indLinNbr]).reshape(1, 2)])
    
                        fx = (np.abs(a1) - Vlim) / 2.
                        Vout[jj, ii] = Vout[jj, ii] - np.sign(a1)*fx
                        Vout[kr[iNbr], kc[iNbr]] = Vout[kr[iNbr], kc[iNbr]] +\
                                                    np.sign(a1)*fx

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
    # Min voltage limit
    new_inds = np.nonzero(dm.V.flatten()<dm.Vmin)[0]  # linear indices of new actuators breaking their bounds
    new_vals = dm.Vmin*np.ones(new_inds.size)
    dm.pinned = np.hstack([dm.pinned, new_inds])  # Augment the vector of pinned actuator linear indices
    dm.Vpinned = np.hstack([dm.Vpinned, new_vals])  # Augment the vector of pinned actuator values
    # Max voltage limit
    new_inds = np.nonzero(dm.V.flatten() > dm.Vmax)[0] # linear indices of new actuators breaking their bounds
    new_vals = dm.Vmax*np.ones(new_inds.size)
    dm.pinned = np.hstack([dm.pinned, new_inds])     # Augment the vector of pinned actuator linear indices
    dm.Vpinned = np.hstack([dm.Vpinned, new_vals])  # Augment the vector of pinned actuator values
    
    # 2) Enforce pinned (or railed or dead) actuator values
    if(dm.pinned.size > 0):
        Vflat = dm.V.flatten()
        Vflat[dm.pinned.astype(int)] = dm.Vpinned
        dm.V = Vflat.reshape(dm.V.shape)
    
    # 3) Find which actuators violate the DM neighbor rule. (This restricts 
    # the maximum voltage between an actuator and each of its 8 neighbors.) 
    # Add those actuator pairs to the list of tied actuators.
    if(dm.flagNbrRule):
        dm.V, indPair1 = apply_neighbor_rule(dm.V, dm.dVnbr, dm.Nact);
        dm.tied = np.vstack([dm.tied, indPair1])  # Tie together actuators violating the neighbor rule
        
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
    surfaceToFit : numpy ndarray
        2-D array of the surface heights for the DM to fit
    dm : ModelParameters
        Structure containing parameter values for the DM

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
    x = np.linspace(-(N1-1.)/2.,(N1-1.)/2., N1)/actres1
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
    if(hasattr(dm, 'flagZYX')):
        if(dm.flagZYX):
            flagXYZ = False
            
    # Perform the fit
    if(nSurface == dm.Nact):
        gridDerotAtActRes = surfaceToFit
    
    elif(nSurface > dm.Nact):
        # Adjust the centering of the output DM surface. The shift needs to be
        # in units of actuators, not meters
        wArray = nSurface*dm.dx
        cshift = -wArray/2./nSurface/dm.dm_spacing if(dm.centering == 'interpixel') else 0.
    
        gridDerotAtActRes = derotate_resize_surface(surfaceToFit, dm.dx,
        dm.Nact, dm.xc-cshift, dm.yc-cshift, dm.dm_spacing, XTILT=dm.xtilt,
        YTILT=dm.ytilt, ZTILT=dm.zrot, XYZ=flagXYZ, inf_sign=dm.inf_sign,
        inf_fn=dm.inf_fn)
    
    elif(nSurface < dm.Nact):
        raise ValueError('surfaceToFit cannot be smaller than [Nact x Nact].')
    
    [Vout, surfaceOut] = proper.prop_fit_dm(gridDerotAtActRes, infFuncAtActRes)

    return Vout

        
def surf_from_poke_cube(dm, dmFullOrCompact):
    """
    Produce a DM surface by superposing actuators from a datacube.
    
    Parameters
    ----------
    dm : numpy ndarray
        2-D array of DM voltage commands
    dmFullOrCompact : ModelParameters
        Structure containing parameter values for the DM in either the full or
        compact model

    Returns
    -------
    DMsurf : numpy ndarray
        2-D array of DM surface map
    """
    DMsurf = np.zeros((dmFullOrCompact.NdmPad,dmFullOrCompact.NdmPad))  # Initialize the empty array
    for iact in range(dm.NactTotal):
        if(any(any(dmFullOrCompact.inf_datacube[:, :, iact])) and any(dm.VtoH(iact))):
            y_box_ind = np.arange(dmFullOrCompact.xy_box_lowerLeft[0, iact], dmFullOrCompact.xy_box_lowerLeft[0,iact]+dmFullOrCompact.Nbox-1, dtype=np.int) # x-indices in pupil arrays for the box
            x_box_ind = np.arange(dmFullOrCompact.xy_box_lowerLeft[1, iact], dmFullOrCompact.xy_box_lowerLeft[1,iact]+dmFullOrCompact.Nbox-1, dtype=np.int) # y-indices in pupil arrays for the box
            DMsurf[x_box_ind,y_box_ind] +=  dm.V.flatten()[iact]*dm.VtoH.flatten()[iact]*dmFullOrCompact.inf_datacube[: ,:, iact]

    # Adjust the orientation if specified
    if(hasattr(dm, 'fliplr')):
        if(dm.fliplr):
            DMsurf = np.fliplr(DMsurf)
            pass
        pass
    
    if(hasattr(dm, 'flipud')):
        if(dm.flipud):
            DMsurf = np.flipud(DMsurf)
            pass
        pass

    return DMsurf


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
        if(kwargs["inf_sign"] == '+'):
            sign_factor = 1.
        elif(kwargs["inf_sign"] == '-'):
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
    dx_inf = 0.1e-3  # influence function spacing in meters
    dx_dm_inf = 1.e-3  # spacing between DM actuators in meters assumed by influence function
    inf_mag = 10

    dx_dm = spacing

    dx_inf = dx_inf * dx_dm / dx_dm_inf   # Influence function sampling scaled
                                          # to specified DM actuator spacing

    dm_z_commanded = dm_z

    s = dm_z.shape
    nx_dm = s[1]
    ny_dm = s[0]

    # Create subsampled DM grid
    margin = 9 * inf_mag
    nx_grid = nx_dm * inf_mag + 2 * margin
    ny_grid = ny_dm * inf_mag + 2 * margin
    xoff_grid = margin + inf_mag/2  # pixel location of 1st actuator center in subsampled grid
    yoff_grid = xoff_grid
    dm_grid = np.zeros([ny_grid, nx_grid], dtype = np.float64)

    x = np.arange(nx_dm, dtype=np.int16) * int(inf_mag) + int(xoff_grid)
    y = np.arange(ny_dm, dtype=np.int16) * int(inf_mag) + int(yoff_grid)
    dm_grid[np.tile(np.vstack(y), (nx_dm,)), np.tile(x, (ny_dm, 1))] = dm_z_commanded
    dm_grid = ss.fftconvolve(dm_grid, inf, mode='same')

    # 3D rotate DM grid and project orthogonally onto wavefront
    xdim = int(np.round(np.sqrt(2) * nx_grid * dx_inf / dx_surf))  # grid dimensions (pix) projected onto wavefront
    ydim = int(np.round(np.sqrt(2) * ny_grid * dx_inf / dx_surf))

    if xdim > n: xdim = n

    if ydim > n: ydim = n

    x = np.ones((ydim,1), dtype=np.int) * ((np.arange(xdim) - xdim // 2) * dx_surf)
    y = (np.ones((xdim,1), dtype=np.int) * ((np.arange(ydim) - ydim // 2) * dx_surf)).T

    a = xtilt * np.pi / 180
    b = ytilt * np.pi / 180
    g = ztilt * np.pi / 180

    if XYZ:
        m = np.array([ 	[cos(b)*cos(g), -cos(b)*sin(g), sin(b), 0],
            [cos(a)*sin(g) + sin(a)*sin(b)*cos(g), cos(a)*cos(g)-sin(a)*sin(b)*sin(g), -sin(a)*cos(b), 0],
            [sin(a)*sin(g)-cos(a)*sin(b)*cos(g), sin(a)*cos(g)+cos(a)*sin(b)*sin(g), cos(a)*cos(b), 0],
            [0, 0, 0, 1] ])
    else:
        m = np.array([	[cos(b)*cos(g), cos(g)*sin(a)*sin(b)-cos(a)*sin(g), cos(a)*cos(g)*sin(b)+sin(a)*sin(g), 0],
        [cos(b)*sin(g), cos(a)*cos(g)+sin(a)*sin(b)*sin(g), -cos(g)*sin(a)+cos(a)*sin(b)*sin(g), 0],
        [-sin(b), cos(b)*sin(a), cos(a)*cos(b), 0],
        [0, 0, 0, 1] ])

    # Compute xdm0 and ydm0 for use in de-rotating the DM surface
    edge = np.array([[-1.0,-1.0,0.0,0.0], [1.0,-1.0,0.0,0.0], [1.0,1.0,0.0,0.0], [-1.0,1.0,0.0,0.0]])
    new_xyz = edge #np.dot(edge, m)

    # determine backward projection for screen-raster-to-DM-surce computation
    dx_dxs = (new_xyz[0, 0] - new_xyz[1, 0]) / (edge[0, 0] - edge[1, 0])
    dx_dys = (new_xyz[1, 0] - new_xyz[2, 0]) / (edge[1, 1] - edge[2, 1])
    dy_dxs = (new_xyz[0, 1] - new_xyz[1, 1]) / (edge[0, 0] - edge[1, 0])
    dy_dys = (new_xyz[1, 1] - new_xyz[2, 1]) / (edge[1, 1] - edge[2, 1])

    xs = (x/dx_dxs - y*dx_dys/(dx_dxs*dy_dys) ) / ( 1 - dy_dxs*dx_dys/(dx_dxs*dy_dys))
    ys = (y/dy_dys - x*dy_dxs/(dx_dxs*dy_dys) ) / ( 1 - dx_dys*dy_dxs/(dx_dxs*dy_dys))

    xdm0 = (xs + dm_xc * dx_dm) / dx_inf + xoff_grid
    ydm0 = (ys + dm_yc * dx_dm) / dx_inf + yoff_grid
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
    if(N2 % 2 == 0):
        xs2 = np.linspace(-N2/2, (N2/2)-1, N2)/N2*(N2*dx/(Nact*spacing))
    else:
        xs2 = np.linspace(-(N2-1)/2, (N2-1)/2, N2)/N2*(N2*dx/(Nact*spacing))

    interp_spline = RectBivariateSpline(xs2, xs2, gridDerot)
    gridDerotResize = interp_spline(xs1-xOffsetInAct/Nact,
                                    xs1-yOffsetInAct/Nact)
        
    xyOffset = int(np.floor(multipleOfCommandGrid/2.))
    gridDerotAtActRes = gridDerotResize[xyOffset::multipleOfCommandGrid,
                                        xyOffset::multipleOfCommandGrid]
#
#    plt.figure(11); plt.imshow(dm_grid); plt.colorbar(); plt.pause(0.1)
#    plt.figure(12); plt.imshow(gridDerot); plt.colorbar(); plt.pause(0.1)
#    plt.figure(13); plt.imshow(gridDerotResize); plt.colorbar(); plt.pause(0.1)
#    plt.figure(14); plt.imshow(gridDerotAtActRes); plt.colorbar(); plt.pause(0.1)
#    plt.figure(15); plt.imshow(gridDerot-gridDerot[::-1,::-1]); plt.colorbar(); plt.pause(0.1)
#    plt.figure(16); plt.imshow(gridDerotResize-gridDerotResize[::-1,::-1]); plt.colorbar(); plt.pause(0.1)

    return gridDerotAtActRes
