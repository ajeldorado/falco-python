import copy

import numpy as np

import os

from scipy.interpolate import RectBivariateSpline
from scipy import ndimage
try:
    from scipy.ndimage import map_coordinates
except:
    from scipy.ndimage.interpolation import map_coordinates

from falco import fftutils, util, proper, mask, check
from astropy.io import fits



# the functions:
#   make_rotation_matrix
#   promote_3d_transformation_to_homography
#   make_homomorphic_translation_matrix
#   drop_z_3d_transformation
#   pack_xy_to_homographic_points
#   warp
#   apply_homography
#   prepare_fwd_reverse_projection_coordinates
#   apply_precomputed_transfer_function
#   prepare_actuator_lattice
#   fourier_resample
#   forward_ft_unit
#
#   and the contents of the DM class
#
# are copied and adapted to FALCO from prysm
#
# prysm's license is repeated here and applies to the above list of functions
#
# see https://github.com/brandondube/prysm
# file LICENSE.md
#
# The MIT License (MIT) Copyright (c) 2017 Brandon Dube
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS",WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def dm_init_falco_wrapper(dm,dx,Narray,dm_z0, dm_xc, dm_yc, spacing=0.,**kwargs):

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
        if(kwargs["inf_sign"] == '+'):
            sign_factor = 1.
        elif(kwargs["inf_sign"] == '-'):
            sign_factor = -1.
    else:
        sign_factor = 1.

    n = Narray
    dx_surf = dx  # sampling of surface in meters
    beamradius = Narray*dx_surf/2.0

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

    dx_inf = dx_inf * dx_dm / dx_dm_inf  # Influence function sampling scaled
                                         # to specified DM actuator spacing
    #else:
    dm_z_commanded = dm_z
    s = dm_z.shape
    nx_dm = s[1]
    ny_dm = s[0]

    # Create subsampled DM grid
    margin = 9 * inf_mag
    nx_grid = nx_dm * inf_mag + 2 * margin
    ny_grid = ny_dm * inf_mag + 2 * margin

    surf_mag = dx_inf / dx_surf

    #scale and recenter to calculate the new coordinates
    #xnew = (np.arange(xdim) - xdim//2) * dx_inf/dx_surf + nx_inf//2
    #ynew = (np.arange(ydim) - ydim//2) * dx_inf/dx_surf + ny_inf//2
    #[Xnew,Ynew] = np.meshgrid(xnew,ynew)
    #intrpolate tine infuence function to the new coordinates
    #inf_scaled = ndimage.map_coordinates(inf, (Ynew[:], Xnew[:]))

    shiftx = -1*int((dm.xc - dm.Nact//2 + 0.5) * inf_mag)
    shifty = -1*int((dm.yc - dm.Nact//2 + 0.5) * inf_mag)

    inf_pad = util.pad_crop(inf, int(nx_grid))

    if XYZ:
        rot_order = 'xyz'
    else:
        rot_order = 'zyx'

    dmModel = DM(
        inf_pad, NoutDefault=Narray, Nact=dm.Nact, sep=inf_mag,
        upsample=surf_mag, shift=(shiftx, shifty),
        rot=(-1*ztilt, ytilt, xtilt), rot_order=rot_order)
    dmModel.update(dm_z_commanded)

    return dmModel

def make_rotation_matrix(angles_zyx, use_zyx_order=True,radians=False):
    """Build a rotation matrix.

    Parameters
    ----------
    angles_zyx : tuple of float
        Z, Y, X rotation angles in that order
    use_zyx_order: bool, optional
        if True, apply the rotations in the order Z, Y, X. If False, apply the
        rotations in the order X, Y, Z.
    radians : bool, optional
        if True, abg are assumed to be radians.  If False, abg are
        assumed to be degrees.

    Returns
    -------
    numpy.ndarray
        3x3 rotation matrix

    """
    ZYX = np.zeros(3)
    ZYX[:len(angles_zyx)] = angles_zyx
    angles_zyx = ZYX
    if not radians:
        angles_zyx = np.radians(angles_zyx)

    # alpha, beta, gamma = abg
    gamma, beta, alpha = angles_zyx
    cos1 = np.cos(alpha)
    cos2 = np.cos(beta)
    cos3 = np.cos(gamma)
    sin1 = np.sin(alpha)
    sin2 = np.sin(beta)
    sin3 = np.sin(gamma)

    Rx = np.asarray([
        [1,    0,  0   ],  # NOQA
        [0, cos1, -sin1],
        [0, sin1,  cos1]
    ])
    Ry = np.asarray([
        [cos2,  0, sin2],
        [    0, 1,    0],  # NOQA
        [-sin2, 0, cos2],
    ])
    Rz = np.asarray([
        [cos3, -sin3, 0],
        [sin3,  cos3, 0],
        [0,        0, 1],
    ])
    
    if use_zyx_order:
        m = Rx@Ry@Rz
    else:
        m = Rz@Ry@Rx
        
    return m


def promote_3d_transformation_to_homography(M):
    """Convert a 3D transformation to 4D homography."""
    out = np.zeros((4, 4), dtype=np.float64)
    out[:3, :3] = M
    out[3, 3] = 1
    return out


def make_homomorphic_translation_matrix(tx=0, ty=0, tz=0):
    out = np.eye(4, dtype=np.float64)
    out[0, -1] = tx
    out[1, -1] = ty
    out[2, -1] = tz
    return out


def drop_z_3d_transformation(M):
    """Drop the Z entries of a 3D homography.

    Drops the third row and third column of 4D transformation matrix M.

    Parameters
    ----------
    M : numpy.ndarray
        4x4 ndarray for (x, y, z, w)

    Returns
    -------
    numpy.ndarray
        3x3 array, (x, y, w)

    """
    mask = [0, 1, 3]
    # first bracket: drop output Z row, second bracket: drop input Z column
    M = M[mask][:, mask]
    return np.ascontiguousarray(M)  # assume this will get used a million times


def pack_xy_to_homographic_points(x, y):
    """Pack (x, y) vectors into a vector of coordinates in homogeneous form.

    Parameters
    ----------
    x : numpy.ndarray
        x points
    y : numpy.ndarray
        y points

    Returns
    -------
    numpy.ndarray
        3xN array (x, y, w)

    """
    out = np.empty((3, x.size), dtype=x.dtype)
    out[0, :] = x.ravel()
    out[1, :] = y.ravel()
    out[2, :] = 1
    return out


def warp(img, xnew, ynew):
    """Warp an image, via "pull" and not "push".

    Parameters
    ----------
    img : numpy.ndarray
        2D ndarray
    xnew : numpy.ndarray
        2D array containing x or column coordinates to look up in img
    ynew : numpy.ndarray
        2D array containing y or row    coordinates to look up in img

    Returns
    -------
    numpy.ndarray
        "pulled" warped image

    Notes
    -----
    The meaning of pull is that the indices of the output array indices
    are the output image coordinates, in other words xnew/ynew specify
    the coordinates in img, at which each output pixel is looked up

    this is a dst->src mapping, aka "pull" in common image processing
    vernacular

    """
    # user provides us (x, y), we provide scipy (row, col) = (y, x)
    return ndimage.map_coordinates(img, (ynew, xnew))


def apply_homography(M, x, y):
    points = pack_xy_to_homographic_points(x, y)
    xp, yp, w = M @ points
    xp /= w
    yp /= w
    if x.ndim > 1:
        xp = np.reshape(xp, x.shape)
        yp = np.reshape(yp, x.shape)
    return xp, yp


def prepare_fwd_reverse_projection_coordinates(shape, rot, use_zyx_order=True):
    # 1. make the matrix that describes the rigid body transformation
    # 2. make the coordinate grid (in "pixels") for the data
    # 3. project the coordinates "forward" (for forward_model())
    # 4. project the coordinates "backwards" (for backprop)
    R = make_rotation_matrix(rot,use_zyx_order)
    oy, ox = [(s)/2 for s in shape]
    y, x = [np.arange(s, dtype=np.float64) for s in shape]
    x, y = np.meshgrid(x, y)
    Tin = make_homomorphic_translation_matrix(-ox, -oy)
    Tout = make_homomorphic_translation_matrix(ox, oy)
    R = promote_3d_transformation_to_homography(R)
    Mfwd = Tout@(R@Tin)
    Mfwd = drop_z_3d_transformation(Mfwd)
    Mifwd = np.linalg.inv(Mfwd)
    xfwd, yfwd = apply_homography(Mifwd, x, y)
    xrev, yrev = apply_homography(Mfwd, x, y)
    return (xfwd, yfwd), (xrev, yrev)


def apply_precomputed_transfer_function(arr, tf):
    ARR = fftutils.fft2(arr)
    ARR *= tf
    # real before fftshift -> drop imaginary part is fftshifting less data,
    # which we would discard anyway (faster)
    arr_influenced = fftutils.fftshift(fftutils.ifft2(ARR).real)
    return arr_influenced


def prepare_actuator_lattice(shape, Nact, sep, dtype):
    # Prepare a lattice of actuators.
    #
    # Usage guide:
    # returns a dict of
    # {
    #     mask, shape Nact
    #     actuators, shape Nact
    #     poke_arr, shape shape
    #     ixx, shape (truthy part of mask)
    #     iyy, shape (truthy part of mask)
    # }
    #
    # assign poke_arr[iyy, ixx] = actuators[mask] in the next step
    actuators = np.zeros(Nact, dtype=dtype)

    cy, cx = [s//2 for s in shape]
    Nactx, Nacty = Nact
    skip_samples_x, skip_samples_y = sep
    # python trick; floor division (//) rounds to negative inf, not zero
    # because FFT grid alignment biases things to the left, if Nact is odd
    # we want more on the negative side;
    # this will make that so
    offx = 0
    offy = 0
    # TODO: does falco have an is_odd?  Did not see it in util
    if Nactx % 2 == 0:
        offx = skip_samples_x // 2
    if Nacty % 2 == 0:
        offy = skip_samples_y // 2

    neg_extreme_x = cx + -Nactx//2 * skip_samples_x + offx
    neg_extreme_y = cy + -Nacty//2 * skip_samples_y + offy
    pos_extreme_x = cx + Nactx//2 * skip_samples_x + offx
    pos_extreme_y = cy + Nacty//2 * skip_samples_y + offy

    ix = slice(neg_extreme_x, pos_extreme_x, skip_samples_x)
    iy = slice(neg_extreme_y, pos_extreme_y, skip_samples_y)
    ixx = ix
    iyy = iy

    poke_arr = np.zeros(shape, dtype=dtype)
    return {
        'actuators': actuators,
        'poke_arr': poke_arr,
        'ixx': ixx,
        'iyy': iyy,
    }


def spline_resample(array_in, zoom):
    """
    Translate, rotate, and downsample a pixel-centered mask.

    Parameters
    ----------
    array_in : np.ndarray
        2-D, pixel-centered array containing the pupil mask.
    zoom : int/float or tuple of two ints/floats

    Returns
    -------
    arrayOut : np.ndarray
        2-D, even-sized, square array containing the resized mask.
    """
    check.twoD_array(array_in, 'array_in', TypeError)
    # check.real_positive_scalar(nBeamIn, 'nBeamIn', TypeError)

    # if mag <= 0:
    #     raise ValueError('This function is for downsampling only.')
    # else:
    #     if array_in.shape[0] % 2 == 0:  # if in an even-sized array
    #         # Crop assuming mask is pixel centered with row 0 and column 0 empty.
    #         array_in = array_in[1::, 1::]

    if zoom == 1:
        return array_in

    if isinstance(zoom, (float, int)):
        zoom = (zoom, zoom)
    elif not isinstance(zoom, tuple):
        zoom = tuple(float(zoom) for zoom in zoom)

    array_in = util.pad_crop(array_in, int(np.max(array_in.shape)))

    # Array sizes
    narray_in = array_in.shape[0]
    narray_out = util.ceil_odd(narray_in*np.max(zoom)+2)
    # 2 pixels added to guarantee the offset mask is fully contained in
    # the output array.
    nBeamIn = 1  # Only relative values matter
    nBeamOutX = nBeamIn*zoom[1]
    nBeamOutY = nBeamIn*zoom[0]
    dxIn = 1./nBeamIn
    dYin = 1./nBeamIn
    dxOut = 1./nBeamOutX
    dyOut = 1./nBeamOutY

    # array-centered coordinates of input matrix [pupil diameters]
    x0 = np.arange(-(narray_in-1.)/2., (narray_in)/2., 1)*dxIn
    y0 = np.arange(-(narray_in-1.)/2., (narray_in)/2., 1)*dYin

    if False: #zoom[0] < 1 and zoom[1] < 1:
        [X0, Y0] = np.meshgrid(x0, y0)
        R2 = X0**2 + Y0**2
        Window = np.zeros_like(R2)
        Window[R2 <= (dxOut/2.)**2 + (dyOut/2.)**2] = 1
        # R0 = np.sqrt(X0**2 + Y0**2)
        # Window = 0*R0
        # Window[R0 <= dxOut/2.] = 1
        Window = Window/np.sum(Window)

        # To get good grayscale edges, convolve with the correct window
        # before downsampling.
        f_window = np.fft.ifft2(np.fft.ifftshift(Window))*narray_in
        f_array_in = np.fft.ifft2(np.fft.ifftshift(array_in))*narray_in
        A = np.fft.fftshift(np.fft.fft2(f_window*f_array_in))
        A = np.real(A)
    else:
        A = array_in

    x1 = (np.arange(-(narray_out-1.)/2., narray_out/2., 1) - 0)*dxOut
    y1 = (np.arange(-(narray_out-1.)/2., narray_out/2., 1) - 0)*dyOut

    # RectBivariateSpline is faster in 2-D than interp2d
    interp_spline = RectBivariateSpline(y0, x0, A)
    Atemp = interp_spline(y1, x1)

    arrayOut = Atemp

    # arrayOut = np.zeros((narray_out+1, narray_out+1))
    # arrayOut[1::, 1::] = np.real(Atemp)

    return arrayOut


def map_resample(f, zoom):
    # print(f'zoom = {zoom}')
    if zoom == 1:
        return f

    if isinstance(zoom, (float, int)):
        zoom = (zoom, zoom)
    elif not isinstance(zoom, tuple):
        zoom = tuple(float(zoom) for zoom in zoom)

    N0 = util.ceil_odd(np.max(f.shape) + np.max(zoom) + 1)
    f = util.pad_crop(f, N0)

    M = int(util.ceil_odd(N0*zoom[0]+1))
    N = int(util.ceil_odd(N0*zoom[1]+1))

    x = (np.arange(N, dtype=np.float64) - np.floor(N/2))/zoom[1] + np.floor(N0/2)
    y = (np.arange(M, dtype=np.float64) - np.floor(M/2))/zoom[0] + np.floor(N0/2)
    xygrid = np.meshgrid(x, y)
    fprime = map_coordinates(f.T, xygrid, order=3, mode="nearest")

    # fprime = proper.prop_magnify(f, zoom, size_out0=N1, AMP_CONSERVE=True)

    return fprime


def fourier_resample(f, zoom):
    print(f'zoom = {zoom}')
    if zoom == 1:
        return f

    if isinstance(zoom, (float, int)):
        zoom = (zoom, zoom)
    elif not isinstance(zoom, tuple):
        zoom = tuple(float(zoom) for zoom in zoom)

    m, n = f.shape
    M = int(np.round(m*zoom[0]))
    N = int(np.round(n*zoom[1]))

    # commented out below, an alternative that does not use the fft2 norm keyword argument
    # doing it this way is mildly preferrable;
    F = fftutils.fftshift(fftutils.fft2(fftutils.ifftshift(f), norm='ortho'))
    # F = fftutils.fftshift(fftutils.fft2(fftutils.ifftshift(f)))

    Mx, My = setup_mft_matricies_scalars((M, N), F.shape, 1, 1)
    fprime = imft2_core(F, Mx, My).real
    fprime *= np.sqrt((zoom[0]*zoom[1]))
    # fprime *= np.sqrt((zoom[0]*zoom[1]))/(np.sqrt(f.size))
    # fprime *= np.sum(fprime)/np.sum(f)

    return fprime


def forward_ft_unit(dx, samples, shift=True):
    """Compute the units resulting from a fourier transform.

    Parameters
    ----------
    dx : float
        center-to-center spacing of samples in an array
    samples : int
        number of samples in the data
    shift : bool, optional
        whether to shift the output.  If True, first element is a negative freq
        if False, first element is 0 freq.

    Returns
    -------
    numpy.ndarray
        array of sample frequencies in the output of an fft

    """
    unit = fftutils.fftfreq(samples, dx)

    if shift:
        return fftutils.fftshift(unit)
    else:
        return unit


def mft2_core(a, Mx, My):
    return My @ (a @ Mx)


def imft2_core(A, Mx, My):
    Mx = Mx.conj().T
    My = My.conj().T
    return My @ (A @ Mx)


def setup_mft_matricies_scalars(shape_space, shape_frequency, zoomx, zoomy):
    # the sampling increment of an FFT is 1/Nin
    # zoom applies to someone's expectation of FFT sampling, so the spatial array
    # shape (Nin) is used instead of the frequency array shape (Nout)
    incy = shape_space[0]*zoomy
    incx = shape_space[1]*zoomx

    y, x = [util.create_axis(S, 1) for S in shape_space]
    fy, fx = [util.create_axis(S, 1/inc) for S, inc in zip(shape_frequency, (incy, incx))]

    Ex = np.exp(-2j * np.pi * np.outer(x, fx))
    Ey = np.exp(-2j * np.pi * np.outer(y, fy).T)

    normx = np.sqrt(1/incx)
    normy = np.sqrt(1/incy)
    Ex *= normx
    Ey *= normy
    return Ex, Ey


class DM:
    """A DM whose actuators fill a rectangular region on a perfect grid, and have the same influence function."""
    def __init__(self, ifn, NoutDefault, Nact=50, sep=10, shift=(0, 0), rot=(0, 0, 0), upsample=1, rot_order='zyx'):
        """Create a new DM model.

        This model is based on convolution of a 'poke lattice' with the influence
        function.  It has the following idiosyncracies:

            1.  The poke lattice is always "FFT centered" on the array, i.e.
                centered on the sample which would contain the DC frequency bin
                after an FFT.
            2.  The rotation is applied in the same sampling as ifn
            3.  Shifts and resizing are applied using a Fourier method and not
                subject to quantization

        Parameters
        ----------
        ifn : numpy.ndarray
            influence function; assumes the same for all actuators.
            Assumed centered on N//2th sample of x, y.
            Assumed to be well-conditioned for use in convolution, i.e.
            compact compared to the array holding it
        Nout : int or tuple of int, length 2
            number of samples in the output array; see notes for details
        Nact : int or tuple of int, length 2
            (X, Y) actuator counts
        sep : int or tuple of int, length 2
            (X, Y) actuator separation, samples of influence function
        shift : tuple of float, length 2
            (X, Y) shift of the actuator grid to (x, y), units of x influence
            function sampling.  E.g., influence function on 0.1 mm grid, shift=1
            = 0.1 mm shift.  Positive numbers describe (rightward, downward)
            shifts in image coordinates (origin lower left).
        rot : tuple of int, length <= 3
            (Z, Y, X) rotations; see coordinates.make_rotation_matrix
        upsample : float
            upsampling factor used in determining output resolution, if it is different
            to the resolution of ifn.
        project_centering : str, {'fft', 'interpixel'}
            how to deal with centering when projecting the surface into the beam normal
            fft = the N/2 th sample, rounded to the right, defines the origin.
            interpixel = the N/2 th sample, without rounding, defines the origin
        rot_order : string
            if 'zyx', rotations will be applied around axis Z, then Y, then X,
            in that order. If 'xyz', rotations will be appied in that order.
            This convention agrees with FALCO/PROPER.

        Notes
        -----
        If ifn is 500x500 and upsample=0.5, then the nominal output array is
        250x250.  If this is supposed to line up with a pupil embedded in a
        512x512 array, then the user would have to call pad2d after, which is
        slightly worse than one stop shop.

        The Nout parameter allows the user to specify Nout=512, and the DM's
        render method will internally do the zero-pad or crop necessary to
        achieve the desired array size.

        """
        if isinstance(NoutDefault, int):
            NoutDefault = (NoutDefault, NoutDefault)
        if isinstance(Nact, int):
            Nact = (Nact, Nact)
        if isinstance(sep, int):
            sep = (sep, sep)

        s = ifn.shape

        # stash inputs and some computed values on self
        self.ifn = ifn
        self.Ifn = fftutils.fft2(ifn)
        self.NoutDefault = NoutDefault
        self.Nact = Nact
        self.sep = sep
        self.shift = shift
        self.obliquity = np.cos(np.radians(np.linalg.norm(rot)))
        self.rot = rot
        self.upsample = upsample
        self.rot_order = rot_order

        # prepare the poke array and supplimentary integer arrays needed to
        # copy it into the working array
        out = prepare_actuator_lattice(ifn.shape, Nact, sep, dtype=ifn.dtype)
        self.actuators = out['actuators']
        self.actuators_work = np.zeros_like(self.actuators)
        self.poke_arr = out['poke_arr']
        self.ixx = out['ixx']
        self.iyy = out['iyy']

        self.needs_rot = True
        if np.allclose(rot, [0, 0, 0]):
            self.needs_rot = False
            self.projx = None
            self.projy = None
            self.invprojx = None
            self.invprojy = None
        else:
            if self.rot_order == 'zyx':
                use_zyx_order = True
            elif self.rot_order == 'xyz':
                use_zyx_order = False
            else:
                raise ValueError("rot_order must be \'zyx\' or \'xyz\'")

            fwd, rev = prepare_fwd_reverse_projection_coordinates(s, rot, use_zyx_order)
            self.projx, self.projy = fwd
            self.invprojx, self.invprojy = rev

        # shift data
        if shift[0] != 0 or shift[1] != 0:
            # caps = Fourier variable (x -> X, y -> Y)
            # make 2pi/px phase ramps in 1D (much faster)
            # then broadcast them to 2D when they're used as transfer functions
            # in a Fourier convolution
            Y, X = [forward_ft_unit(1, s, shift=False) for s in s]
            Xramp = np.exp(X * (-2j * np.pi * shift[0]))
            Yramp = np.exp(Y * (-2j * np.pi * shift[1]))
            shpx = s
            shpy = tuple(reversed(s))
            Xramp = np.broadcast_to(Xramp, shpx)
            Yramp = np.broadcast_to(Yramp, shpy).T
            self.Xramp = Xramp
            self.Yramp = Yramp
            self.tf = self.Ifn * self.Xramp * self.Yramp
        else:
            self.tf = self.Ifn

    def copy(self):
        """Make a (deep) copy of this DM."""
        return copy.deepcopy(self)

    def update(self, actuators):
        # semantics for update:
        # the mask is non-none, then actuators is a 1D vector of the same size
        # as the nonzero elements of the mask
        #
        # or mask is None, and actuators is 2D
        self.actuators[:] = actuators[:]
        self.poke_arr[self.iyy, self.ixx] = self.actuators
        return

    def render(self, Nout=None, wfe=True):
        """Render the DM's surface figure or wavefront error.

        Parameters
        ----------
        wfe : bool, optional
            if True, converts the "native" surface figure error into
            reflected wavefront error, by multiplying by 2 times the obliquity.
            obliquity is the cosine of the rotation vector.

        Returns
        -------
        numpy.ndarray
            surface figure error or wfe, projected into the beam normal
            by self.rot

        """
        if Nout is None:
            Nout = self.NoutDefault

        if isinstance(Nout, int):
            Nout = (Nout, Nout)

        # self.dx is unused inside apply tf, but :shrug:
        sfe = apply_precomputed_transfer_function(self.poke_arr, self.tf)
        if self.needs_rot:
            warped = warp(sfe, self.projx, self.projy)
        else:
            warped = sfe

        if wfe:
            warped *= (2*self.obliquity)

        if self.upsample != 1:
            warped = fourier_resample(warped, self.upsample)
            # warped = spline_resample(warped, self.upsample)

        self.Nintermediate = warped.shape
        warped = util.pad_crop(warped, Nout)
        return warped

    def render_backprop(self, protograd, wfe=True):
        """Gradient backpropagation for render().

        Parameters
        ----------
        protograd : numpy.ndarray
            "prototype gradient"
            the array holding the work-in-progress towards the gradient.
            For example, in a problem fitting actuator commands to a surface,
            you might have:

            render() returns a 512x512 array, for 48x48 actuators.
            y contains a 512x512 array of target surface heights

            The euclidean distance between the two as a cost function:
            cost = np.sum(abs(render() - y)**2)

            Then the first step in computing the gradient is
            diff = 2 * (render() - y)

            and you would call
            dm.render_backprop(diff)
        gain_map : numpy.ndarray
            nact x nact map of actuator gains in units of meters/Volt.
        wfe : bool, optional
            if True, the return is scaled as for a wavefront error instead
            of surface figure error

        Returns
        -------
        numpy.ndarray
            analytic gradient, shape Nact x Nact

        Notes
        -----
        Not compatible with complex valued protograd

        """
        """Gradient backpropagation for self.render."""
        protograd = util.pad_crop(protograd, self.Nintermediate)
        if self.upsample != 1:
            upsample = self.ifn.shape[0]/protograd.shape[0]
            protograd = fourier_resample(protograd, upsample)
            # protograd = spline_resample(protograd, upsample)
            protograd /= upsample**2

        if wfe:
            protograd *= (2*self.obliquity)

        # return protograd
        if self.needs_rot:
            protograd = warp(protograd, self.invprojx, self.invprojy)

        # return protograd
        in_actuator_space = apply_precomputed_transfer_function(protograd, np.conj(self.tf))

        return in_actuator_space[self.iyy, self.ixx]
