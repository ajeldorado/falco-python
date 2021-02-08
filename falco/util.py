"""FALCO utilities."""
import time
import numpy as np
import itertools
import math
from . import check


class TicToc(object):
    """Class for timing."""

    def __init__(self, name=None):
        """Initialize."""
        self.name = name

    def __enter__(self):
        """Start timer."""
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        """End timer."""
        if self.name:
            print('[%s]\t' % self.name, end='')
        print('Elapsed: %s' % (time.time() - self.tstart))


def cart2pol(x, y):
    """
    Convert Cartesian coordinate(s) into polar coordinate(s).

    Parameters
    ----------
    x : float or numpy.ndarray
        x-axis coordinate(s)
    y : float or numpy.ndarray
        y-axis coordinate(s)

    Returns
    -------
    rho : float or numpy.ndarray
        radial coordinate(s)
    theta : float or numpy.ndarray
        azimuthal coordinate(s)
    """
    if(type(x) == np.ndarray and type(y) == np.ndarray):
        if not x.shape == y.shape:
            raise ValueError('The two inputs must have the same shape.')

    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    return(rho, theta)


def sind(thetaDeg):
    """
    Compute the sine of the input given in degrees.

    Parameters
    ----------
    thetaDeg : float or numpy.ndarray
        Angle in degrees

    Returns
    -------
    float or numpy.ndarray
        sine of the input value
    """
    return math.sin(math.radians(thetaDeg))


def cosd(thetaDeg):
    """
    Compute the cosine of the input given in degrees.

    Parameters
    ----------
    thetaDeg : float or numpy.ndarray
        Angle in degrees

    Returns
    -------
    float or numpy.ndarray
        cosine of the input value
    """
    return math.cos(math.radians(thetaDeg))


def nextpow2(N):
    """Return exponent for the smallest power of 2 that satisfies 2^p ≥ |N|."""
    check.real_scalar(N, 'N', TypeError)
    if N == 0:
        p = int(0)
    else:
        p = int(np.ceil(np.log2(np.abs(N))))
    
    return p


def ceil_even(x_in):
    """
    Compute the next highest even integer above the input.

    Parameters
    ----------
    x_in : float
        Scalar value

    Returns
    -------
    x_out : integer
        Even-valued integer
    """
    check.real_scalar(x_in, 'x_in', TypeError)
    return int(2 * np.ceil(0.5 * x_in))


def ceil_odd(x_in):
    """
    Compute the next highest odd integer above the input.

    Parameters
    ----------
    x_in : float
        Scalar value

    Returns
    -------
    x_out : integer
        Odd-valued integer
    """
    check.real_scalar(x_in, 'x_in', TypeError)
    x_out = int(np.ceil(x_in))
    if x_out % 2 == 0:
        x_out += 1

    return x_out


def pad_crop(arrayIn, outsize, extrapval=0):
    """
    Insert a 2D array into another array, centered and zero-padded.

    Given an array and a tuple/list denoting the dimensions of a second array,
    this places the smaller array in the center of the larger array and
    returns that larger array.  The smaller array will be zero-padded to the
    size of the larger.  If ``arrayIn.shape`` is larger than ``outsize`` in
    either dimension, this dimension will be truncated.

    If both sizes of a dimension are even or odd, the array will be centered
    in that dimension.  If the input size is even and the output size is odd,
    the smaller array will be shifted one element toward the start of the
    array.  If the input is odd and the output is even, the smaller array will
    be shifted one element toward the end of the array.  This sequence is
    intended to ensure transitivity, so several ``pad_crop()`` calls can be
    chained in no particular order without changing the final result.

    The output array will be of the same type as the input array. It will be
    a copy even if the arrays are the same size.

    Parameters
    ----------
     arrayIn: numpy ndarray
         input array to be padded or cropped
     outsize: array_like, int
         A positive integer or 2-element tuple/list/ndarray of positive
         integers giving dimensions of output array. If outsize is an int, the
         output has square dimensions of (outsize, outsize)

    Returns
    -------
     arrayOut: numpy ndarray
        an ndarray of the same size as ``outsize`` and type as ``arrayIn``

    """
    check.twoD_array(arrayIn, 'arrayIn', TypeError)
    check.real_scalar(extrapval, 'extrapval', TypeError)
    sh0 = arrayIn.shape    # np.shape(arrayIn) #
    int_types = (int, np.integer)  # Int check support
    if isinstance(outsize, int_types):
        sh1 = (outsize, outsize)
    else:
        sh1 = outsize

    try:
        if len(sh1) != 2:
            raise TypeError('Output dimensions must have 2 elements')
        if (not isinstance(sh1[0], int)) or (not isinstance(sh1[1], int)):
            raise TypeError('Output dimensions must be integers')
        if (sh1[0] <= 0) or (sh1[1] <= 0):
            raise TypeError('Output dimensions must be positive ' + 'integers')
    except TypeError:
        raise TypeError('outsize must be an iterable')

    arrayOut = extrapval * np.ones(sh1, dtype=arrayIn.dtype)

    xneg = min(sh0[1]//2, sh1[1]//2)
    xpos = min(sh0[1] - sh0[1]//2, sh1[1] - sh1[1]//2)
    yneg = min(sh0[0]//2, sh1[0]//2)
    ypos = min(sh0[0] - sh0[0]//2, sh1[0] - sh1[0]//2)

    slice0 = (slice(sh0[0]//2-yneg, sh0[0]//2+ypos),
              slice(sh0[1]//2-xneg, sh0[1]//2+xpos))
    slice1 = (slice(sh1[0]//2-yneg, sh1[0]//2+ypos),
              slice(sh1[1]//2-xneg, sh1[1]//2+xpos))

    arrayOut[slice1] = arrayIn[slice0]

    return arrayOut


def allcomb(*args, **kwargs):
    """
    Compute the Cartesian product of a series of iterables.

    Compute the Cartesian product of a series of iterables, i.e. the list
    consisting of all n-tuples formed by choosing one element from each of the
    n inputs. The output list will have length (P1 x P2 x ... x PN), where
    P1, P2, ..., PN are the lengths of the N input lists.

    Examples
    --------
        allcomb([1, 3, 5], [-3, 8], [0, 1]) % numerical input:
            [(1, -3, 0), (1, -3, 1), (1, 8, 0), (1, 8, 1), (3, -3, 0),
             (3, -3, 1), (3, 8, 0),(3, 8, 1), (5, -3, 0), (5, -3, 1),
             (5, 8, 0), (5, 8, 1)]

        allcomb('abc','XY') % character arrays
            [('a', 'X'), ('a', 'Y'), ('b', 'X'), ('b', 'Y'), ('c', 'X'),
             ('c', 'Y')]

        allcomb('xy', [65, 66]) % a combination
            [('x', 65), ('x', 66), ('y', 65), ('y', 66)]
            # a 4-by-2 character array

    Parameters
    ----------
        An arbitrary long series of iterables.  May be of different lengths and
        types.

    Returns
    -------
    list of tuple
        Cartesian product of input lists (explained above)
    """
    return list(itertools.product(*args))


def _spec_arg(k, kwargs, v):
    """
    Specify default argument for class constructors created from .mat files.

    Used in autogenerated classes like ModelParameters.

    Parameters
    ----------
    k : string
        Name of variable whose value will be assigned

    kwargs : dict
        Dictionary of keyword arguments, which may or may not specify a value
        for k.  If the "mat_struct" field exists, then the value of k will be
        obtained by accessing the corresponding value inside the given MATLAB
        struct via kwargs["mat_struct"].

    v : any
        Default value for the variable k.

    Returns
    -------
    The value to initialze the class with.
    """
    if k in kwargs:
        return kwargs[k]

    elif "mat_struct" in kwargs:
        return eval("kwargs[\"mat_struct\"]." + k)

    else:
        return v


def broadcast(axis):
    """
    Use numpy array broadcasting in two dimensions.

    Use numpy array broadcasting to return two views of the input axis that
    behave like a row vector (x) and a column vector (y), and which can be used
    to build memory-efficient coordinate grids without using meshgrid.

    Given an axis with length N, the naive approach using meshgrid requires
    building two NxN arrays, and combining them (e.g. to obtain a radial
    coordinate grid) produces a third NxN array. Using array broadcasting, one
    only needs to create two separate views into the original Nx1 vector to
    create the final NxN array.

    Parameters
    ----------
    axis : array_like
        1D coordinate axis

    Returns
    -------
    array_like, array_like
        Two views into axis that behave like a row and column vector,
        respectively.

    """
    x = axis[None, :]
    y = axis[:, None]
    return x, y


def azimuthal_grid(axis, xStretch=1., yStretch=1.):
    """
    Compute a memory-efficient radial grid using array broadcasting.

    Parameters
    ----------
    axis : array_like
        1D coordinate axis

    Returns
    -------
    array_like
        2D grid with azimuthal coordinates generated by axis
    """
    check.oneD_array(axis, 'axis', TypeError)
    check.real_scalar(xStretch, 'xStretch', TypeError)
    check.real_scalar(yStretch, 'yStretch', TypeError)
    x, y = broadcast(axis)
    return np.arctan2(y/yStretch, x/xStretch)


def radial_grid(axis, xStretch=1., yStretch=1.):
    """
    Compute a memory-efficient radial grid using array broadcasting.

    Parameters
    ----------
    axis : array_like
        1D coordinate axis

    Returns
    -------
    array_like
        2D grid with radial coordinates generated by axis
    """
    check.oneD_array(axis, 'axis', TypeError)
    check.real_scalar(xStretch, 'xStretch', TypeError)
    check.real_scalar(yStretch, 'yStretch', TypeError)

    x, y = broadcast(axis)
    return np.sqrt((x/xStretch)**2 + (y/yStretch)**2)


def radial_grid_squared(axis, xStretch=1., yStretch=1.):
    """
    Compute a memory-efficient squared radial grid using array broadcasting.

    Parameters
    ----------
    axis : array_like
        1D coordinate axis

    Returns
    -------
    array_like
        2D grid with squared radial coordinates generated by axis
    """
    check.oneD_array(axis, 'axis', TypeError)
    check.real_scalar(xStretch, 'xStretch', TypeError)
    check.real_scalar(yStretch, 'yStretch', TypeError)

    x, y = broadcast(axis)
    return (x/xStretch)**2 + (y/yStretch)**2


def create_axis(N, step, centering='pixel'):
    """
    Create a one-dimensional coordinate axis with a given size and step size.

    Can be constructed to follow either the FFT (pixel-centered) interpixel-
    centered convention, which differ by half a pixel for even-sized arrays.
    For odd-sized arrays, both values of centering put the center on the center
    pixel.

    Parameters
    ----------
    N : int
        Number of pixels in output axis
    step : float
        Physical step size between axis elements
    centering : 'pixel' or 'interpixel'
        Centering of the coordinates in the array.  Note that if N is odd, the
        result will be pixel-centered regardless of the value of this keyword.

    Returns
    -------
    array_like
        The output coordinate axis
    """
    check.positive_scalar_integer(N, 'N', TypeError)
    check.real_positive_scalar(step, 'step', TypeError)
    check.centering(centering)

    axis = np.arange(-N // 2, N // 2, dtype=np.float64) * step
    even = not N % 2  # Even number of samples?

    if even and (centering == 'interpixel'):
        # Inter-pixel-centering onlyif the number of samples is even
        axis += 0.5 * step

    return axis


def offcenter_crop(arrayIn, centerRow, centerCol, nRowOut, nColOut):
    """
    Crop a 2-D array to be centered at the specified pixel.

    This function crops a 2-D array about the center pixel specified by
    centerRow and centerCol. The input array can be
    rectangular with even or odd side lengths. The output will is rectangular
    with dimensions nRowOut, nColOut. If the output array includes
    regions outside the input array, those pixels are included
    and set to zero. If the specified cropping region is fully outside the
    input array, then the output is all zeros.

    The center pixel of an odd-sized array is the array
    center, and the center pixel of an even-sized array follows the FFT
    center pixel convention.

    Parameters
    ----------
    arrayIn : array_like
        2-D input array
    centerRow, centerCol : float or int
        Indices of the pixel to be used as the output array's center.
        Floating point values are rounded to the nearest integer.
        Convention in this function is that y is the first axis.
        Values can be negative and/or lie outside the input array.
    nRowOut, nColOut : int
        Height and width of the 2-D output array in pixels.

    Returns
    -------
    recentered_image : numpy ndarray
        2-D square array

    Notes
    -----
    All alignment units are in detector pixels.
    """
    check.twoD_array(arrayIn, 'arrayIn', TypeError)
    check.real_scalar(centerRow, 'centerRow', TypeError)
    check.real_scalar(centerCol, 'centerCol', TypeError)
    check.positive_scalar_integer(nRowOut, 'nRowOut', TypeError)
    check.positive_scalar_integer(nColOut, 'nColOut', TypeError)

    [nRowIn, nColIn] = arrayIn.shape
    centerRow = int(np.round(centerRow))
    centerCol = int(np.round(centerCol))

    # Compute how much to pad the array in y (if any)
    rowPadPre = 0
    rowPadPost = 0
    if np.ceil(-nRowOut/2.) + centerRow < 0:
        rowPadPre = np.abs(np.ceil(-nRowOut/2.) + centerRow)
    if np.ceil(nRowOut/2.) + centerRow > (nRowIn - 1):
        rowPadPost = np.ceil(nRowOut/2.) + centerRow - \
          (nRowIn)
    y_pad = int(np.max((rowPadPre, rowPadPost)))

    # Compute how much to pad the array in x (if any)
    colPadPre = 0
    colPadPost = 0
    if np.ceil(-nColOut/2.) + centerCol < 0:
        colPadPre = np.abs(np.ceil(-nColOut/2.) + centerCol)
    if np.ceil(nColOut/2.) + centerCol > (nColIn - 1):
        colPadPost = np.ceil(nColOut/2.) + centerCol - \
                     (nColIn)
    x_pad = int(np.max((colPadPre, colPadPost)))

    arrayPadded = pad_crop(arrayIn, (nRowIn+2*y_pad,
                                      nColIn+2*x_pad))

    centerCol += x_pad
    centerRow += y_pad

    # Buffer needed to keep output array correct size
    if nRowOut % 2 == 1:
        rowBuffer = 1
    else:
        rowBuffer = 0
    if nColOut % 2 == 1:
        colBuffer = 1
    else:
        colBuffer = 0

    arrayOut = arrayPadded[
       centerRow-nRowOut//2:centerRow+nRowOut//2 + rowBuffer,
       centerCol-nColOut//2:centerCol+nColOut//2 + colBuffer]

    return arrayOut
    

def bin_downsample(Ain, dsfac):
    """
    Downsample an array by binning.
    
    Parameters
    ----------
    Ain : 2-D array
        The matrix to be downsampled
    dsfac : int
        Downsampling factor for the matrix
    
    Returns
    -------
    Aout : 2-D array
        Downsampled array
    """
    # Error checks on inputs
    check.twoD_array(Ain, 'Ain', ValueError)
    check.positive_scalar_integer(dsfac, 'dsfac', ValueError)
    
    # Array Sizes
    ny0, nx0 = Ain.shape
    if (nx0 % dsfac != 0) or (ny0 % dsfac != 0):
        ValueError('The size of Ain must be divisible by dsfac.')

    nx1 = int(nx0/dsfac)
    ny1 = int(ny0/dsfac)
    
    # Bin and average values from the high-res array into the low-res array
    Aout = np.zeros((ny1, nx1))
    for ix in range(nx1):
        for iy in range(ny1):
            Aout[iy, ix] = np.sum(Ain[dsfac*iy:dsfac*(iy+1),
                                      dsfac*ix:dsfac*(ix+1)])/dsfac/dsfac
            
    return Aout
