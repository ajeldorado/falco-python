import time
import numpy as np
import itertools
import falco
import math

class TicToc(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]\t' % self.name,end='')
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
    --------
    rho : float or numpy.ndarray
        radial coordinate(s)
    theta : float or numpy.ndarray
        azimuthal coordinate(s)    
    """
    if(type(x)==np.ndarray and type(y)==np.ndarray):
        if not x.shape==y.shape:
            raise ValueError('The two inputs must have the same shape.')
    
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)
    
    return(rho,theta)

def sind(thetaDeg):
    """
    Compute the sine of the input given in degrees.

    Parameters
    ----------
    thetaDeg : float or numpy.ndarray
        Angle in degrees

    Returns
    --------
    x : float or numpy.ndarray
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
    --------
    x : float or numpy.ndarray
        cosine of the input value
    """

    return math.cos(math.radians(thetaDeg))


def nextpow2(N): 
    """
    P = nextpow2(N) returns the exponents for the smallest powers of two that satisfy
    2^p ≥ |N|
    """   
    p = np.ceil(np.log2(abs(N)))    
    return p


def ceil_even(x_in):
    """
    Compute the next highest even integer above the input

    Parameters
    ----------
    x_in : float
        Scalar value

    Returns
    --------
    x_out : integer
        Even-valued integer
    """

    return int(2 * np.ceil(0.5 * x_in))


def ceil_odd(x_in):
    """
    Compute the next highest odd integer above the input

    Parameters
    ----------
    x_in : float
        Scalar value

    Returns
    --------
    x_out : integer
        Odd-valued integer
    """
    x_out = int(np.ceil(x_in))
    if x_out % 2 == 0:
        x_out += 1
    return x_out


def padOrCropEven(Ain, Ndes, **kwargs):
    """
    Pad or crop an even-sized input matrix to the desired size.

    Parameters
    ----------
    Ain : np.ndarray
        Rectangular or square input array with even size along each dimension
    Ndes : int
        Desired, even number of points across output array.  The output array will be
        padded/cropped to a square shape.

    Returns
    -------
    Aout : np.ndarray
        Square, even-sized padded or cropped array
    """
    extrapval = kwargs.get('extrapval', 0)  # Value to use for extrapolated points
    Ny0, Nx0 = Ain.shape

    if Nx0 % 2 or Ny0 % 2:  # Size of input array is odd along at least one dimension
        raise ValueError('Input is not an even-sized array')
    elif Nx0 != Ny0:
        raise ValueError('Input is not square')
    elif not isinstance(Ndes, int):
        raise ValueError('Wrong number of dimensions specified for output')

    if min(Nx0, Ny0) > Ndes:  # Output array is smaller than input, so crop
        Aout = Ain[(Ny0 - Ndes) // 2:(Ny0 + Ndes) // 2, (Nx0 - Ndes) // 2:(Nx0 + Ndes) // 2]
    elif max(Nx0, Ny0) < Ndes:  # Output array is bigger than input, so pad
        pad_x = (Ndes - Nx0) // 2
        pad_y = (Ndes - Ny0) // 2
        Aout = np.pad(Ain, (pad_y, pad_x), mode='constant', constant_values=extrapval)
    else:  # Do nothing
        Aout = Ain

    return Aout


def allcomb(*args, **kwargs):
    """
    Compute the Cartesian product of a series of iterables, i.e. the list consisting of all n-tuples
    formed by choosing one element from each of the n inputs.  The output list will have
    have length (P1 x P2 x ... x PN), where P1, P2, ..., PN are the lengths of the N input lists.

    Examples:
        allcomb([1, 3, 5], [-3, 8], [0, 1]) % numerical input:
            [(1, -3, 0), (1, -3, 1), (1, 8, 0), (1, 8, 1), (3, -3, 0), (3, -3, 1), (3, 8, 0),
            (3, 8, 1), (5, -3, 0), (5, -3, 1), (5, 8, 0), (5, 8, 1)]

        allcomb('abc','XY') % character arrays
            [('a', 'X'), ('a', 'Y'), ('b', 'X'), ('b', 'Y'), ('c', 'X'), ('c', 'Y')]

        allcomb('xy', [65, 66]) % a combination
            [('x', 65), ('x', 66), ('y', 65), ('y', 66)]  % a 4-by-2 character array

    Parameters
    ----------
    args
        An arbitrary long series of iterables.  May be of different lengths and types.

    Returns
    -------
    list of tuple
        Cartesian product of input lists (explained above)
    """
    return list(itertools.product(*args))


def _spec_arg(k, kwargs, v):
    """
    Specify a default argument for constructors of classes created from .mat files.
    Used in autogenerated classes like ModelParameters.

    Parameters
    ----------
    k : string
        Name of variable whose value will be assigned

    kwargs : dict
        Dictionary of keyword arguments, which may or may not specify a value for k.  If the
        "mat_struct" field exists, then the value of k will be obtained by accessing the
        corresponding value inside the given MATLAB struct via kwargs["mat_struct"].k

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
    Use numpy array broadcasting to return two views of the input axis that behave like a row
    vector (x) and a column vector (y), and which can be used to build memory-efficient
    coordinate grids without using meshgrid.

    Given an axis with length N, the naive approach using meshgrid requires building two NxN arrays,
    and combining them (e.g. to obtain a radial coordinate grid) produces a third NxN array.
    Using array broadcasting, one only needs to create two separate views into the original Nx1
    vector to create the final NxN array.

    Parameters
    ----------
    axis : array_like
        1D coordinate axis

    Returns
    -------
    array_like, array_like
        Two views into axis that behave like a row and column vector, respectively

    """
    x = axis[None, :]
    y = axis[:, None]
    return x, y


def radial_grid(axis):
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
    x, y = broadcast(axis)
    return np.sqrt(x ** 2 + y ** 2)

def radial_grid_squared(axis):
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
    x, y = broadcast(axis)
    return (x ** 2 + y ** 2)

def create_axis(N, step, centering='pixel'):
    """
    Create a one-dimensional coordinate axis with a given size and step size.  Can be constructed
    to follow either the FFT (pixel-centered) or MFT (inter-pixel-centered) convention,
    which differ by half a pixel.

    Parameters
    ----------
    N : int
        Number of pixels in output axis
    step : float
        Physical step size between axis elements
    centering : str
        Either 'pixel' (pixel-centered) or 'interpixel' (inter-pixel-centered).  Note that if N is
        odd, the result will be pixel-centered regardless of the value of this keyword.

    Returns
    -------
    array_like
        The output coordinate axis
    """
    axis = np.arange(-N // 2, N // 2, dtype=np.float64) * step
    even = not N % 2  # Even number of samples?

    if even and (centering == 'interpixel'):
        # Inter-pixel-centering only makes sense if the number of samples is even
        axis += 0.5 * step

    return axis

def falco_compute_thput(mp):
    """
    Function to compute the off-axis throughput of the coronagraph

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters

    Returns
    -------
    thput : float
        The off-axis throughput of the coronagraph at the specified field location
    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')


    ImSimOffaxis = falco.imaging.falco_sim_image_compact_offaxis(mp, mp.thput_eval_x, mp.thput_eval_y,EVAL=True)
    #if(mp.flagPlot): figure(324); imagesc(mp.Fend.eval.xisDL,mp.Fend.eval.etasDL,ImSimOffaxis); axis xy equal tight; title('Off-axis PSF for Throughput Calculation','Fontsize',20); set(gca,'Fontsize',20); colorbar; drawnow;  end

    if(mp.thput_metric.lower()=='hmi'): #--Absolute energy within half-max isophote(s)
        maskHM = np.zeros(mp.Fend.eval.RHOS.shape,dtype=bool)
        maskHM[ImSimOffaxis>=0.5*np.max(ImSimOffaxis)] = True
        # figure(325); imagesc(mp.Fend.eval.xisDL,mp.Fend.eval.etasDL,maskHM); axis xy equal tight; drawnow;
        thput = np.sum(ImSimOffaxis[maskHM==1])/mp.sumPupil*np.mean(mp.Fend.eval.I00);
        print('Core throughput within the half-max isophote(s) = %.2f%% \tat separation = (%.1f, %.1f) lambda0/D.' % (100*thput,mp.thput_eval_x,mp.thput_eval_y))
            
    elif( (mp.thput_metric.lower()=='ee') or (mp.thput_metric.lower()=='e.e.') ): #--Absolute energy encircled within a given radius
        # (x,y) location [lambda_c/D] in dark hole at which to evaluate throughput
        maskEE = np.zeros(mp.Fend.eval.RHOS.shape,dtype=bool)
        maskEE[mp.Fend.eval.RHOS<=mp.thput_radius] = True
        # figure(325); imagesc(mp.Fend.eval.xisDL,mp.Fend.eval.etasDL,maskEE); axis xy equal tight; drawnow;
        thput = np.sum(ImSimOffaxis[maskEE==1])/mp.sumPupil*np.mean(mp.Fend.eval.I00);
        print('E.E. throughput within a %.2f lambda/D radius = %.2f%% \tat separation = (%.1f, %.1f) lambda/D.\n'%(mp.thput_radius,100*thput,mp.thput_eval_x,mp.thput_eval_y))
            
    return thput,ImSimOffaxis
