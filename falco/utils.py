import numpy as np

def ceil_even(x_in):
    """Compute the next highest even integer above the input

    Parameters
    ----------
    x_in : float
        Scalar value

    Returns
    --------
    x_out : integer
        Even-valued integer
    """

    return int(2*np.ceil(0.5*x_in))

def ceil_odd(x_in):
    """Compute the next highest odd integer above the input

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
    if x_out % 2 == 0: x_out +=1
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