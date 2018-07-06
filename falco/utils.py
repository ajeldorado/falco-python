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

def padOrCropEven(Ain,Ndes,extrapval=0):
    Ny0, Nx0 = Ain.shape

    if Nx0 <> Ny0:
        raise ValueError("padOrCropEven: Input is not a square array")
    if Nx0%2:
        raise ValueError("padOrCropEven: Input is not an even-sized array")


    if Nx0 > Ndes: #Crop
        ifrom = (Nx0-Ndes)/2
        ito = (Nx0+Ndes)/2
        return Ain[ifrom:ito, ifrom:ito]
    elif Nx0 < Ndes: #Pad
        pad = (Ndes-Nx0)/2
        return np.pad(A,((pad,pad),(pad,pad)),"constant", constant_values=(1, 1))
    else: #Leave same size
        return Ain[:,:]

