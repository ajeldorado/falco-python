import numpy as np
import logging
from falco import utils
from falco.masks import falco_gen_vortex_mask
from scipy.signal import tukey

log = logging.getLogger(__name__)

_VALID_CENTERING = ['pixel', 'interpixel']
_CENTERING_ERR = 'Invalid centering specification. Options: {}'.format(_VALID_CENTERING)


def propcustom_relay(E_in, Nrelay,centering='pixel'):
    """
    Propagate a field using two successive Fourier transforms Nrelay times, without any 
    intermediate mask multiplications. This results in a 180-degree rotation of the array 
    for each optical relay. Correct centering of the array must be maintained.

    Parameters
    ----------
    E_in : array_like
        Input electric field
    Nrelay: int
        Number of times to relay (and rotate by 180 degrees)
    centering : string
        Whether the input field is pixel-centered or inter-pixel-centered. If the array is 
        pixel-centered, the output is shifted by 1 pixel in both axes after an odd number 
        of relays to ensure that the origin remains at the same pixel as in the input array.

    Returns
    -------
    array_like
        The input array, after propagation with two Fourier transforms.

    """
    if centering not in _VALID_CENTERING:
        raise ValueError(_CENTERING_ERR)

    #--Only rotate if an odd number of 180-degree rotations. If even, no change.
    if(np.mod(Nrelay,2)==1):
        E_out = E_in[::-1, ::-1]  # Reverse and scale input to account for propagation
        if centering == 'pixel':
            E_out = np.roll(E_out, (1, 1), axis=(0, 1))  # Move the DC pixel back to the right place
    else:
        E_out = E_in
        
    return E_out
    
def propcustom_2FT(E_in, centering='pixel'):
    """
    Propagate a field using two successive Fourier transforms, without any intermediate mask
    multiplications.

    Parameters
    ----------
    E_in : numpy ndarray
        Input electric field
    centering : string
        Whether the input field is pixel-centered or inter-pixel-centered.  If
        inter-pixel-centered, then the output is simply a scaled version of the input, flipped in
        the vertical and horizontal directions.  If pixel-centered, the output is also shifted by 1
        pixel in both directions after flipping, to ensure that the origin remains at the same
        pixel as in the input array.

    Returns
    -------
    array_like
        The input array, after propagation with two Fourier transforms.

    """
    if centering not in _VALID_CENTERING:
        raise ValueError(_CENTERING_ERR)

    E_out = E_in[::-1, ::-1]  # Reverse and scale input to account for propagation

    if centering == 'pixel':
        E_out = np.roll(E_out, (1, 1), axis=(0, 1))  # Move the DC pixel back to the right place

    return E_out


def propcustom_PTP(E_in, full_width, lambda_, dz):
    """
    Propagate an electric field array using the angular spectrum technique.

    Parameters
    ----------
    E_in : array_like
        Square (i.e. NxN) input array.
    full_width : float
        The width along each side of the array [meters]
    lambda_ : float
        Propagation wavelength [meters]
    dz : float
        Propagation distance [meters]

    Returns
    -------
    array_like
        Field after propagating over distance dz.

    """
    M, N = E_in.shape
    dx = full_width / N
    N_critical = int(np.floor(lambda_ * np.abs(dz) / (dx ** 2)))  # Critical sampling

    if M != N:  # Input array is not square
        raise ValueError('Input array is not square')

    elif N < N_critical:
        log.warning(
             '''
             Input array is undersampled.
                Minimum required samples:  {}
                                  Actual:  {}
             '''.format(N_critical, N))

    fx = np.arange(-N // 2, N // 2) / full_width
    rho = utils.radial_grid(fx)  # Spatial frequency coordinate grid

    kernel = np.fft.fftshift(np.exp(-1j * np.pi * lambda_ * dz * (rho ** 2)))
    intermediate = np.fft.fftn(np.fft.fftshift(E_in))

    return np.fft.ifftshift(np.fft.ifftn(kernel * intermediate))


def propcustom_mft_FtoP(E_foc, fl, lambda_, dxi, deta, dx, N, centering='pixel'):
    """
    Propagate a field from a focal plane to a pupil plane, using a matrix-multiply DFT.

    Parameters
    ----------
    E_foc : array_like
        Electric field array in focal plane
    fl : float
        Focal length of Fourier transforming lens
    lambda_ : float
        Propagation wavelength
    dxi : float
        Step size along horizontal axis of focal plane
    deta : float
        Step size along vertical axis of focal plane
    dx : float
        Step size along either axis of focal plane.  The vertical and horizontal step sizes are
        assumed to be equal.
    N : int
        Number of datapoints along each side of the pupil-plane (output) array
    centering : string
        Whether the input and output arrays are pixel-centered or inter-pixel-centered.
        Possible values: 'pixel', 'interpixel'

    Returns
    -------
    array_like
        Field in pupil plane, after propagating through Fourier transforming lens

    """
    if centering not in _VALID_CENTERING:
        raise ValueError(_CENTERING_ERR)

    Neta, Nxi = E_foc.shape
    dy = dx  # Assume equal sample spacing along both directions

    # Focal-plane coordinates
    xi = utils.create_axis(Nxi, dxi, centering=centering)[:, None]  # Broadcast to column vector
    eta = utils.create_axis(Neta, dxi, centering=centering)[None, :]  # Broadcast to row vector

    # Pupil-plane coordinates
    x = utils.create_axis(N, dx, centering=centering)[None, :]  # Broadcast to row vector
    y = x.T  # Column vector

    # Fourier transform matrices
    pre = np.exp(-2 * np.pi * 1j * (y * eta) / (lambda_ * fl))
    post = np.exp(-2 * np.pi * 1j * (xi * x) / (lambda_ * fl))

    # Constant scaling factor in front of Fourier transform
    scaling = np.sqrt(dx * dy * dxi * deta) / (1 * lambda_ * fl)

    return scaling * np.linalg.multi_dot([pre, E_foc, post])


def propcustom_mft_PtoF(E_pup, fl, lambda_, dx, dxi, Nxi, deta, Neta, centering='pixel'):
    """
    Propagate a field from a pupil plane to a focal plane, using a matrix-multiply DFT.

    Parameters
    ----------
    E_pup : array_like
        Electric field array in pupil plane
    fl : float
        Focal length of Fourier transforming lens
    lambda_ : float
        Propagation wavelength
    dx : float
        Step size along either axis of focal plane.  The vertical and horizontal step sizes are
        assumed to be equal.
    dxi : float
        Step size along horizontal axis of focal plane
    Nxi : int
        Number of samples along horizontal axis of focal plane.
    deta : float
        Step size along vertical axis of focal plane
    Neta : int
        Number of samples along vertical axis of focal plane.
    centering : string
        Whether the input and output arrays are pixel-centered or inter-pixel-centered.
        Possible values: 'pixel', 'interpixel'

    Returns
    -------
    array_like
        Field in pupil plane, after propagating through Fourier transforming lens

    """
    if centering not in _VALID_CENTERING:
        raise ValueError(_CENTERING_ERR)

    M, N = E_pup.shape
    dy = dx

    if M != N:
        raise ValueError('Input array is not square')

    # Pupil-plane coordinates
    x = utils.create_axis(N, dx, centering=centering)[:, None]  # Broadcast to column vector
    y = x.T  # Row vector

    # Focal-plane coordinates
    xi = utils.create_axis(Nxi, dxi, centering=centering)[None, :]  # Broadcast to row vector
    eta = utils.create_axis(Neta, deta, centering=centering)[:, None]  # Broadcast to column vector

    # Fourier transform matrices
    pre = np.exp(-2 * np.pi * 1j * (eta * y) / (lambda_ * fl))
    post = np.exp(-2 * np.pi * 1j * (x * xi) / (lambda_ * fl))

    # Constant scaling factor in front of Fourier transform
    scaling = np.sqrt(dx * dy * dxi * deta) / (1 * lambda_ * fl)

    return scaling * np.linalg.multi_dot([pre, E_pup, post])


def propcustom_mft_Pup2Vortex2Pup( IN, charge, apRad,  inVal, outVal):
    """
    Function to propagate from the pupil plane before a vortex FPM to the pupil 
    plane after it.
    """

    # showPlots2debug = False 

    D = 2.0*apRad
    lambdaOverD = 4. # samples per lambda/D
    
    NA = IN.shape[1]
    NB = lambdaOverD*D
    
    [X,Y] = np.meshgrid(np.arange(-NB/2.,NB/2,dtype=float),np.arange(-NB/2.,NB/2,dtype=float))
    [RHO,THETA] = utils.cart2pol(Y,X)    
   
    windowKnee = 1.-inVal/outVal
    
    windowMASK1 = falco_gen_Tukey4vortex( 2*outVal*lambdaOverD, RHO, windowKnee )
    windowMASK2 = falco_gen_Tukey4vortex( NB, RHO, windowKnee )

    # DFT vectors 
    x = np.arange(-NA/2,NA/2,dtype=float)/D   #(-NA/2:NA/2-1)/D
    u1 = np.arange(-NB/2,NB/2,dtype=float)/lambdaOverD #(-NB/2:NB/2-1)/lambdaOverD
    u2 = np.arange(-NB/2,NB/2,dtype=float)*2*outVal/NB # (-NB/2:NB/2-1)*2*outVal/N
    
    FPM = falco_gen_vortex_mask( charge, NB )

    #if showPlots2debug; figure;imagesc(abs(IN));axis image;colorbar; title('pupil'); end;

    ## Low-sampled DFT of entire region

    FP1 = 1/(1*D*lambdaOverD)*np.exp(-1j*2*np.pi*u1.reshape(u1.size,1) @ x.reshape(1,x.size)) @ IN @ np.exp(-1j*2*np.pi*x.reshape(x.size,1) @ u1.reshape(1,u1.size))
    #if showPlots2debug; figure;imagesc(log10(abs(FP1).^2));axis image;colorbar; title('Large scale DFT'); end;

    LP1 = 1/(1*D*lambdaOverD)*np.exp(-1j*2*np.pi*x.reshape(x.size,1) @ u1.reshape(1,u1.size)) @ (FP1*FPM*(1-windowMASK1)) @ np.exp(-1j*2*np.pi*u1.reshape(u1.size,1) @ x.reshape(1,x.size))
    #if showPlots2debug; figure;imagesc(abs(FP1.*(1-windowMASK1)));axis image;colorbar; title('Large scale DFT (windowed)'); end;
    
    ## Fine sampled DFT of innter region
    FP2 = 2*outVal/(1*D*NB)*np.exp(-1j*2*np.pi*u2.reshape(u2.size,1)
    @ x.reshape(1,x.size)) @ IN @ np.exp(-1j*2*np.pi*x.reshape(x.size,1) @ u2.reshape(1,u2.size))
    #if showPlots2debug; figure;imagesc(log10(abs(FP2).^2));axis image;colorbar; title('Fine sampled DFT'); end;
    FPM = falco_gen_vortex_mask(charge, NB)
    LP2 = 2.0*outVal/(1*D*NB)*np.exp(-1j*2*np.pi*x.reshape(x.size,1) @ u2.reshape(1,u2.size)) @ (FP2*FPM*windowMASK2) @ np.exp(-1j*2*np.pi*u2.reshape(u2.size,1) @ x.reshape(1,x.size))       
    #if showPlots2debug; figure;imagesc(abs(FP2.*windowMASK2));axis image;colorbar; title('Fine sampled DFT (windowed)'); end;
    OUT = LP1 + LP2;
    #if showPlots2debug; figure;imagesc(abs(OUT));axis image;colorbar; title('Lyot plane'); end;

    return OUT


def falco_gen_Tukey4vortex( Nwindow, RHO, alpha ):
#% REQUIRED INPUTS: 
#% Nwindow =
#% RHO = 
#% alpha  = 
#%
#% OUTPUTS:
#%  w:     2-D square array of the specified Tukey window
#%
#% Written by Garreth Ruane.
#
#function w = falco_gen_Tukey4vortex( Nwindow, RHO, alpha )

    Nlut = int(10*Nwindow)
    rhos0 = np.linspace(-Nwindow/2,Nwindow/2,Nlut)
    lut = tukey(Nlut,alpha)#,left=0,right=0)
    
    w = np.interp(RHO,rhos0,lut)

    return w