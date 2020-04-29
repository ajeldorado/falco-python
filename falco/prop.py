import numpy as np
import logging
from falco import util, check
from falco.mask import falco_gen_vortex_mask
from scipy.signal import tukey

log = logging.getLogger(__name__)

_VALID_CENTERING = ['pixel', 'interpixel']
_CENTERING_ERR = 'Invalid centering specification. Options: {}'.format(_VALID_CENTERING)


def relay(E_in, Nrelay, centering='pixel'):
    """
    Perform re-imaging of the input E-field through optical relays.
    
    Propagate a field through Nrelay optical relays, without any intermediate
    mask multiplications. This results in a 180-degree rotation of the array 
    for each optical relay. Correct centering of the array must be maintained.

    Parameters
    ----------
    E_in : array_like
        Input electric field
    Nrelay: int
        Number of times to relay (and rotate by 180 degrees)
    centering : string
        Whether the input field is pixel-centered or inter-pixel-centered. If
        the array is pixel-centered, the output is shifted by 1 pixel in both
        axes after an odd number of relays.

    Returns
    -------
    E_out : array_like
        The output E-field. Same as the input E-field but rotated by 180
        degrees times the number of optical relays.

    """
    if centering not in _VALID_CENTERING:
        raise ValueError(_CENTERING_ERR)
    check.twoD_array(E_in, 'E_in', TypeError)
    check.scalar_integer(Nrelay, 'Nrelay', TypeError)

    #--Only rotate if odd number of 180-degree rotations. If even, no change.
    if(np.mod(Nrelay,2)==1):
        # Reverse and scale input to account for propagation
        E_out = E_in[::-1, ::-1]  
        if centering == 'pixel':
            # Move the DC pixel back to the right place
            E_out = np.roll(E_out, (1, 1), axis=(0, 1))  
    else:
        E_out = E_in
        
    return E_out


def ptp(E_in, full_width, wavelength, dz):
    """
    Propagate an electric field array using the angular spectrum technique.

    Parameters
    ----------
    E_in : array_like
        Square (i.e. NxN) input array.
    full_width : float
        The width along each side of the array [meters]
    wavelength : float
        Propagation wavelength [meters]
    dz : float
        Axial propagation distance [meters]

    Returns
    -------
    array_like
        Field after propagating over distance dz.

    """
    check.twoD_array(E_in, 'E_in', TypeError)
    check.real_positive_scalar(full_width, 'full_width', TypeError)
    check.real_positive_scalar(wavelength, 'wavelength', TypeError)
    check.real_scalar(dz, 'dz', TypeError)
    
    M, N = E_in.shape
    dx = full_width / N
    N_critical = int(np.floor(wavelength * np.abs(dz) / (dx ** 2)))  # Critical sampling

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
    rho = util.radial_grid(fx)  # Spatial frequency coordinate grid

    kernel = np.fft.fftshift(np.exp(-1j * np.pi * wavelength * dz * (rho ** 2)))
    intermediate = np.fft.fftn(np.fft.fftshift(E_in))

    return np.fft.ifftshift(np.fft.ifftn(kernel * intermediate))


def mft_f2p(E_foc, fl, wavelength, dxi, deta, dx, N, centering='pixel'):
    """
    Propagate a field from a focal plane to a pupil plane, using a matrix-multiply DFT.

    Parameters
    ----------
    E_foc : array_like
        Electric field array in focal plane
    fl : float
        Focal length of Fourier transforming lens
    wavelength : float
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
    check.twoD_array(E_foc, 'E_foc', TypeError)
    check.real_scalar(fl, 'fl', TypeError)
    check.real_positive_scalar(wavelength, 'wavelength', TypeError)
    check.real_positive_scalar(dxi, 'dxi', TypeError)
    check.real_positive_scalar(deta, 'deta', TypeError)
    check.real_positive_scalar(dx, 'dx', TypeError)
    check.positive_scalar_integer(N, 'N', TypeError)

    Neta, Nxi = E_foc.shape
    dy = dx  # Assume equal sample spacing along both directions

    # Focal-plane coordinates
    xi = util.create_axis(Nxi, dxi, centering=centering)[:, None]  # Broadcast to column vector
    eta = util.create_axis(Neta, dxi, centering=centering)[None, :]  # Broadcast to row vector

    # Pupil-plane coordinates
    x = util.create_axis(N, dx, centering=centering)[None, :]  # Broadcast to row vector
    y = x.T  # Column vector

    # Fourier transform matrices
    pre = np.exp(-2 * np.pi * 1j * (y * eta) / (wavelength * fl))
    post = np.exp(-2 * np.pi * 1j * (xi * x) / (wavelength * fl))

    # Constant scaling factor in front of Fourier transform
    scaling = np.sqrt(dx * dy * dxi * deta) / (1 * wavelength * fl)

    return scaling * np.linalg.multi_dot([pre, E_foc, post])


def mft_p2f(E_pup, fl, wavelength, dx, dxi, Nxi, deta, Neta, centering='pixel'):
    """
    Propagate a pupil to a focus using a matrix-multiply DFT.

    Parameters
    ----------
    E_pup : array_like
        Electric field array in pupil plane
    fl : float
        Focal length of Fourier transforming lens
    wavelength : float
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
    check.twoD_array(E_pup, 'E_pup', TypeError)
    check.real_scalar(fl, 'fl', TypeError)
    check.real_positive_scalar(wavelength, 'wavelength', TypeError)
    check.real_positive_scalar(dx, 'dx', TypeError)
    check.real_positive_scalar(dxi, 'dxi', TypeError)
    check.positive_scalar_integer(Nxi, 'Nxi', TypeError)
    check.real_positive_scalar(deta, 'deta', TypeError)
    check.positive_scalar_integer(Neta, 'Neta', TypeError)

    dy = dx
    M, N = E_pup.shape
    if M != N:
        raise ValueError('Input array is not square')

    # Pupil-plane coordinates
    x = util.create_axis(N, dx, centering=centering)[:, None]  # Broadcast to column vector
    y = x.T  # Row vector

    # Focal-plane coordinates
    xi = util.create_axis(Nxi, dxi, centering=centering)[None, :]  # Broadcast to row vector
    eta = util.create_axis(Neta, deta, centering=centering)[:, None]  # Broadcast to column vector

    # Fourier transform matrices
    pre = np.exp(-2 * np.pi * 1j * (eta * y) / (wavelength * fl))
    post = np.exp(-2 * np.pi * 1j * (x * xi) / (wavelength * fl))

    # Constant scaling factor in front of Fourier transform
    scaling = np.sqrt(dx * dy * dxi * deta) / (1 * wavelength * fl)

    return scaling * np.linalg.multi_dot([pre, E_pup, post])


def mft_p2v2p(pupilPre, charge, beamRadius, inVal, outVal):
    """
    Propagate from the pupil plane before a vortex FPM to pupil plane after it.
    
    Compute a radial Tukey window for propagating through a vortex coroangraph.

    Parameters
    ----------
    pupilPre : array_like
        2-D E-field at pupil plane before the vortex focal plane mask
    charge : int, float
        Charge of the vortex mask
    beamRadius : float
        Beam radius at pupil plane. Units of pixels.
    inVal : float
        Ask Gary
    outVal : float
        Ask Gary
        
    Returns
    -------
    pupilPost : array_like
        2-D E-field at pupil plane after the vortex focal plane mask

    """
    check.twoD_array(pupilPre, 'pupilPre', TypeError)
    check.scalar_integer(charge, 'charge', TypeError)
    check.real_positive_scalar(beamRadius, 'beamRadius', TypeError)
    check.real_positive_scalar(inVal, 'inVal', TypeError)
    check.real_positive_scalar(outVal, 'outVal', TypeError)
    
    # showPlots2debug = False 

    D = 2.0*beamRadius
    lambdaOverD = 4. # samples per lambda/D
    
    NA = pupilPre.shape[1]
    NB = util.ceil_even(lambdaOverD*D)
    
    # [X,Y] = np.meshgrid(np.arange(-NB/2., NB/2., dtype=float),np.arange(-NB/2., NB/2., dtype=float))
    # [RHO,THETA] = util.cart2pol(Y,X)
    RHO = util.radial_grid(np.arange(-NB/2., NB/2., dtype=float))
   
    windowKnee = 1.-inVal/outVal
    
    windowMask1 = gen_tukey_for_vortex(2*outVal*lambdaOverD, RHO, windowKnee)
    windowMask2 = gen_tukey_for_vortex(NB, RHO, windowKnee)

    # DFT vectors 
    x = np.arange(-NA/2,NA/2,dtype=float)/D   #(-NA/2:NA/2-1)/D
    u1 = np.arange(-NB/2,NB/2,dtype=float)/lambdaOverD #(-NB/2:NB/2-1)/lambdaOverD
    u2 = np.arange(-NB/2,NB/2,dtype=float)*2*outVal/NB # (-NB/2:NB/2-1)*2*outVal/N
    
    FPM = falco_gen_vortex_mask(charge, NB)

    #if showPlots2debug; figure;imagesc(abs(pupilPre));axis image;colorbar; title('pupil'); end;

    ## Low-sampled DFT of entire region

    FP1 = 1/(1*D*lambdaOverD)*np.exp(-1j*2*np.pi*np.outer(u1,x)) @ pupilPre @ np.exp(-1j*2*np.pi*np.outer(x,u1))
    #if showPlots2debug; figure;imagesc(log10(abs(FP1).^2));axis image;colorbar; title('Large scale DFT'); end;

    LP1 = 1/(1*D*lambdaOverD)*np.exp(-1j*2*np.pi*np.outer(x,u1)) @ (FP1*FPM*(1-windowMask1)) @ np.exp(-1j*2*np.pi*np.outer(u1,x))
    #if showPlots2debug; figure;imagesc(abs(FP1.*(1-windowMask1)));axis image;colorbar; title('Large scale DFT (windowed)'); end;
    
    ## Fine sampled DFT of innter region
    FP2 = 2*outVal/(1*D*NB)*np.exp(-1j*2*np.pi*np.outer(u2,x)) @ pupilPre @ np.exp(-1j*2*np.pi*np.outer(x,u2))
    #if showPlots2debug; figure;imagesc(log10(abs(FP2).^2));axis image;colorbar; title('Fine sampled DFT'); end;
    FPM = falco_gen_vortex_mask(charge, NB)
    LP2 = 2.0*outVal/(1*D*NB)*np.exp(-1j*2*np.pi*np.outer(x,u2)) @ (FP2*FPM*windowMask2) @ np.exp(-1j*2*np.pi*np.outer(u2,x))       
    #if showPlots2debug; figure;imagesc(abs(FP2.*windowMask2));axis image;colorbar; title('Fine sampled DFT (windowed)'); end;
    pupilPost = LP1 + LP2;
    #if showPlots2debug; figure;imagesc(abs(pupilPost));axis image;colorbar; title('Lyot plane'); end;

    return pupilPost


def gen_tukey_for_vortex(Nwindow, RHO, alpha):
    """
    Compute a radial Tukey window for propagating through a vortex coroangraph.

    Parameters
    ----------
    Nwindow : float, int
        Ask Gary
    RHO : array_like
        Radial coordinates over which to compute a Tukey function
    alpha : float
        Shape parameter of the Tukey window, representing the fraction of the
        window inside the cosine tapered region.

    Returns
    -------
    windowTukey : array_like
        Tukey window of same size as input RHO 

    """
    check.real_scalar(Nwindow, 'Nwindow', TypeError)
    check.real_scalar(alpha, 'alpha', TypeError)

    Nlut = int(10*Nwindow)
    rhos0 = np.linspace(-Nwindow/2, Nwindow/2, Nlut)
    lut = tukey(Nlut, alpha) #,left=0,right=0)
    windowTukey = np.interp(RHO, rhos0, lut)

    return windowTukey
