import numpy as np
import logging
from falco import utils

log = logging.getLogger(__name__)


def propcustom_2FT(E_in, centering='pixel'):
    """
    Propagate a field using two successive Fourier transforms, without any intermediate mask
    multiplications.   Used in a semi-analytical propagation to compute the component of the field
    in the Lyot stop plane that was not diffracted by the occulter.

    Parameters
    ----------
    E_in : array_like
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
    _valid_centering = ['pixel', 'interpixel']

    if centering not in _valid_centering:
        raise ValueError('propcustom_2FT: invalid centering specification.' +
                         'Options: {}'.format(_valid_centering))

    E_out = (1 / 1j) ** 2 * E_in[::-1, ::-1]  # Reverse and scale input to account for propagation

    if centering == 'pixel':
        E_out = np.roll(E_in, (1, 1), axis=(0, 1))  # Move the DC pixel back to the right place

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
        raise ValueError('propcustom_PTP: input array is not square')

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
