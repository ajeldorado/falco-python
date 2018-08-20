import numpy as np


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
