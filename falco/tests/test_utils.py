import falco.utils as utils
import numpy as np

def test_ceil_even():
    assert utils.ceil_even(1)==2
    assert utils.ceil_even(1.5)==2
    assert utils.ceil_even(4)==4
    assert utils.ceil_even(3.14159)==4
    assert utils.ceil_even(2001)==2002

def test_ceil_odd():
    assert utils.ceil_odd(1)==1
    assert utils.ceil_odd(1.5)==3
    assert utils.ceil_odd(4)==5
    assert utils.ceil_odd(3.14159)==5
    assert utils.ceil_odd(2001)==2001

def test_padOrCropEven():
    # Test case parameters
    sizes = [2, 10, 98]
    pads = [2, 6, 8, 10, 40, 98, 256]

    for size in sizes:
        for pad in pads:
            input = np.ones((size, size), dtype=np.complex128)
            output = utils.padOrCropEven(input, pad)

            assert output.shape == (pad, pad)  # Check correct shape

            if pad > size:  # If padding, check that output is padded equally on all sides
                assert np.allclose(output, output[::-1, ::-1])
            elif pad == size:  # If input and output are same size, make sure input is unmodified
                assert np.allclose(input, output)

