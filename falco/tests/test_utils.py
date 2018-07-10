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


def test_allcomb():
    def assert_lists_equal(first, second):
        assert all([a == b for a, b in zip(first, second)])

    # Test cases adapted from documentation for MATLAB version
    assert_lists_equal(utils.allcomb([1, 3, 5], [-3, 8], [0, 1]),
                       [(1, -3, 0), (1, -3, 1), (1, 8, 0), (1, 8, 1), (3, -3, 0), (3, -3, 1), (3, 8, 0),
                        (3, 8, 1), (5, -3, 0), (5, -3, 1), (5, 8, 0), (5, 8, 1)])

    assert_lists_equal(utils.allcomb('abc','XY'),
                       [('a', 'X'), ('a', 'Y'), ('b', 'X'), ('b', 'Y'), ('c', 'X'), ('c', 'Y')])

    assert_lists_equal(utils.allcomb('abc', [-3, 8]),
                       [('a', -3), ('a', 8), ('b', -3), ('b', 8), ('c', -3), ('c', 8)])
