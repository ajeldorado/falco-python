import falco.masks as masks
import numpy as np


def test_annular_fpm():
    """ Quick and dirty spot check of annular_fpm
    This test is very partial at best.
    """

    # test some semi-random cases - is the array size as expected? 
    assert masks.annular_fpm(3, 2, np.inf).shape == (3*2*2, 3*2*2)
    assert masks.annular_fpm(3, 5, np.inf).shape == (3*5*2, 3*5*2)
    assert masks.annular_fpm(3, 5, 10).shape == (3*10*2, 3*10*2)
    assert masks.annular_fpm(3, 5, 11).shape == (3*11*2, 3*11*2)

    # test some pixel values are as expected. 
    mask = masks.annular_fpm(3, 2, 10)
    assert mask[0,0]==0 # corner is black
    assert mask[5*10, 5*10]==1 # in between is white
    assert mask[3*10, 3*10]==0 # center is black
