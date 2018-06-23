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


