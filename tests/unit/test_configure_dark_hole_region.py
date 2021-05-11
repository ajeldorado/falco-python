"""Unit test suite for falco.setup.falco_configure_dark_hole_region()"""
import numpy as np
from math import isclose
import copy

import falco


def test_single_region():

    mp = falco.config.ModelParameters()
    mp.Fend = falco.config.Object()
    mp.Fend.corr = falco.config.Object()
    mp.Fend.score = falco.config.Object()
    mp.Fend.eval = falco.config.Object()

    # Correction and scoring region definition
    mp.Fend.corr.Rin = 2  # inner radius [lambda0/D]
    mp.Fend.corr.Rout = 10  # outer radius [lambda0/D]
    mp.Fend.corr.ang = 180  # angular opening per side [degrees]
    mp.Fend.score = copy.deepcopy(mp.Fend.corr)

    mp.centering = 'pixel'
    mp.Fend.sides = 'lr'
    mp.Fend.shape = 'circle'
    mp.Fend.res = 10

    mp.Fend.eval.res = 20
    mp.thput_eval_x = 7
    mp.thput_eval_y = 0

    falco.setup.falco_configure_dark_hole_region(mp)

    area = np.sum(mp.Fend.corr.maskBool.astype(int))

    areaExpected = (np.pi * (mp.Fend.corr.Rout**2 - mp.Fend.corr.Rin**2) *
                    (2*mp.Fend.corr.ang/360) * mp.Fend.res**2)

    assert isclose(area, areaExpected, rel_tol=1e-3)


def test_double_region():

    mp = falco.config.ModelParameters()
    mp.Fend = falco.config.Object()
    mp.Fend.corr = falco.config.Object()
    mp.Fend.score = falco.config.Object()
    mp.Fend.eval = falco.config.Object()

    # Correction and scoring region definition
    mp.Fend.corr.Rin = [2, 2]  # inner radius [lambda0/D]
    mp.Fend.corr.Rout = [5, 5]  # outer radius [lambda0/D]
    mp.Fend.corr.ang = [150, 180]  # angular opening per side [degrees]
    mp.Fend.score = copy.deepcopy(mp.Fend.corr)

    mp.centering = 'pixel'
    mp.Fend.sides = ['lr', 'lr']
    mp.Fend.shape = ['circle', 'square']
    mp.Fend.res = 10
    mp.Fend.xiOffset = [0, 20]

    mp.Fend.eval.res = 20
    mp.thput_eval_x = 7
    mp.thput_eval_y = 0

    falco.setup.falco_configure_dark_hole_region(mp)

    area = np.sum(mp.Fend.corr.maskBool.astype(int))

    areaExpected = (np.pi * (mp.Fend.corr.Rout**2 - mp.Fend.corr.Rin**2) *
                    (2*mp.Fend.corr.ang/360) * mp.Fend.res**2)

    areaExpected = (np.pi*(mp.Fend.corr.Rout[0]**2 - mp.Fend.corr.Rin[0]**2) *
                    (2*mp.Fend.corr.ang[0]/360)*(mp.Fend.res**2) +
                    (4*mp.Fend.corr.Rout[1]**2 -
                     np.pi*mp.Fend.corr.Rin[1]**2)*(mp.Fend.res**2))

    assert isclose(area, areaExpected, rel_tol=2e-2)


if __name__ == '__main__':
    test_single_region()
    test_double_region()
