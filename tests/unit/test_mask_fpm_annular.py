"""Unit test suite for falco.mask.gen_annular_fpm()."""
import numpy as np
from math import isclose

from falco.mask import gen_annular_fpm
from falco.util import pad_crop


def test_area_spot():
    inputs = {"pixresFPM": 6,
              "rhoInner": 3,
              "rhoOuter": np.Inf,
              "centering": "pixel",
              }
    fpm = gen_annular_fpm(inputs)

    area = np.sum(1 - fpm)
    areaExpected = np.pi * inputs["rhoInner"]**2 * inputs["pixresFPM"]**2
    assert isclose(area, areaExpected, rel_tol=1e-3)


def test_area_annulus():
    inputs = {"pixresFPM": 6,
              "rhoInner": 3,
              "rhoOuter": 10,
              "centering": "pixel",
              }
    fpm = gen_annular_fpm(inputs)

    area = np.sum(fpm)
    areaExpected = (np.pi * inputs["pixresFPM"]**2 *
                    (inputs["rhoOuter"]**2 - inputs["rhoInner"]**2))
    assert isclose(area, areaExpected, rel_tol=1e-3)


def test_translation_spot():
    res = 6
    inputs = {"pixresFPM": res,
              "rhoInner": 3,
              "rhoOuter": np.Inf,
              "centering": "pixel",
              }
    fpm = gen_annular_fpm(inputs)

    inputs.update({"xOffset": 5.5, "yOffset": -10})
    fpmOffset = gen_annular_fpm(inputs)

    fpmRecentered = np.roll(fpmOffset,
                            [int(-res*inputs["yOffset"]),
                             int(-res*inputs["xOffset"])],
                            axis=(0, 1))
    fpmPad = pad_crop(fpm, fpmRecentered.shape, extrapval=1)
    assert np.allclose(fpmPad, fpmRecentered, atol=1e-7)


def test_translation_annulus():
    res = 6
    inputs = {"pixresFPM": res,
              "rhoInner": 3,
              "rhoOuter": 10,
              "centering": "pixel",
              }
    fpm = gen_annular_fpm(inputs)

    inputs.update({"xOffset": 5.5, "yOffset": -10})
    fpmOffset = gen_annular_fpm(inputs)

    fpmRecentered = np.roll(fpmOffset,
                            [int(-res*inputs["yOffset"]),
                             int(-res*inputs["xOffset"])],
                            axis=(0, 1))
    fpmPad = pad_crop(fpm, fpmRecentered.shape)
    assert np.allclose(fpmPad, fpmRecentered, atol=1e-7)


if __name__ == '__main__':
    test_area_spot()
    test_area_annulus()
    test_translation_spot()
    test_translation_annulus()
