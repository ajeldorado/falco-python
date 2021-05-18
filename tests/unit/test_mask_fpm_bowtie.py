"""Unit test suite for falco.mask.gen_bowtie_fpm()."""
import numpy as np
from math import isclose

from falco.mask import gen_bowtie_fpm
from falco.util import pad_crop


def test_area():
    inputs = {"pixresFPM": 6,
              "rhoInner": 2.6,
              "rhoOuter": 9.4,
              "ang": 65,
              }
    fpm = gen_bowtie_fpm(inputs)

    area = np.sum(fpm)
    areaExpected = (np.pi * inputs["pixresFPM"]**2 * inputs["ang"]/180 *
                    (inputs["rhoOuter"]**2 - inputs["rhoInner"]**2))
    assert isclose(area, areaExpected, rel_tol=1e-3)


def test_translation():
    res = 6
    inputs = {"pixresFPM": res,
              "rhoInner": 2.6,
              "rhoOuter": 9.4,
              "ang": 65,
              "clocking": 20,
              }
    fpm = gen_bowtie_fpm(inputs)

    inputs.update({"xOffset": 5.5, "yOffset": -10})
    fpmOffset = gen_bowtie_fpm(inputs)

    fpmRecentered = np.roll(fpmOffset,
                            [int(-res*inputs["yOffset"]),
                             int(-res*inputs["xOffset"])],
                            axis=(0, 1))
    fpmPad = pad_crop(fpm, fpmRecentered.shape)
    assert np.allclose(fpmPad, fpmRecentered, atol=1e-7)


def test_rotation():
    res = 6
    inputs = {"pixresFPM": res,
              "rhoInner": 2.6,
              "rhoOuter": 9.4,
              "ang": 65,
              }
    fpm = gen_bowtie_fpm(inputs)
    fpmRot = np.zeros_like(fpm)
    fpmRot[1::, 1::] = np.rot90(fpm[1::, 1::], -1)

    inputs.update({"xOffset": 5.5, "yOffset": -10, "clocking": 90})
    fpmRotOffset = gen_bowtie_fpm(inputs)

    fpmRotRecentered = np.roll(fpmRotOffset,
                               [int(-res*inputs["yOffset"]),
                                int(-res*inputs["xOffset"])],
                               axis=(0, 1))
    fpmRotPad = pad_crop(fpmRot, fpmRotRecentered.shape)
    assert np.allclose(fpmRotPad, fpmRotRecentered, atol=1e-4)


if __name__ == '__main__':
    test_area()
    test_translation()
    test_rotation()
