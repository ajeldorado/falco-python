import pytest
import numpy as np
from math import isclose
# import matplotlib.pyplot as plt

import falco


def test_area_annulus():
    inputs = {"shape": "circle", "pixresFP": 9, "whichSide": "top",
              "rhoInner": 2.5, "rhoOuter": 10.0, "angDeg": 160}
    (swMask, xis, etas) = falco.mask.falco_gen_SW_mask(inputs)
    areaExpected = np.pi*(inputs["rhoOuter"]**2 - inputs["rhoInner"]**2) * \
        inputs["angDeg"]/360 * inputs["pixresFP"]**2
    area = np.sum(swMask)

    assert isclose(area, areaExpected, rel_tol=1e-3)


def test_area_square():
    inputs = {"shape": "square", "pixresFP": 10, "whichSide": "right",
              "rhoInner": 2.5, "rhoOuter": 10.0, "angDeg": 180}
    (swMask, xis, etas) = falco.mask.falco_gen_SW_mask(inputs)
    areaExpected = (4*inputs["rhoOuter"]**2-np.pi*inputs["rhoInner"]**2) * \
        inputs["angDeg"]/360 * inputs["pixresFP"]**2
    area = np.sum(swMask)

    # plt.figure()
    # plt.imshow(swMask)
    # plt.gca().invert_yaxis()
    # plt.pause(0.2)

    assert isclose(area, areaExpected, rel_tol=3e-2)


def test_area_rectangle():
    inputs = {"shape": "rectangle", "pixresFP": 10, "whichSide": "right",
              "rhoInner": 2.5, "rhoOuter": 10.0, "angDeg": 180}
    (swMask, xis, etas) = falco.mask.falco_gen_SW_mask(inputs)
    areaExpected = (2*inputs["rhoOuter"] *
                    (inputs["rhoOuter"]-inputs["rhoInner"]) *
                    inputs["pixresFP"]**2)
    area = np.sum(swMask)

    # plt.figure()
    # plt.imshow(swMask)
    # plt.gca().invert_yaxis()
    # plt.pause(0.2)

    assert isclose(area, areaExpected, rel_tol=3e-2)


@pytest.mark.parametrize("inputs", [
    {"whichSide": "l", "shape": "circle", "clockAngDeg": 10},
    {"whichSide": "r", "shape": "square", "clockAngDeg": 30},
    {"whichSide": "rl", "shape": "rect", "clockAngDeg": 135},
    {"whichSide": "b", "shape": "d", "clockAngDeg": -22.2},
])
def test_translation(inputs):
    inputsFixed = {"pixresFP": 4, "rhoInner": 2.5, "rhoOuter": 10.0,
                   "angDeg": 160}
    inputs.update(inputsFixed)

    (swMaskCentered, xis, etas) = falco.mask.falco_gen_SW_mask(inputs)

    inputs["xiOffset"] = 2
    inputs["etaOffset"] = -3
    (swMaskOffcenter, _, _) = falco.mask.falco_gen_SW_mask(inputs)
    swMaskRecenter = np.roll(swMaskOffcenter,
                             (-round(inputs["etaOffset"]*inputs["pixresFP"]),
                              -round(inputs["xiOffset"]*inputs["pixresFP"])),
                             axis=(0, 1))
    swMaskRecenter = falco.util.pad_crop(swMaskRecenter, swMaskCentered.shape)

    maxAbsDiff = np.max(np.abs(swMaskRecenter.astype(int) -
                               swMaskCentered.astype(int)))
    assert maxAbsDiff < 1e-6


@pytest.mark.parametrize("inputs", [
    {"shape": "circle", "whichSide": "r", "clockAngDeg": 0},
    {"shape": "circle", "whichSide": "right", "clockAngDeg": 0},
    {"shape": "circle", "whichSide": "t", "clockAngDeg": -90},
    {"shape": "circle", "whichSide": "top", "clockAngDeg": -90},
    {"shape": "circle", "whichSide": "u", "clockAngDeg": -90},
    {"shape": "circle", "whichSide": "up", "clockAngDeg": -90},
    {"shape": "circle", "whichSide": "l", "clockAngDeg": 180},
    {"shape": "circle", "whichSide": "left", "clockAngDeg": 180},
    {"shape": "circle", "whichSide": "d", "clockAngDeg": 90},
    {"shape": "circle", "whichSide": "down", "clockAngDeg": 90},
    {"shape": "circle", "whichSide": "b", "clockAngDeg": 90},
    {"shape": "circle", "whichSide": "bottom", "clockAngDeg": 90},
])
def test_degeneracy_of_rotation_and_one_side(inputs):
    inputsFixed = {"pixresFP": 4, "rhoInner": 2.5, "rhoOuter": 10.0,
                   "angDeg": 160}
    inputs.update(inputsFixed)
    (swMask, _, _) = falco.mask.falco_gen_SW_mask(inputs)

    # plt.figure()
    # plt.imshow(swMask)
    # plt.title(inputs["whichSide"])
    # plt.gca().invert_yaxis()
    # plt.pause(0.2)

    # Reference mask
    inputs["whichSide"] = "right"
    inputs["clockAngDeg"] = 0
    (swMaskRef, xis, etas) = falco.mask.falco_gen_SW_mask(inputs)
    sumAbsDiff = np.sum(np.abs(swMaskRef.astype(int) - swMask.astype(int)))
    assert sumAbsDiff == 0


@pytest.mark.parametrize("inputs", [
    {"shape": "circle", "whichSide": "lr", "clockAngDeg": 0},
    {"shape": "circle", "whichSide": "rl", "clockAngDeg": 0},
    {"shape": "circle", "whichSide": "leftright", "clockAngDeg": 0},
    {"shape": "circle", "whichSide": "rightleft", "clockAngDeg": 0},
    {"shape": "circle", "whichSide": "both", "clockAngDeg": 0},
    {"shape": "circle", "whichSide": "ud", "clockAngDeg": 90},
    {"shape": "circle", "whichSide": "du", "clockAngDeg": 90},
    {"shape": "circle", "whichSide": "updown", "clockAngDeg": 90},
    {"shape": "circle", "whichSide": "downup", "clockAngDeg": 90},
    {"shape": "circle", "whichSide": "bt", "clockAngDeg": 90},
    {"shape": "circle", "whichSide": "tb", "clockAngDeg": 90},
    {"shape": "circle", "whichSide": "bottomtop", "clockAngDeg": 90},
    {"shape": "circle", "whichSide": "topbottom", "clockAngDeg": 90},
])
def test_degeneracy_of_rotation_and_two_sides(inputs):
    inputsFixed = {"pixresFP": 4, "rhoInner": 2.5, "rhoOuter": 10.0,
                   "angDeg": 160}
    inputs.update(inputsFixed)
    (swMask, _, _) = falco.mask.falco_gen_SW_mask(inputs)

    # plt.figure()
    # plt.imshow(swMask)
    # plt.title(inputs["whichSide"])
    # plt.gca().invert_yaxis()
    # plt.pause(0.2)

    # Reference mask
    inputs["whichSide"] = "lr"
    inputs["clockAngDeg"] = 0
    (swMaskRef, xis, etas) = falco.mask.falco_gen_SW_mask(inputs)
    sumAbsDiff = np.sum(np.abs(swMaskRef.astype(int) - swMask.astype(int)))
    assert sumAbsDiff == 0


if __name__ == '__main__':
    test_area_annulus()
    test_area_square()
    test_area_rectangle()
