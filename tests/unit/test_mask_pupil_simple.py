import numpy as np
from math import isclose

import falco


def test_area_circle():
    inputs = {"Nbeam": 100, "Npad": 180, "OD": 1.0}
    pupil = falco.mask.falco_gen_pupil_Simple(inputs)
    areaExpected = np.pi/4*(inputs["OD"]*inputs["Nbeam"])**2
    area = np.sum(pupil)
    assert isclose(area, areaExpected, rel_tol=1e-5)


def test_area_annulus():
    inputs = {"Nbeam": 100, "Npad": 180, "OD": 1.0, "ID": 0.20}
    pupil = falco.mask.falco_gen_pupil_Simple(inputs)
    areaExpected = np.pi/4*(inputs["OD"]**2 -
                            inputs["ID"]**2)*inputs["Nbeam"]**2
    area = np.sum(pupil)
    assert isclose(area, areaExpected, rel_tol=1e-5)


def test_translation():
    inputs = {"Nbeam": 100, "Npad": 180, "OD": 1.0, "ID": 0.20,
              "wStrut": 0.02, "angStrut": (10 + np.array([90, 180, 270, 315]))}
    pupilCentered = falco.mask.falco_gen_pupil_Simple(inputs)

    inputs["xShear"] = -11/inputs["Nbeam"]
    inputs["yShear"] = 19/inputs["Nbeam"]
    pupilOffcenter = falco.mask.falco_gen_pupil_Simple(inputs)
    pupilRecentered = np.roll(pupilOffcenter,
                              (round(-inputs["Nbeam"]*inputs["yShear"]),
                               round(-inputs["Nbeam"]*inputs["xShear"])),
                              axis=(0, 1))

    maxAbsDiff = np.max(np.abs(pupilRecentered - pupilCentered))
    assert maxAbsDiff < 1e-6


def test_translation_and_rotation():
    inputs = {"Nbeam": 100, "Npad": 180, "OD": 1.0, "ID": 0.20,
              "wStrut": 0.02, "angStrut": (10 + np.array([90, 180, 270, 315]))}
    pupilCentered = falco.mask.falco_gen_pupil_Simple(inputs)
    pupilCentered = pupilCentered[1:, 1:]

    inputs["xShear"] = -11/inputs["Nbeam"]
    inputs["yShear"] = 19/inputs["Nbeam"]
    inputs["clocking"] = 90
    pupilOffcenterRot = falco.mask.falco_gen_pupil_Simple(inputs)
    pupilOffcenterRot = pupilOffcenterRot[1:, 1:]
    pupilRecentered = np.rot90(np.roll(pupilOffcenterRot,
                               (round(-inputs["Nbeam"]*inputs["yShear"]),
                                round(-inputs["Nbeam"]*inputs["xShear"])),
                               axis=(0, 1)), 1)

    maxAbsDiff = np.max(np.abs(pupilRecentered - pupilCentered))
    assert maxAbsDiff < 1e-2


if __name__ == '__main__':
    test_area_circle()
    test_area_annulus()
    test_translation()
    test_translation_and_rotation()
