import numpy as np

from falco.mask import falco_gen_pupil_LUVOIR_B
from falco.util import pad_crop


def test_translation():
    Nbeam = 200
    inputs = {"Nbeam": Nbeam}
    pupil = falco_gen_pupil_LUVOIR_B(inputs)

    inputs.update({"xShear": -11/100, "yShear": 19/100})
    pupilOffset = falco_gen_pupil_LUVOIR_B(inputs)
    pupilRecentered = np.roll(pupilOffset,
                              [int(-Nbeam*inputs["yShear"]),
                               int(-Nbeam*inputs["xShear"])],
                              axis=(0, 1))
    diff = pad_crop(pupil, pupilOffset.shape) - pupilRecentered
    assert np.max(np.abs(diff)) <= 1e-8


def test_translation_and_rotation():
    Nbeam = 200
    inputs = {"Nbeam": Nbeam}
    pupil = falco_gen_pupil_LUVOIR_B(inputs)
    pupilRot = np.zeros_like(pupil)
    pupilRot[1::, 1::] = np.rot90(pupil[1::, 1::], -1)

    inputs.update({"xShear": -11/100, "yShear": 19/100, "clock_deg": 90})
    pupilRotOffset = falco_gen_pupil_LUVOIR_B(inputs)
    pupilRotRecentered = np.roll(pupilRotOffset,
                                 [int(-Nbeam*inputs["yShear"]),
                                  int(-Nbeam*inputs["xShear"])], axis=(0, 1))

    diff = pad_crop(pupilRot, pupilRotOffset.shape) - pupilRotRecentered
    assert np.max(np.abs(diff)) < 1/33


if __name__ == '__main__':
    test_translation()
    test_translation_and_rotation()
