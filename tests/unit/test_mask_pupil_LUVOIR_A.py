import numpy as np

from falco.mask import falco_gen_pupil_LUVOIR_A_final
from falco.util import pad_crop


def test_translation():
    Nbeam = 200
    inputs = {"Nbeam": Nbeam}
    pupil = falco_gen_pupil_LUVOIR_A_final(inputs)

    inputs.update({"xShear": -11/100, "yShear": 19/100})
    pupilOffset = falco_gen_pupil_LUVOIR_A_final(inputs)
    pupilRecentered = np.roll(pupilOffset,
                              [int(-Nbeam*inputs["yShear"]),
                               int(-Nbeam*inputs["xShear"])],
                              axis=(0, 1))
    diff = pad_crop(pupil, pupilOffset.shape) - pupilRecentered
    assert np.max(np.abs(diff)) <= 1/33


def test_translation_and_rotation():
    Nbeam = 200
    inputs = {"Nbeam": Nbeam}
    pupil = falco_gen_pupil_LUVOIR_A_final(inputs)
    pupilRot = np.zeros_like(pupil)
    pupilRot[1::, 1::] = np.rot90(pupil[1::, 1::], -1)

    inputs.update({"xShear": -11/100, "yShear": 19/100, "clock_deg": 90})
    pupilRotOffset = falco_gen_pupil_LUVOIR_A_final(inputs)
    pupilRotRecentered = np.roll(pupilRotOffset,
                                 [int(-Nbeam*inputs["yShear"]),
                                  int(-Nbeam*inputs["xShear"])], axis=(0, 1))

    diff = pad_crop(pupilRot, pupilRotOffset.shape) - pupilRotRecentered
    assert np.max(np.abs(diff)) < 1/33


def test_lyot_stop_translation():
    Nbeam = 200
    inputs = {"Nbeam": Nbeam, "flagLyot": True, "ID": 0.30, "OD": 0.80}
    pupil = falco_gen_pupil_LUVOIR_A_final(inputs)

    # Translation test
    inputs.update({"xShear": -11/100, "yShear": 19/100})
    pupilOffset = falco_gen_pupil_LUVOIR_A_final(inputs)
    pupilRecentered = np.roll(pupilOffset,
                              [int(-Nbeam*inputs["yShear"]),
                               int(-Nbeam*inputs["xShear"])],
                              axis=(0, 1))
    diff = pad_crop(pupil, pupilOffset.shape) - pupilRecentered
    assert np.sum(np.abs(diff)) < 1e-8


def test_lyot_stop_translation_and_rotation():
    Nbeam = 200
    inputs = {"Nbeam": Nbeam, "flagLyot": True, "ID": 0.30, "OD": 0.80}
    pupil = falco_gen_pupil_LUVOIR_A_final(inputs)
    pupilRot = np.zeros_like(pupil)
    pupilRot[1::, 1::] = np.rot90(pupil[1::, 1::], -1)

    # Translation test
    inputs.update({"xShear": -11/100, "yShear": 19/100, "clock_deg": 90})
    pupilRotOffset = falco_gen_pupil_LUVOIR_A_final(inputs)
    pupilRotRecentered = np.roll(pupilRotOffset,
                                 [int(-Nbeam*inputs["yShear"]),
                                  int(-Nbeam*inputs["xShear"])],
                                 axis=(0, 1))
    diff = pad_crop(pupilRot, pupilRotOffset.shape) - pupilRotRecentered
    assert np.max(np.abs(diff)) < 1/33


if __name__ == '__main__':
    test_translation()
    test_translation_and_rotation()
    test_lyot_stop_translation()
    test_lyot_stop_translation_and_rotation()
