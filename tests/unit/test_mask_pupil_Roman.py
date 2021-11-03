import os
import numpy as np
from matplotlib.image import imread

from falco.mask import falco_gen_pupil_Roman_CGI_20200513
from falco.util import pad_crop, bin_downsample


def test_translation():
    Nbeam = 100
    centering = 'pixel'
    pupil = falco_gen_pupil_Roman_CGI_20200513(Nbeam, centering)

    changes = {"xShear": -11/100, "yShear": 19/100}
    pupilOffset = falco_gen_pupil_Roman_CGI_20200513(Nbeam, centering, changes)
    pupilRecentered = np.roll(pupilOffset,
                              [int(-Nbeam*changes["yShear"]),
                               int(-Nbeam*changes["xShear"])],
                              axis=(0, 1))
    diff = pad_crop(pupil, pupilOffset.shape) - pupilRecentered

    assert np.sum(np.abs(diff)) < 1e-8


def test_translation_and_rotation():

    Nbeam = 100
    centering = 'pixel'
    pupil = falco_gen_pupil_Roman_CGI_20200513(Nbeam, centering)

    # Translation test
    changes = {"xShear": -11/100, "yShear": 19/100}
    pupilOffset = falco_gen_pupil_Roman_CGI_20200513(Nbeam, centering, changes)

    # Test rotation (and translation)
    changes["clock_deg"] = 90
    pupilRotOffset = falco_gen_pupil_Roman_CGI_20200513(Nbeam,
                                                        centering,
                                                        changes)
    pupilRotRecentered = np.roll(pupilRotOffset,
                                 [int(-Nbeam*changes["yShear"]),
                                  int(-Nbeam*changes["xShear"])],
                                 axis=(0, 1))

    pupilRot = np.zeros_like(pupil)
    pupilRot[1::, 1::] = np.rot90(pupil[1::, 1::], -1)

    diff = pad_crop(pupilRot, pupilOffset.shape) - pupilRotRecentered

    assert np.max(np.abs(diff)) < 1e-4


def test_roman_pupil_against_file():

    localpath = os.path.dirname(os.path.abspath(__file__))
    fn = os.path.join(localpath, 'testdata',
                      'pupil_CGI-20200513_8k_binary_noF.png')
    pupil0 = imread(fn).astype(float)

    pupilFromFile = pupil0/np.max(pupil0)
    Narray = pupilFromFile.shape[1]
    downSampFac = 16
    Nbeam = 2*4027.25
    NbeamDS = Nbeam/downSampFac
    NarrayDS = int(Narray/downSampFac)
    pupilFromFileDS = bin_downsample(pupilFromFile, downSampFac)

    # Generate pupil representation in FALCO
    shift = downSampFac/2 - 0.5
    changes = {"xShear": -0.5/Nbeam - shift/Nbeam,
               "yShear": -52.85/Nbeam - shift/Nbeam}
    pupilFromFALCODS = falco_gen_pupil_Roman_CGI_20200513(NbeamDS,
                                                          'pixel',
                                                          changes)
    pupilFromFALCODS = pad_crop(pupilFromFALCODS, NarrayDS)
    diff = pupilFromFileDS - pupilFromFALCODS

    percentvalue = np.sum(np.abs(diff))/np.sum(pupilFromFileDS)*100
    assert percentvalue < 0.1


if __name__ == '__main__':
    test_translation()
    test_translation_and_rotation()
    test_roman_pupil_against_file()
