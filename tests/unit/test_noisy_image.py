import numpy as np
from math import isclose

import falco
from falco.imaging import add_noise_to_subband_image


def test_photon_shot_noise():

    trueMean = 1e-4  # [normalized intensity]
    imageIn = trueMean * np.ones((1000, 1000))
    peakFluxCoef = 1e10  # [counts/pixel/second]
    readNoiseStd = 0
    tExp = 38
    gain = 4.2
    darkCurrentRate = 0  # [e-/pixel/second]
    Nexp = 11
    shotNoiseStd = np.sqrt(gain*tExp*trueMean*peakFluxCoef/Nexp)

    mp = falco.config.ModelParameters()
    mp.detector = falco.config.Object()
    mp.Nsbp = 3
    iSubband = 2
    mp.detector.gain = gain  # [e-/count]
    mp.detector.darkCurrentRate = darkCurrentRate  # [e-/pixel/second]
    mp.detector.readNoiseStd = readNoiseStd  # [e-/count]
    mp.detector.peakFluxVec = (peakFluxCoef *
                               np.ones(mp.Nsbp))  # [counts/pixel/second]
    mp.detector.tExpVec = tExp * np.ones(mp.Nsbp)  # [seconds]
    mp.detector.Nexp = Nexp

    imageOut = add_noise_to_subband_image(mp, imageIn, iSubband)

    imageStd = np.std(imageOut - trueMean)*mp.detector.gain*peakFluxCoef*tExp
    print('Expected photon shot noise = %.5e' % shotNoiseStd)
    print('Meas photon shot noise     = %.5e' % imageStd)

    assert isclose(imageStd, shotNoiseStd, rel_tol=1e-2)


def test_that_mean_stays_same():
    trueMean = 1.123e-3  # [normalized intensity]
    imageIn = trueMean * np.ones((1000, 1000))

    mp = falco.config.ModelParameters()
    mp.detector = falco.config.Object()
    mp.Nsbp = 3
    iSubband = 2
    mp.detector.gain = 4  # [e-/count]
    mp.detector.darkCurrentRate = 0.1  # [e-/pixel/second]
    mp.detector.readNoiseStd = 5  # [e-/count]
    mp.detector.peakFluxVec = 1e10 * np.ones(mp.Nsbp)  # [counts/pixel/second]
    mp.detector.tExpVec = 10.0 * np.ones(mp.Nsbp)  # [seconds]
    mp.detector.Nexp = 3

    imageOut = add_noise_to_subband_image(mp, imageIn, iSubband)
    imageMean = np.mean(imageOut)

    print('Mean before noise = %.5e' % trueMean)
    print('Mean after noise  = %.5e' % imageMean)
    assert isclose(imageMean, trueMean, rel_tol=1e-3)


def test_read_noise_std():
    trueMean = 0  # [normalized intensity]
    imageIn = trueMean * np.ones((1000, 1000))
    peakFluxCoef = 0.3  # [counts/pixel/second]
    readNoiseStdPerFrame = 55
    tExp = 0.1
    mp = falco.config.ModelParameters()
    mp.detector = falco.config.Object()
    mp.detector.Nexp = 7

    readNoiseStd = readNoiseStdPerFrame / np.sqrt(mp.detector.Nexp)

    mp.Nsbp = 3
    iSubband = 2
    mp.detector.gain = 4  # [e-/count]
    mp.detector.darkCurrentRate = 0  # [e-/pixel/second]
    mp.detector.readNoiseStd = readNoiseStdPerFrame  # [e-/count]
    mp.detector.peakFluxVec = (peakFluxCoef *
                               np.ones(mp.Nsbp))  # [counts/pixel/second]
    mp.detector.tExpVec = tExp * np.ones(mp.Nsbp)  # [seconds]

    imageOut = add_noise_to_subband_image(mp, imageIn, iSubband)

    imageStd = np.std(imageOut - trueMean)*mp.detector.gain*peakFluxCoef*tExp
    print('True read noise std = %.5e' % readNoiseStd)
    print('Meas read noise std = %.5e' % imageStd)
    assert isclose(imageStd, readNoiseStd, rel_tol=1e-2)


def test_dark_current():
    trueMean = 0  # [normalized intensity]
    imageIn = trueMean * np.ones((1000, 1000))
    peakFluxCoef = 1e-3  # [counts/pixel/second]
    readNoiseStd = 0
    tExp = 1e4
    darkCurrentRate = 5.3  # [e-/pixel/second]
    mp = falco.config.ModelParameters()
    mp.detector = falco.config.Object()
    mp.detector.Nexp = 3
    darkCurrentStd = np.sqrt(tExp * darkCurrentRate / mp.detector.Nexp)

    mp.Nsbp = 3
    iSubband = 2
    mp.detector.gain = 4  # [e-/count]
    mp.detector.darkCurrentRate = darkCurrentRate  # [e-/pixel/second]
    mp.detector.readNoiseStd = readNoiseStd  # [e-/count]
    mp.detector.peakFluxVec = (peakFluxCoef *
                               np.ones(mp.Nsbp))  # [counts/pixel/second]
    mp.detector.tExpVec = tExp * np.ones(mp.Nsbp)  # [seconds]

    imageOut = add_noise_to_subband_image(mp, imageIn, iSubband)

    imageStd = np.std(imageOut - trueMean)*mp.detector.gain*peakFluxCoef*tExp
    print('Expected dark current noise = %.5e' % darkCurrentStd)
    print('Meas dark current noise     = %.5e' % imageStd)
    assert isclose(imageStd, darkCurrentStd, rel_tol=1e-2)


if __name__ == '__main__':
    test_photon_shot_noise()
    test_that_mean_stays_same()
    test_read_noise_std()
    test_dark_current()
