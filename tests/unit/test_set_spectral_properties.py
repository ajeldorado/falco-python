"""Unit test suite for falco.setup.falco_set_spectral_properties()"""
import numpy as np

import falco


def test_1subband_1wavelength():

    mp = falco.config.ModelParameters()
    mp.full = falco.config.Object()
    mp.lambda0 = 1e-6
    mp.flagSim = False
    mp.Nsbp = 1
    mp.Nwpsbp = 1
    mp.fracBW = 0.20
    falco.setup.falco_set_spectral_properties(mp)

    assert np.array_equal(mp.sbp_weights, np.ones((1, 1)))
    assert np.array_equal(mp.sbp_centers, np.array((1e-6, )))
    assert np.array_equal(mp.full.lambda_weights_all, np.array((1, )))
    assert np.array_equal(mp.full.lambdas, np.array((1e-6, )))


def test_1subband_3wavelengths():

    mp = falco.config.ModelParameters()
    mp.full = falco.config.Object()
    mp.lambda0 = 1e-6
    mp.flagSim = False
    mp.Nsbp = 1
    mp.Nwpsbp = 3
    mp.fracBW = 0.20
    falco.setup.falco_set_spectral_properties(mp)

    assert np.array_equal(mp.sbp_weights, np.ones((1, 1)))
    assert np.array_equal(mp.sbp_centers, np.array((1e-6, )))
    assert np.allclose(mp.full.lambda_weights_all, np.array([0.25, 0.5, 0.25]),
                       atol=np.finfo(float).eps)
    assert np.allclose(mp.full.lambdas, 1e-6*np.linspace(0.9, 1.1, 3),
                       atol=np.finfo(float).eps)


def test_5subbands_1wavelength_sim():

    mp = falco.config.ModelParameters()
    mp.full = falco.config.Object()
    mp.lambda0 = 1e-6
    mp.flagSim = True
    mp.Nsbp = 5
    mp.Nwpsbp = 1
    mp.fracBW = 0.20
    falco.setup.falco_set_spectral_properties(mp)

    sbp_weights = np.array([0.125, 0.25, 0.25, 0.25, 0.125]).reshape((5, 1))
    assert np.allclose(mp.sbp_weights,
                       sbp_weights,
                       atol=np.finfo(float).eps)
    assert np.allclose(mp.sbp_centers,
                       np.linspace(0.9, 1.1, 5)*1e-6,
                      atol=np.finfo(float).eps)
    assert np.allclose(mp.full.lambda_weights_all,
                       np.array([0.1250, 0.25, 0.25, 0.25, 0.125]),
                       atol=np.finfo(float).eps)
    assert np.allclose(mp.full.lambdas,
                       np.linspace(0.9, 1.1, 5)*1e-6,
                       atol=np.finfo(float).eps)


def test_5subbands_1wavelength_lab():

    mp = falco.config.ModelParameters()
    mp.full = falco.config.Object()
    mp.lambda0 = 1e-6
    mp.flagSim = False
    mp.Nsbp = 5
    mp.Nwpsbp = 1
    mp.fracBW = 0.20
    falco.setup.falco_set_spectral_properties(mp)

    assert np.allclose(mp.sbp_weights,
                       0.2*np.ones((5, 1)),
                       atol=np.finfo(float).eps)
    assert np.allclose(mp.sbp_centers,
                       np.linspace(0.92, 1.08, 5)*1e-6,
                      atol=np.finfo(float).eps)
    assert np.allclose(mp.full.lambda_weights_all,
                       0.2*np.ones((5, )),
                       atol=np.finfo(float).eps)
    assert np.allclose(mp.full.lambdas,
                       np.linspace(0.92, 1.08, 5)*1e-6,
                       atol=np.finfo(float).eps)


def test_5subbands_3wavelengths_sim():

    mp = falco.config.ModelParameters()
    mp.full = falco.config.Object()
    mp.lambda0 = 1e-6
    mp.flagSim = True
    mp.Nsbp = 5
    mp.Nwpsbp = 3
    mp.fracBW = 0.20
    falco.setup.falco_set_spectral_properties(mp)

    assert np.allclose(mp.sbp_weights,
                       0.2*np.ones((5, 1)),
                       atol=np.finfo(float).eps)
    assert np.allclose(mp.sbp_centers,
                       np.linspace(0.92, 1.08, 5)*1e-6,
                      atol=np.finfo(float).eps)
    lambda_weights_all = 0.1*np.ones((11, ))
    lambda_weights_all[0] = 0.05
    lambda_weights_all[-1] = 0.05
    assert np.allclose(mp.full.lambda_weights_all,
                       lambda_weights_all,
                       atol=np.finfo(float).eps)
    assert np.allclose(mp.full.lambdas,
                       np.linspace(0.9, 1.1, 11)*1e-6,
                       atol=np.finfo(float).eps)


if __name__ == '__main__':
    test_1subband_1wavelength()
    test_1subband_3wavelengths()
    test_5subbands_1wavelength_sim()
    test_5subbands_1wavelength_lab()
