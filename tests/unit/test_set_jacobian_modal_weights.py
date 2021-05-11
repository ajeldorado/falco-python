"""Unit test suite for falco.setup.falco_set_jacobian_modal_weights()"""
import numpy as np

import falco


def test_errors_a():
    pass


def test_errors_b():
    pass


def test_weights_a():
    mp = falco.config.ModelParameters()
    mp.jac = falco.config.Object()
    mp.compact = falco.config.Object()
    mp.compact.star = falco.config.Object()

    mp.estimator = 'perfect'
    mp.jac.zerns = 1
    mp.jac.Zcoef = 1e-9
    mp.Nsbp = 1
    mp.si_ref = np.floor(mp.Nsbp/2).astype(int)
    mp.compact.star.count = 1
    falco.setup.falco_set_jacobian_modal_weights(mp)

    print(mp.jac.zern_inds)

    assert mp.jac.Nzern == 1
    assert mp.jac.Nmode == 1
    assert np.allclose(mp.jac.zerns, np.array([1, ]))
    assert np.allclose(mp.jac.Zcoef, np.array([1, ]))
    assert np.allclose(mp.jac.weights, np.array([1, ]))
    assert np.allclose(mp.jac.weights, np.ones((1, 1)))
    assert np.allclose(mp.jac.sbp_inds, [0])
    assert np.allclose(mp.jac.zern_inds, [1])


def test_weights_b():
    mp = falco.config.ModelParameters()
    mp.jac = falco.config.Object()
    mp.compact = falco.config.Object()
    mp.compact.star = falco.config.Object()

    mp.estimator = 'perfect'
    mp.jac.zerns = [1, 5, 6]
    mp.jac.Zcoef = 1e-9*np.ones((3, ))
    mp.Nsbp = 3
    mp.si_ref = np.floor(mp.Nsbp/2).astype(int)
    mp.compact.star.count = 1
    falco.setup.falco_set_jacobian_modal_weights(mp)

    assert mp.jac.Nzern == 3
    assert mp.jac.Nmode == 9
    assert np.allclose(mp.jac.zerns, [1, 5, 6])
    assert np.allclose(mp.jac.Zcoef, [1, 1e-9, 1e-9])
    assert np.allclose(mp.jac.weights, [0.25, 0.25, 0.25, 0.5, 0.5, 0.5,
                                        0.25, 0.25, 0.25])
    assert np.allclose(mp.jac.weightMat, [[0.25, 0.25, 0.25],
                                          [0.5, 0.5, 0.5],
                                          [0.25, 0.25, 0.25]])
    assert np.allclose(mp.jac.sbp_inds, [0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert np.allclose(mp.jac.zern_inds, [1, 5, 6, 1, 5, 6, 1, 5, 6])


def test_weights_c():
    mp = falco.config.ModelParameters()
    mp.jac = falco.config.Object()
    mp.compact = falco.config.Object()
    mp.compact.star = falco.config.Object()

    mp.estimator = 'pwp-bp'
    mp.jac.zerns = [1, 5, 6]
    mp.jac.Zcoef = 1e-9*np.ones((3, ))
    mp.Nsbp = 3
    mp.si_ref = np.floor(mp.Nsbp/2).astype(int)
    mp.compact.star.count = 1
    falco.setup.falco_set_jacobian_modal_weights(mp)

    assert mp.jac.Nzern == 3
    assert mp.jac.Nmode == 9
    assert np.allclose(mp.jac.zerns, [1, 5, 6])
    assert np.allclose(mp.jac.Zcoef, [1, 1e-9, 1e-9])
    assert np.allclose(mp.jac.weights, 1/3*np.ones(9),
                       atol=np.finfo(float).eps)
    assert np.allclose(mp.jac.weightMat, 1/3*np.ones((3, 3)),
                       atol=np.finfo(float).eps)
    assert np.allclose(mp.jac.sbp_inds, [0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert np.allclose(mp.jac.zern_inds, [1, 5, 6, 1, 5, 6, 1, 5, 6])


if __name__ == '__main__':
    test_weights_a()
    test_weights_b()
    test_weights_c()
