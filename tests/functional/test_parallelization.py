"""FALCO regression test of WFSC with a vortex coronagraph."""
from copy import deepcopy
import numpy as np
import os
# from math import isclose
# import matplotlib.pyplot as plt

import falco

import config_wfsc_vc as CONFIG


# plt.figure(1)
# plt.title('Serial')
# plt.imshow(np.log10(imageSerial))
# plt.gca().invert_yaxis()
# plt.colorbar()
# plt.clim(-10, -5)
# plt.pause(0.1)

#     # assert np.allclose(out.log10regHist, np.array([-4.5, -4, -4]), rtol=1e-2)

def test_parallel_images():

    # del mp
    mp = deepcopy(CONFIG.mp)
    mp.fracBW = 0.20
    mp.Nsbp = 4

    # mp.P1.compact.E = np.ones((256, 256, mp.Nsbp))
    # mp.P1.full.E = np.ones((256, 256, mp.Nwpsbp, mp.Nsbp))

    mp.flagPlot = False
    LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
    mp.path.falco = os.path.dirname(os.path.dirname(LOCAL_PATH))

    # Generate the label associated with this trial
    mp.runLabel = 'testing_wfsc_vc'

    # Perform the Wavefront Sensing and Control

    mp.flagParallel = False
    out = falco.setup.flesh_out_workspace(mp)

    with falco.util.TicToc('Taking image in serial'):
        imageSerial = falco.imaging.get_summed_image(mp)

    mp.flagParallel = True
    with falco.util.TicToc('Taking image in parallel'):
        imageParallel = falco.imaging.get_summed_image(mp)

    diff = imageSerial - imageParallel

    assert np.max(np.abs(diff)) < 10*np.finfo(float).eps


def test_parallel_jacobian():
    """Verify that Jacobian calculation gives the same result in parallel."""
    mp = deepcopy(CONFIG.mp)
    mp.runLabel = 'test_lc'

    mp.fracBW = 0.20
    mp.Nsbp = 1

    mp.path = falco.config.Object()
    LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
    mp.path.falco = os.path.dirname(os.path.dirname(LOCAL_PATH))
    _ = falco.setup.flesh_out_workspace(mp)

    # Fast Jacobian calculation
    mp.dm1.V = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
    mp.dm2.V = np.zeros((mp.dm2.Nact, mp.dm2.Nact))

    mp.flagParallel = False
    with falco.util.TicToc('Computing Jacobian in serial'):
        jacStructSerial = falco.model.jacobian(mp)

    mp.flagParallel = True
    with falco.util.TicToc('Computing Jacobian in parallel'):
        jacStructParallel = falco.model.jacobian(mp)

    G1Serial = jacStructSerial.G1
    G2Serial = jacStructSerial.G2

    G1Parallel = jacStructParallel.G1
    G2Parallel = jacStructParallel.G2

    diff1 = G1Serial - G1Parallel
    diff2 = G2Serial - G2Parallel

    assert np.max(np.abs(diff1)) < 10*np.finfo(float).eps
    assert np.max(np.abs(diff2)) < 10*np.finfo(float).eps


def test_parallel_perfect_estimate():
    """Verify that perfect estimator gives the same result in parallel."""
    mp = deepcopy(CONFIG.mp)
    mp.runLabel = 'test_lc'

    mp.fracBW = 0.10
    mp.Nsbp = 3
    mp.Nwpsbp = 5

    mp.path = falco.config.Object()
    LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
    mp.path.falco = os.path.dirname(os.path.dirname(LOCAL_PATH))
    _ = falco.setup.flesh_out_workspace(mp)

    #
    mp.dm1.V = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
    mp.dm2.V = np.zeros((mp.dm2.Nact, mp.dm2.Nact))

    mp.flagParallel = False
    ev = falco.config.Object()
    ev.Itr = 0
    falco.est.wrapper(mp, ev, [])
    evSerial = deepcopy(ev)

    mp.flagParallel = True
    ev = falco.config.Object()
    ev.Itr = 0
    falco.est.wrapper(mp, ev, [])
    evParallel = deepcopy(ev)

    diff1 = evSerial.Eest - evParallel.Eest
    diff2 = evSerial.imageArray - evParallel.imageArray

    assert np.max(np.abs(diff1)) < 10*np.finfo(float).eps
    assert np.max(np.abs(diff2)) < 10*np.finfo(float).eps


def test_parallel_controller():
    pass


if __name__ == '__main__':
    test_parallel_images()
    test_parallel_jacobian()
    test_parallel_perfect_estimate()
