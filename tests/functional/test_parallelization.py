"""FALCO regression test of WFSC with a vortex coronagraph."""
from copy import copy, deepcopy
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


def test_parallel_grid_search_efc_controller():

    mp = deepcopy(CONFIG.mp)
    mp.controller = 'gridsearchefc'
    mp.fracBW = 0.10
    mp.Nsbp = 1

    mp.flagPlot = False
    LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
    mp.path.falco = os.path.dirname(os.path.dirname(LOCAL_PATH))

    mp.runLabel = 'testing_parallel_controller'

    mp.flagParallel = False
    _ = falco.setup.flesh_out_workspace(mp)

    cvar = falco.config.Object()
    ev = falco.config.Object()

    cvar.Itr = 0
    ev.Itr = 0

    cvar.flagRelin = True
    cvar.flagCullAct = True

    # Re-compute the Jacobian weights
    falco.setup.falco_set_jacobian_modal_weights(mp)

    # Compute the control Jacobians for each DM

    jacStruct = falco.model.jacobian(mp)
    falco.ctrl.cull_weak_actuators(mp, cvar, jacStruct)

    mp.dm1.V = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
    mp.dm2.V = np.zeros((mp.dm2.Nact, mp.dm2.Nact))

    falco.est.wrapper(mp, ev, jacStruct)

    cvar.Eest = ev.Eest
    cvar.NeleAll = mp.dm1.Nele + mp.dm2.Nele + mp.dm3.Nele + mp.dm4.Nele +\
        mp.dm5.Nele + mp.dm6.Nele + mp.dm7.Nele + mp.dm8.Nele + mp.dm9.Nele

    falco.ctrl.wrapper(mp, cvar, jacStruct)
    V1serial = copy(mp.dm1.V)
    V2serial = copy(mp.dm2.V)

    mp.flagParallel = True
    mp.dm1.V = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
    mp.dm2.V = np.zeros((mp.dm2.Nact, mp.dm2.Nact))

    falco.ctrl.wrapper(mp, cvar, jacStruct)
    V1parallel = copy(mp.dm1.V)
    V2parallel = copy(mp.dm2.V)


    # with falco.util.TicToc('Taking image in serial'):
    #     imageSerial = falco.imaging.get_summed_image(mp)

    # with falco.util.TicToc('Taking image in parallel'):
    #     imageParallel = falco.imaging.get_summed_image(mp)

    diff1 = V1serial - V1parallel
    diff2 = V2serial - V2parallel

    assert np.max(np.abs(diff1)) < 10*np.finfo(float).eps
    assert np.max(np.abs(diff2)) < 10*np.finfo(float).eps


def test_parallel_planned_efc_controller():
    pass


def test_parallel_zern_sens():
    mp = deepcopy(CONFIG.mp)
    mp.fracBW = 0.20
    mp.Nsbp = 2

    mp.flagPlot = False
    LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
    mp.path.falco = os.path.dirname(os.path.dirname(LOCAL_PATH))

    mp.eval.indsZnoll = [2, 3, 4]
    mp.eval.Rsens = np.array([[3., 4.], [4., 8.], [8, 9]])

    mp.runLabel = 'testing_parallel_sensitivities'
    mp.flagParallel = False
    _ = falco.setup.flesh_out_workspace(mp)

    Nannuli = mp.eval.Rsens.shape[0]
    Nzern = len(mp.eval.indsZnoll)
    zernSensSerial = np.zeros((Nzern, Nannuli))
    zernSensParallel = np.zeros((Nzern, Nannuli))

    with falco.util.TicToc('Computing in serial'):
        zernSensSerial = falco.zern.calc_zern_sens(mp)

    mp.flagParallel = True
    with falco.util.TicToc('Computing in parallel'):
        zernSensParallel = falco.zern.calc_zern_sens(mp)

    diff = zernSensSerial - zernSensParallel

    assert np.max(np.abs(diff)) < 10*np.finfo(float).eps


def test_parallel_images():

    mp = deepcopy(CONFIG.mp)
    mp.fracBW = 0.20
    mp.Nsbp = 4

    mp.flagPlot = False
    LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
    mp.path.falco = os.path.dirname(os.path.dirname(LOCAL_PATH))

    mp.runLabel = 'testing_parallel_images'

    mp.flagParallel = False
    _ = falco.setup.flesh_out_workspace(mp)

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
    mp.runLabel = 'testing_jacobian'

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
    mp.runLabel = 'testing_estimator'

    mp.fracBW = 0.10
    mp.Nsbp = 3
    mp.Nwpsbp = 5

    mp.path = falco.config.Object()
    LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
    mp.path.falco = os.path.dirname(os.path.dirname(LOCAL_PATH))
    _ = falco.setup.flesh_out_workspace(mp)

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


if __name__ == '__main__':
    test_parallel_grid_search_efc_controller()
    test_parallel_zern_sens()
    test_parallel_images()
    test_parallel_jacobian()
    test_parallel_perfect_estimate()
