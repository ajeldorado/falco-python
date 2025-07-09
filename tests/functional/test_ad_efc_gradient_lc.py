"""Jacobian accuracy tests for the Lyot coronagraph."""
from copy import deepcopy
import os

import matplotlib.pyplot as plt
import numpy as np

import falco

import config_wfsc_lc2 as CONFIG

show_plots = True


def test_adjoint_model_lc():
    """Lyot Jacobian gradient."""
    mp = deepcopy(CONFIG.mp)
    mp.runLabel = 'test_lc'

    mp.flagRotation = False
    mp.Fend.corr.Rin = 0
    mp.Fend.score.Rin = 0
    # mp.Fend.corr.Rin = 2.5
    # mp.Fend.score.Rin = 2.5
    mp.d_dm1_dm2 = 0.2  # DEBUGGING

    mp.dm1.useDifferentiableModel = True
    mp.dm2.useDifferentiableModel = True

    mp.flagDM2stop = False
    mp.dm1.xtilt = 0
    mp.dm1.ytilt = 0
    # mp.lambda0 *= 2  # DEBUGGING. --> Not the issue.

    # mp.dm1.inf_fn = falco.INFLUENCE_BMC_2K
    # mp.dm2.inf_fn = falco.INFLUENCE_BMC_2K
    mp.dm1.inf_fn = falco.INFLUENCE_XINETICS
    mp.dm2.inf_fn = falco.INFLUENCE_XINETICS

    # mp.dm1.dm_spacing = 1.05*0.9906e-3  # actuator pitch
    # mp.dm2.dm_spacing = 1.05*0.9906e-3  # actuator pitch



    mp.path = falco.config.Object()
    LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
    mp.path.falco = os.path.dirname(os.path.dirname(LOCAL_PATH))
    _ = falco.setup.flesh_out_workspace(mp)

    Nout = 250
    scalefac = 0.5
    mp.dm1.V = scalefac*(np.random.rand(mp.dm1.Nact, mp.dm1.Nact)-0.5)
    mp.dm2.V = scalefac*(np.random.rand(mp.dm2.Nact, mp.dm2.Nact)-0.5)

    # surf_prysm = falco.dm.gen_surf_from_act(mp.dm1, mp.P2.compact.dx, Nout)
    # mp.dm1.useDifferentiableModel = False
    # mp.dm2.useDifferentiableModel = False
    # surf_proper = falco.dm.gen_surf_from_act(mp.dm1, mp.P2.compact.dx, Nout)

    # plt.figure(101)
    # plt.imshow(np.real(surf_prysm))
    # plt.colorbar()
    # plt.title('surf_prysm')

    # plt.figure(102)
    # plt.imshow(np.real(surf_proper))
    # plt.colorbar()
    # plt.title('surf_proper')
    # plt.show()

    # Fast Jacobian calculation
    # mp.dm1.V = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
    # mp.dm2.V = np.zeros((mp.dm2.Nact, mp.dm2.Nact))
    jacStruct = falco.model.jacobian(mp)  # Get structure containing Jacobians

    G1 = np.squeeze(jacStruct.G1)
    G2 = np.squeeze(jacStruct.G2)
    # Nind = 20
    # subinds = np.ix_(np.arange(3, 20*4, 4).astype(int))
    # absG1sum = np.sum(np.abs(G1fastAll), axis=1)
    # indG1 = np.nonzero(absG1sum > 1e-2*np.max(absG1sum))[0]
    # indG1subset = indG1[subinds]  # Take a 20-actuator subset
    # absG2sum = np.sum(np.abs(G2fastAll), axis=1)
    # indG2 = np.nonzero(absG2sum > 1e-2*np.max(absG2sum))[0]
    # indG2subset = indG2[subinds]  # Take a 20-actuator subset
    # G1fast = np.squeeze(G1fastAll[:, indG1subset])
    # G2fast = np.squeeze(G2fastAll[:, indG2subset])

    # Get E-field
    ev = falco.config.Object()
    cvar = falco.config.Object()
    falco.est.wrapper(mp, ev, [])
    e_vec = ev.Eest.reshape((-1,))
    e_vec = e_vec.reshape((-1,))

    # du1 = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
    # du1[0:24, 0:24] = 10

    # du1 = np.eye(mp.dm1.Nact)
    # du2 = np.zeros((mp.dm2.Nact, mp.dm2.Nact))

    du1 = np.eye(mp.dm1.Nact)
    du2 = np.zeros((mp.dm2.Nact, mp.dm2.Nact))
    du2[:, 21] = 1
    du = np.concatenate((du1.ravel(), du2.ravel()))

    # Compute expected response
    jac = np.concatenate((G1, G2), axis=1)
    u_bar_expected = 2 * jac.conj().T @ (e_vec + jac @ du)
    u1_bar_expected_2d = u_bar_expected[0:mp.dm1.NactTotal].reshape((mp.dm1.Nact, mp.dm1.Nact))
    u2_bar_expected_2d = u_bar_expected[mp.dm1.NactTotal::].reshape((mp.dm2.Nact, mp.dm2.Nact))
    # u1_bar_expected = 2 * G1.conj().T @ (e_vec + G1 @ du1.flatten() + G2 @ du2.flatten())
    # u2_bar_expected = 2 * G2.conj().T @ (e_vec + G1 @ du1.flatten() + G2 @ du2.flatten())
    # u1_bar_expected_2d = u1_bar_expected.reshape((mp.dm1.Nact, mp.dm1.Nact))
    # u2_bar_expected_2d = u2_bar_expected.reshape((mp.dm2.Nact, mp.dm2.Nact))

    command_vec = du  # np.concatenate((du1.flatten(), du2.flatten()))

    thresh = 0.001
    mask1 = np.zeros_like(u1_bar_expected_2d)
    mask1[np.abs(u1_bar_expected_2d) > thresh*np.max(np.abs(u1_bar_expected_2d))] = 1
    mask2 = np.zeros_like(u2_bar_expected_2d)
    mask2[np.abs(u2_bar_expected_2d) > thresh*np.max(np.abs(u2_bar_expected_2d))] = 1

    # mask1 = np.ones_like(u1_bar_expected_2d)
    # mask2 = np.ones_like(u2_bar_expected_2d)


    # Compute gradient
    command_vec = np.concatenate((du1.flatten(), du2.flatten()))
    EestAll = ev.Eest
    log10reg = -np.Inf
    EFendPrev = []

    falco.ctrl.init(mp, cvar)

    for iMode in range(mp.jac.Nmode):

        modvar = falco.config.ModelVariables()
        modvar.whichSource = 'star'
        modvar.sbpIndex = mp.jac.sbp_inds[iMode]
        modvar.zernIndex = mp.jac.zern_inds[iMode]
        modvar.starIndex = mp.jac.star_inds[iMode]

        # Calculate E-Field for previous EFC iteration
        EFend = falco.model.compact(mp, modvar, isNorm=True, isEvalMode=False,
                                    useFPM=True, forRevGradModel=False)
        EFendPrev.append(EFend)
    total_cost, u_bar_out = falco.model.compact_reverse_gradient(command_vec, mp, EestAll, EFendPrev, log10reg)
    # print(u_bar_out.shape)

    u1_bar_out = u_bar_out[0:mp.dm1.NactTotal]
    u2_bar_out = u_bar_out[mp.dm1.NactTotal::]
    # print(u1_bar_out.shape)
    # print(u2_bar_out.shape)

    u1_bar_out_2d = u1_bar_out.reshape((mp.dm1.Nact, mp.dm1.Nact))
    u2_bar_out_2d = u2_bar_out.reshape((mp.dm2.Nact, mp.dm2.Nact))

    print(f'mp.Fend.compact.I00 = {mp.Fend.compact.I00}')

    if show_plots:
        plt.figure(1)
        plt.imshow(np.real(u1_bar_expected_2d))
        plt.colorbar()
        plt.title('u1_bar_expected_2d, real')

        plt.figure(11)
        plt.imshow(np.real(u1_bar_out_2d))
        plt.colorbar()
        plt.title('u1_bar_out_2d, real')

        plt.figure(21)
        plt.imshow(np.real(u1_bar_expected_2d)/np.real(u1_bar_out_2d))
        plt.colorbar()
        plt.title('DM1 expected/out, real')

        # plt.figure(31)
        # plt.imshow(np.real(u1_bar_out_2d)/np.real(u1_bar_expected_2d))
        # plt.colorbar()
        # plt.title('DM1 out/expected')



        plt.figure(2)
        plt.imshow(np.real(u2_bar_expected_2d))
        plt.colorbar()
        plt.title('u2_bar_expected_2d, real')

        plt.figure(12)
        plt.imshow(np.real(u2_bar_out_2d))
        plt.colorbar()
        plt.title('u2_bar_out_2d, real')

        plt.figure(22)
        plt.imshow(np.real(u2_bar_expected_2d)/np.real(u2_bar_out_2d))
        plt.colorbar()
        plt.title('DM2 expected/out, real')


        # plt.figure(31)
        # plt.imshow(np.real(u1_bar_out_2d/u1_bar_expected_2d*mask1))
        # plt.colorbar()
        # plt.title('DM1 out/expected, real')

        # plt.figure(32)
        # plt.imshow(np.real(u2_bar_out_2d/u2_bar_expected_2d*mask2))
        # plt.colorbar()
        # plt.title('DM2 out/expected, real')

        plt.show()



    # dDM1Vvec = np.zeros(mp.dm1.NactTotal)
    # dDM1Vvec[mp.dm1.act_ele] = mp.dm1.weight*u1_bar_expected.flatten() #[cvar.uLegend == 1]
    # u1_bar_expected_2d = dDM1Vvec.reshape((mp.dm1.Nact, mp.dm1.Nact))


   # u1_bar_expected = 2 * G.conj().T @ (e_vec + G1fast @ du1.flatten() + G2fast @ du2.flatten())

    # for ii in range(Nind):
    #     # DM1
    #     mp.dm1.V = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
    #     mp.dm2.V = np.zeros((mp.dm2.Nact, mp.dm2.Nact))
    #     mp.dm1.V[np.unravel_index(indG1subset[ii], mp.dm1.V.shape)] = DeltaV
    #     Epoked2D = falco.model.compact(mp, modvar)
    #     Epoked = Epoked2D[mp.Fend.corr.maskBool]

    #     # DM2
    #     mp.dm1.V = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
    #     mp.dm2.V = np.zeros((mp.dm2.Nact, mp.dm2.Nact))
    #     mp.dm2.V[np.unravel_index(indG2subset[ii], mp.dm2.V.shape)] = DeltaV
    #     Epoked2D = falco.model.compact(mp, modvar)
    #     Epoked = Epoked2D[mp.Fend.corr.maskBool]

    # Tests
    # rmsNormErrorDM1 = (np.sqrt(np.sum(np.abs(G1slow - G1fast)**2) /
    #                            np.sum(np.abs(G1slow)**2)))
    # rmsNormErrorDM2 = (np.sqrt(np.sum(np.abs(G2slow - G2fast)**2) /
    #                            np.sum(np.abs(G2slow)**2)))

    # print('rmsNormErrorDM1 = %.3f' % rmsNormErrorDM1)
    # print('rmsNormErrorDM2 = %.3f' % rmsNormErrorDM2)
    # assert rmsNormErrorDM1 < 0.01
    # assert rmsNormErrorDM2 < 0.01


if __name__ == '__main__':
    test_adjoint_model_lc()
