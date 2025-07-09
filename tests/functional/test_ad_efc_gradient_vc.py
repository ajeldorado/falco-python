"""AD-EFC gradient accuracy tests for the Lyot coronagraph with all plane-to-plane rotations."""
from copy import deepcopy
import os

import matplotlib.pyplot as plt
import numpy as np

import falco

import config_wfsc_vc_quick as CONFIG

show_plots = False


def test_adjoint_model_vc():
    """Lyot Jacobian gradient."""

    # Create a Generator instance with a seed
    rng1 = np.random.default_rng(seed=42)
    rng2 = np.random.default_rng(seed=7)

    for case_num in range(6):

        mp = deepcopy(CONFIG.mp)
        mp.runLabel = 'test_vc'

        if case_num == 0:
            mp.flagRotation = False  # Whether to rotate 180 degrees between conjugate planes in the compact model
        elif case_num == 1:
            mp.flagRotation = True  # Whether to rotate 180 degrees between conjugate planes in the compact model
            mp.Nrelay1to2 = 0
            mp.Nrelay2to3 = 0
            mp.Nrelay3to4 = 0
            mp.NrelayFend = 0  # How many times to rotate the final image by 180 degrees
        elif case_num == 2:
            mp.flagRotation = True  # Whether to rotate 180 degrees between conjugate planes in the compact model
            mp.Nrelay1to2 = 1
            mp.Nrelay2to3 = 0
            mp.Nrelay3to4 = 0
            mp.NrelayFend = 0  # How many times to rotate the final image by 180 degrees
        elif case_num == 3:
            mp.flagRotation = True  # Whether to rotate 180 degrees between conjugate planes in the compact model
            mp.Nrelay1to2 = 0
            mp.Nrelay2to3 = 1
            mp.Nrelay3to4 = 0
            mp.NrelayFend = 0  # How many times to rotate the final image by 180 degrees
        elif case_num == 4:
            mp.flagRotation = True  # Whether to rotate 180 degrees between conjugate planes in the compact model
            mp.Nrelay1to2 = 0
            mp.Nrelay2to3 = 0
            mp.Nrelay3to4 = 1
            mp.NrelayFend = 0  # How many times to rotate the final image by 180 degrees
        elif case_num == 5:
            mp.flagRotation = True  # Whether to rotate 180 degrees between conjugate planes in the compact model
            mp.Nrelay1to2 = 0
            mp.Nrelay2to3 = 0
            mp.Nrelay3to4 = 0
            mp.NrelayFend = 1  # How many times to rotate the final image by 180 degrees
        else:
            raise ValueError('Case not defined.')
        
        print('\n*** NEW TEST CASE ***')
        print(f'mp.flagRotation = {mp.flagRotation}')
        print(f'mp.Nrelay1to2 = {mp.Nrelay1to2}')
        print(f'mp.Nrelay2to3 = {mp.Nrelay2to3}')
        print(f'mp.Nrelay3to4 = {mp.Nrelay3to4}')
        print(f'mp.NrelayFend = {mp.NrelayFend}')
        print('')

        mp.dm1.useDifferentiableModel = True
        mp.dm2.useDifferentiableModel = True

        mp.flagDM2stop = False
        # mp.dm1.xtilt = 0
        # mp.dm1.ytilt = 0

        # Input pupil (P1)
        # Inputs common to both the compact and full models
        inputs = {"OD": 1.00}
        inputs['wStrut'] = 0.04
        inputs['angStrut'] = np.array([10])
        # Full model only
        inputs["Nbeam"] = mp.P1.full.Nbeam
        inputs["Npad"] = mp.P1.full.mask.shape[0]
        mp.P1.full.mask *= falco.mask.falco_gen_pupil_Simple(inputs)
        # Compact model only
        inputs["Nbeam"] = mp.P1.compact.Nbeam
        inputs["Npad"] = mp.P1.compact.mask.shape[0]
        mp.P1.compact.mask *= falco.mask.falco_gen_pupil_Simple(inputs)

        # Apodizer (P3)
        # Inputs common to both the compact and full models
        inputs = {"OD": 0.84}
        inputs['wStrut'] = 0.04
        inputs['angStrut'] = np.array([70])
        # Full model only
        inputs["Nbeam"] = mp.P1.full.Nbeam
        inputs["Npad"] = 2**(falco.util.nextpow2(mp.P1.full.Nbeam))
        mp.P3.full.mask = falco.mask.falco_gen_pupil_Simple(inputs)
        # Compact model only
        inputs["Nbeam"] = mp.P1.compact.Nbeam
        inputs["Npad"] = 2**(falco.util.nextpow2(mp.P1.compact.Nbeam))
        mp.P3.compact.mask = falco.mask.falco_gen_pupil_Simple(inputs)

        # Lyot stop (P4) Definition and Generation
        mp.P4.IDnorm = 0  # Lyot stop ID [Dtelescope]
        mp.P4.ODnorm = 0.82  # Lyot stop OD [Dtelescope]
        # Inputs common to both the compact and full models
        inputs = {}
        inputs["ID"] = mp.P4.IDnorm
        inputs["OD"] = mp.P4.ODnorm
        inputs['wStrut'] = 0.04
        inputs['angStrut'] = np.array([160])
        # Full model
        inputs["Nbeam"] = mp.P4.full.Nbeam
        inputs["Npad"] = 2**(falco.util.nextpow2(mp.P4.full.Nbeam))
        mp.P4.full.mask = falco.mask.falco_gen_pupil_Simple(inputs)
        # Compact model
        inputs["Nbeam"] = mp.P4.compact.Nbeam
        inputs["Npad"] = 2**(falco.util.nextpow2(mp.P4.compact.Nbeam))
        mp.P4.compact.mask = falco.mask.falco_gen_pupil_Simple(inputs)


        mp.path = falco.config.Object()
        LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
        mp.path.falco = os.path.dirname(os.path.dirname(LOCAL_PATH))
        _ = falco.setup.flesh_out_workspace(mp)

        scalefac = 0.5
        mp.dm1.V = scalefac*(rng1.random((mp.dm1.Nact, mp.dm1.Nact))-0.5)
        mp.dm2.V = scalefac*(rng2.random((mp.dm2.Nact, mp.dm2.Nact))-0.5)

        # Nout = 250
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

        # Get E-field
        ev = falco.config.Object()
        cvar = falco.config.Object()
        falco.est.wrapper(mp, ev, [])
        e_vec = ev.Eest.reshape((-1,))
        e_vec = e_vec.reshape((-1,))

        du1 = np.eye(mp.dm1.Nact)
        du1 = np.roll(du1, (0, 1), axis=(0, 1))
        du2 = np.zeros((mp.dm2.Nact, mp.dm2.Nact))
        du2[:, mp.dm2.Nact//2-1] = 1
        du = np.concatenate((du1.ravel(), du2.ravel()))

        # Compute expected response
        jac = np.concatenate((G1, G2), axis=1)
        u_bar_expected = 2 * jac.conj().T @ (e_vec + jac @ du)
        u1_bar_expected_2d = u_bar_expected[0:mp.dm1.NactTotal].reshape((mp.dm1.Nact, mp.dm1.Nact))
        u2_bar_expected_2d = u_bar_expected[mp.dm1.NactTotal::].reshape((mp.dm2.Nact, mp.dm2.Nact))


        sumG1sq = np.sum(np.abs(G1)**2, axis=0).reshape((mp.dm1.Nact, mp.dm1.Nact))
        sumG2sq = np.sum(np.abs(G2)**2, axis=0).reshape((mp.dm2.Nact, mp.dm2.Nact))
        sumG1sq /= np.max(sumG1sq)
        sumG2sq /= np.max(sumG2sq)
        jacThresh = 0.8
        mask1 = np.zeros_like(u1_bar_expected_2d, dtype=bool)
        mask2 = np.zeros_like(u2_bar_expected_2d, dtype=bool)
        mask1[sumG1sq > jacThresh] = True
        mask2[sumG2sq > jacThresh] = True

        # Compute gradient
        EestAll = ev.Eest
        log10reg = -np.inf
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
        total_cost, u_bar_out = falco.model.compact_reverse_gradient(du, mp, EestAll, EFendPrev, log10reg)

        u1_bar_out = u_bar_out[0:mp.dm1.NactTotal]
        u2_bar_out = u_bar_out[mp.dm1.NactTotal::]

        u1_bar_out_2d = u1_bar_out.reshape((mp.dm1.Nact, mp.dm1.Nact))
        u2_bar_out_2d = u2_bar_out.reshape((mp.dm2.Nact, mp.dm2.Nact))

        ratio1 = np.real(u1_bar_expected_2d)/np.real(u1_bar_out_2d)
        ratio2 = np.real(u2_bar_expected_2d)/np.real(u2_bar_out_2d)

        # print('DM1 1')
        # print(np.max(ratio1[mask1]))
        # print(np.min(ratio1[mask1]))
        # print(np.median(ratio1[mask1]))
        # print(np.mean(ratio1[mask1]))
        # print(np.std(ratio1[mask1]))

        # print('DM1 2')
        # print(np.max(ratio2[mask2]))
        # print(np.min(ratio2[mask2]))
        # print(np.median(ratio2[mask2]))
        # print(np.mean(ratio2[mask2]))
        # print(np.std(ratio2[mask2]))                    

        if show_plots:

            # plt.figure(91)
            # plt.imshow(sumG1sq)
            # plt.colorbar()
            # plt.title('sumG1sq')

            # plt.figure(92)
            # plt.imshow(sumG2sq)
            # plt.colorbar()
            # plt.title('sumG2sq')

            plt.figure(1)
            plt.imshow(np.real(u1_bar_expected_2d))
            plt.colorbar()
            plt.title('u1_bar_expected_2d, real')

            plt.figure(11)
            plt.imshow(np.real(u1_bar_out_2d))
            plt.colorbar()
            plt.title('u1_bar_out_2d, real')

            # plt.figure(21)
            # plt.imshow(np.real(u1_bar_expected_2d)/np.real(u1_bar_out_2d))
            # plt.colorbar()
            # plt.title('DM1 expected/out, real')

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

            # plt.figure(22)
            # plt.imshow(np.real(u2_bar_expected_2d)/np.real(u2_bar_out_2d))
            # plt.colorbar()
            # plt.title('DM2 expected/out, real')

            # plt.figure(41)
            # plt.imshow(mask1)
            # plt.colorbar()
            # plt.title('mask1')

            # plt.figure(42)
            # plt.imshow(mask2)
            # plt.colorbar()
            # plt.title('mask2')


            plt.figure(51)
            plt.imshow(ratio1*mask1)
            plt.colorbar()
            plt.title('ratio1')

            plt.figure(52)
            plt.imshow(ratio2*mask2)
            plt.colorbar()
            plt.title('ratio2')

            plt.show()

        # Tests
        dm1_median_ratio = np.median(ratio1[mask1])
        dm2_median_ratio = np.median(ratio2[mask2])

        atol = 0.01
        target_ratio = 1.0
        assert np.isclose(dm1_median_ratio, target_ratio, atol)
        assert np.isclose(dm2_median_ratio, target_ratio, atol)

        del mp


if __name__ == '__main__':
    test_adjoint_model_vc()
