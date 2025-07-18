"""Jacobian accuracy tests for the Lyot coronagraph."""
from copy import deepcopy
import numpy as np
import os

import falco

import config_wfsc_lc_quick as CONFIG


def test_jacobian_lc():
    """Lyot Jacobian test using FFTs between DMs."""

    # Create a Generator instance with a seed
    rng1 = np.random.default_rng(seed=42)
    rng2 = np.random.default_rng(seed=7)

    for case_num in range(6):

        mp = deepcopy(CONFIG.mp)
        mp.runLabel = 'test_vc'

        if case_num == 0:
            mp.flagRotation = False  # Whether to rotate 180 degrees between conjugate planes in the compact model
            mp.NrelayFend = 0  # How many times to rotate the final image by 180 degrees
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

        mp = deepcopy(CONFIG.mp)
        mp.runLabel = 'test_lc'

        scalefac = 1  # 0.5
        dm1v0 = scalefac*(rng1.random((mp.dm1.Nact, mp.dm1.Nact))-0.5)
        dm2v0 = scalefac*(rng2.random((mp.dm2.Nact, mp.dm2.Nact))-0.5)

        mp.path = falco.config.Object()
        LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
        mp.path.falco = os.path.dirname(os.path.dirname(LOCAL_PATH))
        _ = falco.setup.flesh_out_workspace(mp)

        # Fast Jacobian calculation
        mp.dm1.V = dm1v0.copy()  # np.zeros((mp.dm1.Nact, mp.dm1.Nact))
        mp.dm2.V = dm2v0.copy()  # np.zeros((mp.dm2.Nact, mp.dm2.Nact))
        jacStruct = falco.model.jacobian(mp)  # Get structure containing Jacobians

        G1fastAll = jacStruct.G1
        G2fastAll = jacStruct.G2
        Nind = 20
        subinds = np.ix_(np.arange(3, 20*4, 4).astype(int))
        absG1sum = np.sum(np.abs(G1fastAll), axis=1)
        indG1 = np.nonzero(absG1sum > 1e-2*np.max(absG1sum))[0]
        indG1subset = indG1[subinds]  # Take a 20-actuator subset
        absG2sum = np.sum(np.abs(G2fastAll), axis=1)
        indG2 = np.nonzero(absG2sum > 1e-2*np.max(absG2sum))[0]
        indG2subset = indG2[subinds]  # Take a 20-actuator subset
        G1fast = np.squeeze(G1fastAll[:, indG1subset])
        G2fast = np.squeeze(G2fastAll[:, indG2subset])

        # Compute Jacobian via differencing (slower)
        modvar = falco.config.ModelVariables()
        modvar.whichSource = 'star'
        modvar.sbpIndex = 0
        modvar.starIndex = 0
        Eunpoked2D = falco.model.compact(mp, modvar)
        Eunpoked = Eunpoked2D[mp.Fend.corr.maskBool]
        DeltaV = 1e-4
        G1slow = np.zeros((mp.Fend.corr.Npix, Nind), dtype=complex)
        G2slow = np.zeros((mp.Fend.corr.Npix, Nind), dtype=complex)

        for ii in range(Nind):
            # DM1
            mp.dm1.V = dm1v0.copy()  # np.zeros((mp.dm1.Nact, mp.dm1.Nact))
            mp.dm2.V = dm2v0.copy()  # np.zeros((mp.dm2.Nact, mp.dm2.Nact))
            mp.dm1.V[np.unravel_index(indG1subset[ii], mp.dm1.V.shape)] += DeltaV
            Epoked2D = falco.model.compact(mp, modvar)
            Epoked = Epoked2D[mp.Fend.corr.maskBool]
            G1slow[:, ii] = (Epoked - Eunpoked) / DeltaV

            # DM2
            mp.dm1.V = dm1v0.copy()  # np.zeros((mp.dm1.Nact, mp.dm1.Nact))
            mp.dm2.V = dm2v0.copy()  # np.zeros((mp.dm2.Nact, mp.dm2.Nact))
            mp.dm2.V[np.unravel_index(indG2subset[ii], mp.dm2.V.shape)] += DeltaV
            Epoked2D = falco.model.compact(mp, modvar)
            Epoked = Epoked2D[mp.Fend.corr.maskBool]
            G2slow[:, ii] = (Epoked - Eunpoked) / DeltaV

        # Tests
        rmsNormErrorDM1 = (np.sqrt(np.sum(np.abs(G1slow - G1fast)**2) /
                                np.sum(np.abs(G1slow)**2)))
        rmsNormErrorDM2 = (np.sqrt(np.sum(np.abs(G2slow - G2fast)**2) /
                                np.sum(np.abs(G2slow)**2)))

        print('rmsNormErrorDM1 = %.3f' % rmsNormErrorDM1)
        print('rmsNormErrorDM2 = %.3f' % rmsNormErrorDM2)
        assert rmsNormErrorDM1 < 0.01
        assert rmsNormErrorDM2 < 0.01


def test_jacobian_lc_no_fpm():
    """Test accuracy of no-FPM Jacobian calculation."""
    mp = deepcopy(CONFIG.mp)
    mp.jac.minimizeNI = True
    mp.runLabel = 'test_lc_no_fpm'
    iMode = 0

    mp.path = falco.config.Object()
    LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
    mp.path.falco = os.path.dirname(os.path.dirname(LOCAL_PATH))
    _ = falco.setup.flesh_out_workspace(mp)

    # Pre-compute the DM surfaces to save time
    NdmPad = int(mp.compact.NdmPad)
    if any(mp.dm_ind == 1):
        mp.dm1.compact.surfM = falco.dm.gen_surf_from_act(mp.dm1,
                                                    mp.dm1.compact.dx, NdmPad)
    else:
        mp.dm1.compact.surfM = np.zeros((NdmPad, NdmPad))
    if any(mp.dm_ind == 2):
        mp.dm2.compact.surfM = falco.dm.gen_surf_from_act(mp.dm2,
                                                    mp.dm2.compact.dx, NdmPad)
    else:
        mp.dm2.compact.surfM = np.zeros((NdmPad, NdmPad))

    # No-FPM Jacobian calculation
    mp.dm1.V = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
    mp.dm2.V = np.zeros((mp.dm2.Nact, mp.dm2.Nact))
    falco.model.jacobians.precomp(mp)
    G1fastAll = np.zeros((1, mp.dm1.Nele), dtype=complex)
    G2fastAll = np.zeros((1, mp.dm2.Nele), dtype=complex)
    for index, iact in enumerate(mp.dm1.act_ele):
        G1fastAll[0, index] = falco.model.jacobians.no_fpm(mp, iMode, 1, iact)
    for index, iact in enumerate(mp.dm2.act_ele):
        G2fastAll[0, index] = falco.model.jacobians.no_fpm(mp, iMode, 2, iact)    
#     G1fastAll = falco.model.jacobians.no_fpm(mp, iMode, 1)
#     G2fastAll = falco.model.jacobians.no_fpm(mp, iMode, 2)

    Nind = 20
    thresh = 1e-1
    subinds = np.ix_(np.arange(3, 20*4, 4).astype(int))
    absG1sum = np.sum(np.abs(G1fastAll), axis=0)
    indG1 = np.nonzero(absG1sum > thresh*np.max(absG1sum))[0]
    indG1subset = indG1[subinds]  # Take a 20-actuator subset
    absG2sum = np.sum(np.abs(G2fastAll), axis=0)
    indG2 = np.nonzero(absG2sum > thresh*np.max(absG2sum))[0]
    indG2subset = indG2[subinds]  # Take a 20-actuator subset
    G1fast = np.squeeze(G1fastAll[:, indG1subset])
    G2fast = np.squeeze(G2fastAll[:, indG2subset])

    # Compute Jacobian via differencing (slower)

    # Get the unocculted peak E-field and coronagraphic E-field
    if mp.jac.minimizeNI:
        modvar = falco.config.ModelVariables()
        modvar.sbpIndex = mp.jac.sbp_inds[iMode]
        modvar.zernIndex = mp.jac.zern_inds[iMode]
        modvar.starIndex = mp.jac.star_inds[iMode]
        modvar.whichSource = 'star'
        Eunocculted = falco.model.compact(mp, modvar, useFPM=False)
        indPeak = np.unravel_index(np.argmax(np.abs(Eunocculted), axis=None),
                                   Eunocculted.shape)

    modvar = falco.config.ModelVariables()
    modvar.whichSource = 'star'
    modvar.sbpIndex = 0
    modvar.starIndex = 0
    Eunpoked2D = falco.model.compact(mp, modvar, useFPM=False)
    Eunpoked = Eunpoked2D[indPeak]
    DeltaV = 1e-4
    G1slow = np.zeros((1, Nind), dtype=complex)
    G2slow = np.zeros((1, Nind), dtype=complex)

    for ii in range(Nind):
        # DM1
        mp.dm1.V = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
        mp.dm2.V = np.zeros((mp.dm2.Nact, mp.dm2.Nact))
        mp.dm1.V[np.unravel_index(indG1subset[ii], mp.dm1.V.shape)] = DeltaV
        Epoked2D = falco.model.compact(mp, modvar, useFPM=False)
        Epoked = Epoked2D[indPeak]
        G1slow[:, ii] = (Epoked - Eunpoked) / DeltaV

        # DM2
        mp.dm1.V = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
        mp.dm2.V = np.zeros((mp.dm2.Nact, mp.dm2.Nact))
        mp.dm2.V[np.unravel_index(indG2subset[ii], mp.dm2.V.shape)] = DeltaV
        Epoked2D = falco.model.compact(mp, modvar, useFPM=False)
        Epoked = Epoked2D[indPeak]
        G2slow[:, ii] = (Epoked - Eunpoked) / DeltaV

    # Tests
    rmsNormErrorDM1 = (np.sqrt(np.sum(np.abs(G1slow - G1fast)**2) /
                               np.sum(np.abs(G1slow)**2)))
    rmsNormErrorDM2 = (np.sqrt(np.sum(np.abs(G2slow - G2fast)**2) /
                               np.sum(np.abs(G2slow)**2)))

    assert rmsNormErrorDM1 < 1e-3
    assert rmsNormErrorDM2 < 1e-3


def test_jacobian_lc_mft():
    """Lyot Jacobian test using MFTs between DMs."""
    pass


if __name__ == '__main__':
    test_jacobian_lc()
    test_jacobian_lc_mft()
    test_jacobian_lc_no_fpm()
