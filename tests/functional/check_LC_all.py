"""Jacobian accuracy tests for the Lyot coronagraph."""
import numpy as np
import os
# from math import isclose

import falco

import config_wfsc_lc as CONFIG


def test_jacobian_lc_no_fpm():
    """Test accuracy of no-FPM Jacobian calculation."""
    mp = CONFIG.mp
    mp.jac.minimizeNI = True
    mp.runLabel = 'test_lc_no_fpm'
    iMode = 0

    mp.path = falco.config.Object()
    LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
    mp.path.falco = os.path.dirname(os.path.dirname(LOCAL_PATH))
    _ = falco.setup.flesh_out_workspace(mp)

    # Fast Jacobian calculation
    mp.dm1.V = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
    mp.dm2.V = np.zeros((mp.dm2.Nact, mp.dm2.Nact))
    jacStruct = falco.model.jacobian(mp)  # Get structure containing Jacobians

    G1fastAll = jacStruct.G1
    G2fastAll = jacStruct.G2
    Nind = 20
    thresh = 1e-2
    subinds = np.ix_(np.arange(3, 20*4, 4).astype(int))
    absG1sum = np.sum(np.abs(G1fastAll), axis=1)
    indG1 = np.nonzero(absG1sum > thresh*np.max(absG1sum))[0]
    indG1subset = indG1[subinds]  # Take a 20-actuator subset
    absG2sum = np.sum(np.abs(G2fastAll), axis=1)
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
    G1slow = np.zeros((mp.Fend.corr.Npix, Nind), dtype=complex)
    G2slow = np.zeros((mp.Fend.corr.Npix, Nind), dtype=complex)

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

    print('rmsNormErrorDM1 = %.5f' % rmsNormErrorDM1)
    print('rmsNormErrorDM2 = %.5f' % rmsNormErrorDM2)
    assert rmsNormErrorDM1 < 1e-3
    assert rmsNormErrorDM2 < 1e-3



if __name__ == '__main__':
    # test_jacobian_lc()
    # test_jacobian_lc_mft()
    test_jacobian_lc_no_fpm()
