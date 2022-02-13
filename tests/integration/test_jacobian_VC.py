import numpy as np
import os
# from math import isclose

import falco

import testing_config_VC as CONFIG


mp = CONFIG.mp
mp.runLabel = 'test_VC'
mp.jac.mftToVortex = False

mp.path = falco.config.Object()
LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
mp.path.falco = os.path.dirname(os.path.dirname(LOCAL_PATH))
out = falco.setup.flesh_out_workspace(mp)

# Fast Jacobian calculation
mp.dm1.V = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
mp.dm2.V = np.zeros((mp.dm2.Nact, mp.dm2.Nact))
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
    mp.dm1.V = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
    mp.dm2.V = np.zeros((mp.dm2.Nact, mp.dm2.Nact))
    mp.dm1.V[np.unravel_index(indG1subset[ii], mp.dm1.V.shape)] = DeltaV
    Epoked2D = falco.model.compact(mp, modvar)
    Epoked = Epoked2D[mp.Fend.corr.maskBool]
    G1slow[:, ii] = (Epoked - Eunpoked) / DeltaV

    # DM2
    mp.dm1.V = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
    mp.dm2.V = np.zeros((mp.dm2.Nact, mp.dm2.Nact))
    mp.dm2.V[np.unravel_index(indG2subset[ii], mp.dm2.V.shape)] = DeltaV
    Epoked2D = falco.model.compact(mp, modvar)
    Epoked = Epoked2D[mp.Fend.corr.maskBool]
    G2slow[:, ii] = (Epoked - Eunpoked) / DeltaV

# Tests
rmsNormErrorDM1 = (np.sqrt(np.sum(np.abs(G1slow - G1fast)**2) /
                           np.sum(np.abs(G1slow)**2)))
rmsNormErrorDM2 = (np.sqrt(np.sum(np.abs(G2slow - G2fast)**2) /
                           np.sum(np.abs(G2slow)**2)))

assert rmsNormErrorDM1 < 0.01
assert rmsNormErrorDM2 < 0.01


# def test_jacobian_vc_no_fpm():
#     pass


# def test_jacobian_vc_mft():
#     pass


# def test_jacobian_vc_fft():

#     mp = CONFIG.mp
#     mp.runLabel = 'test_VC'
#     mp.jac.mftToVortex = False

#     mp.path = falco.config.Object()
#     LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
#     mp.path.falco = os.path.dirname(os.path.dirname(LOCAL_PATH))
#     out = falco.setup.flesh_out_workspace(mp)

#     # Fast Jacobian calculation
#     mp.dm1.V = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
#     mp.dm2.V = np.zeros((mp.dm2.Nact, mp.dm2.Nact))
#     jacStruct = falco.model.jacobian(mp)  # Get structure containing Jacobians

#     G1fastAll = jacStruct.G1
#     G2fastAll = jacStruct.G2
#     Nind = 20
#     subinds = np.ix_(np.arange(3, 20*4, 4).astype(int))
#     absG1sum = np.sum(np.abs(G1fastAll))
#     indG1 = np.nonzero(absG1sum > 1e-2*np.max(absG1sum))[0]
#     indG1subset = indG1[subinds]  # Take a 20-actuator subset
#     absG2sum = np.sum(np.abs(G2fastAll))
#     indG2 = np.nonzero(absG2sum > 1e-2*np.max(absG2sum))[0]
#     indG2subset = indG2[subinds]  # Take a 20-actuator subset
#     G1fast = G1fastAll[:, indG1subset]
#     G2fast = G2fastAll[:, indG2subset]

#     # Compute Jacobian via differencing (slower)
#     modvar = falco.config.ModelVariables()
#     modvar.whichSource = 'star'
#     modvar.sbpIndex = 0
#     modvar.starIndex = 0
#     Eunpoked2D = falco.model.compact(mp, modvar)
#     Eunpoked = Eunpoked2D(mp.Fend.corr.maskBool)
#     DeltaV = 1e-4
#     G1slow = np.zeros(mp.Fend.corr.Npix, Nind)
#     G2slow = np.zeros(mp.Fend.corr.Npix, Nind)

#     for ii in range(Nind):
#         # DM1
#         mp.dm1.V = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
#         mp.dm2.V = np.zeros((mp.dm2.Nact, mp.dm2.Nact))
#         mp.dm1.V[indG1subset(ii)] = DeltaV
#         Epoked2D = falco.model.compact(mp, modvar)
#         Epoked = Epoked2D(mp.Fend.corr.maskBool)
#         G1slow[:, ii] = (Epoked - Eunpoked) / DeltaV

#         # DM2
#         mp.dm1.V = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
#         mp.dm2.V = np.zeros((mp.dm2.Nact, mp.dm2.Nact))
#         mp.dm2.V[indG2subset(ii)] = DeltaV
#         Epoked2D = falco.model.compact(mp, modvar)
#         Epoked = Epoked2D(mp.Fend.corr.maskBool)
#         G2slow[:, ii] = (Epoked - Eunpoked) / DeltaV

#     # Tests
#     rmsNormErrorDM1 = (np.sqrt(np.sum(np.abs(G1slow - G1fast)**2) /
#                                np.sum(np.abs(G1slow)**2)))
#     rmsNormErrorDM2 = (np.sqrt(np.sum(np.abs(G2slow - G2fast)**2) /
#                                np.sum(np.abs(G2slow)**2)))

#     assert rmsNormErrorDM1 < 0.01
#     assert rmsNormErrorDM2 < 0.01


# if __name__ == '__main__':
#     test_jacobian_vc_fft()
#     test_jacobian_vc_mft()
#     test_jacobian_vc_no_fpm()
