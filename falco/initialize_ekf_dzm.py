"""
Python implementation of initialize_ekf_matrices function from FALCO MATLAB.
"""

import numpy as np

import falco
from .est_utils import rearrange_jacobians

def initialize_ekf_dzm(mp, ev, jacStruct):
    """
    Initialize the EKF maintenance estimator.

    Parameters
    ----------
    mp : ModelParameters
        Object containing optical model parameters
    ev : FALCO object
        Structure containing estimation variables.
    jacStruct : ModelParameters
        Structure containing control Jacobians for each specified DM.

    Returns
    -------
    ev : FALCO object
        Updated structure containing estimation variables.
    """

    if mp.controller.lower() == 'ad-efc':
        cvar = falco.config.Object()
        cvar.Itr = ev.Itr
        cvar.flagRelin = True

        # Re-compute the Jacobian weights
        falco.setup.falco_set_jacobian_modal_weights(mp)

        # Compute the control Jacobians for each DM
        jacStruct = falco.model.jacobian(mp)

        falco.ctrl.cull_weak_actuators(mp, cvar, jacStruct)

    # Check if sim mode to avoid calling tb obj in sim mode
    if mp.flagSim:
        sbp_texp = mp.detector.tExpVec  # exposure times for non-pairwise-probe images in each subband
        psf_peaks = mp.detector.peakFluxVec
    else:
        sbp_texp = mp.tb.info.sbp_texp
        psf_peaks = mp.tb.info.PSFpeaks

    # Find values to convert images back to counts rather than normalized intensity
    ev.peak_psf_counts = np.zeros(mp.Nsbp)
    ev.e_scaling = np.zeros(mp.Nsbp)

    for iSubband in range(mp.Nsbp):
        # potentially set mp.detector.peakFluxVec[iSubband] * mp.detector.tExpUnprobedVec[iSubband]
        # to mp.tb.info.sbp_texp[iSubband]*mp.tb.info.PSFpeaks[iSubband] for cleaner setup
        ev.peak_psf_counts[iSubband] = sbp_texp[iSubband] * psf_peaks[iSubband]
        ev.e_scaling[iSubband] = np.sqrt(psf_peaks[iSubband])

    # Rearrange jacobians
    ev.G_tot_cont = rearrange_jacobians(mp, jacStruct, mp.dm_ind)
    ev.G_tot_drift = rearrange_jacobians(mp, jacStruct, mp.dm_drift_ind)

    # Initialize EKF matrices
    initialize_ekf_matrices(mp, ev, sbp_texp)

    # Initialize pinned actuator check
    ev.dm1_initial_pinned_actuators = mp.dm1.pinned.copy()
    if np.any(mp.dm_ind == 2):
        ev.dm2_initial_pinned_actuators = mp.dm2.pinned.copy()
    ev.dm1_new_pinned_actuators = []
    ev.dm2_new_pinned_actuators = []
    ev.dm1_act_ele_pinned = []
    ev.dm2_act_ele_pinned = []

    return ev


def initialize_ekf_matrices(mp, ev, sbp_texp):
    """
    Initialize EKF matrices for the estimator.

    This function creates and initializes all the matrices needed for
    the Extended Kalman Filter (EKF) estimator.

    Parameters
    ----------
    mp : ModelParameters
        Object containing optical model parameters
    ev : FALCO object
        Structure containing estimation variables.
    sbp_texp : array_like
        Exposure times for each subband

    Returns
    -------
    ev : FALCO object
        Updated structure containing estimation variables.
    """
    ## Below are the defnitions of the EKF matrices. There are multiple EKFs
    ## defined in parallel.

    # To get an idea of what the code below does, it's easier to play with
    # the toy example at https://github.com/leonidprinceton/DHMaintenanceExample
    ev.SS = 2  # Pixel state size. Two for real and imaginary parts of the electric field. If incoherent intensity is not ignored, SS should be 3 and the EKF modified accordingly.
    ev.BS = ev.SS * 1  # EKF block size - number of pixels per EKF (currently 1). Computation time grows as the cube of BS.
    ev.SL = ev.SS * mp.Fend.corr.Npix  # Total length of the sate vector (all pixels).

    # 3D matrices that include all the 2D EKF matrices for all pixels at once
    ev.H = np.zeros((ev.SL // ev.BS, ev.BS // ev.SS, ev.BS))
    ev.R = np.zeros((ev.SL // ev.BS, ev.BS // ev.SS, ev.BS // ev.SS))
    ev.H_indices = np.where(
        [np.kron(np.eye(ev.BS // ev.SS), np.ones(ev.SS)) for _ in range(len(ev.H))])
    ev.R_indices = np.where([np.eye(ev.BS // ev.SS) for _ in range(len(ev.H))])

    # The drift covariance matrix for each pixel (or block of pixels). Needs
    # to be estimated if the model is not perfectly known.
    # Need to convert jacobian from contrast units to counts.
    # if self.drift_dm == 'boston':
    ev.Q = {}
    ev.P = {}
    for iSubband in range(mp.Nsbp):
        G_reordered = ev.G_tot_drift[:, :, iSubband].T
        dm_drift_covariance = np.eye(G_reordered.shape[0]) * (mp.drift.presumed_dm_std ** 2)
        ev.Q[iSubband] = np.array([G_reordered[:, i * ev.BS:(i + 1) * ev.BS].T.dot(dm_drift_covariance).dot(G_reordered[:, i * ev.BS:(i + 1) * ev.BS]) for i in range(ev.SL // ev.BS)]) * sbp_texp[iSubband] * (ev.e_scaling[iSubband] ** 2)

        ev.P[iSubband] = np.zeros_like(ev.Q[iSubband])

    # Initialize state vector
    ev.x_hat = np.zeros((ev.SL, mp.Nsbp))  # Units: sqrt(counts)
    ev.x_hat0 = np.zeros((ev.SL, mp.Nsbp))  # Units: sqrt(counts)

    # Initialize EKF for each subband
    for iSubband in range(mp.Nsbp):
        # Try to use existing estimate if available
        try:
            # MATLAB: E_hat = mp.est.Eest(:,iSubband) * ev.e_scaling(iSubband) * sqrt(mp.tb.info.sbp_texp(iSubband));
            E_hat = mp.est.Eest[:, iSubband] * ev.e_scaling[iSubband] * np.sqrt(sbp_texp[iSubband])
        except (AttributeError, IndexError):
            # MATLAB: E_hat = zeros(ev.SL/ev.BS,1);
            E_hat = np.zeros(int(ev.SL / ev.BS), dtype=complex)

        # Save initial ev state:
        # MATLAB: ev.x_hat0(1:ev.SS:end,iSubband) = real(E_hat) * ev.e_scaling(iSubband) * sqrt(sbp_texp(iSubband));
        ev.x_hat0[0::ev.SS, iSubband] = np.real(E_hat) * ev.e_scaling[iSubband] * np.sqrt(sbp_texp[iSubband])
        ev.x_hat0[1::ev.SS, iSubband] = np.imag(E_hat) * ev.e_scaling[iSubband] * np.sqrt(sbp_texp[iSubband])

        # The EKF state is scaled such that the intensity is measured in photons:
        ev.x_hat[0::ev.SS, iSubband] = np.real(E_hat) * ev.e_scaling[iSubband] * np.sqrt(sbp_texp[iSubband])
        ev.x_hat[1::ev.SS, iSubband] = np.imag(E_hat) * ev.e_scaling[iSubband] * np.sqrt(sbp_texp[iSubband])

    return ev

#
# def initialize_ekf_matrices(mp, ev, sbp_texp):
#     """
#     Initialize EKF matrices for the estimator.
#
#     This function creates and initializes all the matrices needed for
#     the Extended Kalman Filter (EKF) estimator.
#
#     Parameters
#     ----------
#     mp : ModelParameters
#         Object containing optical model parameters
#     ev : FALCO object
#         Structure containing estimation variables.
#     sbp_texp : array_like
#         Exposure times for each subband
#
#     Returns
#     -------
#     ev : FALCO object
#         Updated structure containing estimation variables.
#     """
#     # Below are the definitions of the EKF matrices. There are multiple EKFs
#     # defined in parallel.
#
#     # To get an idea of what the code below does, it's easier to play with
#     # the toy example at https://github.com/leonidprinceton/DHMaintenanceExample
#     ev.SS = 2  # Pixel state size. Two for real and imaginary parts of the electric field.
#                # If incoherent intensity is not ignored, SS should be 3 and the EKF modified accordingly.
#     ev.BS = ev.SS * 1  # EKF block size - number of pixels per EKF (currently 1).
#                         # Computation time grows as the cube of BS.
#
#     ev.SL = ev.SS * mp.Fend.corr.Npix  # Total length of the state vector (all pixels)
#
#     # 3D matrices that include all the 2D EKF matrices for all pixels at once
#     BS_SS_ratio = int(np.floor(ev.BS/ev.SS))
#     SL_BS_ratio = int(np.floor(ev.SL/ev.BS))
#
#     ev.H = np.zeros((BS_SS_ratio, ev.BS, SL_BS_ratio))
#     ev.R = np.zeros((BS_SS_ratio, BS_SS_ratio, SL_BS_ratio))
#
#     # Assemble Q matrix
#     # MATLAB: ev.H_indices = find(kron(eye(floor(ev.BS/ev.SS)),ones(1,ev.SS)).*ones(floor(ev.BS/ev.SS),floor(ev.BS/ev.SS)*ev.SS,floor(ev.SL/ev.BS)));
#     kron_product = np.kron(np.eye(BS_SS_ratio), np.ones((1, ev.SS)))
#     ones_mat = np.ones((BS_SS_ratio, BS_SS_ratio*ev.SS, SL_BS_ratio))
#     ev.H_indices = np.nonzero(kron_product[:, :, np.newaxis] * ones_mat)
#
#     # MATLAB: ev.R_indices = logical(eye(floor(ev.BS/ev.SS)).*ones(floor(ev.BS/ev.SS),floor(ev.BS/ev.SS),floor(ev.SL/ev.BS)));
#     eye_3d = np.eye(BS_SS_ratio)[:, :, np.newaxis] * np.ones((1, 1, SL_BS_ratio))
#     ev.R_indices = eye_3d.astype(bool)
#
#     # The drift covariance matrix for each pixel (or block of pixels).
#     # Needs to be estimated if the model is not perfectly known.
#     # This is a 4D matrix (re | im | px | wavelength).
#     # Need to convert jacobian from contrast units to counts.
#     ev.Q = np.zeros((ev.SS, ev.SS, SL_BS_ratio, mp.Nsbp))
#
#     for iSubband in range(mp.Nsbp):
#         print(f"Assembling Q for subband {iSubband+1}")
#
#         G_reordered = ev.G_tot_drift[:, :, iSubband]
#         dm_drift_covariance = np.eye(G_reordered.shape[1]) * (mp.drift.presumed_dm_std**2)
#
#         # MATLAB: for i = 0:1:floor(ev.SL/ev.BS)-1
#         for i in range(SL_BS_ratio):
#             # MATLAB: ev.Q(:,:,i+1,iSubband) = G_reordered((i)*ev.BS+1:(i+1)*ev.BS,:)*dm_drift_covariance*G_reordered(i*ev.BS+1:(i+1)*ev.BS,:).'*sbp_texp(iSubband)*(ev.e_scaling(iSubband)^2);
#             start_idx = i * ev.BS
#             end_idx = (i + 1) * ev.BS
#
#             G_slice = G_reordered[start_idx:end_idx, :]
#             Q_term = G_slice @ dm_drift_covariance @ G_slice.T * sbp_texp[iSubband] * (ev.e_scaling[iSubband]**2)
#             ev.Q[:, :, i, iSubband] = Q_term
#
#     # Initialize P matrix
#     ev.P = np.zeros_like(ev.Q)  # MATLAB: ev.P = ev.Q*0.0;
#
#     # Initialize state vector
#     ev.x_hat = np.zeros((ev.SL, mp.Nsbp))  # Units: sqrt(counts)
#     ev.x_hat0 = np.zeros((ev.SL, mp.Nsbp))  # Units: sqrt(counts)
#
#     # Initialize EKF for each subband
#     for iSubband in range(mp.Nsbp):
#         # Try to use existing estimate if available
#         try:
#             # MATLAB: E_hat = mp.est.Eest(:,iSubband) * ev.e_scaling(iSubband) * sqrt(mp.tb.info.sbp_texp(iSubband));
#             E_hat = mp.est.Eest[:, iSubband] * ev.e_scaling[iSubband] * np.sqrt(sbp_texp[iSubband])
#         except (AttributeError, IndexError):
#             # MATLAB: E_hat = zeros(ev.SL/ev.BS,1);
#             E_hat = np.zeros(int(ev.SL/ev.BS), dtype=complex)
#
#         # Save initial ev state:
#         # MATLAB: ev.x_hat0(1:ev.SS:end,iSubband) = real(E_hat) * ev.e_scaling(iSubband) * sqrt(sbp_texp(iSubband));
#         ev.x_hat0[0::ev.SS, iSubband] = np.real(E_hat) * ev.e_scaling[iSubband] * np.sqrt(sbp_texp[iSubband])
#         ev.x_hat0[1::ev.SS, iSubband] = np.imag(E_hat) * ev.e_scaling[iSubband] * np.sqrt(sbp_texp[iSubband])
#
#         # The EKF state is scaled such that the intensity is measured in photons:
#         ev.x_hat[0::ev.SS, iSubband] = np.real(E_hat) * ev.e_scaling[iSubband] * np.sqrt(sbp_texp[iSubband])
#         ev.x_hat[1::ev.SS, iSubband] = np.imag(E_hat) * ev.e_scaling[iSubband] * np.sqrt(sbp_texp[iSubband])
#
#     return ev
