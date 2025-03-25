"""
Extended Kalman Filter (EKF) estimator for maintaining dark holes.

This module implements the EKF maintenance estimator for FALCO Python,
converted from the MATLAB implementation.
"""

import os
import time
import numpy as np
import astropy.io.fits as fits
import falco





def get_open_loop_data(mp, ev):
    """
    Get open loop data by removing control and dither from DM command.
    
    Parameters
    ----------
    mp : ModelParameters
        Object containing optical model parameters
    ev : FALCO object
        Structure containing estimation variables.
        
    Returns
    -------
    mp : ModelParameters
        Updated object
    ev : FALCO object
        Updated structure
    """
    # If DM is used for drift and control, apply V_dz and Vdrift, if DM is only
    # used for control, apply V_dz
    if (np.any(mp.dm_drift_ind == 1) and np.any(mp.dm_ind == 1)) or np.any(mp.dm_drift_ind == 1):
        mp.dm1.V = mp.dm1.V_dz + mp.dm1.V_drift
        mp.dm1 = falco.dm.enforce_constraints(mp.dm1)
    elif np.any(mp.dm_ind == 1) or np.any(mp.dm_ind_static == 1):
        mp.dm1.V = mp.dm1.V_dz
        mp.dm1 = falco.dm.enforce_constraints(mp.dm1)
    
    if (np.any(mp.dm_drift_ind == 2) and np.any(mp.dm_ind == 2)) or np.any(mp.dm_drift_ind == 2):
        mp.dm2.V = mp.dm2.V_dz + mp.dm2.V_drift
        mp.dm2 = falco.dm.enforce_constraints(mp.dm2)
    elif np.any(mp.dm_ind == 2) or np.any(mp.dm_ind_static == 2):
        mp.dm2.V = mp.dm2.V_dz
        mp.dm2 = falco.dm.enforce_constraints(mp.dm2)
    
    # Do safety check for pinned actuators
    # print('OL DM safety check.')
    # ev = pinned_act_safety_check(mp, ev)
    #
    if ev.Itr == 1:
        ev.IOLScoreHist = np.zeros((mp.Nitr, mp.Nsbp))
    
    I_OL = np.zeros((ev.imageArray.shape[0], ev.imageArray.shape[1], mp.Nsbp))
    for iSubband in range(mp.Nsbp):
        I0 = falco.imaging.get_sbp_image(mp, iSubband)
        I_OL[:, :, iSubband] = I0
        
        ev.IOLScoreHist[ev.Itr, iSubband] = np.mean(I0[mp.Fend.score.maskBool])
    
    ev.normI_OL_sbp = I_OL
    
    print(f"mean OL contrast: {np.mean(ev.IOLScoreHist[ev.Itr, :])}")
    return mp, ev


def save_ekf_data(mp, ev, DM1Vdither, DM2Vdither):
    """
    Save EKF data to FITS files.
    
    Parameters
    ----------
    mp : ModelParameters
        Object containing optical model parameters
    ev : FALCO object
        Structure containing estimation variables.
    DM1Vdither, DM2Vdither : ndarray
        Dither commands for DM1 and DM2
    """
    drift = np.zeros((mp.dm1.Nact, mp.dm1.Nact, len(mp.dm_drift_ind)))
    dither = np.zeros((mp.dm1.Nact, mp.dm1.Nact, len(mp.dm_ind)))
    efc = np.zeros((mp.dm1.Nact, mp.dm1.Nact, len(mp.dm_ind)))
    
    if mp.dm_drift_ind[0] == 1:
        drift[:, :, 0] = mp.dm1.V_drift
    if mp.dm_drift_ind[0] == 2:
        drift[:, :, 0] = mp.dm2.V_drift
    else:
        drift[:, :, 1] = mp.dm2.V_drift
    
    if mp.dm_ind[0] == 1:
        dither[:, :, 0] = DM1Vdither
    if mp.dm_ind[0] == 2:
        dither[:, :, 0] = DM2Vdither
    else:
        dither[:, :, 1] = DM2Vdither
    
    if mp.dm_ind[0] == 1:
        efc[:, :, 0] = mp.dm1.dV
    if mp.dm_ind[0] == 2:
        efc[:, :, 0] = mp.dm2.dV
    else:
        efc[:, :, 1] = mp.dm2.dV
    
    # TODO: move to plot_progress_iact
    fits.writeto(os.path.join(mp.path.config, f'drift_command_it{ev.Itr}.fits'), drift, overwrite=True)
    fits.writeto(os.path.join(mp.path.config, f'dither_command_it{ev.Itr}.fits'), dither, overwrite=True)
    fits.writeto(os.path.join(mp.path.config, f'efc_command_it{ev.Itr-1}.fits'), efc, overwrite=True)
    
    if ev.Itr == 1:
        dz_init = np.zeros((mp.dm1.Nact, mp.dm1.Nact, len(mp.dm_ind)))
        if mp.dm_ind[0] == 1:
            dz_init[:, :, 0] = mp.dm1.V_dz
        if mp.dm_ind[0] == 2:
            dz_init[:, :, 0] = mp.dm2.V_dz
        else:
            dz_init[:, :, 1] = mp.dm2.V_dz
        
        fits.writeto(os.path.join(mp.path.config, 'dark_zone_command_0_pwp.fits'), dz_init, overwrite=True)


def est_ekf_dzm(mp, ev, jacStruct=None):
    """
    EKF maintenance estimator for FALCO.
    
    Parameters
    ----------
    mp : ModelParameters
        Object containing optical model parameters
    ev : FALCO object
        Structure containing estimation variables.
    jacStruct : ModelParameters, optional
        Structure containing control Jacobians for each specified DM.
    
    Returns
    -------
    ev : FALCO object
        Updated structure containing estimation variables.
    """
    start_time = time.time()
    
    # Get iteration number
    Itr = ev.Itr
    
    # Get which DM to use for probing
    whichDM = mp.est.probe.whichDM
    
    if not isinstance(mp.est.probe, falco.config.Probe):
        raise TypeError('mp.est.probe must be an instance of class Probe')
    
    # Augment which DMs are used if the probing DM isn't used for control
    dm_ind = mp.dm_ind.copy()
    if whichDM == 1 and not np.any(mp.dm_ind == 1):
        mp.dm_ind = np.append(mp.dm_ind, 1)
    elif whichDM == 2 and not np.any(mp.dm_ind == 2):
        mp.dm_ind = np.append(mp.dm_ind, 2)
    
    # Select number of actuators across based on chosen DM for the probing
    if whichDM == 1:
        Nact = mp.dm1.Nact
    elif whichDM == 2:
        Nact = mp.dm2.Nact
    
    # Select number of actuators being changed
    Nact_delta = 0
    if np.any(mp.dm_ind == 1) or (whichDM == 1):
        Nact_delta += mp.dm1.Nele
    if np.any(mp.dm_ind == 2) or (whichDM == 2):
        Nact_delta += mp.dm2.Nele
    
    # Initialize output arrays
    ev.imageArray = np.zeros((mp.Fend.Neta, mp.Fend.Nxi, 1, mp.Nsbp))
    ev.Eest = np.zeros((mp.Fend.corr.Npix, mp.Nsbp*mp.compact.star.count), dtype=complex)
    ev.IincoEst = np.zeros((mp.Fend.corr.Npix, mp.Nsbp*mp.compact.star.count))
    ev.IprobedMean = 0
    ev.Im = np.zeros((mp.Fend.Neta, mp.Fend.Nxi))
    
    if whichDM == 1:
        ev.dm1_Vall = np.zeros((mp.dm1.Nact, mp.dm1.Nact, 1, mp.Nsbp))
    if whichDM == 2:
        ev.dm2_Vall = np.zeros((mp.dm2.Nact, mp.dm2.Nact, 1, mp.Nsbp))
    
    # Get dither command
    # Set random number generator seed
    # Dither commands get re-used every dither_cycle_iters iterations
    if (Itr - 1) % mp.est.dither_cycle_iters == 0 or Itr == 0:
        ev.dm1_seed_num = 0
        ev.dm2_seed_num = 1000  # Don't want same random commands on DM1 and DM2
        print(f"Dither random seed reset at iteration {Itr}")
    else:
        ev.dm1_seed_num += 1
        ev.dm2_seed_num += 1
    
    # Generate random dither command
    if np.any(mp.dm_ind == 1):
        # TODO: FIX ALL OF THE THINGS TO MATCH THIS
        np.random.seed(ev.dm1_seed_num)
        # DM1Vdither = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
        DM1Vdither = np.zeros((mp.dm1.Nact**2, 1))
        DM1Vdither[np.reshape(mp.dm1.act_ele, (1, mp.dm1.Nele))] = np.random.normal(0, mp.est.dither, mp.dm1.Nele).reshape((mp.dm1.Nele, 1))
        DM1Vdither = np.reshape(DM1Vdither, (mp.dm1.Nact, mp.dm1.Nact))
    else:
        DM1Vdither = np.zeros_like(mp.dm1.V)  # The 'else' block would mean we're only using DM2
    
    if np.any(mp.dm_ind == 2):
        np.random.seed(ev.dm2_seed_num)
        DM2Vdither = np.zeros((mp.dm2.Nact, mp.dm2.Nact))
        DM2Vdither[mp.dm2.act_ele] = np.random.normal(0, mp.est.dither, mp.dm2.Nele)
    else:
        DM2Vdither = np.zeros_like(mp.dm2.V)  # The 'else' block would mean we're only using DM1
    
    dither = get_dm_command_vector(mp, DM1Vdither, DM2Vdither)
    
    # Set total command for estimator image
    if Itr > 1:
        if not hasattr(mp.dm1, 'dV'):
            mp.dm1.dV = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
        if not hasattr(mp.dm2, 'dV'):
            mp.dm2.dV = np.zeros((mp.dm2.Nact, mp.dm2.Nact))
        efc_command = get_dm_command_vector(mp, mp.dm1.dV, mp.dm2.dV)
    else:
        efc_command = np.zeros_like(dither)
        mp.dm1.dV = np.zeros_like(DM1Vdither)
        mp.dm2.dV = np.zeros_like(DM2Vdither)
    
    # Generate command to apply to DMs
    # Note if dm_drift_ind != i, the command is set to zero in falco_drift_injection
    mp = set_constrained_full_command(mp, DM1Vdither, DM2Vdither)
    
    # # Do safety check to make sure no actuators are pinned
    # ev = pinned_act_safety_check(mp, ev)
    
    closed_loop_command = dither + efc_command + get_dm_command_vector(mp, mp.dm1.V_shift, mp.dm2.V_shift)
    
    # Get images
    y_measured = np.zeros((mp.Fend.corr.Npix, mp.Nsbp))
    for iSubband in range(mp.Nsbp):
        ev.imageArray[:, :, 0, iSubband] = falco.imaging.get_sbp_image(mp, iSubband)
        I0 = ev.imageArray[:, :, 0, iSubband] * ev.peak_psf_counts[iSubband]
        y_measured[:, iSubband] = I0[mp.Fend.corr.maskBool]
    
    # Perform the estimation
    ev = ekf_estimate(mp, ev, jacStruct, y_measured, closed_loop_command, DM1Vdither, DM2Vdither)
    
    # Reset DM commands
    mp = set_constrained_full_command(mp, DM1Vdither, DM2Vdither)
    
    # Save out the estimate
    if mp.flagSim:
        sbp_texp = mp.detector.tExpUnprobedVec  # exposure times for non-pairwise-probe images in each subband
    else:
        sbp_texp = mp.tb.info.sbp_texp
    
    ev.Im = np.zeros((mp.Fend.Neta, mp.Fend.Nxi))
    for iSubband in range(mp.Nsbp):
        ev.Eest[:, iSubband] = (ev.x_hat[0::2, iSubband] + 1j*ev.x_hat[1::2, iSubband]) / (ev.e_scaling[iSubband] * np.sqrt(sbp_texp[iSubband]))
        if np.any(mp.dm_ind == 1):
            ev.dm1_Vall[:, :, 0, iSubband] = mp.dm1.V
        if np.any(mp.dm_ind == 2):
            ev.dm2_Vall[:, :, 0, iSubband] = mp.dm2.V
        
        ev.Im += mp.sbp_weights[iSubband] * ev.imageArray[:, :, 0, iSubband]
    
    I0vec = y_measured / ev.peak_psf_counts[:, np.newaxis].T
    ev.IincoEst = I0vec - np.abs(ev.Eest)**2  # incoherent light
    
    # Other data to save out
    ev.ampSqMean = np.mean(I0vec)  # Mean probe intensity
    
    # ev.Im = ev.imageArray[:, :, 0, mp.si_ref]
    ev.IprobedMean = np.mean(ev.imageArray)
    
    mp.isProbing = False
    
    # If itr = itr_OL get OL data. NOTE THIS SHOULD BE BEFORE THE
    # "Remove control from DM command so that controller images are correct" block
    if np.any(mp.est.itr_ol == ev.Itr):
        mp, ev = get_open_loop_data(mp, ev)
    else:
        ev.IOLScoreHist[ev.Itr, :] = ev.IOLScoreHist[ev.Itr-1, :]
    
    # Remove control from DM command so that controller images are correct
    if np.any(mp.dm_ind == 1):
        mp.dm1.V = mp.dm1.V_dz + mp.dm1.V_drift + DM1Vdither + mp.dm1.V_shift
        mp.dm1 = falco.dm.enforce_constraints(mp.dm1)
    elif np.any(mp.dm_ind_static == 1):
        mp.dm1.V = mp.dm1.V_dz
        mp.dm1 = falco.dm.enforce_constraints(mp.dm1)
    
    if np.any(mp.dm_ind == 2):
        mp.dm2.V = mp.dm2.V_dz + mp.dm2.V_drift + DM2Vdither + mp.dm2.V_shift
        mp.dm2 = falco.dm.enforce_constraints(mp.dm2)
    elif np.any(mp.dm_ind_static == 2):
        mp.dm2.V = mp.dm2.V_dz
        mp.dm2 = falco.dm.enforce_constraints(mp.dm2)
    
    save_ekf_data(mp, ev, DM1Vdither, DM2Vdither)
    
    print(f" done. Time: {time.time() - start_time:.3f}")
    
    # Restore original dm_ind value
    mp.dm_ind = dm_ind
    
    return ev, mp


def ekf_estimate(mp, ev, jacStruct, y_measured, closed_loop_command, DM1Vdither, DM2Vdither):
    """
    Perform the EKF estimation.
    
    Parameters
    ----------
    mp : ModelParameters
        Object containing optical model parameters
    ev : FALCO object
        Structure containing estimation variables.
    jacStruct : ModelParameters
        Structure containing control Jacobians for each specified DM.
    y_measured : ndarray
        Measured intensity values
    closed_loop_command : ndarray
        Combined command vector
    DM1Vdither, DM2Vdither : ndarray
        Dither commands for DM1 and DM2
        
    Returns
    -------
    ev : FALCO object
        Updated structure containing estimation variables.
    """
    # Estimation part. All EKFs are advanced in parallel
    if mp.flagSim:
        sbp_texp = mp.detector.tExpUnprobedVec
    else:
        sbp_texp = mp.tb.info.sbp_texp
    
    for iSubband in range(mp.Nsbp):
        # Get gdu
        gdu = get_gdu(mp, ev, iSubband, y_measured, closed_loop_command, DM1Vdither, DM2Vdither)
        
        # Estimate of the closed loop electric field
        x_hat_CL = ev.x_hat[:, iSubband] + gdu * ev.e_scaling[iSubband] * np.sqrt(sbp_texp[iSubband])
        
        # Estimate of the measurement
        y_hat = x_hat_CL[0::ev.SS]**2 + x_hat_CL[1::ev.SS]**2 + (mp.est.dark_current * sbp_texp[iSubband])
        
        # Update R matrix
        ev.R[ev.R_indices] = y_hat.reshape(ev.R[ev.R_indices].shape) + (mp.est.read_noise)**2
        
        # Update H matrix
        ev.H[ev.H_indices] = 2 * x_hat_CL
        
        # Transpose H matrix
        H_T = np.transpose(ev.H, (1, 0, 2))
        
        # Update P
        ev.P[:, :, :, iSubband] = ev.P[:, :, :, iSubband] + ev.Q[:, :, :, iSubband]
        
        # Matrix multiplications for Kalman gain calculation
        P_H_T = np.matmul(ev.P[:, :, :, iSubband], H_T)
        S = np.matmul(ev.H, P_H_T) + ev.R
        S_inv = np.linalg.inv(S)
        
        # Calculate Kalman gain
        K = np.matmul(P_H_T, S_inv)
        
        # Update P
        ev.P[:, :, :, iSubband] = ev.P[:, :, :, iSubband] - np.matmul(P_H_T, np.transpose(K, (1, 0, 2)))
        
        # EKF correction
        dy = y_measured[:, iSubband] - y_hat
        
        # Stack the measurement differences
        dy_hat_stacked = np.zeros_like(K)
        dy_hat_stacked[0, :, :] = dy
        dy_hat_stacked[1, :, :] = dy
        
        # Apply Kalman gain
        dx_hat_stacked = K * dy_hat_stacked
        
        # Unstack the state corrections
        dx_hat = np.zeros_like(x_hat_CL)
        dx_hat[0::ev.SS] = dx_hat_stacked[0, :, :]
        dx_hat[1::ev.SS] = dx_hat_stacked[1, :, :]
        
        # Update state estimate
        ev.x_hat[:, iSubband] = ev.x_hat[:, iSubband] + dx_hat
    
    return ev


def get_gdu(mp, ev, iSubband, y_measured, closed_loop_command, DM1Vdither, DM2Vdither):
    """
    Get the gradient of the electric field with respect to DM commands.
    
    Parameters
    ----------
    mp : ModelParameters
        Object containing optical model parameters
    ev : FALCO object
        Structure containing estimation variables.
    iSubband : int
        Subband index
    y_measured : ndarray
        Measured intensity values
    closed_loop_command : ndarray
        Combined command vector
    DM1Vdither, DM2Vdither : ndarray
        Dither commands for DM1 and DM2
        
    Returns
    -------
    gdu : ndarray
        Gradient vector
    """
    modvar = falco.config.ModelVariables()
    modvar.starIndex = 0
    modvar.whichSource = 'star'
    
    if mp.est.flagUseJacAlgDiff:
        gdu = ev.G_tot_cont[:, :, iSubband] @ closed_loop_command
    else:
        # For unprobed field based on model
        if np.any(mp.dm_ind == 1) or np.any(mp.dm_ind_static == 1):
            mp.dm1.V = mp.dm1.V_dz
            mp.dm1 = falco.dm.enforce_constraints(mp.dm1)
        if np.any(mp.dm_ind == 2) or np.any(mp.dm_ind_static == 2):
            mp.dm2.V = mp.dm2.V_dz
            mp.dm2 = falco.dm.enforce_constraints(mp.dm2)
        
        E0 = falco.model.compact(mp, modvar)
        E0vec = E0[mp.Fend.corr.maskBool]
        
        # For probed fields based on model
        gdu = np.zeros(2 * len(y_measured[:, iSubband]))
        
        if np.any(mp.dm_ind == 1) or np.any(mp.dm_ind_static == 1):
            mp.dm1.V = mp.dm1.V_dz + mp.dm1.dV + DM1Vdither + mp.dm1.V_shift
            mp.dm1 = falco.dm.enforce_constraints(mp.dm1)
        if np.any(mp.dm_ind == 2) or np.any(mp.dm_ind_static == 2):
            mp.dm2.V = mp.dm2.V_dz + mp.dm2.dV + DM2Vdither + mp.dm2.V_shift
            mp.dm2 = falco.dm.enforce_constraints(mp.dm2)
        
        Edither = falco.model.compact(mp, modvar)
        Edithervec = Edither[mp.Fend.corr.maskBool]
        
        gdu_comp = Edithervec - E0vec
        
        gdu[0::2] = np.real(gdu_comp)
        gdu[1::2] = np.imag(gdu_comp)
    
    return gdu


def get_dm_command_vector(mp, command1, command2):
    """
    Combine DM commands into a single vector.
    
    Parameters
    ----------
    mp : ModelParameters
        Object containing optical model parameters
    command1, command2 : ndarray
        Commands for DM1 and DM2
        
    Returns
    -------
    comm_vector : ndarray
        Combined command vector
    """
    if np.any(mp.dm_ind == 1):
        comm1 = command1.flat[mp.dm1.act_ele]
    else:
        comm1 = np.array([])  # The 'else' block would mean we're only using DM2
    
    if np.any(mp.dm_ind == 2):
        comm2 = command2.flat[mp.dm2.act_ele]
    else:
        comm2 = np.array([])
    
    comm_vector = np.concatenate((comm1, comm2))
    
    return comm_vector


def set_constrained_full_command(mp, DM1Vdither, DM2Vdither):
    """
    Set constrained voltage commands to the DMs.
    
    Parameters
    ----------
    mp : ModelParameters
        Object containing optical model parameters
    DM1Vdither, DM2Vdither : ndarray
        Dither commands for DM1 and DM2
        
    Returns
    -------
    mp : ModelParameters
        Updated object
    """
    if np.any(mp.dm_ind == 1):
        # note falco_enforce_constraints does not apply the command to the DM
        mp.dm1.V = mp.dm1.V_dz + mp.dm1.V_drift + mp.dm1.dV + DM1Vdither + mp.dm1.V_shift
        mp.dm1 = falco.dm.enforce_constraints(mp.dm1)
    elif np.any(mp.dm_drift_ind == 1):
        mp.dm1.V = mp.dm1.V_dz + mp.dm1.V_drift
        mp.dm1 = falco.dm.enforce_constraints(mp.dm1)
    elif np.any(mp.dm_ind_static == 1):
        mp.dm1.V = mp.dm1.V_dz
        mp.dm1 = falco.dm.enforce_constraints(mp.dm1)
    
    if np.any(mp.dm_ind == 2):
        mp.dm2.V = mp.dm2.V_dz + mp.dm2.V_drift + mp.dm2.dV + DM2Vdither + mp.dm2.V_shift
        mp.dm2 = falco.dm.enforce_constraints(mp.dm2)
    elif np.any(mp.dm_drift_ind == 2):
        mp.dm2.V = mp.dm2.V_dz + mp.dm2.V_drift
        mp.dm2 = falco.dm.enforce_constraints(mp.dm2)
    elif np.any(mp.dm_ind_static == 2):
        mp.dm2.V = mp.dm2.V_dz
        mp.dm2 = falco.dm.enforce_constraints(mp.dm2)
    
    return mp


def pinned_act_safety_check(mp, ev):
    """
    Check for newly pinned actuators.
    
    Parameters
    ----------
    mp : ModelParameters
        Object containing optical model parameters
    ev : FALCO object
        Structure containing estimation variables.
        
    Returns
    -------
    ev : FALCO object
        Updated structure
    """
    # Update new pinned actuators
    if np.any(mp.dm_ind == 1) or np.any(mp.dm_drift_ind == 1):
        ev.dm1_new_pinned_actuators = np.setdiff1d(mp.dm1.pinned, ev.dm1_initial_pinned_actuators)
        mask = np.isin(ev.dm1_new_pinned_actuators, mp.dm1.act_ele)
        ev.dm1_act_ele_pinned = ev.dm1_new_pinned_actuators[mask]
    
    if np.any(mp.dm_ind == 2) or np.any(mp.dm_drift_ind == 2):
        ev.dm2_new_pinned_actuators = np.setdiff1d(mp.dm2.pinned, ev.dm2_initial_pinned_actuators)
        mask = np.isin(ev.dm2_new_pinned_actuators, mp.dm2.act_ele)
        ev.dm2_act_ele_pinned = ev.dm2_new_pinned_actuators[mask]
    
    # Check that no new actuators have been pinned
    if len(ev.dm1_new_pinned_actuators) > 0 or len(ev.dm2_new_pinned_actuators) > 0:
        # Print error warning
        print(f"New DM1 pinned: [{','.join(map(str, ev.dm1_new_pinned_actuators))}]")
        print(f"New DM2 pinned: [{','.join(map(str, ev.dm2_new_pinned_actuators))}]")
        
        # If actuators are used in jacobian, quit
        if len(ev.dm1_act_ele_pinned) > 0 or len(ev.dm2_act_ele_pinned) > 0:
            fits.writeto(os.path.join(mp.path.config, f'/ev_exit_{ev.Itr}.mat'), ev, overwrite=True)
            fits.writeto(os.path.join(mp.path.config, f'/mp_exit_{ev.Itr}.mat'), mp, overwrite=True)
            
            raise RuntimeError('New actuators in act_ele pinned, exiting loop')
    
    return ev




# def initialize_ekf_matrices(mp, ev, sbp_texp):
#     """
#     Initialize EKF matrices for the estimator.
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
#
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
#     # Create kronecker product for H_indices
#     eye_mat = np.eye(BS_SS_ratio)
#     ones_vec = np.ones((1, ev.SS))
#     kron_product = np.kron(eye_mat, ones_vec)
#
#     ones_mat1 = np.ones((BS_SS_ratio, BS_SS_ratio*ev.SS, SL_BS_ratio))
#     # Find indices where product is non-zero
#     ev.H_indices = np.nonzero(kron_product[:, :, np.newaxis] * ones_mat1)
#
#     # For R_indices
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
#         print(f"assembling Q for subband {iSubband+1}")
#
#         G_reordered = ev.G_tot_drift[:, :, iSubband]
#         dm_drift_covariance = np.eye(G_reordered.shape[1]) * (mp.drift.presumed_dm_std**2)
#
#         for i in range(SL_BS_ratio):
#             start_idx = i * ev.BS
#             end_idx = (i + 1) * ev.BS
#
#             G_slice = G_reordered[start_idx:end_idx, :]
#             Q_term = G_slice @ dm_drift_covariance @ G_slice.T * sbp_texp[iSubband] * (ev.e_scaling[iSubband]**2)
#             ev.Q[:, :, i, iSubband] =