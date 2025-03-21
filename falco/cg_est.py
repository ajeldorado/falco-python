import numpy as np
import falco


def falco_est_ekf_maintenance(mp, ev, jacStruct=None):
    """
    Extended Kalman Filter (EKF) maintenance estimation function.
    """
    Itr = ev.Itr
    whichDM = mp.est.probe.whichDM

    if not isinstance(mp.est.probe, falco.config.Probe):
        raise TypeError("mp.est.probe must be an instance of Probe")

    if jacStruct is None:
        jacStruct = {}

    if whichDM == 1 and 1 not in mp.dm_ind:
        mp.dm_ind.append(1)
    elif whichDM == 2 and 2 not in mp.dm_ind:
        mp.dm_ind.append(2)

    Nact = mp.dm1.Nact if whichDM == 1 else mp.dm2.Nact

    ev.imageArray = np.zeros((mp.Fend.Neta, mp.Fend.Nxi, 1, mp.Nsbp))
    ev.Eest = np.zeros((mp.Fend.corr.Npix, mp.Nsbp * mp.compact.star.count))
    ev.IincoEst = np.zeros((mp.Fend.corr.Npix, mp.Nsbp * mp.compact.star.count))
    ev.IprobedMean = 0
    ev.Im = np.zeros((mp.Fend.Neta, mp.Fend.Nxi))
    if whichDM == 1:
        ev.dm1.Vall = np.zeros((mp.dm1.Nact, mp.dm1.Nact, 1, mp.Nsbp))
    if whichDM == 2:
        ev.dm2.Vall = np.zeros((mp.dm2.Nact, mp.dm2.Nact, 1, mp.Nsbp))

    np.random.seed(Itr)
    DM1Vdither = np.random.normal(0, mp.est.dither, (mp.dm1.Nact, mp.dm1.Nact)) if 1 in mp.dm_ind else np.zeros_like(
        mp.dm1.V)
    DM2Vdither = np.random.normal(0, mp.est.dither, (mp.dm2.Nact, mp.dm2.Nact)) if 2 in mp.dm_ind else np.zeros_like(
        mp.dm2.V)

    mp = set_constrained_full_command(mp, DM1Vdither, DM2Vdither)

    y_measured = np.zeros((mp.Fend.corr.Npix, mp.Nsbp))
    for iSubband in range(mp.Nsbp):
        ev.imageArray[:, :, 0, iSubband] = falco.imaging.get_sbp_image(mp, iSubband)
        I0 = ev.imageArray[:, :, 0, iSubband] * ev.peak_psf_counts[iSubband]
        y_measured[:, iSubband] = I0[mp.Fend.corr.mask]

    ev = ekf_estimate(mp, ev, jacStruct, y_measured, DM1Vdither, DM2Vdither)

    mp = set_constrained_full_command(mp, DM1Vdither, DM2Vdither)

    return ev


def ekf_estimate(mp, ev, jacStruct, y_measured, DM1Vdither, DM2Vdither):
    """
    Perform Extended Kalman Filter (EKF) estimation.
    """
    for iSubband in range(mp.Nsbp):
        gdu = get_gdu(mp, ev, iSubband, y_measured, DM1Vdither, DM2Vdither)
        x_hat_CL = ev.x_hat[:, iSubband] + gdu * ev.e_scaling[iSubband] * np.sqrt(mp.tb.info.sbp_texp[iSubband])
        y_hat = x_hat_CL[::ev.SS] ** 2 + x_hat_CL[1::ev.SS] ** 2 + (mp.est.dark_current * mp.tb.info.sbp_texp[iSubband])
        ev.x_hat[:, iSubband] += np.linalg.pinv(ev.H) @ (y_measured[:, iSubband] - y_hat)
    return ev


def get_gdu(mp, ev, iSubband, y_measured, DM1Vdither, DM2Vdither):
    """
    Compute gain disturbance update (GDU) for EKF.
    """
    E0 = falco.model.compact(mp, falco.config.ModelVariables())
    E0vec = E0[mp.Fend.corr.maskBool]
    Edither = falco.model.compact(mp, falco.config.ModelVariables())
    Edithervec = Edither[mp.Fend.corr.maskBool]
    gdu_comp = Edithervec - E0vec
    return np.concatenate([np.real(gdu_comp), np.imag(gdu_comp)])


def set_constrained_full_command(mp, DM1Vdither, DM2Vdither):
    """
    Apply dither and drift commands to deformable mirrors.
    """
    if 1 in mp.dm_ind:
        mp.dm1.V = mp.dm1.V_dz + mp.dm1.V_drift + DM1Vdither
    if 2 in mp.dm_ind:
        mp.dm2.V = mp.dm2.V_dz + mp.dm2.V_drift + DM2Vdither
    return mp


def pinned_act_safety_check(mp, ev):
    """
    Check for newly pinned actuators and update status.
    """
    if 1 in mp.dm_ind:
        ev.dm1.new_pinned_actuators = set(mp.dm1.pinned) - set(ev.dm1.initial_pinned_actuators)
    if 2 in mp.dm_ind:
        ev.dm2.new_pinned_actuators = set(mp.dm2.pinned) - set(ev.dm2.initial_pinned_actuators)
    return ev


def initialize_ekf_matrices(mp, ev):
    """
    Initialize EKF matrices for estimation.
    """
    ev.SS = 2
    ev.BS = ev.SS * 1
    ev.SL = ev.SS * mp.Fend.corr.Npix
    ev.H = np.zeros((ev.BS // ev.SS, ev.BS, ev.SL // ev.BS))
    ev.R = np.zeros((ev.BS // ev.SS, ev.BS // ev.SS, ev.SL // ev.BS))
    ev.x_hat = np.zeros((ev.SL, mp.Nsbp))
    return ev
