import numpy as np
from .est_utils import get_dm_command_vector

def drift_injection(mp, ev):
    """
    Python implementation of FALCO's drift injection function

    Parameters:
        mp: Model parameters dictionary
        ev: Estimation variables dictionary

    Returns:
        mp: Updated model parameters
        ev: Updated estimation variables
    """
    if mp.drift.type.lower() == 'rand_walk':
        # Only apply drift to active actuators:
        if any(mp.dm_drift_ind == 1):

            # Create an empty array for DM1 drift at the final shape directly
            DM1Vdrift = np.zeros((mp.dm1.Nact, mp.dm1.Nact))

            # Generate random normal values for only the active elements
            drift_values = np.random.normal(0, mp.drift.magnitude, mp.dm1.Nele)

            # Convert 1D indices to 2D coordinates
            act_rows, act_cols = np.unravel_index(mp.dm1.act_ele, (mp.dm1.Nact, mp.dm1.Nact))

            # Directly assign values to active elements
            DM1Vdrift[act_rows, act_cols] = drift_values

            # Add this iteration drift to accumulated command
            mp.dm1.V_drift = mp.dm1.V_drift + DM1Vdrift

        else:  # The 'else' block would mean we're only using DM2
            mp.dm1.V_drift = np.zeros_like(mp.dm1.V)

        if any(mp.dm_drift_ind == 2):
            # Create an empty array for DM1 drift at the final shape directly
            DM2Vdrift = np.zeros((mp.dm2.Nact, mp.dm2.Nact))

            # Generate random normal values for only the active elements
            drift_values2 = np.random.normal(0, mp.drift.magnitude, mp.dm2.Nele)

            # Convert 1D indices to 2D coordinates
            act_rows2, act_cols2 = np.unravel_index(mp.dm2.act_ele, (mp.dm2.Nact, mp.dm2.Nact))

            # Directly assign values to active elements
            DM2Vdrift[act_rows2, act_cols2] = drift_values2

            # Add this iteration drift to accumulated command
            mp.dm2.V_drift = mp.dm2.V_drift + DM2Vdrift

        else:  # The 'else' block would mean we're only using DM1
            mp.dm2.V_drift = np.zeros_like(mp.dm2.V)

    # TODO: eventually move estimator reset to different function and put in main loop
    # before estimator
    if any(np.isin(mp.est.itr_reset, ev.Itr)):
        if not hasattr(mp.dm1, 'dV'):
            mp.dm1.dV = np.zeros((mp.dm1.Nact, mp.dm1.Nact))
        if not hasattr(mp.dm2, 'dV'):
            mp.dm2.dV = np.zeros((mp.dm2.Nact, mp.dm2.Nact))

        efc_command = get_dm_command_vector(mp, mp.dm1.dV, mp.dm2.dV)

        # Check if sim mode to avoid calling tb obj in sim mode
        if mp.flagSim:
            sbp_texp = mp.detector.tExpUnprobedVec  # exposure times for non-pairwise-probe images in each subband
        else:
            sbp_texp = mp.tb.info.sbp_texp

        for iSubband in range(mp.Nsbp):
            ev.x_hat[:, iSubband] = ev.x_hat[:, iSubband] + \
                                    (ev.G_tot[:, :, iSubband] * ev.e_scaling[iSubband]) * \
                                    np.sqrt(sbp_texp[iSubband]) * efc_command

        mp.dm1.V_shift = mp.dm1.dV
        mp.dm2.V_shift = mp.dm2.dV

        # mp.dm1.V_dz = mp.dm1.V_dz + mp.dm1.dV
        # mp.dm2.V_dz = mp.dm2.V_dz + mp.dm2.dV

        mp.dm1.dV = np.zeros_like(mp.dm1.V_dz)
        mp.dm2.dV = np.zeros_like(mp.dm2.V_dz)

    # TODO: save each drift command
    return mp, ev


