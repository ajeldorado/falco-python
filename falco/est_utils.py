
import falco
import numpy as np


def rearrange_jacobians(mp, jacStruct, dm_inds):
    """
    Rearrange Jacobians for EKF.

    Parameters
    ----------
    mp : ModelParameters
        Object containing optical model parameters
    jacStruct : ModelParameters
        Structure containing control Jacobians for each specified DM.
    dm_inds : array_like
        Indices of DMs to include in the Jacobian

    Returns
    -------
    G_tot : ndarray
        Rearranged Jacobian with real and imaginary parts separated.
    """

    if np.any(dm_inds == 1):
        G1 = np.zeros((2 * jacStruct.G1.shape[0], mp.dm1.Nele, mp.Nsbp))
    else:
        G1 = np.array([])

    if np.any(dm_inds == 2):
        G2 = np.zeros((2 * jacStruct.G2.shape[0], mp.dm2.Nele, mp.Nsbp))
    else:
        G2 = np.array([])

    # Set up jacobian so real and imag components alternate and jacobian from each DM is stacked
    for iSubband in range(mp.Nsbp):

        if np.any(dm_inds == 1):
            G1_comp = jacStruct.G1[:, :, iSubband]
            G1_split = np.zeros((2 * jacStruct.G1.shape[0], mp.dm1.Nele))
            G1_split[0::2, :] = np.real(G1_comp)
            G1_split[1::2, :] = np.imag(G1_comp)

            G1[:, :, iSubband] = G1_split

        if np.any(dm_inds == 2):
            G2_comp = jacStruct.G2[:, :, iSubband]
            G2_split = np.zeros((2 * jacStruct.G2.shape[0], mp.dm2.Nele))
            G2_split[0::2, :] = np.real(G2_comp)
            G2_split[1::2, :] = np.imag(G2_comp)

            G2[:, :, iSubband] = G2_split

    # Combine the Jacobians
    if np.any(dm_inds == 1) and np.any(dm_inds == 2):
        G_tot = np.concatenate((G1, G2), axis=1)
    elif np.any(dm_inds == 1):
        G_tot = G1
    elif np.any(dm_inds == 2):
        G_tot = G2
    else:
        G_tot = np.array([])

    return G_tot


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
        comm1 = np.ravel(command1)[mp.dm1.act_ele]
    else:
        comm1 = np.array([])  # The 'else' block would mean we're only using DM2

    if np.any(mp.dm_ind == 2):
        comm2 = np.ravel(command2)[mp.dm2.act_ele]
    else:
        comm2 = np.array([])

    comm_vector = np.concatenate((comm1, comm2))

    return comm_vector