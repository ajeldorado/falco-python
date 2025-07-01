import numpy as np

# Copyright 2018-2021, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged. Any
# commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# -------------------------------------------------------------------------
# 
# Update the DM gain map based on the total voltage command.
# Accounts for nonlinear displacement vs voltage.
#
# INPUTS
# ------
# dm : structure of DM parameters. Is either mp.dm1 or mp.dm2
#
# OUTPUTS
# -------
# dm : structure of DM parameters. Is either mp.dm1 or mp.dm2


def falco_update_dm_gain_map(dm):
    """
    Update the deformable mirror (DM) gain map based on the specified fit type.

    Parameters:
    -----------
    dm : object
        A deformable mirror object with various attributes

    Returns:
    --------
    dm : object
        The updated deformable mirror object
    """

    if dm.fitType.lower() in ['linear', 'poly1']:
        # No change to dm.VtoH
        pass

    elif dm.fitType.lower() in ['quadratic', 'poly2']:
        if not hasattr(dm, 'p1') or not hasattr(dm, 'p2') or not hasattr(dm, 'p3'):
            error_msg = ("The fields p1, p2, and p3 must exist when dm.fitType == 'quadratic'.\n"
                         "Those fields satisfy the formula:\n"
                         "height = p1*V*V + p2*V + p3")
            raise ValueError(error_msg)

        Vtotal = dm.V + dm.biasMap
        dm.VtoH = 2 * dm.p1 * Vtotal + dm.p2

    elif dm.fitType.lower() == 'fourier2':
        if (not hasattr(dm, 'a0') or not hasattr(dm, 'a1') or not hasattr(dm, 'a2') or
                not hasattr(dm, 'b1') or not hasattr(dm, 'b2') or not hasattr(dm, 'w')):
            error_msg = ("The fields a0, a1, a2, b1, b2, and w must exist when dm.fitType == 'fourier2'.\n"
                         "Those fields satisfy the formula:\n"
                         "height = a0 + a1*cos(V*w) + b1*sin(V*w) + a2*cos(2*V*w) + b2*sin(2*V*w)")
            raise ValueError(error_msg)

        Vtotal = dm.V + dm.biasMap
        dm.VtoH = dm.w * (-dm.a1 * np.sin(Vtotal * dm.w) + dm.b1 * np.cos(Vtotal * dm.w) +
                          -2 * dm.a2 * np.sin(2 * Vtotal * dm.w) + 2 * dm.b2 * np.cos(2 * Vtotal * dm.w))

    else:
        raise ValueError('Value of dm.fitType not recognized.')

    return dm