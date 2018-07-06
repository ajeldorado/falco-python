import numpy as np

from falco import models


def falco_est_perfect_Efield(mp, DM, which_model='full'):
    """
    Function to return the perfect-knowledge E-field and summed intensity for the full model.

    Parameters
    ----------
    mp : ModelParameters
        Parameter structure for current model.
    DM : DeformableMirrorParameters (placeholder class for now)
        Parameter structure for deformable mirrors
    which_model: string
        Coronagraph model used to compute the detector-plane electric field. Either 'full' or
        'compact'.
    Returns
    -------
    Emat : np.ndarray
        Exact electric field inside dark hole
    Isum2D : float
        Total intensity inside dark hole
    """

    if which_model == 'full':
        Icube_shape = (mp.F4.full.Neta, mp.F4.full.Nxim, mp.Nttlam)
        Emat_shape = (mp.F4.full.corr.inds.shape[0], mp.Nttlam)
        model = models.model_full
    elif which_model == 'compact':
        Icube_shape = (mp.F4.compact.Neta, mp.F4.compact.Nxim, mp.Nttlam)
        Emat_shape = (mp.F4.compact.corr.inds.shape[0], mp.Nttlam)
        model = models.model_compact
    else:
        raise ValueError('Invalid model specified.  Try \'full\' or \'compact\'.')

    Icube = np.zeros(Icube_shape, dtype=np.float64)
    Emat = np.zeros(Emat_shape, dtype=np.float64)

    # Initialize model variable dictionary
    modvar = {
        'flagCalcJac': 0,
        'wpsbpIndex': mp.wi_ref,
        'whichSource': 'star'
    }

    # Execute model to obtain electric field and total intensity
    for tsi in range(mp.Nttlam):
        modvar['sbpIndex'] = mp.Wttlam_si[tsi]
        modvar['ttIndex'] = mp.Wttlam_ti[tsi]

        E2D = model(mp, DM, modvar)
        Emat[:, tsi] = E2D[mp.F4.corr.inds]  # Exact field inside estimation area
        Icube[:, :, tsi] = (np.abs(E2D) ** 2) * mp.WttlamVec(tsi) / mp.Wsum

    Isum2D = Icube.sum(axis=2)
    return Emat, Isum2D