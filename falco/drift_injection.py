import numpy as np
from scipy.interpolate import RectBivariateSpline
import astropy
import scipy


from .est_utils import get_dm_command_vector
from falco import util


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
    if mp.drift.dm_drift:
        if mp.drift.type.lower() == 'rand_walk':
            mp, ev = dm_rand_walk(mp, ev)
        else:
            mp.dm2.V_drift = np.zeros_like(mp.dm2.V)
    else:
        mp.dm2.V_drift = np.zeros_like(mp.dm2.V)

    if mp.drift.pupil_drift:
        if mp.drift.pupil_drift_type.lower() == 'stop':
            mp, ev = pupil_stop_drift(mp, ev)

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


def pupil_stop_drift(mp, ev):
    # check if we are assuming perfect initial WFE or real initial WFE
    if mp.drift.pupil_delta:
        base_wfe = mp.drift.pupil_drift_scaling * mp.drift.wfe_data[mp.drift.opd_keys[0]]
    else:
        base_wfe = 0*mp.drift.wfe_data[mp.drift.opd_keys[0]]

    delta_wfe = mp.drift.pupil_drift_scaling * mp.drift.wfe_data[mp.drift.opd_keys[1]][:, :, ev.Itr] - base_wfe

    mp = update_input_e_field(mp, delta_wfe)

    return mp, ev


def update_input_e_field(mp, wfe):
    wfe_resampled = resample_wfe(mp, wfe, verbose=False) * 1e-6  # um to m

    input_E = np.exp(2j * np.pi * (wfe_resampled) / (mp.lambda0))

    # full needs 4d, compact needs 3d
    mp.P1.compact.E = np.ones((mp.P1.compact.mask.shape[0], mp.P1.compact.mask.shape[1], mp.Nsbp), dtype=complex)
    mp.P1.full.E = np.ones((mp.P1.full.mask.shape[0], mp.P1.compact.mask.shape[1], mp.Nwpsbp, mp.Nsbp), dtype=complex)
    mp.P1.compact.E[:, :, 0] = input_E
    mp.P1.full.E[:, :, 0, 0] = input_E

    return mp


def resample_wfe(mp, wfe_data, verbose=False):

    cds_max_dim = mp.drift.cds_pix_size * mp.drift.cds_grid_size  # 7.2 meters
    c5_max_dim = mp.drift.c5_grid_size * mp.drift.c5_pix_size  # 6.016 meters

    # Your original 256x256 array
    original_array = wfe_data  # 6.016m x 6.016m physical size

    # Step 1: Create physical coordinate systems
    # C5 coordinates (centered at 0)
    c5_coords = np.linspace(-c5_max_dim / 2, c5_max_dim / 2, mp.drift.c5_grid_size)

    # CDS coordinates (centered at 0)
    cds_coords = np.linspace(-cds_max_dim / 2, cds_max_dim / 2, mp.drift.cds_grid_size)

    # Step 2: Create interpolator on the C5 grid
    interp = RectBivariateSpline(c5_coords, c5_coords, original_array, kx=3, ky=3)

    # Step 3: Evaluate on the CDS grid
    # Points outside the C5 aperture will be extrapolated (or you can set them to 0)
    resampled_array = interp(cds_coords, cds_coords)

    # Step 4: Mask points outside the original C5 aperture
    xx, yy = np.meshgrid(cds_coords, cds_coords, indexing='ij')
    distance_from_center = np.sqrt(xx ** 2 + yy ** 2)
    outside_c5_aperture = distance_from_center > (c5_max_dim / 2)
    resampled_array[outside_c5_aperture] = 0  # Set to zero outside original aperture

    if mp.P1.compact.mask.shape[0] != mp.P1.compact.Nbeam:
        if verbose:
            print('Array padded')
        resampled_array = util.pad_crop(resampled_array, mp.P1.compact.mask.shape[0], extrapval=0)

    if verbose:
        print(f"Final array shape: {resampled_array.shape}")  # (513, 513)
        print(f"CDS physical size: {mp.drift.cds_grid_size * mp.drift.cds_pix_size:.6f} meters")  # 7.200000 meters
        print(f"C5 physical size preserved: {mp.drift.c5_max_dim:.6f} meters")  # 6.016000 meters

        # Verify pixel scale
        print(f"CDS pixel size: {mp.drift.cds_pix_size:.10f} meters")

    return resampled_array

def dm_rand_walk(mp, ev):

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

    return mp, ev

