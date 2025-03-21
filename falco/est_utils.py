


def initialize_ekf_maintenance(mp, ev, jacStruct):

    # Check if sim mode to avoid calling tb obj in sim mode
    if mp.flagSim:
        sbp_texp = mp.detector.tExpUnprobedVec # exposure times for non-pairwise-probe images in each subband.
        psf_peaks = mp.detector.peakFluxVec
    else:
        sbp_texp  = mp.tb.info.sbp_texp
        psf_peaks = mp.tb.info.PSFpeaks

    # Find values to convert images back to counts rather than normalized intensity
    ev.peak_psf_counts = np.zeros([1, mp.Nsbp])
    ev.e_scaling = np.zeros([1, mp.Nsbp])

    for iSubband in range(mp.Nsbp): # TODO: missing +1 in range?
        # potentially set mp.detector.peakFluxVec(si) * mp.detector.tExpUnprobedVec(si) set to mp.tb.info.sbp_texp(si)*mp.tb.info.PSFpeaks(si);
        # to have cleaner setup
        ev.peak_psf_counts[iSubband] = sbp_texp[iSubband]*psf_peaks[iSubband]
        ev.e_scaling[iSubband] = np.sqrt(psf_peaks[iSubband])


    # Rearrange jacobians
    ev.G_tot_cont = rearrange_jacobians(mp, jacStruct, mp.dm_ind)
    ev.G_tot_drift = rearrange_jacobians(mp, jacStruct, mp.dm_drift_ind)

    # Initialize EKF matrices
    ev = initialize_ekf_matrices(mp, ev, sbp_texp)

    # Initialize pinned actuator check
    ev.dm1.initial_pinned_actuators = mp.dm1.pinned
    if np.any(mp.dm_ind == 2):
        ev.dm2.initial_pinned_actuators = mp.dm2.pinned
    ev.dm1.new_pinned_actuators = []
    ev.dm2.new_pinned_actuators = []
    ev.dm1.act_ele_pinned = []
    ev.dm2.act_ele_pinned = []


    return None



def rearrange_jacobians(mp, jacStruct, mp.dm_ind):
    self.jacobians_boston_pixel = {}
    for wavelength in self.wavelengths.keys():
        G_boston = self.jacobians_boston[
            wavelength]  # / np.sqrt(self.counts_per_photons[wavelength] / self.direct_photons)  # ********************************************
        G_reordered = np.zeros(G_boston.shape, G_boston.dtype)
        G_reordered[:, ::2] = G_boston[:, :G_boston.shape[1] // 2]
        G_reordered[:, 1::2] = G_boston[:, G_boston.shape[1] // 2:]
        self.jacobians_boston_pixel[wavelength] = G_reordered



    G1 = zeros(mp.dm1.Nele,mp.Nsbp, 2*size(jacStruct.G1,1));
    G2 = zeros(2*size(jacStruct.G2,1),mp.dm2.Nele,mp.Nsbp);
