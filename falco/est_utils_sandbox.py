function
gdu = get_gdu(mp, ev, iSubband, y_measured, closed_loop_command, DM1Vdither, DM2Vdither)

modvar = ModelVariables
modvar.starIndex = 1
modvar.whichSource = 'star'

if mp.est.flagUseJacAlgDiff
    gdu = ev.G_tot_cont(:,:, iSubband)*closed_loop_command

else % --Get the probe phase from the model and the probe amplitude from the measurements
% For
unprobed
field
based
on
model:
if any(mp.dm_ind == 1) or any(mp.dm_ind_static == 1) 
    mp.dm1 = falco_set_constrained_voltage(mp.dm1, mp.dm1.V_dz) 
if any(mp.dm_ind == 2) or any(mp.dm_ind_static == 2) 
    mp.dm2 = falco_set_constrained_voltage(mp.dm2, mp.dm2.V_dz)
E0 = model_compact(mp, modvar)
E0vec = E0(mp.Fend.corr.maskBool)

% --For
probed
fields
based
on
model:
gdu = zeros(2 * size(y_measured(:, iSubband), 1), 1)
if any(mp.dm_ind == 1) | | any(
        mp.dm_ind_static == 1) mp.dm1 = falco_set_constrained_voltage(mp.dm1, mp.dm1.V_dz + mp.dm1.dV + DM1Vdither + mp.dm1.V_shift) end
if any(mp.dm_ind == 2) | | any(
        mp.dm_ind_static == 2) mp.dm2 = falco_set_constrained_voltage(mp.dm2, mp.dm2.V_dz + mp.dm2.dV + DM2Vdither + mp.dm2.V_shift) end
Edither = model_compact(mp, modvar)
Edithervec = Edither(mp.Fend.corr.maskBool)

gdu_comp = Edithervec - E0vec

gdu(1: 2:end, 1) = real(gdu_comp)
gdu(2: 2:end, 1) = imag(gdu_comp)

% Split
gdu
into
re and imag and convert
units
% Reset
DMS
end

end

function
comm_vector = get_dm_command_vector(mp, command1, command2)

if any(mp.dm_ind == 1) comm1 = command1(mp.dm1.act_ele) else comm1 =[] end % The 'else' block would mean we're only using DM2
if any(mp.dm_ind == 2) comm2 = command2(mp.dm2.act_ele) else comm2 =[] end
comm_vector = [comm1 comm2]

end

function
mp = set_constrained_full_command(mp, DM1Vdither, DM2Vdither)

if any(mp.dm_ind == 1)
    % note
    falco_set_constrained_voltage
    does
    not apply
    the
    command
    to
    the
    % DM
    mp.dm1 = falco_set_constrained_voltage(mp.dm1,
                                           mp.dm1.V_dz + mp.dm1.V_drift + mp.dm1.dV + DM1Vdither + mp.dm1.V_shift)
elseif
any(mp.dm_drift_ind == 1)
mp.dm1 = falco_set_constrained_voltage(mp.dm1, mp.dm1.V_dz + mp.dm1.V_drift)
elseif
any(mp.dm_ind_static == 1)
mp.dm1 = falco_set_constrained_voltage(mp.dm1, mp.dm1.V_dz)
end

if any(mp.dm_ind == 2)
    mp.dm2 = falco_set_constrained_voltage(mp.dm2,
                                           mp.dm2.V_dz + mp.dm2.V_drift + mp.dm2.dV + DM2Vdither + mp.dm2.V_shift)
elseif
any(mp.dm_drift_ind == 2)
mp.dm2 = falco_set_constrained_voltage(mp.dm2, mp.dm2.V_dz + mp.dm2.V_drift)
elseif
any(mp.dm_ind_static == 2)
mp.dm2 = falco_set_constrained_voltage(mp.dm2, mp.dm2.V_dz)
end

end

function
ev = pinned_act_safety_check(mp, ev)
% Update
new
pinned
actuators
if any(mp.dm_ind == 1) | | any(mp.dm_drift_ind == 1)
    ev.dm1.new_pinned_actuators = setdiff(mp.dm1.pinned, ev.dm1.initial_pinned_actuators)
    ev.dm1.act_ele_pinned = mp.dm1.pinned(ismember(ev.dm1.new_pinned_actuators, mp.dm1.act_ele))
end
if any(mp.dm_ind == 2) | | any(mp.dm_drift_ind == 2)
    ev.dm2.new_pinned_actuators = setdiff(mp.dm2.pinned, ev.dm2.initial_pinned_actuators)
    ev.dm2.act_ele_pinned = mp.dm2.pinned(ismember(ev.dm2.new_pinned_actuators, mp.dm2.act_ele))

end

% Check
that
no
new
actuators
have
been
pinned
if size(ev.dm1.new_pinned_actuators, 2) > 0 | | size(ev.dm2.new_pinned_actuators, 2) > 0

    % Print
    error
    warning
    fprintf('New DM1 pinned: [%s]\n', join(string(ev.dm1.new_pinned_actuators), ','))
    fprintf('New DM2 pinned: [%s]\n', join(string(ev.dm2.new_pinned_actuators), ','))

    % If
    actuators
    are
    used in jacobian, quit.
    if size(ev.dm1.act_ele_pinned, 2) > 0 | | size(ev.dm2.act_ele_pinned, 2) > 0
        save(fullfile([mp.path.config, '/', '/ev_exit_', num2str(ev.Itr), '.mat']), 'ev')
        save(fullfile([mp.path.config, '/', '/mp_exit_', num2str(ev.Itr), '.mat']), "mp")

        error('New actuators in act_ele pinned, exiting loop')
    end
end

end

function
out = mypageinv( in)

dim = size( in, 3)
out = zeros(size( in))
for i = 1:dim
out(:,:, i) = inv( in (:,:, i))
end

end

function
out = mypagemtimes(X, Y)

dim1 = size(X, 3)
dim2 = size(Y, 3)
if (dim1~=dim2) error('X and Y need to be the same size.') end
out = zeros(size(X, 1), size(Y, 2), dim1)
for i = 1:dim1
out(:,:, i) = X(:,:, i)*Y(:,:, i)
end

end

function[mp, ev] = get_open_loop_data(mp, ev)
% % Remove
control and dither
from DM command

% If
DM is used
for drift and control, apply V_dz and Vdrift, if DM is only
% used
for control, apply V_dz
    if (any(mp.dm_drift_ind == 1) & & any(mp.dm_ind == 1)) | | any(mp.dm_drift_ind == 1)
        mp.dm1 = falco_set_constrained_voltage(mp.dm1, mp.dm1.V_dz + mp.dm1.V_drift)
elseif
any(mp.dm_ind == 1) | | any(mp.dm_ind_static == 1)
mp.dm1 = falco_set_constrained_voltage(mp.dm1, mp.dm1.V_dz)
end

if (any(mp.dm_drift_ind == 2) & & any(mp.dm_ind == 2)) | | any(mp.dm_drift_ind == 2)
    mp.dm2 = falco_set_constrained_voltage(mp.dm2, mp.dm2.V_dz + mp.dm2.V_drift)
elseif
any(mp.dm_ind == 2) | | any(mp.dm_ind_static == 2)
mp.dm2 = falco_set_constrained_voltage(mp.dm2, mp.dm2.V_dz)
end

% Do
safety
check
for pinned actuators
    disp('OL DM safety check.')
ev = pinned_act_safety_check(mp, ev)

if ev.Itr == 1
    ev.IOLScoreHist = zeros(mp.Nitr, mp.Nsbp)
end

I_OL = zeros(size(ev.imageArray(:,:, 1, 1), 1), size(ev.imageArray(:,:, 1, 1), 2), mp.Nsbp)
for iSubband = 1:mp.Nsbp
I0 = falco_get_sbp_image(mp, iSubband)
I_OL(:,:, iSubband) = I0

ev.IOLScoreHist(ev.Itr, iSubband) = mean(I0(mp.Fend.score.mask))

end

ev.normI_OL_sbp = I_OL

disp(['mean OL contrast: ', num2str(mean(ev.IOLScoreHist(ev.Itr,:)))])
end

function
save_ekf_data(mp, ev, DM1Vdither, DM2Vdither)
drift = zeros(mp.dm1.Nact, mp.dm1.Nact, length(mp.dm_drift_ind))
dither = zeros(mp.dm1.Nact, mp.dm1.Nact, length(mp.dm_ind))
efc = zeros(mp.dm1.Nact, mp.dm1.Nact, length(mp.dm_ind))

if mp.dm_drift_ind(1) == 1 drift(:,:, 1) = mp.dm1.V_drift
end
if mp.dm_drift_ind(1) == 2 drift(:,:, 1) = mp.dm2.V_drift else drift(:,:, 2) = mp.dm2.V_drift
end

if mp.dm_ind(1) == 1 dither(:,:, 1) = DM1Vdither
end
if mp.dm_ind(1) == 2 dither(:,:, 1) = DM2Vdither else dither(:,:, 2) = DM2Vdither
end

if mp.dm_ind(1) == 1 efc(:,:, 1) = mp.dm1.dV
end
if mp.dm_ind(1) == 2 efc(:,:, 1) = mp.dm2.dV else efc(:,:, 2) = mp.dm2.dV
end

% TODO: move
to
plot_progress_iact
fitswrite(drift, fullfile([mp.path.config, '/', '/drift_command_it', num2str(ev.Itr), '.fits']))
fitswrite(dither, fullfile([mp.path.config, '/', 'dither_command_it', num2str(ev.Itr), '.fits']))
fitswrite(efc, fullfile([mp.path.config, '/', 'efc_command_it', num2str(ev.Itr - 1), '.fits']))

if ev.Itr == 1
    dz_init = zeros(mp.dm1.Nact, mp.dm1.Nact, length(mp.dm_ind))
    if mp.dm_ind(1) == 1 dz_init(:,:, 1) = mp.dm1.V_dz
    end
    if mp.dm_ind(1) == 2 dz_init(:,:, 1) = mp.dm2.V_dz else dz_init(:,:, 2) = mp.dm2.V_dz
    end

    fitswrite(dz_init, fullfile([mp.path.config, '/', 'dark_zone_command_0_pwp.fits']))
end

end





def ekf_estimate():
    ## Estimation part.All EKFs are advanced in parallel

    if mp.flagSim:
        sbp_texp = mp.detector.tExpUnprobedVec
    else:
        sbp_texp = mp.tb.info.sbp_texp


    for iSubband in 1:1: mp.Nsbp:

        # Get gdu
        gdu = get_gdu(mp, ev, iSubband, y_measured, closed_loop_command, DM1Vdither, DM2Vdither)

        # Estimate of the CL open loop electric field
        x_hat_CL = ev.x_hat[:, iSubband] + gdu * ev.e_scaling[iSubband] * sqrt(sbp_texp[iSubband])

        # --Estimate of the measurement:
        y_hat = x_hat_CL[1:ev.SS: end].^ 2 + x_hat_CL[2: ev.SS:end].^ 2 + (mp.est.dark_current * sbp_texp[iSubband])

        ev.R[ev.R_indices] = reshape(y_hat + (mp.est.read_noise) ^ 2, size(ev.R[ev.R_indices])]

        ev.H[ev.H_indices) = 2 * x_hat_CL

        % H_T = H.transpose(0, 2, 1)

        H_T = permute(ev.H, [2, 1, 3])

        ev.P[:,:,:, iSubband] = ev.P[:,:,:, iSubband] + ev.Q[:,:,:, iSubband]

        P_H_T = mypagemtimes(ev.P[:,:,:, iSubband], H_T)
        S = mypagemtimes(ev.H, P_H_T) + ev.R
        S_inv = mypageinv(S)

        % S_inv = np.linalg.pinv(S)
        K = mypagemtimes(P_H_T, S_inv)
        ev.P[:,:,:, iSubband] = ev.P[:,:,:, iSubband] - mypagemtimes(P_H_T, permute(K, [2, 1, 3]))

        # EKF correction:
        dy = (y_measured[:, iSubband] - y_hat)

        dy_hat_stacked = zeros(size(K))
        dy_hat_stacked[1,:,:] = dy.T

        dy_hat_stacked[2,:,:] = dy.T


        dx_hat_stacked = K. * dy_hat_stacked

        dx_hat = zeros(size(x_hat_CL))
        dx_hat[1: ev.SS:end] = dx_hat_stacked[1,:,:]
        dx_hat[2: ev.SS:end] = dx_hat_stacked[2,:,:]


        ev.x_hat[:, iSubband] = ev.x_hat[:, iSubband] + dx_hat

    return None



def ekf_estimate_py_sfr1():

    if mp.flagSim:
        sbp_texp = mp.detector.tExpUnprobedVec
    else:
        sbp_texp = mp.tb.info.sbp_texp
    # Calculate DM command for estimator, EKF does not know about drift

    # Assemble DM command known to estimator
    closed_loop_command = command + ev.periodic_dm_shift

    ## Estimation part. All EKFs are advanced in parallel
    x_hat_CL_new = {}

    # Estimate of the closed loop electric field:
    # hcipy.write_fits(ev.x_hat[wavelength], os.path.join(ev.experiment.output_path, f'x_hat_init.fits'))
    x_hat_CL = ev.x_hat[wavelength] + closed_loop_command.dot(
        ev.jacobians_boston_pixel[wavelength] * ev.e_scaling[wavelength]) * np.sqrt(
        ev.exposure_time_coron)
    # sqrt(photons) = sqrt(photons) + nm * sqrt(contrast)/nm * sqrt(photons/s)/sqrt(contrast) * sqrt(s)

    # hcipy.write_fits(x_hat_CL, os.path.join(ev.experiment.output_path, f'x_hat_CL.fits'))
    if ev.low_photon_regime:
        if ev.noise_model == 'rst':
            y_hat = ev.emccd_simple.get_light_noise_mean(x_hat_CL[::ev.SS] ** 2 + x_hat_CL[1::ev.SS] ** 2,
                                                           ev.emccd_gain, ev.exposure_time_coron)
            # photons
            ev.R[ev.R_indices] = ev.emccd_simple.get_light_noise_var(
                x_hat_CL[::ev.SS] ** 2 + x_hat_CL[1::ev.SS] ** 2, ev.emccd_gain, ev.exposure_time_coron)
        else:
            y_hat = x_hat_CL[::ev.SS] ** 2 + x_hat_CL[1::ev.SS] ** 2 + (
                    ev.dark_current * ev.exposure_time_coron)  # exposure time is in s for low snr case
            # photons = (sqrt(counts))^2 + (elec/s * s)
            ev.R[ev.R_indices] = y_hat + (ev.read_noise) ** 2
            # = photons + (electrons^2)
    else:
        y_hat = x_hat_CL[::ev.SS] ** 2 + x_hat_CL[1::ev.SS] ** 2 + (
                ev.dark_current * ev.quantum_efficiency * ev.exposure_time_coron * 1e-6)  # exposure time is in us for high snr case
        ev.R[ev.R_indices] = y_hat + (ev.read_noise * ev.quantum_efficiency) ** 2

    ev.H[ev.H_indices] = 2 * x_hat_CL
    H_T = ev.H.transpose(0, 2, 1)

    ev.P[wavelength] = ev.P[wavelength] + ev.Q[wavelength]
    P_H_T = np.matmul(ev.P[wavelength], H_T)
    S = np.matmul(ev.H, P_H_T) + ev.R
    S_inv = np.linalg.inv(S)  # does this need to be a pinv?

    # S_inv = np.linalg.pinv(S)
    K = np.matmul(P_H_T, S_inv)
    ev.P[wavelength] = ev.P[wavelength] - np.matmul(P_H_T, K.transpose(0, 2, 1))

    # EKF correction:
    dx_hat = np.matmul(K, (ev.y_measured[wavelength] - y_hat).reshape((-1, ev.BS // ev.SS, 1))).reshape(
        -1)

    ev.x_hat[wavelength] = ev.x_hat[wavelength] + dx_hat

    # Convert E_hat to contrast units:
    E_hat = (ev.x_hat[wavelength][::ev.SS] + complex(0, 1) * ev.x_hat[wavelength][1::ev.SS]) / (
            ev.e_scaling[wavelength] * np.sqrt(
        ev.exposure_time_coron))  # Estimate of the electric field from EKF state estimate

    E = hcipy.Field(np.zeros(ev.focal_grid.size, dtype='complex'), ev.focal_grid)
    E[ev.dark_zone_arr] = E_hat
    ev.E_estimates[wavelength] = E

    x_hat_CL_stacked = ev.x_hat[wavelength] + closed_loop_command.dot(ev.jacobians_boston_pixel[wavelength]
                                                                        * ev.e_scaling[wavelength]) * np.sqrt(
        ev.exposure_time_coron)

    x_hat_CL_new[wavelength] = hcipy.Field(np.zeros(ev.focal_grid.size, dtype='complex'), ev.focal_grid)
    x_hat_CL_new[wavelength][ev.dark_zone_arr] = (x_hat_CL_stacked[::ev.SS] + complex(0, 1) * x_hat_CL_stacked[
                                                                                                  1::ev.SS]) / (
                                                               ev.e_scaling[wavelength] * np.sqrt(
                                                           ev.exposure_time_coron))
    return None