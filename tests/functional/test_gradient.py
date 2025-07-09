"""Jacobian accuracy tests for the Lyot coronagraph."""
from copy import deepcopy
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

import falco
from falco.util import pad_crop

show_plots = False

n_pupil = 50
n_pupil_total = n_pupil*n_pupil
n_pupil_pad = 128
centering = 'pixel'

res_cam = 3 #3
rho0 = 2.5
rho1 = 9.0
n_cam = falco.util.ceil_even(2*rho1*res_cam + 1)

fl = 1
lam_norm = 1
dx_pupil = 1/n_pupil
dxi_cam = 1/res_cam

d_dm1_dm2 = 0.2  # meters
d_pupil = 46e-3  # meters
dx_pupil_m = d_pupil/n_pupil
lam = 575e-9  # meters
gain = 1e-9
kk = gain*2*np.pi/lam


def make_masks():

    # masks = SimpleNamespace() #falco.config.Object()
    masks = falco.config.Object()

    # Make pupil mask

    inputs = {
        'Nbeam': n_pupil,
        'Npad': n_pupil_total,
        'OD': 1,
        'wStrut': 0.04,
        # 'angStrut': 10 + np.array([0, 120, 240])
    }
    pupil = falco.mask.falco_gen_pupil_Simple(inputs)


    # pupil = falco.mask.falco_gen_pupil_Roman_CGI_20200513(n_pupil, centering)
    pupil = falco.util.pad_crop(pupil, n_pupil_pad)
    masks.pupil = pupil

    # Make software mask at detector
    SCORE = {}
    SCORE["Nxi"] = n_cam
    SCORE["Neta"] = n_cam
    SCORE["pixresFP"] = res_cam
    SCORE["centering"] = centering
    SCORE["rhoInner"] = rho0  # lambda0/D
    SCORE["rhoOuter"] = rho1  # lambda0/D
    SCORE["angDeg"] = 180  # degrees
    SCORE["whichSide"] = 'both'
    SCORE["shape"] = 'circle'
    [maskScore, _, _] = falco.mask.falco_gen_SW_mask(SCORE)
    masks.dh = np.array(maskScore, dtype=bool)

    return masks


def make_jacobian(masks, u1, u2):

    n_pix = np.sum(masks.dh.astype(int))
    jac = np.zeros((n_pix, 2*n_pupil_total), dtype=complex)

    du_flat = np.zeros((n_pupil_total,))
    for ii in range(n_pupil_total):
        du_flat[ii] = 1
        du = du_flat.reshape((n_pupil, n_pupil))

        jac[:, ii] = jac_model(masks, u1, u2, 1, du)
        jac[:, n_pupil_total+ii] = jac_model(masks, u1, u2, 2, du)

        du_flat *= 0  # reset

    return jac


def jac_model(masks, u1, u2, which_dm, du):

    u1_pad = pad_crop(u1, n_pupil_pad)
    u2_pad = pad_crop(u2, n_pupil_pad)
    du_pad = pad_crop(du, n_pupil_pad)
    e_pupil = masks.pupil
    e_dm1 =  e_pupil * np.exp(1j*kk*u1_pad)

    if which_dm == 1:
        de_dm1 = (1j*kk*du_pad) * e_dm1
        de_dm2 = falco.prop.ptp(de_dm1, dx_pupil_m*n_pupil_pad, lam, d_dm1_dm2)
        de_dm2 *= np.exp(1j*kk*u2_pad)
        de_dm1_eff = falco.prop.ptp(de_dm2, dx_pupil_m*n_pupil_pad, lam, -d_dm1_dm2)

    elif which_dm == 2:
        e_dm2 = falco.prop.ptp(e_dm1, dx_pupil_m*n_pupil_pad, lam, d_dm1_dm2)
        e_dm2 *= np.exp(1j*kk*u2_pad)
        de_dm2 =  (1j*kk*du_pad) * e_dm2
        de_dm1_eff = falco.prop.ptp(de_dm2, dx_pupil_m*n_pupil_pad, lam, -d_dm1_dm2)

    else:
        raise ValueError('which_dm must be 1 or 2')

    de_cam = falco.prop.mft_p2f(de_dm1_eff, fl, lam_norm, dx_pupil, dxi_cam, n_cam, dxi_cam, n_cam, centering)
    de_vec = de_cam[masks.dh]

    return de_vec


def forward_model(masks, u1, u2):

    u1_pad = falco.util.pad_crop(u1, n_pupil_pad)
    u2_pad = falco.util.pad_crop(u2, n_pupil_pad)

    e_dm1 = masks.pupil * np.exp(1j*kk*u1_pad)
    e_dm2 = falco.prop.ptp(e_dm1, dx_pupil_m*n_pupil_pad, lam, d_dm1_dm2)
    e_dm2 *= np.exp(1j*kk*u2_pad)
    e_dm1_eff = falco.prop.ptp(e_dm2, dx_pupil_m*n_pupil_pad, lam, -d_dm1_dm2)

    e_cam = falco.prop.mft_p2f(e_dm1_eff, fl, lam_norm, dx_pupil, dxi_cam, n_cam, dxi_cam, n_cam, centering)
    e_vec = e_cam[masks.dh]

    return e_vec, e_cam


def reverse_model(masks, u1, u2, du1, du2, Eest2D, I00):

    _, EFendA = forward_model(masks, u1, u2)
    _, EFendB = forward_model(masks, u1+du1, u2+du2)
    dEend = EFendB - EFendA
    EdhNew = Eest2D + dEend/np.sqrt(I00)

    u1_pad = falco.util.pad_crop(u1, n_pupil_pad)
    u2_pad = falco.util.pad_crop(u2, n_pupil_pad)
    e_dm1_post = masks.pupil * np.exp(1j*kk*u1_pad)
    e_dm2_pre = falco.prop.ptp(e_dm1_post, dx_pupil_m*n_pupil_pad, lam, d_dm1_dm2)
    e_dm2_post = np.exp(1j*kk*u2_pad) * e_dm2_pre

    # Gradient
    cam_grad = 2/np.sqrt(I00)*EdhNew*np.real(masks.dh.astype(float))


    e_dm1_eff_grad = falco.prop.mft_f2p(cam_grad, -fl, lam_norm, dxi_cam, dxi_cam, dx_pupil, n_pupil_pad, centering)
    e_dm2_grad = falco.prop.ptp(e_dm1_eff_grad, dx_pupil_m*n_pupil_pad, lam, d_dm1_dm2)
    e_dm2_grad *= np.conj(np.exp(1j*kk*u2_pad))
    phase_dm2_bar = -kk*np.imag(e_dm2_grad * np.conj(e_dm2_pre))

    
    e_dm1_grad = falco.prop.ptp(e_dm2_grad, dx_pupil_m*n_pupil_pad, lam, -d_dm1_dm2)
    phase_dm1_bar = -kk*np.imag(e_dm1_grad * np.conj(e_dm1_post))

    # e_dm1_eff_grad = falco.prop.mft_f2p(cam_grad, -fl, lam_norm, dxi_cam, dxi_cam, dx_pupil, n_pupil_pad, centering)
    # e_dm2_grad = falco.prop.ptp(e_dm1_eff_grad, dx_pupil_m*n_pupil_pad, lam, d_dm1_dm2)
    # e_dm2_bar = e_dm2_grad
    # # e_dm2_bar = np.conj(e_dm2_pre) * np.exp(1j*kk*u2_pad)
    # phase_dm2_bar = -kk*np.imag(e_dm2_bar * np.conj(e_dm2_post))

    # e_dm2_grad *= np.conj(np.exp(1j*kk*u2_pad))
    # e_dm1_grad = falco.prop.ptp(e_dm2_grad, dx_pupil_m*n_pupil_pad, lam, -d_dm1_dm2)
    # phase_dm1_bar = -kk*np.imag(e_dm1_grad * np.conj(e_dm1_post))

    gradient = np.zeros((2*n_pupil_total,))
    gradient[0:n_pupil_total] = pad_crop(phase_dm1_bar, n_pupil).ravel()
    gradient[n_pupil_total::] = pad_crop(phase_dm2_bar, n_pupil).ravel()

    return gradient


def test_adjoint():
    """Most basic tests of the adjoint model accuracy."""

    masks = make_masks()
    u1 = np.zeros((n_pupil, n_pupil))
    u2 = np.zeros((n_pupil, n_pupil))

    u1 = (np.random.rand(n_pupil, n_pupil)-0.5)
    u2 = (np.random.rand(n_pupil, n_pupil)-0.5)

    # Make Jacobian G1
    jac = make_jacobian(masks, u1, u2)
    G1 = jac[:, 0:n_pupil_total]
    G2 = jac[:, n_pupil_total::]

    # Get true E-field, e_vec
    e_vec, _ = forward_model(masks, u1, u2)

    # Get "estimated" starting E-field
    _, Eest2D = forward_model(masks, u1, u2)
    I00 = np.max(np.abs(Eest2D)**2)
    print(f'kk = {kk}')
    print(f'I00 = {I00}')
    print(f'sqrt(I00) = {np.sqrt(I00)}')
    # I00 = 1  # DEBUGGING

    e_vec /= np.sqrt(I00)
    Eest2D /= np.sqrt(I00)
    G1 /= np.sqrt(I00)
    G2 /= np.sqrt(I00)

    # Assign delta commands
    du1 = np.eye(n_pupil)
    du2 = np.zeros((n_pupil, n_pupil))
    du2[:, 22] = 1

    du = np.concatenate((du1.ravel(), du2.ravel()))
    u_bar_expected = 2 * jac.conj().T @ (e_vec + jac @ du)
    u1_bar_expected_2d = u_bar_expected[0:n_pupil_total].reshape((n_pupil, n_pupil))
    u2_bar_expected_2d = u_bar_expected[n_pupil_total::].reshape((n_pupil, n_pupil))

    # # Compute expected response
    # u1_bar_expected = 2 * G1.conj().T @ (e_vec + G1 @ du1.ravel() + G2 @ du2.ravel())
    # u2_bar_expected = 2 * G2.conj().T @ (e_vec + G1 @ du1.ravel() + G2 @ du2.ravel())
    # u1_bar_expected_2d = u1_bar_expected.reshape((n_pupil, n_pupil))
    # u2_bar_expected_2d = u2_bar_expected.reshape((n_pupil, n_pupil))

    # Compute gradient
    gradient = reverse_model(masks, u1, u2, du1, du2, Eest2D, I00)

    u1_bar_out_2d = gradient[0:n_pupil_total].reshape((n_pupil, n_pupil))
    u2_bar_out_2d = gradient[n_pupil_total::].reshape((n_pupil, n_pupil))

    # breakpoint()

    if show_plots:

        plt.figure(1)
        plt.imshow(np.real(u1_bar_expected_2d))
        plt.colorbar()
        plt.title('u1_bar_expected_2d, real')

        plt.figure(11)
        plt.imshow(np.real(u1_bar_out_2d))
        plt.colorbar()
        plt.title('u1_bar_out_2d, real')

        # plt.figure(21)
        # plt.imshow(np.real(u1_bar_expected_2d - u1_bar_out_2d))
        # plt.colorbar()
        # plt.title('u1_bar_expected_2d - u1_bar_out_2d, real')

        plt.figure(31)
        plt.imshow(np.real(u1_bar_expected_2d / u1_bar_out_2d))
        plt.colorbar()
        plt.title('u1_bar_expected_2d / u1_bar_out_2d, real')


        plt.figure(2)
        plt.imshow(np.real(u2_bar_expected_2d))
        plt.colorbar()
        plt.title('u2_bar_expected_2d, real')

        plt.figure(12)
        plt.imshow(np.real(u2_bar_out_2d))
        plt.colorbar()
        plt.title('u2_bar_out_2d, real')

        plt.figure(32)
        plt.imshow(np.real(u2_bar_expected_2d / u2_bar_out_2d))
        plt.colorbar()
        plt.title('u2_bar_expected_2d / u2_bar_out_2d, real')

        # plt.figure(51)
        # plt.imshow(np.abs(G1))
        # plt.colorbar()
        # plt.title('abs(G1)')

        # # plt.figure(52)
        # # plt.imshow(np.abs(G2))
        # # plt.colorbar()
        # # plt.title('abs(G2)')

        # plt.figure(53)
        # plt.imshow(np.abs(G1-G2))
        # plt.colorbar()
        # plt.title('abs(G1-G2)')

        plt.show()

        pass


if __name__ == '__main__':
    test_adjoint()
