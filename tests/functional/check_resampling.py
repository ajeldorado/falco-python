"""Jacobian accuracy tests for the Lyot coronagraph."""
from copy import deepcopy
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

import falco
from falco.util import pad_crop, ceil_odd
from falco.diff_dm import spline_resample, map_resample, fourier_resample


show_plots = True

n_pupil = 50
n_pupil_total = n_pupil*n_pupil
n_pupil_pad = 100
centering = 'pixel'

res_cam = 3 #3
rho0 = 2.5
rho1 = 9.0
n_cam = falco.util.ceil_even(2*rho1*res_cam + 1)

fl = 1
lam_norm = 1
diam_pupil_m = 46e-3
dx_pupil = 1/n_pupil
dxi_cam = 1/res_cam

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
        # 'centering': 'interpixel',
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


def make_jacobian(masks, u1):

    n_pix = np.sum(masks.dh.astype(int))
    jac = np.zeros((n_pix, n_pupil_total), dtype=complex)

    du1_flat = np.zeros((n_pupil_total,))
    for ii in range(n_pupil_total):
        du1_flat[ii] = 1
        du1 = du1_flat.reshape((n_pupil, n_pupil))

        jac[:, ii] = jac_model(masks, u1, du1)

        du1_flat *= 0  # reset

    return jac


def jac_model(masks, u1, du1):

    u1_pad = pad_crop(u1, n_pupil_pad)
    du1_pad = pad_crop(du1, n_pupil_pad)
    de_pupil = masks.pupil * np.exp(1j*kk*u1_pad) * (1j*kk*du1_pad)

    de_cam = falco.prop.mft_p2f(de_pupil, fl, lam_norm, dx_pupil, dxi_cam, n_cam, dxi_cam, n_cam, centering)
    de_vec = de_cam[masks.dh]

    return de_vec


def forward_model(masks, u1):

    u1_pad = falco.util.pad_crop(u1, n_pupil_pad)
    e_dm1_post = np.exp(1j*kk*u1_pad) * masks.pupil

    e_cam = falco.prop.mft_p2f(e_dm1_post, fl, lam_norm, dx_pupil, dxi_cam, n_cam, dxi_cam, n_cam, centering)
    e_vec = e_cam[masks.dh]

    return e_vec, e_cam


def reverse_model(masks, u1, du1, Eest2D, I00):

    _, EFendA = forward_model(masks, u1)
    _, EFendB = forward_model(masks, u1+du1)
    dEend = EFendB - EFendA
    EdhNew = Eest2D + dEend/np.sqrt(I00)

    u1_pad = falco.util.pad_crop(u1, n_pupil_pad)
    e_dm1_post = np.exp(1j*kk*u1_pad) * masks.pupil
    # e_dm1_post = masks.pupil  # DEBUGGING

    # Gradient
    # cam_grad = EdhNew*np.real(masks.dh.astype(float)) / np.sum(np.abs(Eest2D[masks.dh])**2)
    cam_grad = 2/np.sqrt(I00)*EdhNew*masks.dh #np.real(masks.dh.astype(float))
    # cam_grad = np.zeros_like(EdhNew)
    # cam_grad[masks.dh] = 2*EdhNew[masks.dh]

    pupil_grad = falco.prop.mft_f2p(cam_grad, -fl, lam_norm, dxi_cam, dxi_cam, dx_pupil, n_pupil_pad, centering)

    phase_DM2_bar = -kk*np.imag(pupil_grad * np.conj(e_dm1_post))
    gradient = pad_crop(phase_DM2_bar, n_pupil).ravel()

    return gradient


def test_resampling():
    masks = make_masks()

    usfac = 2.9
    dsfac = 0.33

    pupil0 = masks.pupil
    pupil0 = pad_crop(pupil0, int(ceil_odd(pupil0.shape[0])))

    pupil_spline_up = spline_resample(pupil0, usfac)
    pupil_spline_down = spline_resample(pupil0, dsfac)
    # pupil_spline_up = map_resample(pupil0, usfac)
    # pupil_spline_down = map_resample(pupil0, dsfac)

    pupil_fourier_up = fourier_resample(pupil0, usfac)
    pupil_fourier_down = fourier_resample(pupil0, dsfac)

    pupil_fourier_up = pad_crop(pupil_fourier_up, pupil_spline_up.shape)
    pupil_fourier_down = pad_crop(pupil_fourier_down, pupil_spline_down.shape)

    print(f'pupil0.shape = {pupil0.shape}')
    print(f'pupil_spline_up.shape = {pupil_spline_up.shape}')
    print(f'pupil_spline_down.shape = {pupil_spline_down.shape}')
    # print(f'pupil_fourier_up.shape = {pupil_fourier_up.shape}')
    # print(f'pupil_fourier_down.shape = {pupil_fourier_down.shape}')

    if show_plots:

        plt.figure(1)
        plt.imshow(pupil0)
        plt.colorbar()
        plt.title('pupil0')

        plt.figure(2)
        plt.imshow(pupil_spline_up)
        plt.colorbar()
        plt.title('pupil_spline_up')

        plt.figure(3)
        plt.imshow(pupil_spline_down)
        plt.colorbar()
        plt.title('pupil_spline_down')

        plt.figure(4)
        plt.imshow(pupil_fourier_up - pupil_spline_up)
        plt.colorbar()
        plt.title('pupil_fourier_up')

        plt.figure(5)
        plt.imshow(pupil_fourier_down - pupil_spline_down)
        plt.colorbar()
        plt.title('pupil_fourier_down')


        plt.show()


def test_adjoint():
    """Most basic tests of the adjoint model accuracy."""

    masks = make_masks()
    # u1 = np.zeros((n_pupil, n_pupil))
    u1 = 0.8 * (np.random.rand(n_pupil, n_pupil)-0.5)

    # Make Jacobian G1
    G1 = make_jacobian(masks, u1)

    # Get true E-field, e_vec
    e_vec, _ = forward_model(masks, u1)

    # Get "estimated" starting E-field
    _, Eest2D = forward_model(masks, u1)
    I00 = np.max(np.abs(Eest2D)**2)
    # I00 = 1  # DEBUGGING
    print(f'I00 = {I00}')
    print(f'sqrt(I00) = {np.sqrt(I00)}')
    print(f'kk = {kk}')

    e_vec /= np.sqrt(I00)
    Eest2D /= np.sqrt(I00)
    G1 /= np.sqrt(I00)

    # Assign delta command 
    du1 = np.eye(n_pupil)

    # Compute expected response
    u1_bar_expected = 2 * G1.conj().T @ (e_vec + G1 @ du1.ravel())
    u1_bar_expected_2d = u1_bar_expected.reshape((n_pupil, n_pupil))

    # Compute gradient
    gradient = reverse_model(masks, u1, du1, Eest2D, I00)

    u1_bar_out_2d = gradient.reshape((n_pupil, n_pupil))

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

        # plt.figure(2)
        # plt.imshow(np.real(u2_bar_expected_2d))
        # plt.colorbar()
        # plt.title('u2_bar_expected_2d, real')

        # plt.figure(12)
        # plt.imshow(np.real(u2_bar_out_2d))
        # plt.colorbar()
        # plt.title('u2_bar_out_2d, real')


        plt.figure(10)
        plt.imshow(u1)
        plt.colorbar()
        plt.title('u1')

        plt.figure(20)
        plt.imshow(np.log10(np.abs(Eest2D)))
        plt.colorbar()
        plt.title('log10(abs(Eest2D))')

        plt.show()


if __name__ == '__main__':
    # test_adjoint()
    test_resampling()
