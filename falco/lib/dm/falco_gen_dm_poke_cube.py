# python version of falco_gen_dm_poke_cube.m
import sys
import os
import numpy as np
import scipy
import scipy.io as sio
from falco import utils  # ceil_even, ceil_odd


sys.path.append('y:/src/Falco/falco-python/falco/')


def sind(tdeg):
    return np.sin((np.pi / 180.) * tdeg)


def cosd(tdeg):
    return np.cos((np.pi / 180.) * tdeg)


def falco_gen_dm_poke_cube(dm, mp, dx_dm, flagGenCube=True, **kwds):

    # flagGenCube = True

    # check 'flag_hex_array' in dm
    if 'flag_hex_array' not in dm.keys():
        dm['flag_hex_array'] = False

    #%-- set the order of operations
    flagZYX = dm['flagZYX'] if 'flagZYX' in dm.keys() else False

    #%--Compute sampling of the pupil. Assume that it is square.
    # we will reconstruct dm[] for output at the end
    # dm['dx_dm'] = dx_dm
    # dm['dx'] = dx_dm

    #%--Default to being centered on a pixel (FFT-style) if no centering is specified
    if 'centering' not in dm.keys():
        dm['centering'] = 'pixel'

    #%--Compute coordinates of original influence function
    Ninf0 = dm['inf0'].shape  # rows, cols ( = (91,91) )
    x_inf0 = (dm['dx_inf0'] * np.arange((-Ninf0[1] + 1) / 2,
                                        (Ninf0[1] - 1) / 2 + 1)).astype('float64')

    Ndm0 = utils.ceil_even(Ninf0[0] + (dm['Nact'] - 1) * (dm['dm_spacing'] / dm['dx_inf0']))
    dm['NdmMin'] = utils.ceil_even(Ndm0 * (dm['dx_inf0'] / dx_dm)) + 2

    #%--Number of points across the array to fully contain the DM surface at new,
    #   desired resolution and z-rotation angle.
    # dm.Ndm = ceil_even( max(abs(...
    #    [sqrt(2)*cosd(45-dm.zrot),sqrt(2)*sind(45-dm.zrot)] ...
    #                              )) *Ndm0*(dm.dx_inf0/dm.dx)...
    #                    )+2;

    dm['Ndm'] = utils.ceil_even(
        np.max(np.abs(
            np.sqrt(2.0) * np.array(
                [cosd(45. - dm['zrot']), sind(45. - dm['zrot'])]
            )
        )) * Ndm0 * (dm['dx_inf0'] / dx_dm)
    ) + 2

    # not used here:
    #[Xinf0, Yinf0] = np.meshgrid(x_inf0, x_inf0)

    #%--Compute list of initial actuator center coordinates (in actutor widths).
    if dm['flag_hex_array']:
        print('Error: flag_hex_array is not implemented')
        raise

    else:
        # meshgrid((0:dm.Nact-1)-dm.xc,(0:dm.Nact-1)-dm.yc); % in actuator widths
        Xact, Yact = np.meshgrid(
            np.arange(0, dm['Nact'], dtype='float64') - dm['xc'],
            np.arange(0, dm['Nact'], dtype='float64') - dm['yc']
        )

    NactTotal = Xact.size

    # tilt section copied right from the Matlab function
    tlt = np.zeros((1, 3))
    tlt[0, 0] = dm['xtilt']
    tlt[0, 1] = dm['ytilt']
    tlt[0, 2] = -dm['zrot']

    sa = sind(tlt[0, 0])
    ca = cosd(tlt[0, 0])
    sb = sind(tlt[0, 1])
    cb = cosd(tlt[0, 1])
    sg = sind(tlt[0, 2])
    cg = cosd(tlt[0, 2])

    if flagZYX:
        Mrot = np.array([[cb * cg,               -cb * sg,       sb, 0.0],
                         [ca * sg + sa * sb * cg, ca * cg - sa * sb * sg, -sa * cb, 0.0],
                         [sa * sg - ca * sb * cg, sa * cg + ca * sb * sg,  ca * cb, 0.0],
                         [0.0,                    0.0,      0.0, 1.0]])
    else:
        Mrot = np.array([[cb * cg, sa * sb * cg - ca * sg, ca * sb * cg + sa * sg, 0.0],
                         [cb * sg, sa * sb * sg + ca * cg, ca * sb * sg - sa * cg, 0.0],
                         [-sb,                sa * cb,                ca * cb, 0.0],
                         [0.0,                    0.0,                    0.0, 1.0]])

    #xy_cent_act = np.zeros((2,NactTotal))

    # Xact.T to conform to the Matlab column-first ordering
    Atmp = np.vstack((Xact.T.flatten(), Yact.T.flatten(),
                      np.zeros((Xact.size,)), np.ones((Xact.size,))))
    # Atmp.shape = (4, 2304)
    # save only x,y:
    xy_cent_act = np.matmul(Mrot, Atmp)[:2, :]

    N0 = np.array(Ninf0).max()
    Npad = utils.ceil_odd(np.sqrt(2.) * N0)

    inf0pad = np.zeros((Npad, Npad))

    inf0pad[int(np.ceil(Npad/2)-np.floor(N0/2)-1):int(np.ceil(Npad/2)+np.floor(N0/2)),
            int(np.ceil(Npad/2)-np.floor(N0/2)-1):int(np.ceil(Npad/2)+np.floor(N0/2))] = dm['inf0']

    # ydim, xdim = inf0pad.shape # redundant to Npad, Npad

    xd2 = np.fix(Npad / 2)
    yd2 = np.fix(Npad / 2)
    # cx   = ([0 : Npad] - xd2) ;
    cx = np.arange(0, Npad) - xd2
    cy = np.arange(0, Npad) - yd2
    Xs0, Ys0 = np.meshgrid(cx, cy)

    xyzValsRot = np.matmul(Mrot, np.vstack((
        Xs0.T.flatten(), Ys0.T.flatten(), np.zeros((Xs0.size,)), np.ones((Xs0.size,))
    ))
    )
    xsNew = xyzValsRot[0, :].reshape((Npad, Npad), order='F')
    ysNew = xyzValsRot[1, :].reshape((Npad, Npad), order='F')

    infMaster = scipy.interpolate.griddata(
        (xsNew.flatten(), ysNew.flatten()), inf0pad.flatten(),
        (Xs0, Ys0), method='cubic', fill_value=0.0
    )

    #%--Crop down the influence function until it has no zero padding left
    infSum = np.sum(infMaster)
    infDiff = 0
    counter = 0
    while(abs(infDiff) <= 1e-7):
        counter = counter + 2
        #%--Number of points across the rotated, cropped-down influence function at the original resolution
        Ninf0pad = infMaster.shape[0] - counter  # assumes square array
        #%--Subtract an extra 2 to negate the extra step that overshoots.
        infDiff = infSum - np.sum(infMaster[counter // 2:-counter // 2, counter // 2:-counter // 2])

    counter = counter - 2
    Ninf0pad = infMaster.shape[0] - counter  # %Ninf0pad = Ninf0pad+2, assumes square array
    #% padOrCropEven(dm.infMaster,Ncrop); %--The cropped-down Lyot stop for the compact model
    infMaster2 = infMaster[counter // 2:-counter // 2, counter // 2:-counter // 2]

    # check infMaster2 is Ninf0pad x Ninf0pad
    print('check infMaster2.shape == Ninf0pad: ', infMaster2.shape == (Ninf0pad, Ninf0pad))

    # use new coordinate grid from now on
    # % True for even- or odd-sized influence function maps as long as they are centered on the array.
    infMaster = infMaster2.copy()
    Npad = Ninf0pad

    x_inf0 = np.arange(-(Npad - 1) / 2, (Npad - 1) / 2 + 1)*dm['dx_inf0']
    Xinf0, Yinf0 = np.meshgrid(x_inf0, x_inf0)

    ################################################

    # %--Compute the size of the postage stamps.
    # % Number of points across the influence function in the pupil file's spacing. Want as even
    Nbox = utils.ceil_even(((Ninf0pad * dm['dx_inf0']) / dx_dm))
    #dm['Nbox'] = Nbox

    # %--Also compute their padded sizes for the angular spectrum (AS) propagation
    #    between P2 and DM1 or between DM1 and DM2
    # %--Minimum number of points across for accurate angular spectrum propagation
    Nmin = utils.ceil_even(
        np.max(mp['sbp_center_vec']) * np.max(np.abs(
             [mp['d_P2_dm1'], mp['d_dm1_dm2'], (mp['d_P2_dm1'] + mp['d_dm1_dm2'])]
             )) / dx_dm ** 2)
    # % dm.NboxAS = 2^(nextpow2(max([Nbox,Nmin])));  %--Zero-pad for FFTs in A.S. propagation.
    #   Uses a larger array if the max sampling criterion for angular spectrum propagation is violated
    # %--Uses a larger array if the max sampling criterion for angular spectrum propagation is violated
    NboxAS = np.max([Nbox, Nmin])

    # %% Pad the pupil to at least the size of the DM(s) surface(s) to allow all actuators
    # to be located outside the pupil.
    # % (Same for both DMs)

    # %--Find actuator farthest from center:
    r_cent_act = np.sqrt(xy_cent_act[0, :] ** 2 + xy_cent_act[1, :] ** 2)
    rmax = np.max(np.abs(r_cent_act))
    NpixPerAct = dm['dm_spacing'] / dx_dm

    # flag_hex_array not implemented
    # if(dm.flag_hex_array)
    # %dm.NdmPad = 2*ceil(1/2*Nbox*2) + 2*ceil((1/2*2*(dm.rmax)*dm.dx_inf0_act)*Nbox);
    # %2*ceil((dm.rmax+3)*dm.dm_spacing/Dpup*Npup);
    # dm.NdmPad = ceil_even((2*(dm.rmax+2))*NpixPerAct + 1);
    # % padded 2 actuators past the last actuator center to avoid trying to index outside the array
    # else

    # %         dm.NdmPad = ceil_even( sqrt(2)*( 2*(dm.rmax*NpixPerAct + 1)) );
    # % padded 1/2 an actuator past the farthest actuator center (on each side) to prevent
    #   indexing outside the array

    # % DM surface array padded by the width of the padded influence function to prevent indexing
    #   outside the array. The 1/2 term is because the farthest actuator center is still half an
    #   actuator away from the nominal array edge.
    NdmPad = utils.ceil_even((NboxAS + 2 *
                              (1 + (np.max(np.max(np.abs(xy_cent_act))) + 0.5) * NpixPerAct)))

    # %--Compute coordinates (in meters) of the full DM array
    if dm['centering'] == 'pixel':
        # % meters, coords for the full DM arrays. Origin is centered on a pixel
        x_pupPad = np.arange(-NdmPad / 2, NdmPad / 2) * dx_dm

    else:
        # % meters, coords for the full DM arrays. Origin is centered between pixels for an even-sized array
        x_pupPad = np.arange(-(NdmPad - 1) / 2, (NdmPad - 1) / 2 + 1) * dx_dm

    y_pupPad = x_pupPad

    #######
    # %% DM: (use NboxPad-sized postage stamps)

    if flagGenCube:
        if not dm['flag_hex_array']:  # always true, flag_hex_array not implemented
            #
            print('  Influence function padded from %d to %d points for A.S. propagation.' % (Nbox, NboxAS))

        print('Computing datacube of DM influence functions... ')

        # %--Find the locations of the postage stamps arrays in the larger pupilPad array
        #% Convert units to pupil-file pixels
        xy_cent_act_inPix = xy_cent_act * (dm['dm_spacing'] / dx_dm)
        #%--For the half-pixel offset if pixel centered.
        xy_cent_act_inPix = xy_cent_act_inPix + 0.5

        # % Center locations of the postage stamps (in between pixels), in actuator widths
        xy_cent_act_box = np.round(xy_cent_act_inPix)

        # % now in meters
        xy_cent_act_box_inM = xy_cent_act_box * dx_dm

        # % indices of pixel in lower left of the postage stamp within the whole pupilPad array
        xy_box_lowerLeft = xy_cent_act_box + (NdmPad-Nbox) / 2 + 1

        # %--Starting coordinates (in actuator widths) for updated influence function. This is
        # % interpixel centered, so do not translate!
        x_box0 = np.arange(-(Nbox - 1) / 2, (Nbox - 1) / 2 + 1) * dx_dm
        # %--meters, interpixel-centered coordinates for the master influence function
        Xbox0, Ybox0 = np.meshgrid(x_box0, x_box0)

        # %--Limit the actuators used to those within 1 actuator width of the pupil
        r_cent_act_box_inM = np.sqrt(xy_cent_act_box_inM[0, :] ** 2 +
                                     xy_cent_act_box_inM[1, :] ** 2)

        # %--Compute and store all the influence functions:
        # %dm.Nact^2); %--initialize array of influence function "postage stamps"
        inf_datacube = np.zeros((Nbox, Nbox, NactTotal))
        act_ele = list()  # % Indices of actuators to use
        for iact in range(NactTotal):

            # Add actuator index to the keeper list
            act_ele.append(iact)

            # % X = X0 -(x_true_center-x_box_center)
            Xbox = Xbox0 - (xy_cent_act_inPix[0, iact] - xy_cent_act_box[0, iact]) * dx_dm

            # % Y = Y0 -(y_true_center-y_box_center)
            Ybox = Ybox0 - (xy_cent_act_inPix[1, iact] - xy_cent_act_box[1, iact]) * dx_dm

            # interpolation of influence function
            #inf_datacube[:,:,iact] = np.interp2(Xinf0,Yinf0,infMaster,Xbox,Ybox,'spline',0)
            #f = scipy.interpolate.interp2d(Xinf0, Yinf0, infMaster, kind='cubic')
            # assumes square array (y_inf0 == x_inf0)
            # Note: this interpolation is slightly different from the Matlab
            #    in that Matlab interpolates the boundary x=0, y=0 to evaluate to 0
            #    this interpolation gives non-zero results at the boundary
            f = scipy.interpolate.RectBivariateSpline(x_inf0, x_inf0, infMaster)
            inf_datacube[:, :, iact] = f.ev(Xbox, Ybox)

        #fprintf('done.  Time = %.1fs\n',toc);
        print('done.')

        dmoutGenCube = {
            'xy_cent_act_inPix': xy_cent_act_inPix,
            'xy_cent_act_box': xy_cent_act_box,
            'xy_cent_act_box_inM': xy_cent_act_box_inM,
            'xy_box_lowerLeft': xy_box_lowerLeft,
            'x_box0': x_box0,
            'Xbox0': Xbox0,
            'Ybox0': Ybox0,
            'inf_datacube': inf_datacube,
            'Xbox': Xbox,
            'Ybox': Ybox,
        }

    else:
        act_ele = range(NactTotal)
        dmoutGenCube = {}

    # build return dict
    dmout = dict()
    for k in dm.keys():
        dmout[k] = dm[k]

    dmout.update({
        'dx_dm': dx_dm,
        'dx': dx_dm,
        'Xact': Xact,
        'Yact': Yact,
        'NactTotal': NactTotal,
        'xy_cent_act': xy_cent_act,
        'infMaster': infMaster,
        'Nbox': Nbox,
        'NboxAS': NboxAS,
        'r_cent_act': r_cent_act,
        'rmax': rmax,
        'NdmPad': NdmPad,
        'x_pupPad': x_pupPad,
        'y_pupPad': y_pupPad,
        'act_ele': act_ele,
    })
    dmout.update(dmoutGenCube)

    return dmout
