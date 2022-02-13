"""WFSC Loop Function."""
import numpy as np
import os
import time
import pickle
import matplotlib.pyplot as plt

import falco


def loop(mp, out):
    """
    Loop over the estimator and controller for WFSC.

    Parameters
    ----------
    mp : falco.config.ModelParameters
        Structure of model parameters
    out : falco.config.Object
        Output variables

    Returns
    -------
    None
        Outputs are included in the objects mp and out.
    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')

    # # Take initial broadband image
    # Im = falco.imaging.get_summed_image(mp)

    cvar = falco.config.Object()
    ev = falco.config.Object()
    mp.thput_vec = np.zeros(mp.Nitr+1)

    for Itr in range(mp.Nitr):

        # %% Bookkeeping

        # Start of new estimation+control iteration
        print('Iteration: %d / %d\n' % (Itr, mp.Nitr-1), end='')
        cvar.Itr = Itr
        ev.Itr = Itr
        out.Itr = Itr
        out.serialDate[Itr] = time.time()

        print('Zernike modes (Noll indexing) used in this Jacobian:\t', end='')
        print(mp.jac.zerns)

        # Updated DM data
        # Change the selected DMs if using the scheduled EFC controller
        if mp.controller.lower() in ['plannedefc']:
            mp.dm_ind = mp.dm_ind_sched[Itr]

        # Report which DMs are used in this iteration
        print('DMs to be used in this iteration = [', end='')
        for jj in range(len(mp.dm_ind)):
            print(' %d' % (mp.dm_ind[jj]), end='')
        print(' ]')

        store_dm_command_history(mp, out, Itr)

        # %% Normalization and throughput calculations

        falco.imaging.calc_psf_norm_factor(mp)

        thput, ImSimOffaxis = falco.imaging.calc_thput(mp)
        out.thput[Itr] = np.max(thput)
        mp.thput_vec[Itr] = np.max(thput)

        # %% Plotting (OLD)
        # if mp.flagPlot:

        #     # Compute the DM surfaces
        #     if np.any(mp.dm_ind == 1):
        #         DM1surf = falco.dm.gen_surf_from_act(mp.dm1, mp.dm1.compact.dx, mp.dm1.compact.Ndm)
        #     else:
        #         DM1surf = np.zeros((mp.dm1.compact.Ndm, mp.dm1.compact.Ndm))

        #     if np.any(mp.dm_ind == 2):
        #         DM2surf = falco.dm.gen_surf_from_act(mp.dm2, mp.dm2.compact.dx, mp.dm2.compact.Ndm)
        #     else:
        #         DM2surf = np.zeros((mp.dm2.compact.Ndm, mp.dm2.compact.Ndm))

        #     if Itr == 0:
        #         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        #     # else:
        #     #     ax1.clear()
        #     #     ax2.clear()
        #     #     ax3.clear()
        #     #     ax4.clear()

        #     fig.subplots_adjust(hspace=0.4, wspace=0.0)
        #     fig.suptitle(mp.coro+': Iteration %d' % Itr)

        #     im1 = ax1.imshow(np.log10(Im), cmap='magma', interpolation='none',
        #                      extent=[np.min(mp.Fend.xisDL), np.max(mp.Fend.xisDL),
        #                              np.min(mp.Fend.xisDL), np.max(mp.Fend.xisDL)])
        #     ax1.set_title('Stellar PSF: NI=%.2e' % InormHist[Itr])
        #     ax1.tick_params(labelbottom=False)
        #     # cbar1 = fig.colorbar(im1, ax=ax1)

        #     im3 = ax3.imshow(ImSimOffaxis/np.max(ImSimOffaxis), cmap=plt.cm.get_cmap('Blues'),
        #                      interpolation='none', extent=[np.min(mp.Fend.xisDL), np.max(mp.Fend.xisDL),
        #                                                    np.min(mp.Fend.xisDL), np.max(mp.Fend.xisDL)])
        #     ax3.set_title('Off-axis Thput = %.2f%%' % (100*thput))
        #     # cbar3 = fig.colorbar(im3, ax=ax3)
        #     # cbar3.set_ticks(np.array([0.0, 0.5, 1.0]))
        #     # cbar3.set_ticklabels(['0', '0.5', '1'])

        #     im2 = ax2.imshow(1e9*DM1surf, cmap='viridis')
        #     ax2.set_title('DM1 Surface (nm)')
        #     ax2.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        #     # cbar2 = fig.colorbar(im2, ax=ax2)

        #     im4 = ax4.imshow(1e9*DM2surf, cmap='viridis')
        #     ax4.set_title('DM2 Surface (nm)')
        #     ax4.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        #     # cbar4 = fig.colorbar(im4, ax=ax4)

        #     if Itr == 0:
        #         cbar1 = fig.colorbar(im1, ax=ax1)
        #         cbar2 = fig.colorbar(im2, ax=ax2)
        #         cbar3 = fig.colorbar(im3, ax=ax3)
        #         cbar3.set_ticks(np.array([0.0, 0.5, 1.0]))
        #         cbar3.set_ticklabels(['0', '0.5', '1'])
        #         cbar4 = fig.colorbar(im4, ax=ax4)

        #     plt.pause(0.1)

        # %% Control Jacobian

        # Re-compute the Jacobian weights
        falco.setup.falco_set_jacobian_modal_weights(mp)

        # Compute the control Jacobians for each DM
        cvar.flagRelin = np.any(mp.relinItrVec == Itr) or Itr == 0
        if cvar.flagRelin:
            jacStruct = falco.model.jacobian(mp)

        falco.ctrl.cull_weak_actuators(mp, cvar, jacStruct)

        # %% Wavefront Estimation

        if Itr > 0:
            EestPrev = ev.Eest  # save previous estimate for Delta E plot
        falco.est.wrapper(mp, ev, jacStruct)

        store_intensities(mp, out, ev, Itr)

        # %% !!!!!!!!! Move to ctrl.py

        # # Add spatially-dependent weighting to the control Jacobians
        # if np.any(mp.dm_ind == 1):
        #     jacStruct.G1 = jacStruct.G1*np.moveaxis(np.tile(mp.WspatialVec[:, None], [mp.jac.Nmode, 1, mp.dm1.Nele]), 0, -1)
        # if np.any(mp.dm_ind == 2):
        #     jacStruct.G2 = jacStruct.G2*np.moveaxis(np.tile(mp.WspatialVec[:, None], [mp.jac.Nmode, 1, mp.dm2.Nele]), 0 ,-1)
        # if np.any(mp.dm_ind == 8):
        #     jacStruct.G8 = jacStruct.G8*np.moveaxis(np.tile(mp.WspatialVec[:, None], [mp.jac.Nmode, 1, mp.dm8.Nele]), 0, -1)
        # if np.any(mp.dm_ind == 9):
        #     jacStruct.G9 = jacStruct.G9*np.moveaxis(np.tile(mp.WspatialVec[:, None], [mp.jac.Nmode, 1, mp.dm9.Nele]), 0, -1)


        # %% Progress plots (PSF, NI, and DM surfaces)

        # # if Itr == 0:
        # #     # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        # #     figObj = plt.subplots(2, 2)
        # # figObj = plot_progress(mp, out, ev, figObj, Itr, ImSimOffaxis, Im)
        
        # plot_progress(mp, out, Itr, ImSimOffaxis, Im)

        # %% Plot the expected and measured delta E-fields

        # if Itr > 0:
        #     EsimPrev = Esim  # save previous value for Delta E plot
        # Esim = compute_simulated_efield_for_delta_efield_plot(mp)

        # if Itr > 0:
        #     out = falco_plot_DeltaE(mp, out, ev.Eest, EestPrev, Esim, EsimPrev, Itr)

        # %% Compute and Plot the Singular Mode Spectrum of the Electric Field

        if mp.flagSVD:
            pass
            # out = falco_plot_singular_mode_spectrum_of_Efield(mp, out, jacStruct, ev.Eest, Itr)

        # %% Wavefront Control

        cvar.Eest = ev.Eest
        cvar.NeleAll = mp.dm1.Nele + mp.dm2.Nele + mp.dm3.Nele + mp.dm4.Nele +\
            mp.dm5.Nele + mp.dm6.Nele + mp.dm7.Nele + mp.dm8.Nele + mp.dm9.Nele
        falco.ctrl.wrapper(mp, cvar, jacStruct)

        # Store key data in out object
        out.log10regHist[Itr] = cvar.log10regUsed
        if hasattr(cvar, 'Im') and not mp.ctrl.flagUseModel:
            out.IrawScoreHist[Itr+1] = np.mean(cvar.Im[mp.Fend.score.maskBool])
            out.IrawCorrHist[Itr+1] = np.mean(cvar.Im[mp.Fend.corr.maskBool])
            out.InormHist[Itr+1] = out.IrawCorrHist[Itr+1]

        # Enforce constraints on DM commands
        if any(mp.dm_ind == 1):
            mp.dm1 = falco.dm.enforce_constraints(mp.dm1)
        if any(mp.dm_ind == 2):
            mp.dm2 = falco.dm.enforce_constraints(mp.dm2)

        # # Update DM actuator gains for new voltages
        # if any(mp.dm_ind == 1):
        #     falco_update_dm_gain_map(mp.dm1)
        # if any(mp.dm_ind == 2):
        #     falco_update_dm_gain_map(mp.dm2)

        # %% Report Various Stats

        falco_compute_dm_stats(mp, out, Itr)

        # Calculate sensitivities Zernike phase aberrations at entrance pupil.
        if (mp.eval.Rsens.size > 0) and (mp.eval.indsZnoll.size > 0):
            out.Zsens[:, :, Itr] = falco.zern.calc_zern_sens(mp)

        # Report Normalized Intensity
        if np.abs(out.InormHist[Itr+1]) > np.finfo(float).eps:
            print('Prev and New Measured Normalized Intensity:\t\t\t '
                  '%.2e\t->\t%.2e\t (%.2f x smaller)  \n\n' %
                  (out.InormHist[Itr],
                   out.InormHist[Itr+1],
                   out.InormHist[Itr]/out.InormHist[Itr+1]))
            if not mp.flagSim:
                print('')
        else:
            print('Previous Measured NI:\t\t\t %.2e ' % (out.InormHist[Itr]))

        # Save just the 'out' object to a pickle file
        fnSnippet = os.path.join(mp.path.brief, (mp.runLabel + '_snippet.pkl'))
        print('Saving data snippet to:\n\t%s...' % (fnSnippet), end='')
        with open(fnSnippet, 'wb') as f:
            pickle.dump(out, f)
        print('done.')

        # END OF ESTIMATION + CONTROL LOOP

    # %%

    # Update 'out' structure and progress plot one last time
    Itr = Itr + 1

    store_dm_command_history(mp, out, Itr)

    # Calculate the core throughput (at higher resolution to be more accurate)
    thput, ImSimOffaxis = falco.imaging.calc_thput(mp)
    out.thput[Itr] = thput
    mp.thput_vec[Itr] = np.max(thput)

    # Update progress plot using image from controller (if new image was taken)
    if hasattr(cvar, 'Im') and not mp.ctrl.flagUseModel:
        ev.Im = cvar.Im
        # plot_progress(mp, out, Itr, ImSimOffaxis, cvar.Im)

    # Save just the 'out' object to a pickle file
    fnSnippet = os.path.join(mp.path.brief, (mp.runLabel + '_snippet.pkl'))
    print('\nSaving data snippet to:\n\t%s...' % (fnSnippet), end='')
    with open(fnSnippet, 'wb') as f:
        pickle.dump(out, f)
    print('done.\n')

    # Save out the data from the workspace
    if mp.flagSaveWS:
        del cvar
        del G
        del h
        del jacStruct

        # Don't bother saving the large 2-D, floating point maps.
        # (they take up too much space)
        mp.P1.full.mask = 1
        mp.P1.compact.mask = 1
        mp.P3.full.mask = 1
        mp.P3.compact.mask = 1
        mp.P4.full.mask = 1
        mp.P4.compact.mask = 1
        mp.F3.full.mask = 1
        mp.F3.compact.mask = 1

        mp.P1.full.E = 1
        mp.P1.compact.E = 1
        mp.dm1.full.mask = 1
        mp.dm1.compact.mask = 1
        mp.dm2.full.mask = 1
        mp.dm2.compact.mask = 1
        mp.complexTransFull = 1
        mp.complexTransCompact = 1

        mp.dm1.compact.inf_datacube = 0
        mp.dm2.compact.inf_datacube = 0
        mp.dm8.compact.inf_datacube = 0
        mp.dm9.compact.inf_datacube = 0
        mp.dm8.inf_datacube = 0
        mp.dm9.inf_datacube = 0

        fnAll = mp.path.ws + mp.runLabel + '_all.pkl'
        print('Saving entire workspace to file ' + fnAll + '...', end='')
        with open(fnAll, 'wb') as f:
            pickle.dump(mp, f)

        print('done.\n\n')
    else:
        print('Entire workspace NOT saved because mp.flagSaveWS==False')

    # END OF main FUNCTION
    print('*** END OF WFSC LOOP ***')

    return None


def store_dm_command_history(mp, out, Itr):
    """
    Store the latest DM commands in the out object.

    Parameters
    ----------
    mp : falco.config.ModelParameters
        Structure of model parameters
    out : falco.config.Object
        Output variables
    Itr : int
        The current WFSC loop iteration number.

    Returns
    -------
    None.

    """
    # Fill in History of DM commands to Store
    if hasattr(mp, 'dm1'):
        if hasattr(mp.dm1, 'V'):
            out.dm1.Vall[:, :, Itr] = mp.dm1.V
    if hasattr(mp, 'dm2'):
        if hasattr(mp.dm2, 'V'):
            out.dm2.Vall[:, :, Itr] = mp.dm2.V
    if hasattr(mp, 'dm5'):
        if hasattr(mp.dm5, 'V'):
            out.dm5.Vall[:, :, Itr] = mp.dm5.V
    if hasattr(mp, 'dm8'):
        if hasattr(mp.dm8, 'V'):
            out.dm8.Vall[:, Itr] = mp.dm8.V[:]
    if hasattr(mp, 'dm9'):
        if hasattr(mp.dm9, 'V'):
            out.dm9.Vall[:, Itr] = mp.dm9.V[:]

    return None


def falco_compute_dm_stats(mp, out, Itr):
    """
    Compute statistics on the DM1 and DM2 surface actuations.

    Parameters
    ----------
    mp : falco.config.ModelParameters
        Structure of model parameters
    out : falco.config.Object
        Output variables
    Itr : int
        The current WFSC loop iteration number.

    Returns
    -------
    None.

    """
    # Calculate and report updated P-V DM voltages.
    if np.any(mp.dm_ind == 1):
        out.dm1.Vpv[Itr] = np.max(mp.dm1.V) - np.min(mp.dm1.V)
        print(' DM1 P-V in volts: %.3f' % (out.dm1.Vpv[Itr]))
        if(mp.dm1.tied.size > 0):
            print(' DM1 has %d pairs of tied actuators.' % (mp.dm1.tied.shape[0]))

    if np.any(mp.dm_ind == 2):
        out.dm2.Vpv[Itr] = np.max(mp.dm2.V) - np.min(mp.dm2.V)
        print(' DM2 P-V in volts: %.3f' % (out.dm2.Vpv[Itr]))
        if(mp.dm2.tied.size > 0):
            print(' DM2 has %d pairs of tied actuators.' % (mp.dm2.tied.shape[0]))

    # Calculate and report updated RMS DM surfaces.
    if(any(mp.dm_ind == 1)):
        # Pupil-plane coordinates
        dx_dm = mp.P2.compact.dx/mp.P2.D  # Normalized dx [Units of pupil diameters]
        xs = falco.util.create_axis(mp.dm1.compact.Ndm, dx_dm, centering=mp.centering)
        RS = falco.util.radial_grid(xs)
        rmsSurf_ele = np.logical_and(RS >= mp.P1.IDnorm/2., RS <= 0.5)

        DM1surf = falco.dm.gen_surf_from_act(mp.dm1, mp.dm1.compact.dx, mp.dm1.compact.Ndm)
        out.dm1.Spv[Itr] = np.max(DM1surf)-np.min(DM1surf)
        out.dm1.Srms[Itr] = np.sqrt(np.mean(np.abs((DM1surf[rmsSurf_ele]))**2))
        print('RMS surface of DM1 = %.1f nm' % (1e9*out.dm1.Srms[Itr]))
    if(any(mp.dm_ind == 2)):
        # Pupil-plane coordinates
        dx_dm = mp.P2.compact.dx/mp.P2.D  # Normalized dx [Units of pupil diameters]
        xs = falco.util.create_axis(mp.dm2.compact.Ndm, dx_dm, centering=mp.centering)
        RS = falco.util.radial_grid(xs)
        rmsSurf_ele = np.logical_and(RS >= mp.P1.IDnorm/2., RS <= 0.5)

        DM2surf = falco.dm.gen_surf_from_act(mp.dm2, mp.dm2.compact.dx, mp.dm2.compact.Ndm)
        out.dm2.Spv[Itr] = np.max(DM2surf)-np.min(DM2surf)
        out.dm2.Srms[Itr] = np.sqrt(np.mean(np.abs((DM2surf[rmsSurf_ele]))**2))
        print('RMS surface of DM2 = %.1f nm' % (1e9*out.dm2.Srms[Itr]))

    return None


def store_intensities(mp, out, ev, Itr):
    """Store newest intensities in the out object."""

    # ## Calculate the average measured, coherent, and incoherent intensities

    # Apply subband weights and then sum over subbands
    Iest = np.abs(ev.Eest)**2
    Iinco = ev.IincoEst
    for iMode in range(mp.jac.Nmode):
        iSubband = mp.jac.sbp_inds[iMode]
        Iest[:, iMode] = mp.sbp_weights[iSubband] * Iest[:, iMode]
        Iinco[:, iMode] = mp.sbp_weights[iSubband] * Iinco[:, iMode]

    IestAllBands = np.sum(Iest, axis=1)
    IincoAllBands = np.sum(Iinco, axis=1)

    # Put intensities back into 2-D arrays to use correct indexing of scoring region.
    # Modulated
    Iest2D = np.zeros((mp.Fend.Neta, mp.Fend.Nxi))
    Iest2D[mp.Fend.corr.maskBool] = IestAllBands.flatten()
    out.IestScoreHist[Itr] = np.mean(Iest2D[mp.Fend.score.maskBool])
    out.IestCorrHist[Itr] = np.mean(Iest2D[mp.Fend.corr.maskBool])
    # Unmodulated
    Iinco2D = np.zeros((mp.Fend.Neta, mp.Fend.Nxi))
    Iinco2D[mp.Fend.corr.maskBool] = IincoAllBands.flatten()
    out.IincoScoreHist[Itr] = np.mean(Iinco2D[mp.Fend.score.maskBool])
    out.IincoCorrHist[Itr] = np.mean(Iinco2D[mp.Fend.corr.maskBool])

    # Measured
    out.IrawScoreHist[Itr] = np.mean(ev.Im[mp.Fend.score.maskBool])
    out.IrawCorrHist[Itr] = np.mean(ev.Im[mp.Fend.corr.maskBool])
    out.InormHist[Itr] = out.IrawCorrHist[Itr]  # a vestigial variable

    # %% Calculate the measured, coherent, and incoherent intensities by subband

    # measured intensities
    for iSubband in range(mp.Nsbp):
        imageMeas = np.squeeze(ev.imageArray[:, :, 0, iSubband]);
        out.normIntMeasCorr[Itr, iSubband] = np.mean(imageMeas[mp.Fend.corr.maskBool])
        out.normIntMeasScore[Itr, iSubband] = np.mean(imageMeas[mp.Fend.score.maskBool])
        del imageMeas

    # estimated
    for iMode in range(mp.jac.Nmode):

        imageModVec = np.abs(ev.Eest[:, iMode])**2
        imageUnmodVec = ev.IincoEst[:, iMode]

        out.normIntModCorr[Itr, iMode] = np.mean(imageModVec)
        out.normIntModScore[Itr, iMode] = np.mean(imageModVec[mp.Fend.scoreInCorr])

        out.normIntUnmodCorr[Itr, iMode] = np.mean(imageUnmodVec)
        out.normIntUnmodScore[Itr, iMode] = np.mean(imageUnmodVec[mp.Fend.scoreInCorr])

    pass


def plot_progress(mp, out, Itr, ImSimOffaxis, Im):
    # Plotting
    if mp.flagPlot:

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, num=99)
        #fig, ((ax1, ax2), (ax3, ax4)) = figObj

        # Compute the DM surfaces
        if np.any(mp.dm_ind == 1):
            DM1surf = falco.dm.gen_surf_from_act(mp.dm1, mp.dm1.compact.dx, mp.dm1.compact.Ndm)
        else:
            DM1surf = np.zeros((mp.dm1.compact.Ndm, mp.dm1.compact.Ndm))

        if np.any(mp.dm_ind == 2):
            DM2surf = falco.dm.gen_surf_from_act(mp.dm2, mp.dm2.compact.dx, mp.dm2.compact.Ndm)
        else:
            DM2surf = np.zeros((mp.dm2.compact.Ndm, mp.dm2.compact.Ndm))

        # if Itr == 0:
        #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        # # else:
        # #     ax1.clear()
        # #     ax2.clear()
        # #     ax3.clear()
        # #     ax4.clear()

        fig.subplots_adjust(hspace=0.4, wspace=0.0)
        fig.suptitle(mp.coro+': Iteration %d' % Itr)

        im1 = ax1.imshow(np.log10(Im), cmap='magma', interpolation='none',
                         extent=[np.min(mp.Fend.xisDL), np.max(mp.Fend.xisDL),
                                 np.min(mp.Fend.xisDL), np.max(mp.Fend.xisDL)])
        ax1.set_title('Stellar PSF: NI=%.2e' % out.InormHist[Itr])
        ax1.tick_params(labelbottom=False)
        # cbar1 = fig.colorbar(im1, ax=ax1)

        im3 = ax3.imshow(ImSimOffaxis/np.max(ImSimOffaxis), cmap=plt.cm.get_cmap('Blues'),
                         interpolation='none', extent=[np.min(mp.Fend.xisDL), np.max(mp.Fend.xisDL),
                                                       np.min(mp.Fend.xisDL), np.max(mp.Fend.xisDL)])
        ax3.set_title('Off-axis Thput = %.2f%%' % (100*mp.thput_vec[Itr]))
        # cbar3 = fig.colorbar(im3, ax=ax3)
        # cbar3.set_ticks(np.array([0.0, 0.5, 1.0]))
        # cbar3.set_ticklabels(['0', '0.5', '1'])

        im2 = ax2.imshow(1e9*DM1surf, cmap='viridis')
        ax2.set_title('DM1 Surface (nm)')
        ax2.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        # cbar2 = fig.colorbar(im2, ax=ax2)

        im4 = ax4.imshow(1e9*DM2surf, cmap='viridis')
        ax4.set_title('DM2 Surface (nm)')
        ax4.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        # cbar4 = fig.colorbar(im4, ax=ax4)

        cbar1 = fig.colorbar(im1, ax=ax1)
        cbar2 = fig.colorbar(im2, ax=ax2)
        cbar3 = fig.colorbar(im3, ax=ax3)
        cbar3.set_ticks(np.array([0.0, 0.5, 1.0]))
        cbar3.set_ticklabels(['0', '0.5', '1'])
        cbar4 = fig.colorbar(im4, ax=ax4)

        # if Itr == 0:
        #     cbar1 = fig.colorbar(im1, ax=ax1)
        #     cbar2 = fig.colorbar(im2, ax=ax2)
        #     cbar3 = fig.colorbar(im3, ax=ax3)
        #     cbar3.set_ticks(np.array([0.0, 0.5, 1.0]))
        #     cbar3.set_ticklabels(['0', '0.5', '1'])
        #     cbar4 = fig.colorbar(im4, ax=ax4)

        plt.pause(0.1)

        return None
