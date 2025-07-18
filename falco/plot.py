"""FALCO plotting."""
import numpy as np
import pickle
import matplotlib.pyplot as plt

# from . import check
import falco


def wfsc_progress(mp, out, ev, Itr, ImSimOffaxis):
    """Plot WFSC progress (On-axis PSF, off-axis PSF, and DM shapes)."""
    if mp.flagPlot:

        Im = ev.Im

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
        plt.figure(100)
        plt.clf()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, num=100)

        # else:
        # if Itr > 0:
        #     ax1.clear()
        #     ax2.clear()
        #     ax3.clear()
        #     ax4.clear()

        fig.subplots_adjust(hspace=0.4, wspace=0.0)
        fig.suptitle(mp.coro+': Iteration %d' % Itr)

        im1 = ax1.imshow(np.log10(Im), cmap='magma', interpolation='none',
                         origin='lower',
                         extent=[np.min(mp.Fend.xisDL), np.max(mp.Fend.xisDL),
                                 np.min(mp.Fend.xisDL), np.max(mp.Fend.xisDL)])
        ax1.set_title('Stellar PSF: NI=%.2e' % out.InormHist[Itr])
        ax1.tick_params(labelbottom=False)
        # cbar1 = fig.colorbar(im1, ax=ax1)

        im3 = ax3.imshow(ImSimOffaxis/np.max(ImSimOffaxis),
                         cmap=plt.cm.get_cmap('Blues'),
                         interpolation='none',
                         origin='lower',
                         extent=[np.min(mp.Fend.xisDL), np.max(mp.Fend.xisDL),
                                 np.min(mp.Fend.xisDL), np.max(mp.Fend.xisDL)])
        ax3.set_title('Off-axis Thput = %.2f%%' % (100*out.thput[Itr]))
        # cbar3 = fig.colorbar(im3, ax=ax3)
        # cbar3.set_ticks(np.array([0.0, 0.5, 1.0]))
        # cbar3.set_ticklabels(['0', '0.5', '1'])

        im2 = ax2.imshow(1e9*DM1surf, origin='lower', cmap='viridis')
        ax2.set_title('DM1 Surface (nm)')
        ax2.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        # cbar2 = fig.colorbar(im2, ax=ax2)

        im4 = ax4.imshow(1e9*DM2surf, origin='lower', cmap='viridis')
        ax4.set_title('DM2 Surface (nm)')
        ax4.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        # cbar4 = fig.colorbar(im4, ax=ax4)

        # if Itr == 0:
        cbar1 = fig.colorbar(im1, ax=ax1)
        cbar2 = fig.colorbar(im2, ax=ax2)
        cbar3 = fig.colorbar(im3, ax=ax3)
        cbar3.set_ticks(np.array([0.0, 0.5, 1.0]))
        cbar3.set_ticklabels(['0', '0.5', '1'])
        cbar4 = fig.colorbar(im4, ax=ax4)

        plt.pause(0.1)


def plot_trial_output(out):
    """
    Plot a FALCO trial's data using a FALCO object as the input.

    Parameters
    ----------
    out : FALCO object
        Object containing performance data from the FALCO trial.

    Returns
    -------
    None
    """

    plt.figure()
    plt.plot(range(out.Nitr+1), out.thput)
    plt.xlabel('Iteration')
    plt.ylabel('Throughput')

    plt.figure()
    plt.semilogy(range(out.Nitr+1), out.InormHist)
    plt.xlabel('Iteration')
    plt.ylabel('Normalized Intensity')

    plt.figure()
    plt.plot(range(out.Nitr), out.log10regHist)
    plt.xlabel('Iteration')
    plt.ylabel('log10 Regularization')

    plt.figure()
    plt.plot(range(out.Nitr), 1e9*out.dm1.Srms, '-r',
             range(out.Nitr), 1e9*out.dm2.Srms, '-b')
    plt.xlabel('Iteration')
    plt.ylabel('RMS DM Surface (nm)')

    plt.figure()
    plt.imshow(out.dm1.Vall[:, :, -1])
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Final DM1 Voltages')

    plt.figure()
    plt.imshow(out.dm2.Vall[:, :, -1])
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Final DM2 Voltages')

    iterCount = 1
    plt.figure()
    plt.imshow(out.dm1.Vall[:, :, iterCount])
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title(('DM1 Voltages at Iteration %d' % (iterCount)))

    return None


def plot_trial_output_from_pickle(fnPickle):
    """
    Plot a FALCO trial's pickled output data.

    Parameters
    ----------
    fnPickle : str
        Filename of pickle containing performance data from the FALCO trial.

    Returns
    -------
    None
    """

    # with np.load(fnPickle, allow_pickle=True) as data:
    #     out = data['out']
    # out = pickle.load(fnPickle)

    with open(fnPickle, 'rb') as pickle_file:
        out = pickle.load(pickle_file)

    plot_trial_output(out)


def delta_efield(mp, out, Eest, EestPrev, Esim, EsimPrev, Itr):
    """
    Plot the model-based and estimated change in E-field at each subband.

    Parameters
    ----------
    mp : falco.config.ModelParameters()
        Object of FALCO model parameters.
    out : types.SimpleNamespace
        Object containing performance data from the FALCO trial.
    Eest : array_like
        Vectorized E-field estimate from the current iteration.
    EestPrev : array_like
        Vectorized E-field estimate from the previous iteration.
    Esim : array_like
        Vectorized model-based E-field from the current iteration.
    EsimPrev : array_like
        Vectorized model-based E-field from the previous iteration.
    Itr : int
        WFSC iteration number.

    Returns
    -------
    None.

    """
    for iSubband in range(mp.Nsbp):
        dEmeas = np.squeeze(Eest[:, iSubband] - EestPrev[:, iSubband])
        dEmeas2D = np.zeros((mp.Fend.Neta, mp.Fend.Nxi), dtype=complex)
        dEmeas2D[mp.Fend.corr.maskBool] = dEmeas  # 2-D for plotting
        indsNonzero = np.nonzero(dEmeas != 0)[0]
        # Skip zeroed values when computing complex projection and correlation
        dEmeasNonzero = dEmeas[indsNonzero].reshape([-1, 1])

        dEsim = np.squeeze(Esim[:, iSubband] - EsimPrev[:, iSubband])
        dEsim2D = np.zeros((mp.Fend.Neta, mp.Fend.Nxi), dtype=complex)
        dEsim2D[mp.Fend.corr.maskBool] = dEsim  # 2-D for plotting
        dEsimNonzero = dEsim[indsNonzero].reshape([-1, 1])

        out.complexProjection[Itr-1, iSubband] = \
            (np.abs(dEsimNonzero.T @ dEmeasNonzero) / np.abs(dEsimNonzero.T @ dEsimNonzero)).item()
        print('Complex projection of deltaE is %3.2f    for subband %d/%d' %
              (out.complexProjection[Itr-1, iSubband], iSubband, mp.Nsbp-1))
        out.complexCorrelation[Itr-1, iSubband] = \
            (np.abs(dEsimNonzero.T @ dEmeasNonzero/(np.sqrt(np.abs(dEmeasNonzero.T @ dEmeasNonzero))*np.sqrt(np.abs(dEsimNonzero.T @ dEsimNonzero))))).item()
        print('Complex correlation of deltaE is %3.2f    for subband %d/%d' %
              (out.complexCorrelation[Itr-1, iSubband], iSubband, mp.Nsbp-1))

        if mp.flagPlot:

            dEmax = np.max(np.abs(dEsim))  # max value in plots

            figNum = 50+iSubband
            plt.figure(figNum)
            plt.clf()
            fig, axs = plt.subplots(2, 2, num=figNum)
            cmaps = ['viridis', 'hsv']
            titles = [['abs($dE_{model}$)', 'angle($dE_{model}$)'],
                      ['abs($dE_{meas}$)', 'angle($dE_{meas}$)']]
            data = [[np.abs(dEsim2D), np.angle(dEsim2D)],
                    [np.abs(dEmeas2D), np.angle(dEmeas2D)]]

            for col in range(2):
                for row in range(2):
                    ax = axs[row, col]
                    ax.set_title(titles[row][col])
                    ax.invert_yaxis()
                    ax.set_box_aspect(1)
                    if row == 0:
                        ax.tick_params(labelbottom=False, bottom=False)
                    if col == 1:
                        ax.tick_params(labelleft=False, left=False)

                    pcm = ax.pcolormesh(data[row][col],
                                        cmap=cmaps[col])
                fig.colorbar(pcm, ax=axs[:, col]) #, shrink=0.6)


def singular_mode_spectrum_of_Efield(mp, out, jacStruct, Eest, Itr):

    iMode = 0
    Gcomplex = np.concatenate((jacStruct.G1[:, :, iMode],
                               jacStruct.G2[:, :, iMode],
                               jacStruct.G8[:, :, iMode],
                               jacStruct.G9[:, :, iMode]), axis=1)
    Gall = np.zeros((mp.jac.Nmode*Gcomplex.shape[0], Gcomplex.shape[1]),
                    dtype=complex)
    Eall = np.zeros((mp.jac.Nmode*Eest.shape[0], 1), dtype=complex)

    for iMode in range(mp.jac.Nmode):
        N = Gcomplex.shape[0]
        Gcomplex = np.concatenate((jacStruct.G1[:, :, iMode],
                                   jacStruct.G2[:, :, iMode],
                                   jacStruct.G8[:, :, iMode],
                                   jacStruct.G9[:, :, iMode]), axis=1)
        Gall[iMode*N:(iMode+1)*N, :] = Gcomplex
        Eall[iMode*N:(iMode+1)*N, 0] = Eest[:, iMode]

    Eri = np.concatenate((np.real(Eall), np.imag(Eall)), axis=0)
    alpha2 = np.max(np.diag(np.real(Gall.conj().T @ Gall)))
    Gri = np.concatenate((np.real(Gall), np.imag(Gall)), axis=0)
    U, s, _ = np.linalg.svd(Gri, full_matrices=False)

    EriPrime = U.conj().T @ Eri
    IriPrime = np.abs(EriPrime)**2

    # Store data for later analysis
    out.EforSpectra.append(EriPrime)
    out.smspectra.append(IriPrime)
    out.sm.append(s)
    out.alpha2.append(alpha2)

    if mp.flagPlot:
        plt.figure(401)
        if Itr > 0:
            plt.clf()
        for ii in range(Itr+1):
            plt.loglog(
                (out.sm[ii]**2/out.alpha2[ii]).reshape([-1]),
                falco.util.smooth(out.smspectra[ii].reshape([-1]), 31),
                color=(0.3, 1-(0.2+(ii+1)/(Itr+1))/(1.3), 1)
                )
        plt.title('Singular Mode Spectrum')
        plt.xlim(1e-10, 2*np.max(s**2/alpha2))
        plt.ylim(1e-12, 1e-0)
        plt.pause(1e-2)


def pairwise_probes(mp, ev, dDMVplus, ampSq2Dcube, iSubband):

    if mp.flagPlot:

        if mp.est.probe.whichDM == 1:
            VtoH = mp.dm1.VtoH
        elif mp.est.probe.whichDM == 2:
            VtoH = mp.dm2.VtoH

        Npairs = mp.est.probe.Npairs
        Ncols = 4

        # Npairs rows, and 4 columns
        figNum = 90 + ev.iStar
        plt.figure(figNum)
        plt.clf()
        fig, axs = plt.subplots(nrows=Npairs, ncols=Ncols, num=figNum)
        cmaps = Ncols*['gray']
        cmaps[0] = 'viridis'
        titles = ['DM Command (nm)', '+Probe Image', '-Probe Image',
                  'Probe Intensity, $|\Delta p|^2$']

        plusImageCube = ev.imageArray[:, :, 1::2, iSubband]
        minusImageCube = ev.imageArray[:, :, 2::2, iSubband]

        for col in range(Ncols):
            for row in range(Npairs):
                ax = axs[row, col]
                ax.set_title(titles[col])
                ax.invert_yaxis()
                ax.set_box_aspect(1)
                if row != Npairs-1:
                    ax.tick_params(labelbottom=False, bottom=False)
                if col != 0:
                    ax.tick_params(labelleft=False, left=False)

                if col == 0:
                    datacube = 1e9*dDMVplus * \
                        np.moveaxis((np.tile(VtoH, [Npairs, 1, 1])), 0, 2)
                elif col == Ncols-1:
                    datacube = ampSq2Dcube
                elif col > 0 and col % 2 == 1:
                    datacube = plusImageCube
                elif col > 0 and col % 2 == 0:
                    datacube = minusImageCube

                if col == 0:
                    pcm = ax.pcolormesh(datacube[:, :, row], cmap=cmaps[col])
                else:
                    pcm = ax.pcolormesh(log10nonneg(datacube[:, :, row]),
                                        cmap=cmaps[col])

            fig.colorbar(pcm, ax=axs[:, col])  #, shrink=0.6)

        return None


def log10nonneg(array):
    array[array < 0] = 0
    return np.log10(array)
