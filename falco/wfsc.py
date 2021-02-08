"""WFSC Loop Function."""

import numpy as np
import os
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
    pass

    # Sort out file paths and save the config file
    
    # Add the slash or backslash to the FALCO path if it isn't there.
    filesep = os.pathsep

    if mp.path.falco[-1] != '/' and mp.path.falco[-1] != '\\':
        mp.path.falco = mp.path.falco + filesep
        
    # Store minimal data to re-construct the data from the run:
    # the "out" structure after a trial goes here
    if not hasattr(mp.path, 'config'):
        mp.path.config = mp.path.falco + filesep + 'data' + filesep + 'config' + filesep
    
    # Entire final workspace from FALCO gets saved here.
    if not hasattr(mp.path, 'ws'):
        mp.path.ws = mp.path.falco + filesep + 'data' + filesep + 'ws' + filesep
    
    # Measured, mean raw contrast in scoring regino of dark hole.
    InormHist = np.zeros((mp.Nitr+1,))
    
    # Take initial broadband image
    Im = falco.imaging.get_summed_image(mp)
    
    ###########################################################################
    # Begin the Correction Iterations
    ###########################################################################

    mp.flagCullActHist = np.zeros((mp.Nitr+1,), dtype=np.bool)

    for Itr in range(mp.Nitr):
    
        # Start of new estimation+control iteration
        print('Iteration: %d / %d\n' % (Itr, mp.Nitr-1), end='')
        
        # Re-normalize PSF after latest DM commands
        falco.imaging.calc_psf_norm_factor(mp)

        # Updated DM data
        # Change the selected DMs if using the scheduled EFC controller
        if mp.controller.lower() in ['plannedefc']:
            mp.dm_ind = mp.dm_ind_sched[Itr]
    
        # Report which DMs are used in this iteration
        print('DMs to be used in this iteration = [', end='')
        for jj in range(len(mp.dm_ind)):
            print(' %d' % (mp.dm_ind[jj]), end='')
        print(' ]')
        
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
    
        # Compute the DM surfaces
        if np.any(mp.dm_ind == 1):
            DM1surf = falco.dm.gen_surf_from_act(mp.dm1, mp.dm1.compact.dx, mp.dm1.compact.Ndm)
        else:
            DM1surf = np.zeros((mp.dm1.compact.Ndm, mp.dm1.compact.Ndm))
    
        if np.any(mp.dm_ind == 2):
            DM2surf = falco.dm.gen_surf_from_act(mp.dm2, mp.dm2.compact.dx, mp.dm2.compact.Ndm)
        else:
            DM2surf = np.zeros((mp.dm2.compact.Ndm, mp.dm2.compact.Ndm))
    
        # Updated plot and reporting
        # Calculate the core throughput (at higher resolution to be more accurate)
        thput, ImSimOffaxis = falco.imaging.calc_thput(mp)
        if mp.flagFiber:
            mp.thput_vec[Itr] = np.max(thput)
        else:
            mp.thput_vec[Itr] = thput  # record keeping
        
        # Compute the current contrast level
        InormHist[Itr] = np.mean(Im[mp.Fend.corr.maskBool])
        
        if(any(mp.dm_ind == 1)):
            mp.dm1 = falco.dm.enforce_constraints(mp.dm1)
        if(any(mp.dm_ind == 2)):
            mp.dm2 = falco.dm.enforce_constraints(mp.dm2)
        
        # Plotting
        if(mp.flagPlot):
            if Itr == 0:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            # else:
            #     ax1.clear()
            #     ax2.clear()
            #     ax3.clear()
            #     ax4.clear()
                
            fig.subplots_adjust(hspace=0.4, wspace=0.0)
            fig.suptitle(mp.coro+': Iteration %d' % Itr)
            
            im1 = ax1.imshow(np.log10(Im), cmap='magma', interpolation='none',
                             extent=[np.min(mp.Fend.xisDL), np.max(mp.Fend.xisDL),
                                     np.min(mp.Fend.xisDL), np.max(mp.Fend.xisDL)])
            ax1.set_title('Stellar PSF: NI=%.2e' % InormHist[Itr])
            ax1.tick_params(labelbottom=False)
            # cbar1 = fig.colorbar(im1, ax=ax1)

            im3 = ax3.imshow(ImSimOffaxis/np.max(ImSimOffaxis), cmap=plt.cm.get_cmap('Blues'),
                             interpolation='none', extent=[np.min(mp.Fend.xisDL), np.max(mp.Fend.xisDL),
                                                           np.min(mp.Fend.xisDL), np.max(mp.Fend.xisDL)])
            ax3.set_title('Off-axis Thput = %.2f%%' % (100*thput))
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
            
            if Itr == 0:
                cbar1 = fig.colorbar(im1, ax=ax1)
                cbar2 = fig.colorbar(im2, ax=ax2)
                cbar3 = fig.colorbar(im3, ax=ax3)
                cbar3.set_ticks(np.array([0.0, 0.5, 1.0]))
                cbar3.set_ticklabels(['0', '0.5', '1'])
                cbar4 = fig.colorbar(im4, ax=ax4)
            
            plt.pause(0.1)
            
        # Updated selection of Zernike modes targeted by the controller
        # Decide with Zernike modes to include in the Jacobian
        if Itr == 0:
            mp.jac.zerns0 = mp.jac.zerns;
        
        print('Zernike modes (Noll indexing) used in this Jacobian:\t', end='')
        print(mp.jac.zerns)
        
        # Re-compute the Jacobian weights
        falco.setup.falco_set_jacobian_weights(mp)
        
        # Actuator Culling: Initialization of Flag and Which Actuators
        
        # If set of DMs changes or Itr is 0, perform a new cull of actuators.
        cvar = falco.config.Object()
        if Itr == 0:
            cvar.flagCullAct = True
        else:
            if hasattr(mp, 'dm_ind_sched'):
                schedPre = np.sort(mp.dm_ind_sched[Itr-1])
                schedNow = np.sort(mp.dm_ind_sched[Itr])
                if not schedPre.size == schedNow.size:
                    cvar.flagCullAct = True
                else:  # when they are same size
                    doesEachValueMatch = np.not_equal(schedNow, schedPre)
                    if all(doesEachValueMatch):
                        cvar.flagCullAct = False
                    else:
                        cvar.flagCullAct = True
            else:
                cvar.flagCullAct = False
        
        mp.flagCullActHist[Itr] = cvar.flagCullAct
        
        # Before performing new cull, include all actuators again
        if cvar.flagCullAct:
            # Re-include all actuators in the basis set.
            if np.any(mp.dm_ind == 1):
                mp.dm1.act_ele = list(range(mp.dm1.NactTotal))
            if np.any(mp.dm_ind == 2):
                mp.dm2.act_ele = list(range(mp.dm2.NactTotal))
            if np.any(mp.dm_ind == 8):
                mp.dm8.act_ele = list(range(mp.dm8.NactTotal))
            if np.any(mp.dm_ind == 9):
                mp.dm9.act_ele = list(range(mp.dm9.NactTotal))
                
            # Update the number of elements used per DM
            if np.any(mp.dm_ind == 1):
                mp.dm1.Nele = len(mp.dm1.act_ele)
            else:
                mp.dm1.Nele = 0
            if np.any(mp.dm_ind == 2):
                mp.dm2.Nele = len(mp.dm2.act_ele)
            else:
                mp.dm2.Nele = 0
            if np.any(mp.dm_ind == 8):
                mp.dm8.Nele = len(mp.dm8.act_ele)
            else:
                mp.dm8.Nele = 0
            if np.any(mp.dm_ind == 9):
                mp.dm9.Nele = len(mp.dm9.act_ele)
            else:
                mp.dm9.Nele = 0
             
        # Whether to relinearize about the DMs surfaces
        if np.any(mp.relinItrVec == Itr):
            cvar.flagRelin = True
        else:
            cvar.flagRelin = False
        
        # Compute the control Jacobians for each DM
        if Itr == 0 or cvar.flagRelin:
            jacStruct = falco.model.jacobian(mp)
        
        # Modify jacStruct to cull actuators
        falco.ctrl.cull_actuators(mp, cvar, jacStruct)
        
        # Load the improved Jacobian if using the E-M technique
        # if mp.flagUseLearnedJac:
        #     jacStructLearned = load('jacStructLearned.mat');
        #     if np.any(mp.dm_ind == 1):
        #     jacStruct.G1 = jacStructLearned.G1
        #     if np.any(mp.dm_ind == 1):
        #     jacStruct.G2 = jacStructLearned.G2
        
        # Wavefront Estimation
        if mp.estimator.lower() in ['perfect']:
            EfieldVec = falco.est.perfect(mp)
        elif mp.estimator.lower() in ['pwp-bp', 'pwp-kf']:
            if Itr == 0:
                ev = falco.config.Object()
            ev.Itr = Itr
                
            if mp.est.flagUseJac:  # Send in the Jacobian if true
                falco.est.pairwise_probing(mp, ev, jacStruct=jacStruct)
            else:  # Otherwise don't pass the Jacobian
                falco.est.pairwise_probing(mp, ev)
            EfieldVec = ev.Eest
            IincoVec = ev.IincoEst
                
        # Add spatially-dependent weighting to the control Jacobians
        if np.any(mp.dm_ind == 1):
            jacStruct.G1 = jacStruct.G1*np.moveaxis(np.tile(mp.WspatialVec[:, None], [mp.jac.Nmode, 1, mp.dm1.Nele]), 0, -1)
        if np.any(mp.dm_ind == 2):
            jacStruct.G2 = jacStruct.G2*np.moveaxis(np.tile(mp.WspatialVec[:, None], [mp.jac.Nmode, 1, mp.dm2.Nele]), 0 ,-1)
        if np.any(mp.dm_ind == 8):
            jacStruct.G8 = jacStruct.G8*np.moveaxis(np.tile(mp.WspatialVec[:, None], [mp.jac.Nmode, 1, mp.dm8.Nele]), 0, -1)
        if np.any(mp.dm_ind == 9):
            jacStruct.G9 = jacStruct.G9*np.moveaxis(np.tile(mp.WspatialVec[:, None], [mp.jac.Nmode, 1, mp.dm9.Nele]), 0, -1)
                
        # Compute the total number of actuators for all DMs used
        cvar.NeleAll = mp.dm1.Nele + mp.dm2.Nele + mp.dm3.Nele + mp.dm4.Nele +\
            mp.dm5.Nele + mp.dm6.Nele + mp.dm7.Nele + mp.dm8.Nele + mp.dm9.Nele
            
        # Wavefront Control
        cvar.Itr = Itr
        cvar.EfieldVec = EfieldVec
        cvar.InormHist = InormHist[Itr]
        falco.ctrl.wrapper(mp, cvar, jacStruct)
    
        # Save out regularization used.
        if hasattr(cvar, "log10regUsed"):
            out.log10regHist[Itr] = cvar.log10regUsed
    
        # ---------------------------------------------------------------------
        
        # DM Stats
        # Calculate and report updated P-V DM voltages.
        if np.any(mp.dm_ind == 1):
            out.dm1.Vpv[Itr] = np.max(mp.dm1.V) - np.min(mp.dm1.V)
            print(' DM1 P-V in volts: %.3f' % (out.dm1.Vpv[Itr]))
            if(mp.dm1.tied.size > 0):
                print(' DM1 has %d pairs of tied actuators.' % (mp.dm1.tied.shape[0]))
#            Nrail1 = len(np.where( (mp.dm1.V <= -mp.dm1.maxAbsV) | (mp.dm1.V >= mp.dm1.maxAbsV) )) 
#            print(' DM1 P-V in volts: %.3f\t\t%d/%d (%.2f%%) railed actuators \n'%(out.dm1.Vpv(Itr), Nrail1, mp.dm1.NactTotal, 100*Nrail1/mp.dm1.NactTotal))
#            if mp.dm1.tied.shape[0]>0:
#                print(' DM1 has %d pairs of tied actuators.\n'%(mp.dm1.tied.shape[0]))
        if np.any(mp.dm_ind == 2):
            out.dm2.Vpv[Itr] = np.max(mp.dm2.V) - np.min(mp.dm2.V)
            print(' DM2 P-V in volts: %.3f' % (out.dm2.Vpv[Itr]))
            if(mp.dm2.tied.size > 0):
                print(' DM2 has %d pairs of tied actuators.' % (mp.dm2.tied.shape[0]))

#            Nrail2 = len(np.where( (mp.dm2.V <= -mp.dm2.maxAbsV) | (mp.dm2.V >= mp.dm2.maxAbsV) ))
#            print(' DM2 P-V in volts: %.3f\t\t%d/%d (%.2f%%) railed actuators \n'%( out.dm2.Vpv(Itr), Nrail2, mp.dm2.NactTotal, 100*Nrail2/mp.dm2.NactTotal))
#            if mp.dm2.tied.shape[0]>0:  
#                print(' DM2 has %d pairs of tied actuators.\n'%(mp.dm2.tied.shape[0]))
        # if(any(mp.dm_ind == 8))
        #     out.dm8.Vpv(Itr) = (max(max(mp.dm8.V))-min(min(mp.dm8.V)));
        #     Nrail8 = length(find( (mp.dm8.V <= mp.dm8.Vmin) | (mp.dm8.V >= mp.dm8.Vmax) ))
        #     fprintf(' DM8 P-V in volts: #.3f\t\t#d/#d (#.2f##) railed actuators \n', out.dm8.Vpv(Itr), Nrail8,mp.dm8.NactTotal,100*Nrail8/mp.dm8.NactTotal); 
        # end
        # if np.any(mp.dm_ind == 9):
        #     out.dm9.Vpv[Itr] = np.max(np.max(mp.dm9.V))-np.min(np.min(mp.dm9.V))
        #     Nrail9 = len(np.where( (mp.dm9.V <= mp.dm9.Vmin) | (mp.dm9.V >= mp.dm9.Vmax) ))
        #     print(' DM9 P-V in volts: %.3f\t\t%d/%d (%.2f%%) railed actuators \n'%(out.dm9.Vpv(Itr), Nrail9,mp.dm9.NactTotal,100*Nrail9/mp.dm9.NactTotal))
        
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
        
        # Calculate sensitivities to 1nm RMS of Zernike phase aberrations at entrance pupil.
        if (mp.eval.Rsens.size > 0) and (mp.eval.indsZnoll.size > 0):
            out.Zsens[:, :, Itr] = falco.zern.calc_zern_sens(mp)
        
        # Take the next image to check the contrast level (in simulation only)
        with falco.util.TicToc('Getting updated summed image'):
            Im = falco.imaging.get_summed_image(mp)
        
        # REPORTING NORMALIZED INTENSITY
        print('Itr: ', Itr)
        InormHist[Itr+1] = np.mean(Im[mp.Fend.corr.maskBool])
        print('Prev and New Measured Normalized Intensity:\t\t\t %.2e\t->\t%.2e\t (%.2f x smaller)  \n\n' %
              (InormHist[Itr], InormHist[Itr+1], InormHist[Itr]/InormHist[Itr+1]))
    
        # END OF ESTIMATION + CONTROL LOOP
    
    Itr = Itr + 1
    
    # Data to store
    if np.any(mp.dm_ind == 1):
        out.dm1.Vall[:, :, Itr] = mp.dm1.V
    if np.any(mp.dm_ind == 2):
        out.dm2.Vall[:, :, Itr] = mp.dm2.V
    if np.any(mp.dm_ind == 8):
        out.dm8.Vall[:, Itr] = mp.dm8.V
    if np.any(mp.dm_ind == 9):
        out.dm9.Vall[:, Itr] = mp.dm9.V
    
    # Calculate the core throughput (at higher resolution to be more accurate)
    thput, ImSimOffaxis = falco.imaging.calc_thput(mp)
    if mp.flagFiber:
        mp.thput_vec[Itr] = np.max(thput)
    else:
        mp.thput_vec[Itr] = thput  # record keeping
    
    # Save the final DM commands separately for faster reference
    if hasattr(mp, 'dm1'):
        if hasattr(mp.dm1, 'V'):
            out.DM1V = mp.dm1.V
    if hasattr(mp, 'dm2'):
        if hasattr(mp.dm2, 'V'):
            out.DM2V = mp.dm2.V
    if mp.coro.upper() in ['HLC', 'EHLC']:
        if hasattr(mp.dm8, 'V'):
            out.DM8V = mp.dm8.V
        if hasattr(mp.dm9, 'V'):
            out.DM9V = mp.dm9.V
    
    # Save out an abridged workspace
    out.thput = mp.thput_vec
    out.Nitr = mp.Nitr
    out.InormHist = InormHist
    
    fnOut = mp.path.config + mp.runLabel + '_snippet.pkl'
    
    print('\nSaving abridged workspace to file:\n\t%s\n' % (fnOut), end='')
    with open(fnOut, 'wb') as f:
        pickle.dump(out, f)
    print('...done.\n')

    # Save out the data from the workspace
    if mp.flagSaveWS:
        del cvar; del G; del h; del jacStruct
    
        # Don't bother saving the large 2-D, floating point maps in the workspace (they take up too much space)
        mp.P1.full.mask=1; mp.P1.compact.mask=1;
        mp.P3.full.mask=1; mp.P3.compact.mask=1;
        mp.P4.full.mask=1; mp.P4.compact.mask=1;
        mp.F3.full.mask=1; mp.F3.compact.mask=1;
    
        mp.P1.full.E = 1; mp.P1.compact.E=1; mp.Eplanet=1;
        mp.dm1.full.mask = 1; mp.dm1.compact.mask = 1;
        mp.dm2.full.mask = 1; mp.dm2.compact.mask = 1;
        mp.complexTransFull = 1; mp.complexTransCompact = 1;
    
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
        print('Entire workspace NOT saved because mp.flagSaveWS==false')
    
    # END OF main FUNCTION
    print('END OF WFSC LOOP')

    return None
