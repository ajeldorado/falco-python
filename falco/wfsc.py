import numpy as np
import falco
import os
import pickle
import scipy
import psutil # For checking number of cores available
import multiprocessing
from astropy.io import fits 
import matplotlib.pyplot as plt 

def loop(mp, out):

    """
    Quick Description here

    Detailed description here

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    Returns
    -------
    TBD
        Return value descriptio here
    """

    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    pass

    ## Sort out file paths and save the config file    
    
    #--Add the slash or backslash to the FALCO path if it isn't there.
    filesep = os.pathsep

    if mp.path.falco[-1] != '/' and mp.path.falco[-1] != '\\':
        mp.path.falco = mp.path.falco + filesep
    
    mp.path.dummy = 1; #--Initialize the folders structure in case it doesn't already exist
    
    #--Store minimal data to re-construct the data from the run: the config files and "out" structure after a trial go here
    if not hasattr(mp.path,'config'):
        mp.path.config = mp.path.falco + filesep + 'data' + filesep + 'config' + filesep
    
    #--Entire final workspace from FALCO gets saved here.
    if not hasattr(mp.path,'ws'):
        mp.path.ws = mp.path.falco + filesep + 'data' + filesep + 'ws' + filesep
    
    #--Save the config file
#    fn_config = mp.path.config + mp.runLabel + '_config.pkl'
#    #save(fn_config)
#    print('Saved the config file: \t%s\n'%(fn_config))
#    with open(fn_config, 'wb') as f:
#        pickle.dump(mp, f)

    
    ## Get configuration data from a function file
#    if not mp.flagSim:  
#        bench = mp.bench #--Save the testbed structure "mp.bench" into "bench" so it isn't overwritten by falco_init_ws

    #mp, out = falco_init_ws(mp, fn_config)
#    out = falco_init_ws(mp)
#    out = falco.setup.flesh_out_workspace(mp)

#    if not mp.flagSim:  
#        mp.bench = bench


    #print('AFTER INIT: ', mp)
    #print('FlagFiber = ', mp.flagFiber)
    ## Initializations of Arrays for Data Storage 
    
    #--Raw contrast (broadband)
    
    InormHist = np.zeros((mp.Nitr+1,)); # Measured, mean raw contrast in scoring regino of dark hole.
    
    ## Plot the pupil masks
    
    # if(mp.flagPlot); figure(101); imagesc(mp.P1.full.mask);axis image; colorbar; title('pupil');drawnow; end
    # if(mp.flagPlot && (length(mp.P4.full.mask)==length(mp.P1.full.mask))); figure(102); imagesc(mp.P4.full.mask);axis image; colorbar; title('Lyot stop');drawnow; end
    # if(mp.flagPlot && isfield(mp,'P3.full.mask')); figure(103); imagesc(padOrCropEven(mp.P1.full.mask,mp.P3.full.Narr).*mp.P3.full.mask);axis image; colorbar; drawnow; end
    
    ## Take initial broadband image 
    
    Im = falco.imaging.falco_get_summed_image(mp)
    

    ##
    ###########################################################################
    #Begin the Correction Iterations
    ###########################################################################

    mp.flagCullActHist = np.zeros((mp.Nitr+1,),dtype=np.bool)

    for Itr in range(mp.Nitr):
    
        #--Start of new estimation+control iteration
        #print(['Iteration: ' num2str(Itr) '/' num2str(mp.Nitr) '\n' ]);
        print('Iteration: %d / %d\n'%(Itr, mp.Nitr),end='');
        
        #--Re-compute the starlight normalization factor for the compact and full models (to convert images to normalized intensity). No tip/tilt necessary.
        falco.imaging.falco_get_PSF_norm_factor(mp);
           
        ## Updated DM data
        #--Change the selected DMs if using the scheduled EFC controller
        if mp.controller.lower() in ['plannedefc']:
            mp.dm_ind = mp.dm_ind_sched[Itr];
    
        #--Report which DMs are used in this iteration
        print('DMs to be used in this iteration = [',end='')
        for jj in range(len(mp.dm_ind)):
            print(' %d'%(mp.dm_ind[jj]),end='')
        print(' ]')
        
        #--Fill in History of DM commands to Store
        if hasattr(mp,'dm1'): 
            if hasattr(mp.dm1,'V'):  
                out.dm1.Vall[:,:,Itr] = mp.dm1.V  
        if hasattr(mp,'dm2'): 
            if hasattr(mp.dm2,'V'):  
                out.dm2.Vall[:,:,Itr] = mp.dm2.V
        if hasattr(mp,'dm5'): 
            if hasattr(mp.dm5,'V'):  
                out.dm5.Vall[:,:,Itr] = mp.dm5.V
        if hasattr(mp,'dm8'): 
            if hasattr(mp.dm8,'V'):  
                out.dm8.Vallr[:,Itr] = mp.dm8.V[:]
        if hasattr(mp,'dm9'): 
            if hasattr(mp.dm9,'V'):  
                out.dm9.Vall[:,Itr] = mp.dm9.V[:]
    
        #--Compute the DM surfaces
        if np.any(mp.dm_ind==1): 
            DM1surf =  falco.dms.falco_gen_dm_surf(mp.dm1, mp.dm1.compact.dx, mp.dm1.compact.Ndm)
        else: 
            DM1surf = np.zeros((mp.dm1.compact.Ndm,mp.dm1.compact.Ndm))
    
        if np.any(mp.dm_ind==2): 
            DM2surf =  falco.dms.falco_gen_dm_surf(mp.dm2, mp.dm2.compact.dx, mp.dm2.compact.Ndm);  
        else: 
            DM2surf = np.zeros((mp.dm2.compact.Ndm,mp.dm2.compact.Ndm))
    
        ## Updated plot and reporting
        #--Calculate the core throughput (at higher resolution to be more accurate)
        thput,ImSimOffaxis = falco.utils.falco_compute_thput(mp);
        if mp.flagFiber:
            mp.thput_vec[Itr] = np.max(thput)
        else:
            mp.thput_vec[Itr] = thput; #--record keeping
        
        #--Compute the current contrast level
        InormHist[Itr] = np.mean(Im[mp.Fend.corr.maskBool]);
        
        if(any(mp.dm_ind==1)):
            mp.dm1 = falco.dms.falco_enforce_dm_constraints(mp.dm1)
        if(any(mp.dm_ind==2)):
            mp.dm2 = falco.dms.falco_enforce_dm_constraints(mp.dm2)
        
        #--Plotting
        if(mp.flagPlot):
            
#            if(Itr==1):
#                plt.ion()
#                plt.show()
            #plt.figure(1)
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
            fig.subplots_adjust(hspace=0.4, wspace=0.0)
            fig.suptitle(mp.coro+': Iteration %d'%Itr)
            
            im1=ax1.imshow(np.log10(Im),cmap='magma',interpolation='none',extent=[np.min(mp.Fend.xisDL),np.max(mp.Fend.xisDL),np.min(mp.Fend.xisDL),np.max(mp.Fend.xisDL)])
            ax1.set_title('Stellar PSF: NI=%.2e'%InormHist[Itr])
            ax1.tick_params(labelbottom=False)
            cbar1 = fig.colorbar(im1, ax = ax1)#,shrink=0.95)

            im3 = ax3.imshow(ImSimOffaxis/np.max(ImSimOffaxis),cmap=plt.cm.get_cmap('Blues'),interpolation='none',extent=[np.min(mp.Fend.xisDL),np.max(mp.Fend.xisDL),np.min(mp.Fend.xisDL),np.max(mp.Fend.xisDL)])            
#            im3 = ax3.imshow(ImSimOffaxis/np.max(ImSimOffaxis),cmap=plt.cm.get_cmap('Blues', 4),interpolation='none',extent=[np.min(mp.Fend.xisDL),np.max(mp.Fend.xisDL),np.min(mp.Fend.xisDL),np.max(mp.Fend.xisDL)])
            ax3.set_title('Off-axis Thput = %.2f%%'%(100*thput))
            cbar3 = fig.colorbar(im3, ax = ax3)
            cbar3.set_ticks(np.array([0.0, 0.5, 1.0]))
            cbar3.set_ticklabels(['0', '0.5', '1'])
            
            im2 = ax2.imshow(1e9*DM1surf,cmap='viridis')
            ax2.set_title('DM1 Surface (nm)')
            ax2.tick_params(labelbottom=False,labelleft=False,bottom=False,left=False)
            cbar2 = fig.colorbar(im2, ax = ax2)
            
            im4 = ax4.imshow(1e9*DM2surf,cmap='viridis')
            ax4.set_title('DM2 Surface (nm)')
            ax4.tick_params(labelbottom=False,labelleft=False,bottom=False,left=False)
            cbar4 = fig.colorbar(im4, ax = ax4)
            
            #plt.show()
            plt.pause(0.1)
            
        ## Updated selection of Zernike modes targeted by the controller
        #--Decide with Zernike modes to include in the Jacobian
        if Itr==0:
            mp.jac.zerns0 = mp.jac.zerns;
        
        print('Zernike modes (Noll indexing) used in this Jacobian:\t',end=''); 
        print(mp.jac.zerns)
        
        #--Re-compute the Jacobian weights
        falco.setup.falco_set_jacobian_weights(mp)
        
        ## Actuator Culling: Initialization of Flag and Which Actuators
        
        #--If new actuators are added, perform a new cull of actuators.
        cvar = falco.config.Object()
        if Itr==0:
            cvar.flagCullAct = True;
        else:
            if hasattr(mp,'dm_ind_sched'):
                cvar.flagCullAct = np.not_equal(mp.dm_ind_sched[Itr], mp.dm_ind_sched[Itr-1]);
            else:
                cvar.flagCullAct = False;
        
        mp.flagCullActHist[Itr] = cvar.flagCullAct;
        
        #--Before performing new cull, include all actuators again
        if cvar.flagCullAct:
            #--Re-include all actuators in the basis set.
            if np.any(mp.dm_ind==1): 
                mp.dm1.act_ele = list(range(mp.dm1.NactTotal)) 
            if np.any(mp.dm_ind==2): 
                mp.dm2.act_ele = list(range(mp.dm2.NactTotal)) 
            if np.any(mp.dm_ind==5):
                mp.dm5.act_ele = list(range(mp.dm5.NactTotal))
            if np.any(mp.dm_ind==8): 
                mp.dm8.act_ele = list(range(mp.dm8.NactTotal))
            if np.any(mp.dm_ind==9): 
                mp.dm9.act_ele = list(range(mp.dm9.NactTotal))
            #--Update the number of elements used per DM
            if np.any(mp.dm_ind==1): 
                mp.dm1.Nele = len(mp.dm1.act_ele)
            else: 
                mp.dm1.Nele = 0; 
            if np.any(mp.dm_ind==2): 
                mp.dm2.Nele = len(mp.dm2.act_ele)
            else: 
                mp.dm2.Nele = 0; 
            if np.any(mp.dm_ind==5): 
                mp.dm5.Nele = len(mp.dm5.act_ele)
            else: 
                mp.dm5.Nele = 0; 
            if np.any(mp.dm_ind==8): 
                mp.dm8.Nele = len(mp.dm8.act_ele)
            else: 
                mp.dm8.Nele = 0; 
            if np.any(mp.dm_ind==9): 
                mp.dm9.Nele = len(mp.dm9.act_ele)
            else: 
                mp.dm9.Nele = 0; 
        
        ## Compute the control Jacobians for each DM
        
        #--Relinearize about the DMs only at the iteration numbers in mp.relinItrVec.
        if np.any(mp.relinItrVec==Itr):
            cvar.flagRelin=True;
        else:
            cvar.flagRelin=False;
        
        if  Itr==0 or cvar.flagRelin:
            jacStruct =  falco.model.jacobian(mp); #--Get structure containing Jacobians
        
        ## Modify jacStruct to cull actuators, but only if(cvar.flagCullAct && cvar.flagRelin)
        falco_ctrl_cull(mp,cvar,jacStruct);
        
        ## Load the improved Jacobian if using the E-M technique
#        if mp.flagUseLearnedJac:
#            jacStructLearned = load('jacStructLearned.mat');
#            if np.any(mp.dm_ind==1):  
#            jacStruct.G1 = jacStructLearned.G1
#            if np.any(mp.dm_ind==1):  
#            jacStruct.G2 = jacStructLearned.G2
        
        ## Wavefront Estimation
        if mp.estimator.lower() in ['perfect']:
            EfieldVec  = falco_est_perfect_Efield_with_Zernikes(mp)
        elif mp.estimator.lower in ['pwp-bp','pwp-kf']:
            if mp.est.flagUseJac: #--Send in the Jacobian if true
                ev = falco_est_pairwise_probing(mp,jacStruct);
            else: #--Otherwise don't pass the Jacobian
                ev = falco_est_pairwise_probing(mp);
        
            EfieldVec = ev.Eest;
            IincoVec = ev.IincoEst;
                
        ## Add spatially-dependent weighting to the control Jacobians
        if np.any(mp.dm_ind==1): 
            jacStruct.G1 = jacStruct.G1*np.moveaxis(np.tile(mp.WspatialVec[:,None],[mp.jac.Nmode,1,mp.dm1.Nele]),0,-1)
        if np.any(mp.dm_ind==2): 
            jacStruct.G2 = jacStruct.G2*np.moveaxis(np.tile(mp.WspatialVec[:,None],[mp.jac.Nmode,1,mp.dm2.Nele]),0,-1)
        if np.any(mp.dm_ind==8): 
            jacStruct.G8 = jacStruct.G8*np.moveaxis(np.tile(mp.WspatialVec[:,None],[mp.jac.Nmode,1,mp.dm8.Nele]),0,-1)
        if np.any(mp.dm_ind==9): 
            jacStruct.G9 = jacStruct.G9*np.moveaxis(np.tile(mp.WspatialVec[:,None],[mp.jac.Nmode,1,mp.dm9.Nele]),0,-1)
            
        #fprintf('Total Jacobian Calcuation Time: #.2f\n',toc);
    
        #--Compute the number of total actuators for all DMs used. 
        cvar.NeleAll = mp.dm1.Nele + mp.dm2.Nele + mp.dm3.Nele + mp.dm4.Nele + mp.dm5.Nele + mp.dm6.Nele + mp.dm7.Nele + mp.dm8.Nele + mp.dm9.Nele; #--Number of total actuators used 
    
        ## Wavefront Control
    
        cvar.Itr = Itr
        cvar.EfieldVec = EfieldVec
        cvar.InormHist = InormHist[Itr]
        falco_ctrl(mp,cvar,jacStruct)
    
        #--Save out regularization used.
        if hasattr(cvar, "log10regUsed"):
            out.log10regHist[Itr] = cvar.log10regUsed;
    
        #-----------------------------------------------------------------------------------------
        
        ## DM Stats
        #--Calculate and report updated P-V DM voltages.
        if np.any(mp.dm_ind==1):
            out.dm1.Vpv[Itr] = np.max(mp.dm1.V) - np.min(mp.dm1.V)
            print(' DM1 P-V in volts: %.3f'%(out.dm1.Vpv[Itr]))
            if(mp.dm1.tied.size>0):  
                print(' DM1 has %d pairs of tied actuators.' % (mp.dm1.tied.shape[0]) )
#            Nrail1 = len(np.where( (mp.dm1.V <= -mp.dm1.maxAbsV) | (mp.dm1.V >= mp.dm1.maxAbsV) )); 
#            print(' DM1 P-V in volts: %.3f\t\t%d/%d (%.2f%%) railed actuators \n'%(out.dm1.Vpv(Itr), Nrail1, mp.dm1.NactTotal, 100*Nrail1/mp.dm1.NactTotal))
#            if mp.dm1.tied.shape[0]>0:  
#                print(' DM1 has %d pairs of tied actuators.\n'%(mp.dm1.tied.shape[0]))
        if np.any(mp.dm_ind==2):
            out.dm2.Vpv[Itr] = np.max(mp.dm2.V) - np.min(mp.dm2.V)
            print(' DM2 P-V in volts: %.3f'%(out.dm2.Vpv[Itr]))
            if(mp.dm2.tied.size>0):  
                print(' DM2 has %d pairs of tied actuators.' % (mp.dm2.tied.shape[0]) )

#            Nrail2 = len(np.where( (mp.dm2.V <= -mp.dm2.maxAbsV) | (mp.dm2.V >= mp.dm2.maxAbsV) )); 
#            print(' DM2 P-V in volts: %.3f\t\t%d/%d (%.2f%%) railed actuators \n'%( out.dm2.Vpv(Itr), Nrail2, mp.dm2.NactTotal, 100*Nrail2/mp.dm2.NactTotal))
#            if mp.dm2.tied.shape[0]>0:  
#                print(' DM2 has %d pairs of tied actuators.\n'%(mp.dm2.tied.shape[0]))
        # if(any(mp.dm_ind==8))
        #     out.dm8.Vpv(Itr) = (max(max(mp.dm8.V))-min(min(mp.dm8.V)));
        #     Nrail8 = length(find( (mp.dm8.V <= mp.dm8.Vmin) | (mp.dm8.V >= mp.dm8.Vmax) ));
        #     fprintf(' DM8 P-V in volts: #.3f\t\t#d/#d (#.2f##) railed actuators \n', out.dm8.Vpv(Itr), Nrail8,mp.dm8.NactTotal,100*Nrail8/mp.dm8.NactTotal); 
        # end
#        if np.any(mp.dm_ind==9):
#            out.dm9.Vpv[Itr] = np.max(np.max(mp.dm9.V))-np.min(np.min(mp.dm9.V))
#            Nrail9 = len(np.where( (mp.dm9.V <= mp.dm9.Vmin) | (mp.dm9.V >= mp.dm9.Vmax) )); 
#            print(' DM9 P-V in volts: %.3f\t\t%d/%d (%.2f%%) railed actuators \n'%(out.dm9.Vpv(Itr), Nrail9,mp.dm9.NactTotal,100*Nrail9/mp.dm9.NactTotal))
        
        #--Calculate and report updated RMS DM surfaces.               
        if(any(mp.dm_ind==1)):
            # Pupil-plane coordinates
            dx_dm = mp.P2.compact.dx/mp.P2.D #--Normalized dx [Units of pupil diameters]
            xs = falco.utils.create_axis(mp.dm1.compact.Ndm, dx_dm, centering=mp.centering)
            RS = falco.utils.radial_grid(xs)
            rmsSurf_ele = np.logical_and(RS>=mp.P1.IDnorm/2., RS<=0.5)
            
            DM1surf = falco.dms.falco_gen_dm_surf(mp.dm1, mp.dm1.compact.dx, mp.dm1.compact.Ndm)
            out.dm1.Spv[Itr] = np.max(DM1surf)-np.min(DM1surf)
            out.dm1.Srms[Itr] = np.sqrt(np.mean(np.abs( (DM1surf[rmsSurf_ele]) )**2))
            print('RMS surface of DM1 = %.1f nm' % (1e9*out.dm1.Srms[Itr]))
        if(any(mp.dm_ind==2)):
            # Pupil-plane coordinates
            dx_dm = mp.P2.compact.dx/mp.P2.D #--Normalized dx [Units of pupil diameters]
            xs = falco.utils.create_axis(mp.dm2.compact.Ndm, dx_dm, centering=mp.centering)
            RS = falco.utils.radial_grid(xs)
            rmsSurf_ele = np.logical_and(RS>=mp.P1.IDnorm/2., RS<=0.5)
            
            DM2surf = falco.dms.falco_gen_dm_surf(mp.dm2, mp.dm2.compact.dx, mp.dm2.compact.Ndm)
            out.dm2.Spv[Itr] = np.max(DM2surf)-np.min(DM2surf)
            out.dm2.Srms[Itr] = np.sqrt(np.mean(np.abs( (DM2surf[rmsSurf_ele]) )**2))
            print('RMS surface of DM2 = %.1f nm' % (1e9*out.dm2.Srms[Itr]))
        
        #--Calculate sensitivities to 1nm RMS of Zernike phase aberrations at entrance pupil.
        if( (mp.eval.Rsens.size>0) and (mp.eval.indsZnoll.size>0) ):
            out.Zsens[:,:,Itr] = falco.zernikes.falco_get_Zernike_sensitivities(mp)
        
        # Take the next image to check the contrast level (in simulation only)
        with falco.utils.TicToc('Getting updated summed image'):
            Im = falco.imaging.falco_get_summed_image(mp);
        
        #--REPORTING NORMALIZED INTENSITY
        print('Itr: ', Itr)
        InormHist[Itr+1] = np.mean(Im[mp.Fend.corr.maskBool]);
        print('Prev and New Measured Contrast (LR):\t\t\t %.2e\t->\t%.2e\t (%.2f x smaller)  \n\n'%(InormHist[Itr], InormHist[Itr+1], InormHist[Itr]/InormHist[Itr+1]))

    
        # --END OF ESTIMATION + CONTROL LOOP
    
    Itr = Itr + 1;
    
    #--Data to store
    if np.any(mp.dm_ind==1): 
        out.dm1.Vall[:,:,Itr] = mp.dm1.V
    if np.any(mp.dm_ind==2): 
        out.dm2.Vall[:,:,Itr] = mp.dm2.V
    # if(any(mp.dm_ind==5)); out.dm5.Vall(:,:,Itr) = mp.dm5.V; end
    if np.any(mp.dm_ind==8): 
        out.dm8.Vall[:,Itr] = mp.dm8.V
    if np.any(mp.dm_ind==9): 
        out.dm9.Vall[:,Itr] = mp.dm9.V
    
    #--Calculate the core throughput (at higher resolution to be more accurate)
    thput,ImSimOffaxis = falco.utils.falco_compute_thput(mp);
    if mp.flagFiber:
        mp.thput_vec[Itr] = np.max(thput);
    else:
        mp.thput_vec[Itr] = thput; #--record keeping
    
    ## Save the final DM commands separately for faster reference
    if hasattr(mp,'dm1'): 
        if hasattr(mp.dm1,'V'): 
            out.DM1V = mp.dm1.V
    if hasattr(mp,'dm2'): 
        if hasattr(mp.dm2,'V'): 
            out.DM2V = mp.dm2.V
    if mp.coro.upper() in ['HLC','EHLC']:
        if hasattr(mp.dm8,'V'): 
            out.DM8V = mp.dm8.V
        if hasattr(mp.dm9,'V'): 
            out.DM9V = mp.dm9.V
    
    ## Save out an abridged workspace
    
    #--Variables to save out:
    # contrast vs iter
    # regularization history
    #  DM1surf,DM1V, DM2surf,DM2V, DM8surf,DM9surf, fpm sampling, base pmgi thickness, base nickel thickness, dm_tilts, aoi, ...
    # to reproduce your basic design results of NI, throughput, etc..
    
    out.thput = mp.thput_vec;
    out.Nitr = mp.Nitr;
    out.InormHist = InormHist;
    
    fnOut = mp.path.config + mp.runLabel + '_snippet.pkl'
    
    print('\nSaving abridged workspace to file:\n\t%s\n'%(fnOut),end='')
    #save(fnOut,'out');
    with open(fnOut, 'wb') as f:
        pickle.dump(out,f)
    print('...done.\n')

    ## Save out the data from the workspace
    if mp.flagSaveWS:
        #clear cvar G* h* jacStruct; # Save a ton of space when storing the workspace
        del cvar; del G; del h; del jacStruct
    
        # Don't bother saving the large 2-D, floating point maps in the workspace (they take up too much space)
        mp.P1.full.mask=1; mp.P1.compact.mask=1;
        mp.P3.full.mask=1; mp.P3.compact.mask=1;
        mp.P4.full.mask=1; mp.P4.compact.mask=1;
        mp.F3.full.mask=1; mp.F3.compact.mask=1;
    
        mp.P1.full.E = 1; mp.P1.compact.E=1; mp.Eplanet=1;
        mp.dm1.full.mask = 1; mp.dm1.compact.mask = 1; mp.dm2.full.mask = 1; mp.dm2.compact.mask = 1;
        mp.complexTransFull = 1; mp.complexTransCompact = 1;
    
        mp.dm1.compact.inf_datacube = 0;
        mp.dm2.compact.inf_datacube = 0;
        mp.dm8.compact.inf_datacube = 0;
        mp.dm9.compact.inf_datacube = 0;
        mp.dm8.inf_datacube = 0;
        mp.dm9.inf_datacube = 0;
    
        fnAll = mp.path.ws + mp.runLabel + '_all.pkl'
        print('Saving entire workspace to file ' + fnAll + '...',end='')
        #save(fnAll);
        with open(fnAll, 'wb') as f:
            pickle.dump(mp,f)

        print('done.\n\n')
    else:
        print('Entire workspace NOT saved because mp.flagSaveWS==false')
    
    #--END OF main FUNCTION
    print('END OF WFSC LOOP')
    #print('END OF WFSC LOOP: ', mp)

    
def falco_est_perfect_Efield_with_Zernikes(mp):
    """
   Function to return the perfect-knowledge E-field from the full model. Can include 
   Zernike aberrations at the input pupil.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
        
    Returns
    -------
    Emat : numpy ndarray
        2-D array with the vectorized, complex E-field of the dark hole pixels for each 
        mode included in the control Jacobian.
    """  
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    
    if mp.flagMultiproc:
        
        Emat = np.zeros((mp.Fend.corr.Npix, mp.jac.Nmode), dtype=complex)
        
        #--Loop over all modes and wavelengths
        inds_list = [(x,y) for x in range(mp.jac.Nmode) for y in range(mp.Nwpsbp)] #--Make all combinations of the values  
        Nvals = mp.jac.Nmode*mp.Nwpsbp

        pool = multiprocessing.Pool(processes=mp.Nthreads)
        resultsRaw = [pool.apply_async(_est_perfect_Efield_with_Zernikes_in_parallel, args=(mp, ilist, inds_list)) for ilist in range(Nvals) ]
        results = [p.get() for p in resultsRaw] #--All the images in a list
        pool.close()
        pool.join()  
        
#        parfor ni=1:Nval
#            Evecs{ni} = falco_est_perfect_Efield_with_Zernikes_parfor(ni,ind_list,mp)
#        end
        
        #--Re-order for easier indexing
        Ecube = np.zeros((mp.Fend.corr.Npix, mp.jac.Nmode, mp.Nwpsbp), dtype=complex)
        for iv in range(Nvals):
            im = inds_list[iv][0]  #--Index of the Jacobian mode
            wi = inds_list[iv][1]   #--Index of the wavelength in the sub-bandpass
            Ecube[:, im, wi] = results[iv]
        Emat = np.mean(Ecube, axis=2) # Average over wavelengths in the subband
  
#        EmatAll = np.zeros((mp.Fend.corr.Npix, Nval))
#        for iv in range(Nval):
#            EmatAll[:, iv] = results[iv]
#
#        counter = 0;
#        for im=1:mp.jac.Nmode
#            EsbpMean = 0;
#            for wi=1:mp.Nwpsbp
#                counter = counter + 1;
#                EsbpMean = EsbpMean + EmatAll(:,counter)*mp.full.lambda_weights(wi);
#            end
#            Emat(:,im) = EsbpMean;
#        end
    
    else:
    
        Emat = np.zeros((mp.Fend.corr.Npix, mp.jac.Nmode), dtype=complex)
        modvar = falco.config.Object() #--Initialize
        
        for im in range(mp.jac.Nmode):
            modvar.sbpIndex = mp.jac.sbp_inds[im]
            modvar.zernIndex = mp.jac.zern_inds[im]
            modvar.whichSource = 'star'
            
            #--Take the mean over the wavelengths within the sub-bandpass
            EmatSbp = np.zeros((mp.Fend.corr.Npix, mp.Nwpsbp),dtype=complex)
            for wi in range(mp.Nwpsbp):
                modvar.wpsbpIndex = wi
                E2D = falco.model.full(mp, modvar)
                EmatSbp[:,wi] = mp.full.lambda_weights[wi]*E2D[mp.Fend.corr.maskBool] #--Actual field in estimation area. Apply spectral weight within the sub-bandpass
            Emat[:,im] = np.sum(EmatSbp,axis=1)
            
    
    return Emat
    
#%--Extra function needed to use parfor (because parfor can have only a
#%  single changing input argument).
def _est_perfect_Efield_with_Zernikes_in_parallel(mp, ilist, inds_list):

    im = inds_list[ilist][0]  #--Index of the Jacobian mode
    wi = inds_list[ilist][1]   #--Index of the wavelength in the sub-bandpass
    
    modvar = falco.config.Object()
    modvar.sbpIndex = mp.jac.sbp_inds[im]
    modvar.zernIndex = mp.jac.zern_inds[im]
    modvar.wpsbpIndex = wi
    modvar.whichSource = 'star'
    
    E2D = falco.model.full(mp, modvar)
    
    return E2D[mp.Fend.corr.maskBool] # Actual field in estimation area. Don't apply spectral weight here.



    #pass
    #return falco.config.Object()

def falco_est_pairwise_probing(mp, **kwargs):
    
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    pass

    return falco.config.Object()


def falco_ctrl(mp,cvar,jacStruct):
    """
    Outermost wrapper function for all the controller functions.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    cvar : ModelParameters
        Structure containing controller variables
    jacStruct : ModelParameters
        Structure containing control Jacobians for each specified DM.

    Returns
    -------
    None
        Changes are made by reference to mp.
    """     
#    if type(mp) is not falco.config.ModelParameters:
#        raise TypeError('Input "mp" must be of type ModelParameters')
#    pass

    #with falco.utils.TicToc('Using the Jacobian to make other matrices'):
    print('Using the Jacobian to make other matrices...',end='')
    
    #--Compute matrices for linear control with regular EFC
    cvar.GstarG_wsum  = np.zeros((cvar.NeleAll,cvar.NeleAll)) 
    cvar.RealGstarEab_wsum = np.zeros((cvar.NeleAll, 1))

    for im in range(mp.jac.Nmode):

        Gmode = np.zeros((mp.Fend.corr.Npix,1), dtype=complex) #--Initialize a row to concatenate onto
        if(any(mp.dm_ind==1)): Gmode = np.hstack((Gmode,np.squeeze(jacStruct.G1[:,:,im])))
        if(any(mp.dm_ind==2)): Gmode = np.hstack((Gmode,np.squeeze(jacStruct.G2[:,:,im])))
        if(any(mp.dm_ind==8)): Gmode = np.hstack((Gmode,np.squeeze(jacStruct.G8[:,:,im])))
        if(any(mp.dm_ind==9)): Gmode = np.hstack((Gmode,np.squeeze(jacStruct.G9[:,:,im])))
        Gmode = Gmode[:,1:] #--Remove the zero column used for initialization
        #Gstack = [jacStruct.G1[:,:,im], jacStruct.G2[:,:,im], jacStruct.G8[:,:,im], jacStruct.G9[:,:,im] ]

        #--Square matrix part stays the same if no re-linearization has occurrred. 
        cvar.GstarG_wsum += mp.jac.weights[im]*np.real(np.conj(Gmode).T @ Gmode) 

        #--The G^*E part changes each iteration because the E-field changes.
        Eweighted = mp.WspatialVec*cvar.EfieldVec[:,im] #--Apply 2-D spatial weighting to E-field in dark hole pixels.
        cvar.RealGstarEab_wsum += mp.jac.weights[im]*np.real(np.conj(Gmode).T @ Eweighted.reshape(mp.Fend.corr.Npix,1)) #--Apply the Jacobian weights and add to the total.
    
    #--Make the regularization matrix. (Define only the diagonal here to save RAM.)
    cvar.EyeGstarGdiag = np.max(np.diag(cvar.GstarG_wsum))*np.ones(cvar.NeleAll)
    cvar.EyeNorm = np.max(np.diag(cvar.GstarG_wsum))
    print('done.') #fprintf(' done. Time: %.3f\n',toc);

    #--Call the Controller Function
    print('Control beginning ...') # tic
    #--Established, conventional controllers
    if(mp.controller.lower()=='plannedefc'): #--EFC regularization is scheduled ahead of time
        dDM = falco_ctrl_planned_EFC(mp,cvar)
    elif(mp.controller.lower()=='gridsearchefc'):
        dDM = falco_ctrl_grid_search_EFC(mp,cvar)

    #--Experimental controllers
    elif(mp.controller.lower()=='plannedefcts'): #--EFC regularization is scheduled ahead of time. total stroke also minimized
        dDM = falco_ctrl_planned_EFC_TS(mp,cvar)   

    elif(mp.controller.lower()=='plannedefccon'): #--Constrained-EFC regularization is scheduled ahead of time
        dDM = falco_ctrl_planned_EFCcon(mp,cvar)          
        
#    print('done.\n') #print(' done. Time: %.3f sec\n',toc);
    
    #--Update the DM commands by adding the delta control signal
    if(any(mp.dm_ind==1)):  mp.dm1.V += dDM.dDM1V
    if(any(mp.dm_ind==2)):  mp.dm2.V += dDM.dDM2V
    if(any(mp.dm_ind==8)):  mp.dm8.V += dDM.dDM8V
    if(any(mp.dm_ind==9)):  mp.dm9.V += dDM.dDM9V

    #--Save the delta from the previous command
    if(any(mp.dm_ind==1)):  mp.dm1.dV = dDM.dDM1V 
    if(any(mp.dm_ind==2)):  mp.dm2.dV = dDM.dDM2V
    if(any(mp.dm_ind==8)):  mp.dm8.dV = dDM.dDM8V
    if(any(mp.dm_ind==9)):  mp.dm9.dV = dDM.dDM9V    
    

def falco_ctrl_cull(mp, cvar, jacStruct):
    """
    Function that removes weak actuators from the controlled set.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    cvar : ModelParameters
        Structure containing controller variables
    jacStruct : ModelParameters
        Structure containing control Jacobians for each specified DM.

    Returns
    -------
    None
        Changes are made by reference to mp and jacStruct.
    """ 
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    
    #--Reduce the number of actuators used based on their relative strength in the Jacobian
    if(cvar.flagCullAct and cvar.flagRelin):
        
        print('Weeding out weak actuators from the control Jacobian...'); 
        if(any(mp.dm_ind==1)):
            G1intNorm = np.sum( np.mean(np.abs(jacStruct.G1)**2,axis=2), axis=0)
            G1intNorm = G1intNorm/np.max(G1intNorm)
            mp.dm1.act_ele = np.nonzero(G1intNorm>=10**(mp.logGmin))[0]
            del G1intNorm
        if(any(mp.dm_ind==2)):
            G2intNorm = np.sum( np.mean(np.abs(jacStruct.G2)**2,axis=2), axis=0)
            G2intNorm = G2intNorm/np.max(G2intNorm)
            mp.dm2.act_ele = np.nonzero(G2intNorm>=10**(mp.logGmin))[0]
            del G2intNorm
        if(any(mp.dm_ind==8)):
            G8intNorm = np.sum( np.mean(np.abs(jacStruct.G8)**2,axis=2), axis=0)
            G8intNorm = G8intNorm/np.max(G8intNorm)
            mp.dm8.act_ele = np.nonzero(G8intNorm>=10**(mp.logGmin))[0]
            del G8intNorm
        if(any(mp.dm_ind==9)):
            G9intNorm = np.sum( np.mean(np.abs(jacStruct.G9)**2,axis=2), axis=0)
            G9intNorm = G9intNorm/np.max(G9intNorm)
            mp.dm9.act_ele = np.nonzero(G9intNorm>=10**(mp.logGmin))[0]
            del G9intNorm

        #--Add back in all actuators that are tied (to make the tied actuator logic easier)
        if(any(mp.dm_ind==1)):
            for ti in range(mp.dm1.tied.shape[0]):
                if not (any(mp.dm1.act_ele==mp.dm1.tied[ti,0])):  mp.dm1.act_ele = np.hstack([mp.dm1.act_ele, mp.dm1.tied[ti,0]])
                if not (any(mp.dm1.act_ele==mp.dm1.tied[ti,1])):  mp.dm1.act_ele = np.hstack([mp.dm1.act_ele, mp.dm1.tied[ti,1]])
            mp.dm1.act_ele = np.sort(mp.dm1.act_ele) #--Need to sort for the logic in model_Jacobian.m

        if(any(mp.dm_ind==2)):
            for ti in range(mp.dm2.tied.shape[0]):
                if not any(mp.dm2.act_ele==mp.dm2.tied[ti,0]):  mp.dm2.act_ele = np.hstack([mp.dm2.act_ele, mp.dm2.tied[ti,0]])
                if not any(mp.dm2.act_ele==mp.dm2.tied[ti,1]):  mp.dm2.act_ele = np.hstack([mp.dm2.act_ele, mp.dm2.tied[ti,1]])
            mp.dm2.act_ele = np.sort(mp.dm2.act_ele) #--Need to sort for the logic in model_Jacobian.m
#            if(any(mp.dm_ind==8))
#                for ti=1:size(mp.dm8.tied,1)
#                    if(any(mp.dm8.act_ele==mp.dm8.tied(ti,1))==false);  mp.dm8.act_ele = [mp.dm8.act_ele; mp.dm8.tied(ti,1)];  end
#                    if(any(mp.dm8.act_ele==mp.dm8.tied(ti,2))==false);  mp.dm8.act_ele = [mp.dm8.act_ele; mp.dm8.tied(ti,2)];  end
#                end
#                mp.dm8.act_ele = sort(mp.dm8.act_ele);
#            end
#            if(any(mp.dm_ind==9))
#                for ti=1:size(mp.dm9.tied,1)
#                    if(any(mp.dm9.act_ele==mp.dm9.tied(ti,1))==false);  mp.dm9.act_ele = [mp.dm9.act_ele; mp.dm9.tied(ti,1)];  end
#                    if(any(mp.dm9.act_ele==mp.dm9.tied(ti,2))==false);  mp.dm9.act_ele = [mp.dm9.act_ele; mp.dm9.tied(ti,2)];  end
#                end
#                mp.dm9.act_ele = sort(mp.dm9.act_ele);
#            end
            
        #--Update the number of elements used per DM
        if(any(mp.dm_ind==1)): mp.dm1.Nele = mp.dm1.act_ele.size
        if(any(mp.dm_ind==2)): mp.dm2.Nele = mp.dm2.act_ele.size
        if(any(mp.dm_ind==8)): mp.dm8.Nele = mp.dm8.act_ele.size
        if(any(mp.dm_ind==9)): mp.dm9.Nele = mp.dm9.act_ele.size

        if(any(mp.dm_ind==1)): print('  DM1: %d/%d (%.2f%%) actuators kept for Jacobian' % (mp.dm1.Nele, mp.dm1.NactTotal,100*mp.dm1.Nele/mp.dm1.NactTotal))
        if(any(mp.dm_ind==2)): print('  DM2: %d/%d (%.2f%%) actuators kept for Jacobian' % (mp.dm2.Nele, mp.dm2.NactTotal,100*mp.dm2.Nele/mp.dm2.NactTotal))
        if(any(mp.dm_ind==8)): print('  DM8: %d/%d (%.2f%%) actuators kept for Jacobian' % (mp.dm8.Nele, mp.dm8.NactTotal,100*mp.dm8.Nele/mp.dm8.NactTotal))
        if(any(mp.dm_ind==9)): print('  DM9: %d/%d (%.2f%%) actuators kept for Jacobian' % (mp.dm9.Nele, mp.dm9.NactTotal,100*mp.dm9.Nele/mp.dm9.NactTotal))
        
        #--Crop out unused actuators from the control Jacobian
        if(any(mp.dm_ind==1)): jacStruct.G1 = jacStruct.G1[:,mp.dm1.act_ele,:]
        if(any(mp.dm_ind==2)): jacStruct.G2 = jacStruct.G2[:,mp.dm2.act_ele,:]
        if(any(mp.dm_ind==8)): jacStruct.G8 = jacStruct.G8[:,mp.dm8.act_ele,:]
        if(any(mp.dm_ind==9)): jacStruct.G9 = jacStruct.G9[:,mp.dm9.act_ele,:]


    #return jacStruct
    #return falco.config.Object()


def falco_ctrl_grid_search_EFC(mp,cvar):
    """
    Wrapper controller function that performs a grid search over specified variables.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    cvar : ModelParameters
        Structure containing controller variables

    Returns
    -------
    dDM : ModelParameters
        Structure containing the delta DM commands for each DM
    """    
    #--Initializations    
    vals_list = [(x,y) for y in mp.ctrl.dmfacVec for x in mp.ctrl.log10regVec ] #--Make all combinations of the values
    Nvals = mp.ctrl.log10regVec.size*mp.ctrl.dmfacVec.size
    InormVec = np.zeros(Nvals)

    #--Temporarily store computed DM commands so that the best one does not have to be re-computed
    if(any(mp.dm_ind==1)): dDM1V_store = np.zeros((mp.dm1.Nact,mp.dm1.Nact,Nvals))
    if(any(mp.dm_ind==2)): dDM2V_store = np.zeros((mp.dm2.Nact,mp.dm2.Nact,Nvals))
    if(any(mp.dm_ind==8)): dDM8V_store = np.zeros((mp.dm8.NactTotal,Nvals))
    if(any(mp.dm_ind==9)): dDM9V_store = np.zeros((mp.dm9.NactTotal,Nvals))

    ## Empirically find the regularization value giving the best contrast
#    if(mp.flagMultiproc and mp.ctrl.flagUseModel):
#        #--Run the controller in parallel
#        pool = multiprocessing.Pool(processes=mp.Nthreads)
#        results = [pool.apply_async(falco_ctrl_EFC_base, args=(ni,vals_list,mp,cvar)) for ni in np.arange(Nvals,dtype=int) ]
#        results_ctrl = [p.get() for p in results] #--All the Jacobians in a list
#        pool.close()
#        pool.join()
#        
#        #--Convert from a list to arrays:
#        for ni in range(Nvals):
#            InormVec[ni] = results_ctrl[ni][0]
#            if(any(mp.dm_ind==1)): dDM1V_store[:,:,ni] = results_ctrl[ni][1].dDM1V
#            if(any(mp.dm_ind==2)): dDM2V_store[:,:,ni] = results_ctrl[ni][1].dDM2V
#    else:
    for ni in range(Nvals):
        [InormVec[ni],dDM_temp] = falco_ctrl_EFC_base(ni,vals_list,mp,cvar) 
        #--delta voltage commands
        if(any(mp.dm_ind==1)): dDM1V_store[:,:,ni] = dDM_temp.dDM1V
        if(any(mp.dm_ind==2)): dDM2V_store[:,:,ni] = dDM_temp.dDM2V
        if(any(mp.dm_ind==8)): dDM8V_store[:,ni] = dDM_temp.dDM8V
        if(any(mp.dm_ind==9)): dDM9V_store[:,ni] = dDM_temp.dDM9V

    #--Print out results to the command line
    print('Scaling factor:\t',end='')
    for ni in range(Nvals): print('%.2f\t\t' % (vals_list[ni][1]),end='')

    print('\nlog10reg:\t',end='')
    for ni in range(Nvals): print('%.1f\t\t' % (vals_list[ni][0]),end='')
    
    print('\nInorm:  \t',end='')
    for ni in range(Nvals):  print('%.2e\t' % (InormVec[ni]),end='')
    print('\n',end='')

    #--Find the best scaling factor and Lagrange multiplier pair based on the best contrast.
    #[cvar.cMin,indBest] = np.min(InormVec)
    indBest = np.argmin(InormVec)
    cvar.cMin = np.min(InormVec)
    dDM = falco.config.Object()
    #--delta voltage commands
    if(any(mp.dm_ind==1)): dDM.dDM1V = np.squeeze(dDM1V_store[:,:,indBest])
    if(any(mp.dm_ind==2)): dDM.dDM2V = np.squeeze(dDM2V_store[:,:,indBest])
    if(any(mp.dm_ind==8)): dDM.dDM8V = np.squeeze(dDM8V_store[:,indBest])
    if(any(mp.dm_ind==9)): dDM.dDM9V = np.squeeze(dDM9V_store[:,indBest])

    cvar.log10regUsed = vals_list[indBest][0]
    dmfacBest = vals_list[indBest][1]
    if(mp.ctrl.flagUseModel):
        print('Model-based grid search gives log10reg, = %.1f,\t dmfac = %.2f,\t %4.2e contrast.' % (cvar.log10regUsed, dmfacBest, cvar.cMin) )
    else:
        print('Empirical grid search gives log10reg, = %.1f,\t dmfac = %.2f,\t %4.2e contrast.' % (cvar.log10regUsed, dmfacBest, cvar.cMin) )
    
    return dDM


def falco_ctrl_EFC_base(ni,vals_list,mp,cvar):
    """
    Function that computes the main EFC equation. Called by a wrapper controller function. 

    Parameters
    ----------
    ni : int
        index for the set of possible combinations of variables to do a grid search over
    vals_list : list
        the set of possible combinations of values to do a grid search over
    mp : ModelParameters
        Structure containing optical model parameters
    cvar : ModelParameters
        Structure containing controller variables

    Returns
    -------
    InormAvg : float
        Normalized intensity averaged over wavelength and over the entire dark hole
    dDM : ModelParameters
        Structure containing the delta DM commands for each DM
    """
    #function [InormAvg,dDM] = falco_ctrl_EFC_base(ni,vals_list,mp,cvar)

    if(any(mp.dm_ind==1)):
        DM1V0 = mp.dm1.V.copy()
#        dDM1V0 = mp.dm1.dV.copy()
    if(any(mp.dm_ind==2)):
        DM2V0 = mp.dm2.V.copy()
        

    #--Initializations
    log10reg = vals_list[ni][0] #--log 10 of regularization value
    dmfac = vals_list[ni][1] #--Scaling factor for entire DM command
    
    #--Save starting point for each delta command to be added to.
    #--Get the indices of each DM's command vector within the single concatenated command vector
    falco_ctrl_setup(mp,cvar) #--Modifies cvar
    
    #--Least-squares solution with regularization:
    duVecNby1 = -dmfac*np.linalg.solve( (10**log10reg*np.diag(cvar.EyeGstarGdiag) + cvar.GstarG_wsum), cvar.RealGstarEab_wsum)
    duVec = duVecNby1.reshape((-1,)) #--Convert to true 1-D array from an Nx1 array
     
    #--Parse the command vector by DM and assign the output commands
    mp,dDM = falco_ctrl_wrapup(mp,cvar,duVec)
    
    #--Take images and compute average intensity in dark hole
    if(mp.ctrl.flagUseModel): #--Perform a model-based grid search using the compact model
        Itotal = falco.imaging.falco_get_expected_summed_image(mp,cvar,dDM)
        InormAvg = np.mean(Itotal[mp.Fend.corr.maskBool])
    else: #--Perform an empirical grid search with actual images from the testbed or full model
        Itotal = falco.imaging.falco_get_summed_image(mp)
        InormAvg = np.mean(Itotal[mp.Fend.corr.maskBool])
        
    #--Reset voltage commands in mp
    if(any(mp.dm_ind==1)): mp.dm1.V = DM1V0
    if(any(mp.dm_ind==2)): mp.dm2.V = DM2V0

    return InormAvg,dDM #dDMtemp


def falco_ctrl_setup(mp,cvar):
    """
    Function to vectorize DM commands and otherwise prepare variables for the controller. 

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    cvar : ModelParameters
        Structure containing controller variables

    Returns
    -------
    None
        Changes are made by reference to structures mp and cvar.
    """
    #--Save starting point for each delta command to be added to.
    if(any(mp.dm_ind==1)): cvar.DM1Vnom = mp.dm1.V
    if(any(mp.dm_ind==2)): cvar.DM2Vnom = mp.dm2.V
    if(any(mp.dm_ind==8)): cvar.DM8Vnom = mp.dm8.V
    if(any(mp.dm_ind==9)): cvar.DM9Vnom = mp.dm9.V
    
    #--Make the vector of total DM commands from before
    u1 = mp.dm1.V.reshape(mp.dm1.NactTotal)[mp.dm1.act_ele] if(any(mp.dm_ind==1)) else np.array([])
    u2 = mp.dm2.V.reshape(mp.dm2.NactTotal)[mp.dm2.act_ele] if(any(mp.dm_ind==2)) else np.array([])
    u8 = mp.dm1.V.reshape(mp.dm8.NactTotal)[mp.dm8.act_ele] if(any(mp.dm_ind==8)) else np.array([])
    u9 = mp.dm1.V.reshape(mp.dm9.NactTotal)[mp.dm9.act_ele] if(any(mp.dm_ind==9)) else np.array([])
    cvar.uVec = np.concatenate((u1,u2,u8,u9))
    cvar.NeleAll = cvar.uVec.size
    
    #--Get the indices of each DM's command within the full command
    u1dummy = 1*np.ones(mp.dm1.Nele,dtype=int) if(any(mp.dm_ind==1)) else np.array([])
    u2dummy = 2*np.ones(mp.dm2.Nele,dtype=int) if(any(mp.dm_ind==2)) else np.array([])
    u8dummy = 8*np.ones(mp.dm8.Nele,dtype=int) if(any(mp.dm_ind==8)) else np.array([])
    u9dummy = 9*np.ones(mp.dm9.Nele,dtype=int) if(any(mp.dm_ind==9)) else np.array([])
    cvar.uLegend = np.concatenate((u1dummy, u2dummy, u8dummy, u9dummy))
    

def falco_ctrl_wrapup(mp,cvar,duVec):
    """
    Function to take the controller commands and apply them to the DMs. 

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    cvar : ModelParameters
        Structure containing controller variables
    duVec : numpy ndarray
        Vector of delta control commands computed by the controller.

    Returns
    -------
    mp : ModelParameters
        Structure containing optical model parameters
    dDM : ModelParameters
        Structure containing the delta DM commands for each DM
    """        
    dDM = falco.config.Object()
    
    #--Parse the command vector by DM
    if(any(mp.dm_ind==1)):
        dDM1Vvec = np.zeros(mp.dm1.NactTotal)
        dDM1Vvec[mp.dm1.act_ele]  = mp.dm1.weight*duVec[cvar.uLegend==1] # Parse the command vector to get component for DM and apply the DM's weight
        dDM.dDM1V = dDM1Vvec.reshape((mp.dm1.Nact,mp.dm1.Nact))
    if(any(mp.dm_ind==2)):  
        dDM2Vvec = np.zeros(mp.dm2.NactTotal)
        dDM2Vvec[mp.dm2.act_ele] = mp.dm2.weight*duVec[cvar.uLegend==2] # Parse the command vector to get component for DM and apply the DM's weight
        dDM.dDM2V = dDM2Vvec.reshape((mp.dm2.Nact,mp.dm2.Nact))
    if(any(mp.dm_ind==8)):  
        dDM.dDM8V = np.zeros(mp.dm8.NactTotal)
        dDM.dDM8V[mp.dm8.act_ele] = mp.dm8.weight*duVec[cvar.uLegend==8] # Parse the command vector to get component for DM and apply the DM's weight
    if(any(mp.dm_ind==9)):  
        dDM.dDM9V = np.zeros(mp.dm9.NactTotal)
        dDM.dDM9V[mp.dm9.act_ele] = mp.dm9.weight*duVec[cvar.uLegend==9] # Parse the command vector to get component for DM and apply the DM's weight
    
    #--Enforce tied actuator pair commands. 
    #   Assign command of first actuator to the second as well.
#    if(any(mp.dm_ind==8)):
#        for ti=1:size(mp.dm8.tied,1)
#            dDM.dDM8V(mp.dm8.tied(ti,2)) = dDM.dDM8V(mp.dm8.tied(ti,1));
#
#    if(any(mp.dm_ind==9)):
#        for ti=1:size(mp.dm9.tied,1)
#            dDM.dDM9V(mp.dm9.tied(ti,2)) = dDM.dDM9V(mp.dm9.tied(ti,1));

    
    #--Combine the delta command with the previous command
    if(any(mp.dm_ind==1)):  mp.dm1.V = cvar.DM1Vnom + dDM.dDM1V
    if(any(mp.dm_ind==2)):  mp.dm2.V = cvar.DM2Vnom + dDM.dDM2V
    if(any(mp.dm_ind==8)):  mp.dm8.V = cvar.DM8Vnom + dDM.dDM8V
    if(any(mp.dm_ind==9)):  mp.dm9.V = cvar.DM9Vnom + dDM.dDM9V


    return mp,dDM
    