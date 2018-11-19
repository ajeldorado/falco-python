# 2018-08-09
# Initial version, starting from scratch, translation of src/Falco-jpl/FALCO/models/model_Jacobian_HLC.m
#
# test with model_jacobian_HLC_test_script.py
# using saved inputs and outputs

# % Copyright 2018, by the California Institute of Technology. ALL RIGHTS
# % RESERVED. United States Government Sponsorship acknowledged. Any
# % commercial use must be negotiated with the Office of Technology Transfer
# % at the California Institute of Technology.
# % -------------------------------------------------------------------------
# %
# % function jac = model_Jacobian_HLC(mp, DM, tsi, whichDM)
# %--Wrapper for the simplified optical models used for the fast Jacobian calculation.
# %  The first-order derivative of the DM pokes are propagated through the system.
# %  Does not include unknown aberrations/errors that are in the full model.
# %  This function is for the DMLC, HLC, APLC, and APHLC coronagraphs.
# %
# % REVISION HISTORY:
# % --------------
# % Modified on 2017-11-13 by A.J. Riggs to be compatible with parfor.
# % Modified on 2017-11-09 by A.J. Riggs to compute only one row of the whole Jacobian. 
# %  This enables much easier parallelization.
# % Modified on 2017-11-09 by A.J. Riggs to have the Jacobian calculation be
# % its own function.
# % Modified on 2017-10-17 by A.J. Riggs to have model_compact.m be a wrapper. All the 
# %  actual compact models have been moved to sub-routines for clarity.
# % Modified on 19 June 2017 by A.J. Riggs to use lower resolution than the
# %   full model.
# % Modified by A.J. Riggs on 18 August 2016 from hcil_model.m to model_compact.m.
# % Modified by A.J. Riggs on 18 Feb 2015 from HCIL_model_lab_BB_v3.m to hcil_model.m.
# % ---------------
# %
# % INPUTS:
# % -mp = structure of model parameters
# % -DM = structure of DM settings
# % -tsi = index of the pair of sub-bandpass index and tip/tilt offset index
# % -whichDM = which DM number
# %
# % OUTPUTS:
# % -Gttlam = Jacobian for the specified DM and specified T/T-wavelength pair
# %

import os, sys
import numpy as np
import multiprocessing
import time
import pdb
import scipy.io as sio

from utils import * # padOrCropEven
import lib.dm.falco_discretize as falco_discretize
import lib.propcustom as propcustom

# function GdmTTlam = model_Jacobian_HLC(mp, DM, tsi, whichDM)
def model_Jacobian_HLC(mp, DM, tsi, whichDM):

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Setup
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # we are assuming for now that all inputs are matlab style and indexes are 1-offset
    # here we create local variables for all the 'arrays of indexes' so that they
    # are 0-offset. We don't want to directly modify the input arrays because in
    # Python all variables are 'passed by ref'

    tsi -= 1 # except scalar constants are passed by value, 
    sbpIndex = mp['Wttlam_si'][tsi] - 1
    ttIndex = mp['Wttlam_ti'][tsi] - 1

    # arrays of indexes change to 0-offset
    dm1_xy_box_lowerLeft = DM['dm1']['compact']['xy_box_lowerLeft'] - 1
    dm2_xy_box_lowerLeft = DM['dm2']['compact']['xy_box_lowerLeft'] - 1
    dm9_xy_box_lowerLeft = DM['dm9']['compact']['xy_box_lowerLeft'] - 1

    listDM1Iact = DM['dm1']['act_ele'] - 1 # 1-offset to 0-offset index iact
    listDM2Iact = DM['dm2']['act_ele'] - 1
    #listDM9Iact = np.arange(DM['dm9']['compact']['NactTotal']) # 0-offset
    listDM9Iact = DM['dm9']['act_ele'] - 1

    # this part of the setup is the original
    lam  = mp['sbp_center_vec'][sbpIndex]
    mirrorFac = 2.; #% Phase change is twice the DM surface height.
    NdmPad = int(DM['compact']['NdmPad'])


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Input E-fields
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # %--Include the tip/tilt in the input wavefront
    x_offset = mp['ttx'][ttIndex]*mp['lambda0']/lam if 'ttx' in mp.keys() else 0.0
    y_offset = mp['tty'][ttIndex]*mp['lambda0']/lam if 'tty' in mp.keys() else 0.0
    
    TTphase = -2.*np.pi*(x_offset*mp['P2']['compact']['XsDL'] + y_offset*mp['P2']['compact']['YsDL'])

    #     Ett = exp(1i*TTphase*mp.lambda0/lam);
    #     Ein = Ett.*mp.P1.compact.E(:,:,modvar.sbpIndex);  

    Ett = np.exp(1j*TTphase*mp['lambda0']/lam);
    Ein = Ett * mp['P1']['compact']['E'][:,:,sbpIndex] ###

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Masks and DM surfaces
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    pupil = padOrCropEven(mp['P1']['compact']['mask'],NdmPad)
    Ein = padOrCropEven(Ein,NdmPad)
    if mp['flagApod'] != 0 and 'compact' in mp['P3'].keys():
        apodRot180 = padOrCropEven( np.rot90(mp['P3']['compact']['mask'],2), NdmPad )
        #if( strcmpi(mp.centering,'pixel') ); apodRot180 = circshift(apodRot180,[1 1]); end %--To undo center offset when pixel centered and rotating by 180 degrees.
        # %--To undo center offset when pixel centered and rotating by 180 degrees.
        if mp['centering'].lower() == 'pixel':
            apodRot180 = np.roll( apodRot180, (1,1), axis=(0,1) )
    else:
        apodRot180 = np.ones((NdmPad,NdmPad))


    # if(mp.flagDM1stop); DM1stop = padOrCropEven(mp.dm1.compact.mask, NdmPad); else; DM1stop = ones(NdmPad); end
    DM1stop = padOrCropEven(mp['dm1']['compact']['mask'], NdmPad) if mp['flagDM1stop'] != 0 \
                 else np.ones((NdmPad,NdmPad))
    DM2stop = padOrCropEven(mp['dm2']['compact']['mask'], NdmPad) if mp['flagDM2stop'] != 0 \
                 else np.ones((NdmPad,NdmPad))
    
    # if(any(DM.dm_ind==1)); DM1surf = padOrCropEven(DM.dm1.compact.surfM, NdmPad);  else; DM1surf = 0; end 
    # if(any(DM.dm_ind==2)); DM2surf = padOrCropEven(DM.dm2.compact.surfM, NdmPad);  else; DM2surf = 0; end 

    DM1surf = padOrCropEven(DM['dm1']['compact']['surfM'], NdmPad) if np.any(DM['dm_ind'] == 1) else 0.
    DM2surf = padOrCropEven(DM['dm2']['compact']['surfM'], NdmPad) if np.any(DM['dm_ind'] == 2) else 0.

    # %--Complex transmission of the FPM. Calculated in model_Jacobian.m.
    # FPM = squeeze(DM.FPMcube(:,:,modvar.sbpIndex)); 
    FPM = np.squeeze(DM['FPMcube'][:,:,sbpIndex])
    
    # %--Complex transmission of the points outside the FPM (just fused silica with neither dielectric nor metal).
    # ilam = modvar.sbpIndex; 
    # ind_metal = falco_discretize_FPM_surf(0, mp.t_metal_nm_vec, mp.dt_metal_nm); %--Obtain the indices of the nearest thickness values in the complex transmission datacube.

    # %--Obtain the indices of the nearest thickness values in the complex transmission datacube.
    ind_metal = falco_discretize.FPM_surf(np.zeros((1,)), mp['t_metal_nm_vec'], mp['dt_metal_nm'])
    # %--Obtain the indices of the nearest thickness values in the complex transmission datacube.
    ind_diel  = falco_discretize.FPM_surf(np.zeros((1,)), mp['t_diel_nm_vec'],  mp['dt_diel_nm']) 

    # %--Complex transmission of the points outside the FPM (just fused silica with neither dielectric nor metal).
    transOuterFPM = mp['complexTransCompact'][ind_diel,ind_metal,sbpIndex]
    
    ## no gpu for now
    # if(mp.useGPU)
    #     pupil = gpuArray(pupil);
    #     Ein = gpuArray(Ein);
    #     if(any(DM.dm_ind==1)); DM1surf = gpuArray(DM1surf); end
    # end
    
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Propagation
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    tstart = time.time()

    # %--Define pupil P1 and Propagate to pupil P2
    EP1 = pupil * Ein # %--E-field at pupil plane P1
    # %--Forward propagate to the next pupil plane (P2) by rotating 180 deg.
    EP2 = propcustom.TwoFT(EP1, mp['centering']) 
    
    # %--Propagate from P2 to DM1, and apply DM1 surface and aperture stop
    # if( abs(mp['d_P2_dm1'])~=0 ); Edm1 = propcustom_PTP(EP2,mp.P2.compact['dx']*NdmPad,lam,mp['d_P2_dm1']); 
    # else; Edm1 = EP2; end  %--E-field arriving at DM1
    if np.abs(mp['d_P2_dm1']) > 10.0*lam:
        Edm1 = propcustom.PTP(EP2,mp['P2']['compact']['dx']*NdmPad,lam,mp['d_P2_dm1'])
    else:
        Edm1 = EP2

    Edm1 = DM1stop * Edm1 * np.exp(1j*mirrorFac*(2.0*np.pi/lam)*DM1surf) # %--E-field leaving DM1

    # %--DM1---------------------------------------------------------
    if whichDM==1:
        #print(mp['F4']['compact']['corr']['inds'].shape)
        npixtmp = (mp['F4']['compact']['corr']['mask'].flatten()>0).sum()
        GdmTTlam = np.zeros((npixtmp, DM['dm1']['NactTotal']), dtype='complex')
    
        #     %--Two array sizes (at same resolution) of influence functions for MFT and angular spectrum
        # %--Smaller array size for MFT to FPM after FFT-AS propagations from DM1->DM2->DM1
        Nbox1 = DM['dm1']['compact']['Nbox'] 
        # % NboxPad1;%2.^ceil(log2(NboxPad1)); %--Power of 2 array size for FFT-AS propagations from DM1->DM2->DM1
        NboxPad1AS = DM['dm1']['compact']['NboxAS']
        # %--Adjust the sub-array location of the influence function for the added zero padding
        dm1_xy_box_lowerLeft_AS = np.round( dm1_xy_box_lowerLeft - (NboxPad1AS-Nbox1)/2 ).astype('int')

        if np.any(DM['dm_ind']==2):
            DM2surf = padOrCropEven(DM2surf,DM['dm1']['compact']['NdmPad'])
        else:
            DM2surf = np.zeros( (DM['dm1']['compact']['NdmPad'],)*2, dtype='complex' )

        if mp['flagDM2stop']:
            DM2stop = padOrCropEven(DM2stop,DM['dm1']['compact']['NdmPad'])
        else:
            DM2stop = np.ones( (DM['dm1']['compact']['NdmPad'],)*2 )

        apodRot180 = padOrCropEven( apodRot180, DM['dm1']['compact']['NdmPad'] )
    
        # %--Pad or crop for expected sub-array indexing
        Edm1pad = padOrCropEven(Edm1,DM['dm1']['compact']['NdmPad']);
    
        # %--Propagate each actuator from DM1 through the optical system
        #for iact in range(DM['dm1']['compact']['NactTotal']):

        #  %--Only compute for acutators specified for use
        #     DM[]['act_ele'] is list of 1-offset indexes
        for iact in listDM1Iact:

            #  skip this actuator if influence functions that are not zeroed out
            if np.max(np.abs(DM['dm1']['compact']['inf_datacube'][:,:,iact])) <= np.finfo('float64').eps:
                continue
                
            # %--x- and y- coordinates of the padded influence function in the full padded pupil
            # % x-indices in pupil arrays for the box
            x_box_AS_slice = slice(dm1_xy_box_lowerLeft_AS[0,iact], \
                                   dm1_xy_box_lowerLeft_AS[0,iact]+NboxPad1AS) 
            # % y-indices in pupil arrays for the box
            y_box_AS_slice = slice(dm1_xy_box_lowerLeft_AS[1,iact], \
                                   dm1_xy_box_lowerLeft_AS[1,iact]+NboxPad1AS)

            
            # %--Propagate from DM1 to DM2, and then back to P2
            # %--Pad influence function at DM1 for angular spectrum propagation.
            dEbox = (mirrorFac*2*np.pi*1j/lam) \
                    * padOrCropEven(DM['dm1']['VtoH'].T.flatten()[iact] \
                                    * DM['dm1']['compact']['inf_datacube'][:,:,iact],
                                    NboxPad1AS) 
            
            #  % forward propagate to DM2 and apply DM2 E-field
            dEbox = propcustom.PTP(dEbox * Edm1pad[y_box_AS_slice, x_box_AS_slice],
                                   mp['P2']['compact']['dx']*NboxPad1AS, lam,
                                   mp['d_dm1_dm2'])
            
            # % back-propagate to DM1 (pupil plane)
            dEP2box = propcustom.PTP(dEbox * DM2stop[y_box_AS_slice,x_box_AS_slice] \
                                     * np.exp(mirrorFac*2*np.pi*1j/lam*DM2surf[y_box_AS_slice,x_box_AS_slice]),
                                     mp['P2']['compact']['dx']*NboxPad1AS, lam,
                                     -1*(mp['d_dm1_dm2'] + mp['d_P2_dm1'] ))
            
            #  %--Crop down from the array size that is a power of 2 to make the MFT faster
            dEP2box = padOrCropEven(dEP2box,Nbox1)
            
            #  %--x- and y- coordinates of the UN-padded influence function in the full padded pupil
            #  % x-indices in pupil arrays for the box
            #  % y-indices in pupil arrays for the box
            x_box_slice = slice(dm1_xy_box_lowerLeft[0,iact], 
                                dm1_xy_box_lowerLeft[0,iact]+Nbox1)
            y_box_slice = slice(dm1_xy_box_lowerLeft[1,iact],
                                dm1_xy_box_lowerLeft[1,iact]+Nbox1)
            
            # % full pupil x-coordinates of the box 
            x_box = DM['dm1']['compact']['x_pupPad'][x_box_slice].T
            # % full pupil y-coordinates of the box
            y_box = DM['dm1']['compact']['y_pupPad'][y_box_slice] 
            
            # % simulate going forward to the next pupil plane (with the apodizer) most efficiently, 
            # % First, back-propagate the apodizer (by rotating 180-degrees) to the previous pupil.
            # % Second, negate the coordinates of the box used.
            
            # %--Apply 180deg-rotated SP mask.
            # %--Forward propagate the cropped box by rotating 180 degrees.
            dEP2box = apodRot180[y_box_slice,x_box_slice] * dEP2box
            dEP3box = (1/1j)**2 * np.rot90(dEP2box,2)
            
            # %--Negate to effectively rotate by 180 degrees
            x_box = np.flipud(-x_box) #np.rot90(-x_box,2)
            y_box = np.flipud(-y_box) #np.rot90(-y_box,2)
            
            # %--Matrices for the MFT from the pupil P3 to the focal plane mask
            # output product to create MFT matrix
            etas_y = np.outer(mp['F3']['compact']['etas'], y_box)
            rect_mat_pre = np.exp(-2*np.pi*1j*etas_y/(lam*mp['fl'])) \
                           * np.sqrt(mp['P2']['compact']['dx'] * mp['P2']['compact']['dx']) \
                           * np.sqrt(mp['F3']['compact']['dxi'] * mp['F3']['compact']['deta']) \
                           / (1j*lam*mp['fl'])
            
            x_xis = np.outer(x_box, mp['F3']['compact']['xis'])
            rect_mat_post  = np.exp(-2*np.pi*1j*x_xis/(lam*mp['fl']))
            
            # %--MFT from pupil P3 to FPM
            # % MFT to FPM
            EF3 = np.matmul( np.matmul(rect_mat_pre, dEP3box), rect_mat_post)
            # %--Propagate through (1-complex FPM) for Babinet's principle
            EF3 = (transOuterFPM-FPM) * EF3
            
            # %--DFT to LS ("Sub" name for Subtrahend part of the Lyot-plane E-field)
            #   %--Subtrahend term for the Lyot plane E-field    
            EP4sub = propcustom.mft_FtoP(
                EF3, mp['fl'], lam, 
                mp['F3']['compact']['dxi'], mp['F3']['compact']['deta'],
                mp['P4']['compact']['dx'],  mp['P4']['compact']['Narr'],
                centering=mp['centering']
            )
            
            # %--Full Lyot plane pupil (for Babinet)
            #  %--Propagating the E-field from P2 to P4 without masks gives the same E-field. 
            EP4noFPM = np.zeros( (DM['dm1']['compact']['NdmPad'],)*2, dtype='complex' )
            EP4noFPM[y_box_slice,x_box_slice] = dEP2box
            EP4noFPM = padOrCropEven(EP4noFPM, mp['P4']['compact']['Narr'])
            # % Babinet's principle to get E-field at Lyot plane
            # mask .* (scalar * Efield - Esub)
            EP4 = mp['P4']['compact']['croppedMask'] * (transOuterFPM*EP4noFPM - EP4sub); 
            
            # % DFT to camera
            EF4 = propcustom.mft_PtoF(
                EP4, mp['fl'], lam,
                mp['P4']['compact']['dx'], 
                mp['F4']['compact']['dxi'],  mp['F4']['compact']['Nxi'], 
                mp['F4']['compact']['deta'], mp['F4']['compact']['Neta'],
                centering=mp['centering'])
            
            # rather than use ['inds'] an array of indexes, and have to worry about whether it's
            # 1-offset or 0-offset, and row-major or column-major, we will use the ['mask']
            # directly.
            # GdmTTlam[:,iact] = \
            #                    DM['dm_weights'][0]*EF4[mp['F4']['compact']['corr']['inds']] \
            #                    / np.sqrt(mp['F4']['compact']['I00'][sbpIndex])
            
            # python style row-major ordering of EF4[mask]
            # GdmTTlam[:,iact] = DM['dm_weights'][0] \
            #                    * EF4.flatten()[mp['F4']['compact']['corr']['mask'].flatten()>0] \
            #                    / np.sqrt(mp['F4']['compact']['I00'][sbpIndex])

            # matlab style column-major ordering of EF4[mask]
            GdmTTlam[:,iact] = DM['dm_weights'][0] \
                               * EF4.T.flatten()[mp['F4']['compact']['corr']['mask'].T.flatten()>0] \
                               / np.sqrt(mp['F4']['compact']['I00'][sbpIndex])

            pass # end for iact
            
        elapsedtime = time.time() - tstart
        print('Done DM1, ', elapsedtime, ' seconds')
        pass # end if DM1


    # %--DM2---------------------------------------------------------
    if whichDM==2:
        npixtmp = (mp['F4']['compact']['corr']['mask'].flatten()>0).sum()
        GdmTTlam = np.zeros((npixtmp, DM['dm2']['NactTotal']), dtype='complex')
    
        #     %--Two array sizes (at same resolution) of influence functions for MFT and angular spectrum
        Nbox2 = DM['dm2']['compact']['Nbox']
        NboxPad2AS = DM['dm2']['compact']['NboxAS']
        # %--Account for the padding of the influence function boxes
        dm2_xy_box_lowerLeft_AS = np.round( dm2_xy_box_lowerLeft - (NboxPad2AS-Nbox2)/2 ).astype('int')

        apodRot180 = padOrCropEven( apodRot180, DM['dm2']['compact']['NdmPad'] )
        DM2stop = padOrCropEven(DM2stop, DM['dm2']['compact']['NdmPad'])
        
        #     %--Propagate full field to DM2 before back-propagating in small boxes
        # % E-field incident upon DM2
        Edm2inc = padOrCropEven( 
            propcustom.PTP(Edm1,NdmPad*mp['P2']['compact']['dx'],lam,mp['d_dm1_dm2']),
            DM['dm2']['compact']['NdmPad']
            )

        # % Initial E-field at DM2 including its own phase contribution
        Edm2 = DM2stop * Edm2inc * \
               np.exp( mirrorFac*(2.*np.pi*1j/lam) * padOrCropEven(DM2surf,DM['dm2']['compact']['NdmPad']) )
    
        #     %--Propagate each actuator from DM2 through the rest of the optical system
        #     for iact=1:DM['dm2']['compact']['NactTotal']
        
        #         if(any(any(DM['dm2']['compact'].inf_datacube(:,:,iact)))  && any(DM['dm2'].act_ele==iact) ) 
        for iact in listDM2Iact:

            #  skip this actuator if influence functions that are not zeroed out
            if np.max(np.abs(DM['dm2']['compact']['inf_datacube'][:,:,iact])) <= np.finfo('float64').eps:
                continue

            # %--x- and y- coordinates of the padded influence function in the full padded pupil
            # % x-indices in pupil arrays for the box
            x_box_AS_slice = slice(dm2_xy_box_lowerLeft_AS[0,iact], \
                                   dm2_xy_box_lowerLeft_AS[0,iact]+NboxPad2AS) 
            # % y-indices in pupil arrays for the box
            y_box_AS_slice = slice(dm2_xy_box_lowerLeft_AS[1,iact], \
                                   dm2_xy_box_lowerLeft_AS[1,iact]+NboxPad2AS)


            # %--the padded influence function at DM2
            dEbox = (mirrorFac*2.*np.pi*1j/lam) \
                    * padOrCropEven(DM['dm2']['VtoH'].T.flatten()[iact] \
                                    * DM['dm2']['compact']['inf_datacube'][:,:,iact],
                                    NboxPad2AS)

            # % back-propagate to pupil P2
            dEP2box = propcustom.PTP(dEbox * Edm2[y_box_AS_slice,x_box_AS_slice],
                                     mp['P2']['compact']['dx']*NboxPad2AS,
                                     lam, -1.*(mp['d_dm1_dm2'] + mp['d_P2_dm1']) )

            #  %--Crop down from the array size that is a power of 2 to make the MFT faster
            dEP2box = padOrCropEven(dEP2box,Nbox2)

            #  %--x- and y- coordinates of the UN-padded influence function in the full padded pupil
            # % x-indices in pupil arrays for the box
            # % y-indices in pupil arrays for the box
            x_box_slice = slice(dm2_xy_box_lowerLeft[0,iact], dm2_xy_box_lowerLeft[0,iact]+Nbox2)
            y_box_slice = slice(dm2_xy_box_lowerLeft[1,iact], dm2_xy_box_lowerLeft[1,iact]+Nbox2)
            
            # % full pupil x-coordinates of the box 
            # % full pupil y-coordinates of the box 
            x_box = DM['dm2']['compact']['x_pupPad'][x_box_slice].T
            y_box = DM['dm2']['compact']['y_pupPad'][y_box_slice].T

            # % simulate going forward to the next pupil plane (with the apodizer) most efficiently, 
            # % First, back-propagate the apodizer (by rotating 180-degrees) to the previous pupil.
            # % Second, negate the coordinates of the box used.

            # %--Apply 180deg-rotated SP mask.
            # %--Forward propagate the cropped box by rotating 180 degrees.
            dEP2box = apodRot180[y_box_slice,x_box_slice] * dEP2box
            dEP3box = (1./1j)**2 * np.rot90(dEP2box,2)

            x_box = np.flipud(-x_box) # rot90(-x_box,2); %--Negate to effectively rotate by 180 degrees
            y_box = np.flipud(-y_box) # rot90(-y_box,2); %--Negate to effectively rotate by 180 degrees
            
            # %--Matrices for the MFT from the pupil P3 to the focal plane mask
            etas_y = np.outer(mp['F3']['compact']['etas'], y_box)
            rect_mat_pre = np.exp(-2.*np.pi*1j*etas_y/(lam*mp['fl'])) \
                           * np.sqrt(mp['P2']['compact']['dx'] * mp['P2']['compact']['dx']) \
                           * np.sqrt(mp['F3']['compact']['dxi'] * mp['F3']['compact']['deta']) \
                           / (1j*lam*mp['fl'])

            x_xis = np.outer(x_box, mp['F3']['compact']['xis'])
            rect_mat_post  = (np.exp(-2.*np.pi*1j*x_xis/(lam*mp['fl'])));

            # %--MFT from pupil P3 to FPM
            # %--Crop back down to make the MFT faster
            # % MFT to FPM
            dEP2box = padOrCropEven(dEP2box,Nbox2)
            EF3 = np.matmul( np.matmul(rect_mat_pre, dEP3box), rect_mat_post)
            # %--Propagate through ( 1 - (complex FPM) ) for Babinet's principle
            EF3 = (transOuterFPM-FPM) * EF3

            # % DFT to LS ("Sub" name for Subtrahend part of the Lyot-plane E-field)
            # %--Subtrahend term for the Lyot plane E-field
            EP4sub = propcustom.mft_FtoP(
                EF3, mp['fl'], lam, 
                mp['F3']['compact']['dxi'], mp['F3']['compact']['deta'],
                mp['P4']['compact']['dx'], mp['P4']['compact']['Narr'],
                centering=mp['centering']
                )

            # %--Propagating the E-field from P2 to P4 without masks gives the same E-field.
            EP4noFPM = np.zeros( (DM['dm2']['compact']['NdmPad'],)*2, dtype='complex')
            EP4noFPM[y_box_slice, x_box_slice] = dEP2box
            EP4noFPM = padOrCropEven(EP4noFPM, mp['P4']['compact']['Narr'])

            # % Babinet's principle to get E-field at Lyot plane
            EP4 = mp['P4']['compact']['croppedMask'] * (transOuterFPM*EP4noFPM - EP4sub)

            # % DFT to camera
            EF4 = propcustom.mft_PtoF(
                EP4, mp['fl'], lam,
                mp['P4']['compact']['dx'], 
                mp['F4']['compact']['dxi'], mp['F4']['compact']['Nxi'],
                mp['F4']['compact']['deta'], mp['F4']['compact']['Neta'],
                centering=mp['centering']
                )

            # # python style row-major ordering of EF4[mask]
            # GdmTTlam[:,iact] = DM['dm_weights'][1] \
            #                    * EF4.flatten()[mp['F4']['compact']['corr']['mask'].flatten()>0] \
            #                    / np.sqrt(mp['F4']['compact']['I00'][sbpIndex])

            # matlab style row-major ordering of EF4[mask]
            GdmTTlam[:,iact] = DM['dm_weights'][1] \
                               * EF4.T.flatten()[mp['F4']['compact']['corr']['mask'].T.flatten()>0] \
                               / np.sqrt(mp['F4']['compact']['I00'][sbpIndex])


            pass # end for iact

        elapsedtime = time.time() - tstart
        print('Done DM2, ', elapsedtime, ' seconds')

        pass # end if DM2






    # %--DM8--------------------------------------------------------- 
    # if(whichDM==8)
    #     GdmTTlam = zeros(length(mp['F4']['compact']['corr']['inds']),DM.dm8['NactTotal']);
    #     Nbox8 = DM.dm8['compact'].Nbox;
    
    #     stepFac = 1; %--Adjust the step size in the Jacobian, then divide back out. Used for helping counteract effect of discretization.
    
    #     %--Propagate from DM1 to DM2, and apply DM2 surface and aperture stop
    #     Edm2 = DM2stop.*exp(mirrorFac*2*pi*1i*DM2surf/lam).*propcustom_PTP(Edm1,mp['P2']['compact']['dx']*NdmPad,lam,mp['d_dm1_dm2']); % Pre-compute the initial DM2 E-field
    
    #     %--Back-propagate to pupil P2
    #     if( mp['d_P2_dm1'] + mp['d_dm1_dm2'] == 0 )
    #         EP2eff = Edm2; 
    #     else
    #         EP2eff = propcustom_PTP(Edm2,mp['P2']['compact']['dx']*NdmPad,lam,-1*(mp['d_dm1_dm2'] + mp['d_P2_dm1'])); 
    #     end
    
    #     %--Rotate 180 degrees to propagate to pupil P3
    #     EP3 = propcustom_2FT(EP2eff, mp['centering']);
    
    #     %--Apply apodizer mask.
    #     if(mp.flagApod)
    #         EP3 = mp.P3['compact'].mask.*padOrCropEven(EP3, mp['P3']['compact']['Narr']); 
    #     end
    
    #     %--MFT from pupil P3 to FPM (at focus F3)
    #     EF3inc = padOrCropEven( propcustom_mft_PtoF(EP3, mp['fl'],lam,mp['P2']['compact']['dx'],mp['F3']['compact']['dxi'],mp['F3']['compact']['Nxi'],mp['F3']['compact']['deta'],mp['F3']['compact']['Neta'],mp['centering']), DM['dm8']['compact']['NdmPad']);
    
    #     %--Coordinates for metal thickness and dielectric thickness
    # %     DM8transIndAll = falco_discretize_FPM_surf(DM.dm8.surf, mp.t_metal_nm_vec, mp.dt_metal_nm); %--All of the mask
    #     DM9transIndAll = falco_discretize_FPM_surf(DM['dm9'].surf, mp.t_diel_nm_vec, mp.dt_diel_nm); %--All of the mask
    
    #     %--Propagate each actuator from DM9 through the rest of the optical system
    #     for iact=1:DM.dm8['compact']['NactTotal']  
    #          if(any(any(DM.dm8['compact'].inf_datacube(:,:,iact)))  && any(DM.dm8.act_ele==iact) )    
    #             %--xi- and eta- coordinates in the full FPM portion of the focal plane
    #             xi_box_slice = DM.dm8['compact'].xy_box_lowerLeft(1,iact):DM.dm8['compact'].xy_box_lowerLeft(1,iact)+DM.dm8['compact'].Nbox-1; % xi-indices in image arrays for the box
    #             eta_box_slice = DM.dm8['compact'].xy_box_lowerLeft(2,iact):DM.dm8['compact'].xy_box_lowerLeft(2,iact)+DM.dm8['compact'].Nbox-1; % eta-indices in image arrays for the box
    #             xi_box = DM.dm8['compact'].x_pupPad(xi_box_slice).'; % full image xi-coordinates of the box 
    #             eta_box = DM.dm8['compact'].y_pupPad(eta_box_slice); % full image eta-coordinates of the box 

    #             %--Obtain values for the "poked" FPM's complex transmission (only in the sub-array where poked)
    #             Nxi = Nbox8;
    #             Neta = Nbox8;
    
    #             DM8surfCropNew = stepFac*DM.dm8.VtoH(iact).*DM.dm8['compact'].inf_datacube(:,:,iact) + DM.dm8.surf(eta_box_slice,xi_box_slice); % New DM8 surface profile in the poked region (meters)
    #             DM8transInd = falco_discretize_FPM_surf(DM8surfCropNew, mp.t_metal_nm_vec,  mp.dt_metal_nm);
    #             DM9transInd = DM9transIndAll(eta_box_slice,xi_box_slice); %--Cropped region of the FPM.

    #             %             DM9surfCropNew = stepFac*DM['dm9'].VtoH(iact).*DM['dm9']['compact'].inf_datacube(:,:,iact) + DM['dm9'].surf(eta_box_slice,xi_box_slice); % New DM9 surface profile in the poked region (meters)
    # %             DM9transInd = falco_discretize_FPM_surf(DM9surfCropNew, mp.t_diel_nm_vec,  mp.dt_diel_nm);
    # %             DM8transInd = DM8transIndAll(eta_box_slice,xi_box_slice); %--Cropped region of the FPM.
            
    #             %--Look up table to compute complex transmission coefficient of the FPM at each pixel
    #             FPMpoked = zeros(Neta, Nxi); %--Initialize output array of FPM's complex transmission    
    #             for ix = 1:Nxi
    #                 for iy = 1:Neta
    #                     ind_metal = DM8transInd(iy,ix);
    #                     ind_diel  = DM9transInd(iy,ix);
    #                     %fprintf('\t%d\t%d\n',ind_metal,ind_diel)
    #                     FPMpoked(iy,ix) = mp.complexTransCompact(ind_diel,ind_metal,sbpIndex);
    #                 end
    #             end            
    
    #             dEF3box = ( (transOuterFPM-FPMpoked) - (transOuterFPM-FPM(eta_box_slice,xi_box_slice)) ).*EF3inc(eta_box_slice,xi_box_slice); % Delta field (in a small region) at the FPM

    #             %--Matrices for the MFT from the FPM stamp to the Lyot stop
    #             rect_mat_pre = (exp(-2*pi*1j*(mp['P4']['compact'].ys*eta_box)/(lam*mp['fl'])))...
    #                 *sqrt(mp['P4']['compact']['dx']*mp['P4']['compact']['dx'])*sqrt(mp['F3']['compact']['dxi']*mp['F3']['compact']['deta'])/(1j*lam*mp['fl']);
    #             rect_mat_post  = (exp(-2*pi*1j*(xi_box*mp['P4']['compact']['xs'])/(lam*mp['fl'])));

    #             %--DFT from FPM to Lyot stop (Nominal term transOuterFPM*EP4noFPM subtracts out to 0 since it ignores the FPM change).
    #             EP4 = 0 - rect_mat_pre*dEF3box*rect_mat_post; % MFT from FPM (F3) to Lyot stop plane (P4)
    #             EP4 = mp['P4']['compact']['croppedMask'].*EP4; %--Apply Lyot stop

    #             %--DFT to final focal plane
    #             EF4 = propcustom_mft_PtoF(EP4,mp['fl'],lam,mp['P4']['compact']['dx'],mp['F4']['compact']['dxi'],mp['F4']['compact']['Nxi'],mp['F4']['compact']['deta'],mp['F4']['compact']['Neta'],mp['centering']);

    #             GdmTTlam(:,iact) = (1/stepFac)*DM['dm_weights'](8)*EF4(mp['F4']['compact']['corr']['inds'])/sqrt(mp['F4']['compact']['I00'][sbpIndex]));
    #         end
    #     end

    # end %%%%%%%%%%%%%%%%%%%




    # %--DM9--------------------------------------------------------- 
    if whichDM==9:
        npixtmp = (mp['F4']['compact']['corr']['mask'].flatten()>0).sum()
        GdmTTlam = np.zeros((npixtmp,  DM['dm9']['NactTotal']), dtype='complex')
    
        Nbox9 = DM['dm9']['compact']['Nbox']
    
        if not 'stepFac' in DM['dm9'].keys():
            # %--Adjust the step size in the Jacobian, then divide back out. 
            # Used for helping counteract effect of discretization.
            stepFac = 20 # ;%10; 
        else:
            stepFac = DM['dm9']['stepFac']

        #     %DM9phasePad = padOrCropEven(DM9phase,DM['dm9']['compact']['NdmPad']);
        #     %FPMampPad = padOrCropEven(mp['F3']['compact'].mask.amp,DM['dm9']['compact']['NdmPad'],'extrapval',1);
    
        # % Pre-compute the initial DM2 E-field
        # %--Propagate from DM1 to DM2, and apply DM2 surface and aperture stop
        Edm2inc = propcustom.PTP(Edm1,NdmPad*mp['P2']['compact']['dx'],lam,mp['d_dm1_dm2'])
        Edm2 = DM2stop * Edm2inc * np.exp(mirrorFac*2.*np.pi*1j*DM2surf/lam)
    
        #     %--Back-propagate to pupil P2
        if mp['d_P2_dm1'] + mp['d_dm1_dm2'] == 0:
            EP2eff = Edm2
        else:
            EP2eff = propcustom.PTP(Edm2,mp['P2']['compact']['dx']*NdmPad,lam,
                                    -1*(mp['d_dm1_dm2'] + mp['d_P2_dm1']))
            
        # %--Rotate 180 degrees to propagate to pupil P3
        EP3 = propcustom.TwoFT(EP2eff, mp['centering'])
        # %--Forward propagate to the next pupil plane (with the SP) by rotating 180 deg.
        # %     EP3 = (1/1j)^2*rot90(EP2eff,2); 
        # %--To undo center offset when beam and mask are pixel centered and rotating by 180 degrees.
        # %     if( strcmpi(mp['centering'],'pixel') ); EP3 = circshift(EP3,[1 1]); end;   
    
        #     %--Apply apodizer mask.
        if mp['flagApod']:
            EP3 = mp['P3']['compact']['mask'] * padOrCropEven(EP3, mp['P3']['compact']['Narr'])
    
        #     %--MFT from pupil P3 to FPM (at focus F3)
        EF3inc = padOrCropEven( 
            propcustom.mft_PtoF(EP3, mp['fl'],lam,
                                mp['P2']['compact']['dx'],mp['F3']['compact']['dxi'],
                                mp['F3']['compact']['Nxi'],mp['F3']['compact']['deta'],
                                mp['F3']['compact']['Neta'],
                                centering=mp['centering']
                            ), DM['dm9']['compact']['NdmPad'])
    
        #     %--Coordinates for metal thickness and dielectric thickness
        # %     [X,Y] = meshgrid(mp.t_metal_nm_vec,mp.t_diel_nm_vec); %--Grid for interpolation
        # % %     DM8surfNM = round(1e9*DM.dm8.surf); % meters -> discretized nanometers
        #  %--All of the mask
        DM8transIndAll = falco_discretize.FPM_surf(DM['dm8']['surf'],
                                                   mp['t_metal_nm_vec'],
                                                   mp['dt_metal_nm'])
    

        #     %--Propagate each actuator from DM9 through the rest of the optical system
        for iact in listDM9Iact:
            if np.max(np.abs(DM['dm9']['compact']['inf_datacube'][:,:,iact])) <= np.finfo('float64').eps:
                continue

            #             %--xi- and eta- coordinates in the full FPM portion of the focal plane
            #  % xi-indices in image arrays for the box
            xi_box_slice = slice(dm9_xy_box_lowerLeft[0,iact],
                                 dm9_xy_box_lowerLeft[0,iact]+DM['dm9']['compact']['Nbox'])
            # ; % eta-indices in image arrays for the box
            eta_box_slice = slice(dm9_xy_box_lowerLeft[1,iact],
                                  dm9_xy_box_lowerLeft[1,iact]+DM['dm9']['compact']['Nbox'])
            
            #  % full image xi-coordinates of the box 
            xi_box = DM['dm9']['compact']['x_pupPad'][xi_box_slice] # .T ??
            #  % full image eta-coordinates of the box 
            eta_box = DM['dm9']['compact']['y_pupPad'][eta_box_slice]
            
            #  %--Obtain values for the "poked" FPM's complex transmission
            # (only in the sub-array where poked)
            Nxi = Nbox9
            Neta = Nbox9
            # % %             DM8surfCropNM = DM8surfNM(eta_box_slice,xi_box_slice);
            # %             DM9surfCropNew = stepFac*DM['dm9'].VtoH(iact).*DM['dm9']['compact'].inf_datacube(:,:,iact) + DM['dm9'].surf(eta_box_slice,xi_box_slice); % New DM9 surface profile in the poked region (meters)
            # % %             DM9surfCropNewNM = round(1e9*DM9surfCropNew); %  meters -> discretized nanometers
            #  % New DM9 surface profile in the poked region (meters)
            DM9surfCropNew = stepFac*DM['dm9']['VtoH'][iact] * DM['dm9']['compact']['inf_datacube'][:,:,iact]\
                             + DM['dm9']['surf'][eta_box_slice,xi_box_slice]
            DM9transInd = falco_discretize.FPM_surf(DM9surfCropNew,
                                                    mp['t_diel_nm_vec'],  mp['dt_diel_nm'])
            #  %--Cropped region of the FPM.
            DM8transInd = DM8transIndAll[eta_box_slice,xi_box_slice]
            
            #%--Look up table to compute complex transmission coefficient of the FPM at each pixel
            #  %--Initialize output array of FPM's complex transmission    
            FPMpoked = np.zeros((Neta, Nxi), dtype='complex')
            for ix in range(Nxi):
                for iy in range(Neta):
                    ind_metal = DM8transInd[iy,ix]
                    ind_diel  = DM9transInd[iy,ix]
                    #  %fprintf('\t%d\t%d\n',ind_metal,ind_diel)
                    FPMpoked[iy,ix] = mp['complexTransCompact'][ind_diel,ind_metal,sbpIndex]

            # %  %--Interpolate (MORE ACCURATE, BUT MUCH SLOWER THAN LOOK-UP TABLE)
            # %             DM8surfCrop = DM['dm8'].surf(eta_box_slice,xi_box_slice);
            # %             DM9surfCrop = DM['dm9'].surf(eta_box_slice,xi_box_slice);
            # %             FPMpoked = zeros(Neta, Nxi); %--Initialize output array of FPM's complex transmission
            # %             for il=1:Nlam        
            # %                 for ix = 1:Nxi
            # %                     for iy = 1:Neta
            # %                         DM9surfNew = DM['dm9']['compact'].inf_datacube(:,:,iact) + DM9surfCrop;
            # %                         FPMpoked(iy,ix) = interp2(X, Y, squeeze(mp.complexTransCompact(:,:,sbpIndex)), DM8surfCrop(iy,ix), DM9surfNew(iy,ix));
            # %                     end
            # %                 end
            # %             end
            
            
            # % Delta field (in a small region) at the FPM
            dEF3box = ( (transOuterFPM-FPMpoked) - (transOuterFPM-FPM[eta_box_slice,xi_box_slice])) * EF3inc[eta_box_slice,xi_box_slice] 
            
            # %             dEF3box = -(2*pi*1j/lam)*(DM['dm9']['VtoH'](iact)*DM['dm9']['compact'].inf_datacube(:,:,iact))...
            # %             .*(FPMampPad(eta_box_slice,xi_box_slice).*exp(2*pi*1i/lam*DM9phasePad(eta_box_slice,xi_box_slice))).*EF3inc(eta_box_slice,xi_box_slice); % Delta field (in a small region) at the FPM
            
            #             %--Matrices for the MFT from the FPM stamp to the Lyot stop
            ### check that this is ys rows x eta_box columns (202 x 28 in the example)
            ys = mp['P4']['compact']['ys'] # should be column vector
            xs = mp['P4']['compact']['xs'] # should be row vector
            rect_mat_pre = (np.exp(-2*np.pi*1j*(np.outer(ys,eta_box))/(lam*mp['fl']))) \
                           *np.sqrt(mp['P4']['compact']['dx']*mp['P4']['compact']['dx']) \
                           *np.sqrt(mp['F3']['compact']['dxi']*mp['F3']['compact']['deta']) \
                           /(1j*lam*mp['fl'])
            # xo_box is column vecotr; xs is row vector (28 x 202 in the example)
            rect_mat_post  = (np.exp(-2*np.pi*1j*(np.outer(xi_box,xs))/(lam*mp['fl'])));
            
            #   %--DFT from FPM to Lyot stop 
            # (Nominal term transOuterFPM*EP4noFPM subtracts out to 0 
            # since it ignores the FPM change).
            # % MFT from FPM (F3) to Lyot stop plane (P4)
            EP4 = 0 - np.matmul( np.matmul(rect_mat_pre,dEF3box), rect_mat_post )
            EP4 = mp['P4']['compact']['croppedMask']*EP4  #; %--Apply Lyot stop
            
            #             %--DFT to final focal plane
            EF4 = propcustom.mft_PtoF(
                EP4, mp['fl'], lam,
                mp['P4']['compact']['dx'],
                mp['F4']['compact']['dxi'],mp['F4']['compact']['Nxi'],
                mp['F4']['compact']['deta'],mp['F4']['compact']['Neta'],
                centering=mp['centering'])
    
            # # matlab
            # GdmTTlam(:,iact) = DM['dm9']['act_sens']*(1/stepFac)*DM['dm_weights'][9-1]\
                #                    *EF4(mp['F4']['compact']['corr']['inds'])\
                #                    /sqrt(mp['F4']['compact']['I00'][sbpIndex]))


            # matlab style row-major ordering of EF4[mask]
            GdmTTlam[:,iact] = DM['dm9']['act_sens']*(1/stepFac)*DM['dm_weights'][9-1] \
                            * EF4.T.flatten()[mp['F4']['compact']['corr']['mask'].T.flatten()>0] \
                            / np.sqrt(mp['F4']['compact']['I00'][sbpIndex])

            pass # end for iact
        
        elapsedtime = time.time() - tstart
        print('Done DM9, ', elapsedtime, ' seconds')

        pass  # end if DM9


    return GdmTTlam




# if(mp.useGPU)
#     GdmTTlam = gather(GdmTTlam);
# end

# end %--END OF FUNCTION

