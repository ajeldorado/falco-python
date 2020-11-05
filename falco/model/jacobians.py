# pylint: disable=E501
"""Functions to compute the Jacobian for EFC."""
import numpy as np
from numpy.fft import fftshift, fft2

import falco
from falco.util import pad_crop
import falco.prop as fp


def lyot(mp, im, idm):
    """
    Differential model used to compute the ctrl Jacobian for Lyot coronagraph.

    Specialized compact model used to compute the DM response matrix, aka the
    control Jacobian for a Lyot coronagraph. Can include an apodizer, making it
    an apodized pupil Lyot coronagraph (APLC). Does not include unknown
    aberrations of the full, "truth" model. This model propagates the
    first-order Taylor expansion of the phase from the poke of each actuator
    of the deformable mirror.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters

    Returns
    -------
    Gzdl : numpy ndarray
        Complex-valued, 2-D array containing the Jacobian for the
        specified Zernike mode, DM number, and wavelength.

    """
    modvar = falco.config.Object()  # Initialize the new structure
    modvar.sbpIndex = mp.jac.sbp_inds[im]
    modvar.zernIndex = mp.jac.zern_inds[im]

    wvl = mp.sbp_centers[modvar.sbpIndex]
    mirrorFac = 2.  # Phase change is twice the DM surface height.
    NdmPad = int(mp.compact.NdmPad)
    
    if mp.coro.upper() in ('LC', 'APLC', 'FLC', 'SPLC'):
        fpm = mp.F3.compact.ampMask
        transOuterFPM = 1.  # transmission of points outside the FPM.
    elif mp.coro.upper() in ('HLC',):
        fpm = np.squeeze(mp.compact.fpmCube[:, :, modvar.sbpIndex])  # complex
        # Complex transmission of the points outside the FPM (just fused silica
        # with optional dielectric and no metal).
        transOuterFPM = fpm[0, 0]

    """Input E-fields"""
    Ein = np.squeeze(mp.P1.compact.E[:, :, modvar.sbpIndex])
    # Apply a Zernike (in amplitude) at input pupil
    # Used only for Zernike sensitivity control, which requires the perfect
    # E-field of the differential Zernike term.
    if not (modvar.zernIndex == 1):
        indsZnoll = modvar.zernIndex  # Just send in 1 Zernike mode
        zernMat = np.squeeze(falco.zern.gen_norm_zern_maps(mp.P1.compact.Nbeam,
                                                    mp.centering, indsZnoll))
        zernMat = pad_crop(zernMat, mp.P1.compact.Narr)
        Ein = Ein*zernMat*(2*np.pi/wvl)*mp.jac.Zcoef[mp.jac.zerns ==
                                                     modvar.zernIndex]

    """ Masks and DM surfaces """
    pupil = pad_crop(mp.P1.compact.mask, NdmPad)
    Ein = pad_crop(Ein, NdmPad)

    # Re-image the apodizer from pupil P3 back to pupil P2.
    if(mp.flagApod):
        apodReimaged = pad_crop(mp.P3.compact.mask, NdmPad)
        apodReimaged = fp.relay(apodReimaged, mp.Nrelay2to3, mp.centering)
    else:
        apodReimaged = np.ones((NdmPad, NdmPad))

    # Compute the DM surfaces for the current DM commands
    if any(mp.dm_ind == 1):
        DM1surf = pad_crop(mp.dm1.compact.surfM, NdmPad)
        # DM1surf = falco.dm.gen_surf_from_act(mp.dm1, mp.dm1.compact.dx, NdmPad)
    else:
        DM1surf = np.zeros((NdmPad, NdmPad))
    if any(mp.dm_ind == 2):
        DM2surf = pad_crop(mp.dm2.compact.surfM, NdmPad)
        # DM2surf = falco.dm.gen_surf_from_act(mp.dm2, mp.dm2.compact.dx, NdmPad)
    else:
        DM2surf = np.zeros((NdmPad, NdmPad))

    if mp.flagDM1stop:
        DM1stop = pad_crop(mp.dm1.compact.mask, NdmPad) 
    else:
        DM1stop = np.ones((NdmPad, NdmPad))
    if(mp.flagDM2stop):
        DM2stop = pad_crop(mp.dm2.compact.mask, NdmPad) 
    else:
        DM2stop = np.ones((NdmPad, NdmPad))

    # This block is for BMC surface error testing
    if mp.flagDMwfe:  # if(mp.flagDMwfe && (mp.P1.full.Nbeam==mp.P1.compact.Nbeam))
        if any(mp.dm_ind == 1):
            Edm1WFE = np.exp(2*np.pi*1j/wvl*pad_crop(mp.dm1.compact.wfe, NdmPad, 'extrapval', 0))
        else:
            Edm1WFE = np.ones((NdmPad, NdmPad))
        if any(mp.dm_ind == 2):
            Edm2WFE = np.exp(2*np.pi*1j/wvl*pad_crop(mp.dm2.compact.wfe, NdmPad, 'extrapval', 0))
        else:
            Edm2WFE = np.ones((NdmPad, NdmPad))
    else:
        Edm1WFE = np.ones((NdmPad, NdmPad))
        Edm2WFE = np.ones((NdmPad, NdmPad))

    """Propagation"""
    # Define pupil P1 and Propagate to pupil P2
    EP1 = pupil*Ein  # E-field at pupil plane P1
    EP2 = fp.relay(EP1, mp.Nrelay1to2, mp.centering)

    # Propagate from P2 to DM1, and apply DM1 surface and aperture stop
    if not abs(mp.d_P2_dm1) == 0:
        Edm1 = fp.ptp(EP2, mp.P2.compact.dx*NdmPad, wvl, mp.d_P2_dm1)
    else:
        Edm1 = EP2
    Edm1out = Edm1*Edm1WFE*DM1stop*np.exp(mirrorFac*2*np.pi*1j*DM1surf/wvl)

    """ ---------- DM1 ---------- """
    if idm == 1:
        Gzdl = np.zeros((mp.Fend.corr.Npix, mp.dm1.Nele), dtype=np.complex)
        
        # Two array sizes (at same resolution) of influence functions for MFT
        # and angular spectrum
        NboxPad1AS = int(mp.dm1.compact.NboxAS)  # array size for FFT-AS propagations from DM1->DM2->DM1
        # Adjust the sub-array location of the influence function for the added zero padding
        mp.dm1.compact.xy_box_lowerLeft_AS = mp.dm1.compact.xy_box_lowerLeft -\
            (mp.dm1.compact.NboxAS-mp.dm1.compact.Nbox)/2.
    
        if any(mp.dm_ind == 2):
            DM2surf = pad_crop(DM2surf, mp.dm1.compact.NdmPad)
        else:
            DM2surf = np.zeros((mp.dm1.compact.NdmPad, mp.dm1.compact.NdmPad))
        if(mp.flagDM2stop):
            DM2stop = pad_crop(DM2stop, mp.dm1.compact.NdmPad)
        else:
            DM2stop = np.ones((mp.dm1.compact.NdmPad, mp.dm1.compact.NdmPad))
        apodReimaged = pad_crop(apodReimaged, mp.dm1.compact.NdmPad)
    
        Edm1pad = pad_crop(Edm1out, mp.dm1.compact.NdmPad)  # Pad or crop for expected sub-array indexing
        Edm2WFEpad = pad_crop(Edm2WFE, mp.dm1.compact.NdmPad)  # Pad or crop for expected sub-array indexing
    
        # Propagate each actuator from DM1 through the optical system
        Gindex = 0  # initialize index counter
        for iact in mp.dm1.act_ele:
            # Compute only for influence functions that are not zeroed out
            if np.sum(np.abs(mp.dm1.compact.inf_datacube[:, :, iact])) > 1e-12:
                
                # x- and y- coordinate indices of the padded influence function in the full padded pupil
                x_box_AS_ind = np.arange(mp.dm1.compact.xy_box_lowerLeft_AS[0,iact], mp.dm1.compact.xy_box_lowerLeft_AS[0, iact]+NboxPad1AS, dtype=np.int) # x-indices in pupil arrays for the box
                y_box_AS_ind = np.arange(mp.dm1.compact.xy_box_lowerLeft_AS[1,iact], mp.dm1.compact.xy_box_lowerLeft_AS[1, iact]+NboxPad1AS, dtype=np.int) # y-indices in pupil arrays for the box
                indBoxAS = np.ix_(y_box_AS_ind, x_box_AS_ind)
                # x- and y- coordinates of the UN-padded influence function in the full padded pupil
                x_box = mp.dm1.compact.x_pupPad[x_box_AS_ind] # full pupil x-coordinates of the box 
                y_box = mp.dm1.compact.y_pupPad[y_box_AS_ind] # full pupil y-coordinates of the box
                
                # Propagate from DM1 to DM2, and then back to P2
                dEbox = (mirrorFac*2*np.pi*1j/wvl)*pad_crop((mp.dm1.VtoH.reshape(mp.dm1.Nact**2)[iact])*np.squeeze(mp.dm1.compact.inf_datacube[:,:,iact]),NboxPad1AS) # Pad influence function at DM1 for angular spectrum propagation.
                dEbox = fp.ptp(dEbox*Edm1pad[np.ix_(y_box_AS_ind,x_box_AS_ind)], mp.P2.compact.dx*NboxPad1AS,wvl, mp.d_dm1_dm2) # forward propagate to DM2 and apply DM2 E-field
                dEP2box = fp.ptp(dEbox*Edm2WFEpad[np.ix_(y_box_AS_ind,x_box_AS_ind)]*DM2stop[np.ix_(y_box_AS_ind,x_box_AS_ind)]*np.exp(mirrorFac*2*np.pi*1j/wvl*DM2surf[np.ix_(y_box_AS_ind,x_box_AS_ind)]), mp.P2.compact.dx*NboxPad1AS,wvl,-1*(mp.d_dm1_dm2 + mp.d_P2_dm1) ) # back-propagate to DM1
#                dEbox = fp.ptp_inf_func(dEbox*Edm1pad[np.ix_(y_box_AS_ind,x_box_AS_ind)], mp.P2.compact.dx*NboxPad1AS,wvl, mp.d_dm1_dm2, mp.dm1.dm_spacing, mp.propMethodPTP) # forward propagate to DM2 and apply DM2 E-field
#                dEP2box = fp.ptp_inf_func(dEbox.*Edm2WFEpad[np.ix_(y_box_AS_ind,x_box_AS_ind)]*DM2stop(y_box_AS_ind,x_box_AS_ind).*exp(mirrorFac*2*np.pi*1j/wvl*DM2surf(y_box_AS_ind,x_box_AS_ind)), mp.P2.compact.dx*NboxPad1AS,wvl,-1*(mp.d_dm1_dm2 + mp.d_P2_dm1), mp.dm1.dm_spacing, mp.propMethodPTP ) # back-propagate to DM1
#
                # To simulate going forward to the next pupil plane (with the apodizer) most efficiently, 
                # First, back-propagate the apodizer (by rotating 180-degrees) to the previous pupil.
                # Second, negate the coordinates of the box used.
                dEP2box = apodReimaged[indBoxAS]*dEP2box # Apply 180deg-rotated SP mask.
                dEP3box = np.rot90(dEP2box, k=2*mp.Nrelay2to3) # Forward propagate the cropped box by rotating 180 degrees mp.Nrelay2to3 times.
                # Negate and reverse coordinate values to effectively rotate by 180 degrees. No change if 360 degree rotation.
                if np.mod(mp.Nrelay2to3, 2) == 1:
                    x_box = -1*x_box[::-1]
                    y_box = -1*y_box[::-1]
               
                # Matrices for the MFT from the pupil P3 to the focal plane mask
                rect_mat_pre = (np.exp(-2*np.pi*1j*np.outer(mp.F3.compact.etas,y_box)/(wvl*mp.fl)))*np.sqrt(mp.P2.compact.dx*mp.P2.compact.dx)*np.sqrt(mp.F3.compact.dxi*mp.F3.compact.deta)/(wvl*mp.fl)
                rect_mat_post = (np.exp(-2*np.pi*1j*np.outer(x_box, mp.F3.compact.xis)/(wvl*mp.fl)))
                
                EF3inc = rect_mat_pre @ dEP3box @ rect_mat_post  # MFT to FPM
                
                if mp.coro.upper() in ('LC', 'APLC', 'HLC'):
                    # Propagate through (1 - FPM) for Babinet's principle
                    EF3 = (transOuterFPM-fpm) * EF3inc
        
                    # MFT to LS ("Sub" name for Subtrahend part of the Lyot-plane E-field)
                    EP4sub = fp.mft_f2p(EF3, mp.fl, wvl, mp.F3.compact.dxi, mp.F3.compact.deta, mp.P4.compact.dx, mp.P4.compact.Narr, mp.centering)
                    EP4sub = fp.relay(EP4sub, mp.Nrelay3to4-1, mp.centering)
                    
                    # Full Lyot plane pupil (for Babinet)
                    EP4noFPM = np.zeros((mp.dm1.compact.NdmPad, mp.dm1.compact.NdmPad),dtype=np.complex)
                    EP4noFPM[indBoxAS] = dEP2box # Propagating the E-field from P2 to P4 without masks gives the same E-field. 
                    EP4noFPM = fp.relay(EP4noFPM, mp.Nrelay2to3+mp.Nrelay3to4, mp.centering) # Get the correct orientation 
                    EP4noFPM = pad_crop(EP4noFPM, mp.P4.compact.Narr)  # Crop down to the size of the Lyot stop opening
                    EP4 = transOuterFPM*EP4noFPM - EP4sub  # Babinet's principle to get E-field at Lyot plane
                
                elif mp.coro.upper() in ('FLC', 'SPLC'):
                    EF3 = fpm * EF3inc  # Apply FPM
                    
                    # MFT to Lyot plane
                    EP4 = fp.mft_f2p(EF3, mp.fl,wvl, mp.F3.compact.dxi, mp.F3.compact.deta, mp.P4.compact.dx, mp.P4.compact.Narr, mp.centering)
                    EP4 = fp.relay(EP4, mp.Nrelay3to4-1, mp.centering)  # Get the correct orientation
                    
                EP4 *= mp.P4.compact.croppedMask  # Apply Lyot stop
    
                # MFT to camera
                EP4 = fp.relay(EP4, mp.NrelayFend, mp.centering) # Rotate the final image 180 degrees if necessary
                EFend = fp.mft_p2f(EP4, mp.fl,wvl, mp.P4.compact.dx, mp.Fend.dxi, mp.Fend.Nxi, mp.Fend.deta, mp.Fend.Neta, mp.centering)
                
                Gzdl[:, Gindex] = EFend[mp.Fend.corr.maskBool]/np.sqrt(mp.Fend.compact.I00[modvar.sbpIndex])

            Gindex += 1

    """ ---------- DM2 ---------- """
    if idm == 2:
        Gzdl = np.zeros((mp.Fend.corr.Npix, mp.dm2.Nele), dtype=np.complex)
        
        # Two array sizes (at same resolution) of influence functions for MFT and angular spectrum
        NboxPad2AS = int(mp.dm2.compact.NboxAS)
        mp.dm2.compact.xy_box_lowerLeft_AS = mp.dm2.compact.xy_box_lowerLeft - (NboxPad2AS-mp.dm2.compact.Nbox)/2 # Account for the padding of the influence function boxes
        
        apodReimaged = pad_crop(apodReimaged, mp.dm2.compact.NdmPad)
        DM2stopPad = pad_crop(DM2stop, mp.dm2.compact.NdmPad)
        Edm2WFEpad = pad_crop(Edm2WFE, mp.dm2.compact.NdmPad)
    
        # Propagate full field to DM2 before back-propagating in small boxes
        Edm2inc = pad_crop(fp.ptp(Edm1out, mp.compact.NdmPad*mp.P2.compact.dx, wvl, mp.d_dm1_dm2), mp.dm2.compact.NdmPad)  # E-field incident upon DM2
        Edm2inc = pad_crop(Edm2inc, mp.dm2.compact.NdmPad)
        Edm2 = DM2stopPad*Edm2WFEpad*Edm2inc*np.exp(mirrorFac*2*np.pi*1j/wvl*pad_crop(DM2surf, mp.dm2.compact.NdmPad))  # Initial E-field at DM2 including its own phase contribution
        
        # Propagate each actuator from DM2 through the rest of the optical system
        Gindex = 0  # initialize index counter
        for iact in mp.dm2.act_ele:
            if np.sum(np.abs(mp.dm2.compact.inf_datacube[:, :, iact])) > 1e-12:  # Only compute for acutators specified for use or for influence functions that are not zeroed out
    
                # x- and y- coordinates of the padded influence function in the full padded pupil
                x_box_AS_ind = np.arange(mp.dm2.compact.xy_box_lowerLeft_AS[0, iact], mp.dm2.compact.xy_box_lowerLeft_AS[0, iact]+NboxPad2AS, dtype=np.int) # x-indices in pupil arrays for the box
                y_box_AS_ind = np.arange(mp.dm2.compact.xy_box_lowerLeft_AS[1, iact], mp.dm2.compact.xy_box_lowerLeft_AS[1, iact]+NboxPad2AS, dtype=np.int) # y-indices in pupil arrays for the box
                indBoxAS = np.ix_(y_box_AS_ind, x_box_AS_ind)
                # x- and y- coordinates of the UN-padded influence function in the full padded pupil
                x_box = mp.dm2.compact.x_pupPad[x_box_AS_ind]  # full pupil x-coordinates of the box 
                y_box = mp.dm2.compact.y_pupPad[y_box_AS_ind]  # full pupil y-coordinates of the box 
                
                dEbox = (mp.dm2.VtoH.reshape(mp.dm2.Nact**2)[iact])*(mirrorFac*2*np.pi*1j/wvl)*pad_crop(np.squeeze(mp.dm2.compact.inf_datacube[:, :, iact]), NboxPad2AS) # the padded influence function at DM2
                dEP2box = fp.ptp(dEbox*Edm2[indBoxAS], mp.P2.compact.dx*NboxPad2AS, wvl, -1*(mp.d_dm1_dm2 + mp.d_P2_dm1)) # back-propagate to pupil P2
#                dEP2box = ptp_inf_func(dEbox.*Edm2(y_box_AS_ind,x_box_AS_ind), mp.P2.compact.dx*NboxPad2AS,wvl,-1*(mp.d_dm1_dm2 + mp.d_P2_dm1), mp.dm2.dm_spacing, mp.propMethodPTP); # back-propagate to pupil P2
                
                # To simulate going forward to the next pupil plane (with the apodizer) most efficiently, 
                # First, back-propagate the apodizer (by rotating 180-degrees) to the previous pupil.
                # Second, negate the coordinates of the box used.
                dEP2box = apodReimaged[indBoxAS]*dEP2box  # Apply 180deg-rotated SP mask.
                dEP3box = np.rot90(dEP2box, k=2*mp.Nrelay2to3)  # Forward propagate the cropped box by rotating 180 degrees mp.Nrelay2to3 times.
                # Negate and rotate coordinates to effectively rotate by 180 degrees. No change if 360 degree rotation.
                if np.mod(mp.Nrelay2to3, 2) == 1:
                    x_box = -1*x_box[::-1]
                    y_box = -1*y_box[::-1]
       
                # Matrices for the MFT from the pupil P3 to the focal plane mask
                rect_mat_pre = np.exp(-2*np.pi*1j*np.outer(mp.F3.compact.etas, y_box)/(wvl*mp.fl))*np.sqrt(mp.P2.compact.dx*mp.P2.compact.dx)*np.sqrt(mp.F3.compact.dxi*mp.F3.compact.deta)/(wvl*mp.fl)
                rect_mat_post = np.exp(-2*np.pi*1j*np.outer(x_box, mp.F3.compact.xis)/(wvl*mp.fl))
    
                EF3inc = rect_mat_pre @ dEP3box @ rect_mat_post  # MFT to FPM
                
                if mp.coro.upper() in ('LC', 'APLC', 'HLC'):
                    # Propagate through (1 - fpm) for Babinet's principle
                    EF3 = (transOuterFPM-fpm) * EF3inc
        
                    # MFT to LS ("Sub" name for Subtrahend part of the Lyot-plane E-field)
                    EP4sub = fp.mft_f2p(EF3, mp.fl, wvl, mp.F3.compact.dxi, mp.F3.compact.deta, mp.P4.compact.dx, mp.P4.compact.Narr, mp.centering) # Subtrahend term for the Lyot plane E-field    
                    EP4sub = fp.relay(EP4sub, mp.Nrelay3to4-1, mp.centering) # Get the correct orientation
                                    
                    EP4noFPM = np.zeros((mp.dm2.compact.NdmPad, mp.dm2.compact.NdmPad), dtype=np.complex)
                    EP4noFPM[indBoxAS] = dEP2box  # Propagating the E-field from P2 to P4 without masks gives the same E-field.
                    EP4noFPM = fp.relay(EP4noFPM, mp.Nrelay2to3+mp.Nrelay3to4, mp.centering) # Get the number or re-imaging relays between pupils P3 and P4. 
                    EP4noFPM = pad_crop(EP4noFPM, mp.P4.compact.Narr)  # Crop down to the size of the Lyot stop opening
                    EP4 = transOuterFPM*EP4noFPM - EP4sub  # Babinet's principle to get E-field at Lyot plane
                
                elif mp.coro.upper() in ('FLC', 'SPLC'):

                    EF3 = fpm * EF3inc  # Apply FPM
        
                    # MFT to LS ("Sub" name for Subtrahend part of the Lyot-plane E-field)
                    EP4 = fp.mft_f2p(EF3, mp.fl, wvl, mp.F3.compact.dxi, mp.F3.compact.deta, mp.P4.compact.dx, mp.P4.compact.Narr, mp.centering)
                    EP4 = fp.relay(EP4, mp.Nrelay3to4-1, mp.centering)
                
                EP4 *= mp.P4.compact.croppedMask  # Apply Lyot stop
    
                # MFT to detector
                EP4 = fp.relay(EP4, mp.NrelayFend, mp.centering)  # Rotate the final image 180 degrees if necessary
                EFend = fp.mft_p2f(EP4, mp.fl, wvl, mp.P4.compact.dx, mp.Fend.dxi, mp.Fend.Nxi, mp.Fend.deta, mp.Fend.Neta, mp.centering)
                
                Gzdl[:, Gindex] = EFend[mp.Fend.corr.maskBool]/np.sqrt(mp.Fend.compact.I00[modvar.sbpIndex])

            Gindex += 1

    """ ---------- DM9 (HLC only) ---------- """
    if idm == 9:
        Gzdl = np.zeros((mp.Fend.corr.Npix, mp.dm9.Nele), dtype=complex)
        Nbox9 = int(mp.dm9.compact.Nbox)

        # Adjust the step size in the Jacobian, then divide back out. Used for
        # helping counteract effect of discretization.
        if not hasattr(mp.dm9, 'stepFac'):
            stepFac = 20
        else:
            stepFac = mp.dm9.stepFac
        
        # Propagate from DM1 to DM2, and apply DM2 surface and aperture stop
        Edm2 = Edm2WFE * DM2stop * np.exp(mirrorFac*2*np.pi*1j*DM2surf/wvl) * \
            fp.ptp(Edm1out, mp.P2.compact.dx*NdmPad, wvl, mp.d_dm1_dm2)
        
        # Back-propagate to pupil P2
        dz2 = mp.d_P2_dm1 + mp.d_dm1_dm2
        if dz2 < 10*wvl:
            EP2eff = Edm2
        else:
            EP2eff = fp.ptp(Edm2, mp.P2.compact.dx*NdmPad, wvl, -dz2)
        
        # Rotate 180 degrees mp.Nrelay2to3 times to go from pupil P2 to P3
        EP3 = fp.relay(EP2eff, mp.Nrelay2to3, mp.centering)
    
        # Apply apodizer mask
        if mp.flagApod:
            EP3 = mp.P3.compact.mask * pad_crop(EP3, mp.P1.compact.Narr)
        
        # MFT from pupil P3 to FPM (at focus F3)
        EF3inc = fp.mft_p2f(EP3, mp.fl, wvl, mp.P2.compact.dx, mp.F3.compact.dxi, mp.F3.compact.Nxi, mp.F3.compact.deta, mp.F3.compact.Neta, mp.centering)
        EF3inc = pad_crop(EF3inc, mp.dm9.compact.NdmPad)
        # Coordinates for metal thickness and dielectric thickness
        DM8transIndAll = falco.hlc.discretize_fpm_surf(mp.dm8.surf, mp.t_metal_nm_vec, mp.dt_metal_nm)  # All of the mask
    
        # Propagate each actuator from DM2 through the rest of the optical system
        Gindex = 0  # initialize index counter
        for iact in mp.dm9.act_ele:
            if np.sum(np.abs(mp.dm9.compact.inf_datacube[:, :, iact])) > 1e-12:  # Only compute for acutators specified for use or for influence functions that are not zeroed out
    
                # xi- and eta- coordinates in the full FPM portion of the focal plane
                xyLL = mp.dm9.compact.xy_box_lowerLeft[:, iact]
                xi_box_ind = np.arange(xyLL[0], xyLL[0]+Nbox9, dtype=int)  # xi-indices in focal arrays for the box
                eta_box_ind = np.arange(xyLL[1], xyLL[1]+Nbox9, dtype=int)  # eta-indices in focal arrays for the box
                indBox = np.ix_(eta_box_ind, xi_box_ind)
                xi_box = mp.dm9.compact.x_pupPad[xi_box_ind]
                eta_box = mp.dm9.compact.y_pupPad[eta_box_ind]
                
                # Obtain values for the "poked" FPM's complex transmission (only in the sub-array where poked)
                Nxi = Nbox9
                Neta = Nbox9
                DM9surfCropNew = stepFac*mp.dm9.VtoH[iact]*mp.dm9.compact.inf_datacube[:, :, iact] + mp.dm9.surf[indBox]  # New DM9 surface profile in the poked region (meters)
                DM9transInd = falco.hlc.discretize_fpm_surf(DM9surfCropNew, mp.t_diel_nm_vec,  mp.dt_diel_nm)
                DM8transInd = DM8transIndAll[indBox]  # Cropped region of the FPM.
    
                # Look up table to compute complex transmission coefficient of the FPM at each pixel
                fpmPoked = np.zeros((Neta, Nxi), dtype=complex)  # Initialize output array of FPM's complex transmission
                for ix in range(Nxi):
                    for iy in range(Neta):
                        ind_metal = DM8transInd[iy, ix]
                        ind_diel = DM9transInd[iy, ix]
                        fpmPoked[iy, ix] = mp.complexTransCompact[ind_diel, ind_metal, modvar.sbpIndex]

    
                dEF3box = ((transOuterFPM-fpmPoked) - (transOuterFPM-fpm[indBox])) * EF3inc[indBox]  # Delta field (in a small region) at the FPM
    
                # Matrices for the MFT from the FPM stamp to the Lyot stop
                rect_mat_pre = np.exp(-2*np.pi*1j*np.outer(mp.P4.compact.ys, eta_box)/(wvl*mp.fl)) *\
                    np.sqrt(mp.P4.compact.dx*mp.P4.compact.dx)*np.sqrt(mp.F3.compact.dxi*mp.F3.compact.deta)/(wvl*mp.fl)
                rect_mat_post = np.exp(-2*np.pi*1j*np.outer(xi_box, mp.P4.compact.xs)/(wvl*mp.fl))
    
                # MFT from FPM to Lyot stop (Nominal term transOuterFPM*EP4noFPM subtracts out to 0 since it ignores the FPM change).
                EP4 = 0 - rect_mat_pre @ dEF3box @ rect_mat_post  # MFT from FPM (F3) to Lyot stop plane (P4)
                EP4 = fp.relay(EP4, mp.Nrelay3to4-1, mp.centering)
                EP4 = mp.P4.compact.croppedMask * EP4  # Apply Lyot stop
    
                # MFT to final focal plane
                EP4 = fp.relay(EP4, mp.NrelayFend, mp.centering)
                EFend = fp.mft_p2f(EP4, mp.fl, wvl, mp.P4.compact.dx, mp.Fend.dxi, mp.Fend.Nxi, mp.Fend.deta, mp.Fend.Neta, mp.centering)
    
                Gzdl[:, Gindex] = mp.dm9.act_sens / stepFac * mp.dm9.weight*EFend[mp.Fend.corr.maskBool] / np.sqrt(mp.Fend.compact.I00[modvar.sbpIndex])

            Gindex += 1
        
    return Gzdl

    
def vortex(mp, im, idm):
    """
    Differential model used to compute ctrl Jacobian for vortex coronagraph.

    Specialized compact model used to compute the DM response matrix, aka the
    control Jacobian for a vortex coronagraph. Can include an apodizer, making
    it an apodized vortex coronagraph (AVC). Does not include unknown
    aberrations of the full, "truth" model. This model propagates the
    first-order Taylor expansion of the phase from the poke of each actuator
    of the deformable mirror.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters

    Returns
    -------
    Gzdl : numpy ndarray
        Complex-valued, 2-D array containing the Jacobian for the
        specified Zernike mode, DM number, and wavelength.

    """
    modvar = falco.config.Object()  # Initialize the new structure
    modvar.sbpIndex = mp.jac.sbp_inds[im]
    modvar.zernIndex = mp.jac.zern_inds[im]
    
    wvl = mp.sbp_centers[modvar.sbpIndex]
    mirrorFac = 2.  # Phase change is twice the DM surface height.
    NdmPad = int(mp.compact.NdmPad)

    # Minimum FPM resolution for Jacobian calculations (in pixels per lambda/D)
    minPadFacVortex = 8

    # Get FPM charge
    if type(mp.F3.VortexCharge) == np.ndarray:
        # Passing an array for mp.F3.VortexCharge with
        # corresponding wavelengths mp.F3.VortexCharge_lambdas
        # represents a chromatic vortex FPM
        if mp.F3.VortexCharge.size == 1:
            charge = mp.F3.VortexCharge
        else:
            np.interp(wvl, mp.F3.VortexCharge_lambdas, mp.F3.VortexCharge,
                      'linear', 'extrap')
    elif type(mp.F3.VortexCharge) == int or type(mp.F3.VortexCharge) == float:
        # single value indicates fully achromatic mask
        charge = mp.F3.VortexCharge
    else:
        raise TypeError("mp.F3.VortexCharge must be an int, float, or numpy ndarray.")

    """Input E-fields"""
    Ein = np.squeeze(mp.P1.compact.E[:, :, modvar.sbpIndex])
    
    # Apply a Zernike (in amplitude) at input pupil
    # Used only for Zernike sensitivity control, which requires the perfect
    # E-field of the differential Zernike term.
    if not modvar.zernIndex == 1:
        indsZnoll = modvar.zernIndex  # Just send in 1 Zernike mode
        zernMat = np.squeeze(falco.zern.gen_norm_zern_maps(mp.P1.compact.Nbeam,
                                                    mp.centering, indsZnoll))
        zernMat = pad_crop(zernMat, mp.P1.compact.Narr)
        Ein = Ein*zernMat*(2*np.pi/wvl) * \
            mp.jac.Zcoef[mp.jac.zerns == modvar.zernIndex]

    """ Masks and DM surfaces """
    pupil = pad_crop(mp.P1.compact.mask, NdmPad)
    Ein = pad_crop(Ein, NdmPad)

    # Re-image the apodizer from pupil P3 back to pupil P2.
    if(mp.flagApod):
        apodReimaged = pad_crop(mp.P3.compact.mask, NdmPad)
        apodReimaged = fp.relay(apodReimaged, mp.Nrelay2to3, mp.centering)
    else:
        apodReimaged = np.ones((NdmPad, NdmPad))

    # Compute the DM surfaces for the current DM commands
    if any(mp.dm_ind == 1):
        DM1surf = pad_crop(mp.dm1.compact.surfM, NdmPad)
        # DM1surf = falco.dm.gen_surf_from_act(mp.dm1, mp.dm1.compact.dx, NdmPad)
    else:
        DM1surf = np.zeros((NdmPad, NdmPad))
    if any(mp.dm_ind == 2):
        DM2surf = pad_crop(mp.dm2.compact.surfM, NdmPad)
        # DM2surf = falco.dm.gen_surf_from_act(mp.dm2, mp.dm2.compact.dx, NdmPad)
    else:
        DM2surf = np.zeros((NdmPad, NdmPad))

    if(mp.flagDM1stop):
        DM1stop = pad_crop(mp.dm1.compact.mask, NdmPad)
    else:
        DM1stop = np.ones((NdmPad, NdmPad))
    if(mp.flagDM2stop):
        DM2stop = pad_crop(mp.dm2.compact.mask, NdmPad)
    else:
        DM2stop = np.ones((NdmPad, NdmPad))

    # This block is for BMC surface error testing
    if(mp.flagDMwfe):
        if any(mp.dm_ind == 1):
            Edm1WFE = np.exp(2*np.pi*1j/wvl*pad_crop(mp.dm1.compact.wfe,
                                                     NdmPad, 'extrapval', 0))
        else:
            Edm1WFE = np.ones((NdmPad, NdmPad))
        if any(mp.dm_ind == 2):
            Edm2WFE = np.exp(2*np.pi*1j/wvl*pad_crop(mp.dm2.compact.wfe,
                                                     NdmPad, 'extrapval', 0))
        else:
            Edm2WFE = np.ones((NdmPad, NdmPad))
    else:
        Edm1WFE = np.ones((NdmPad, NdmPad))
        Edm2WFE = np.ones((NdmPad, NdmPad))
        
    """Propagation"""
    # Define pupil P1 and Propagate to pupil P2
    EP1 = pupil*Ein  # E-field at pupil plane P1
    EP2 = fp.relay(EP1, mp.Nrelay1to2, mp.centering)

    # Propagate from P2 to DM1, and apply DM1 surface and aperture stop
    if not (abs(mp.d_P2_dm1) == 0):  # E-field arriving at DM1
        Edm1 = fp.ptp(EP2, mp.P2.compact.dx*NdmPad, wvl, mp.d_P2_dm1)
    else:
        Edm1 = EP2
    Edm1out = Edm1*Edm1WFE*DM1stop*np.exp(mirrorFac*2*np.pi*1j*DM1surf/wvl)

    """ ---------- DM1 ---------- """
    if idm == 1:
        Gzdl = np.zeros((mp.Fend.corr.Npix, mp.dm1.Nele), dtype=np.complex)
        
        # Array size for planes P3, F3, and P4
        Nfft1 = int(2**falco.util.nextpow2(np.max(np.array([mp.dm1.compact.NdmPad, minPadFacVortex*mp.dm1.compact.Nbox])))) # Don't crop--but do pad if necessary.
        
        # Generate vortex FPM with fftshift already applied
        fftshiftVortex = fftshift(falco.mask.falco_gen_vortex_mask(charge, Nfft1))
    
        # Two array sizes (at same resolution) of influence functions for MFT and angular spectrum
        NboxPad1AS = int(mp.dm1.compact.NboxAS)  # array size for FFT-AS propagations from DM1->DM2->DM1
        mp.dm1.compact.xy_box_lowerLeft_AS = mp.dm1.compact.xy_box_lowerLeft - (mp.dm1.compact.NboxAS-mp.dm1.compact.Nbox)/2. # Adjust the sub-array location of the influence function for the added zero padding
    
        if any(mp.dm_ind == 2):
            DM2surf = pad_crop(DM2surf, mp.dm1.compact.NdmPad)
        else:
            DM2surf = np.zeros((mp.dm1.compact.NdmPad, mp.dm1.compact.NdmPad))
        if(mp.flagDM2stop):
            DM2stop = pad_crop(DM2stop, mp.dm1.compact.NdmPad)
        else:
            DM2stop = np.ones((mp.dm1.compact.NdmPad, mp.dm1.compact.NdmPad))
        apodReimaged = pad_crop(apodReimaged, mp.dm1.compact.NdmPad)
    
        Edm1pad = pad_crop(Edm1out, mp.dm1.compact.NdmPad)  # Pad or crop for expected sub-array indexing
        Edm2WFEpad = pad_crop(Edm2WFE, mp.dm1.compact.NdmPad)  # Pad or crop for expected sub-array indexing
    
        # Propagate each actuator from DM1 through the optical system
        Gindex = 0  # initialize index counter
        for iact in mp.dm1.act_ele:
            # Compute only for influence functions that are not zeroed out
            if np.sum(np.abs(mp.dm1.compact.inf_datacube[:, :, iact])) > 1e-12:
                
                # x- and y- coordinate indices of the padded influence function in the full padded pupil
                x_box_AS_ind = np.arange(mp.dm1.compact.xy_box_lowerLeft_AS[0, iact], mp.dm1.compact.xy_box_lowerLeft_AS[0, iact]+NboxPad1AS, dtype=np.int)  # x-indices in pupil arrays for the box
                y_box_AS_ind = np.arange(mp.dm1.compact.xy_box_lowerLeft_AS[1, iact], mp.dm1.compact.xy_box_lowerLeft_AS[1 ,iact]+NboxPad1AS, dtype=np.int)  # y-indices in pupil arrays for the box
                indBoxAS = np.ix_(y_box_AS_ind, x_box_AS_ind)
                # x- and y- coordinates of the UN-padded influence function in the full padded pupil
                x_box = mp.dm1.compact.x_pupPad[x_box_AS_ind]  # full pupil x-coordinates of the box 
                y_box = mp.dm1.compact.y_pupPad[y_box_AS_ind]  # full pupil y-coordinates of the box
                
                # Propagate from DM1 to DM2, and then back to P2
                dEbox = (mirrorFac*2*np.pi*1j/wvl)*pad_crop((mp.dm1.VtoH.reshape(mp.dm1.Nact**2)[iact])*np.squeeze(mp.dm1.compact.inf_datacube[:, :, iact]), NboxPad1AS) # Pad influence function at DM1 for angular spectrum propagation.
                dEbox = fp.ptp(dEbox*Edm1pad[indBoxAS], mp.P2.compact.dx*NboxPad1AS,wvl, mp.d_dm1_dm2) # forward propagate to DM2 and apply DM2 E-field
                dEP2box = fp.ptp(dEbox*Edm2WFEpad[indBoxAS]*DM2stop[indBoxAS]*np.exp(mirrorFac*2*np.pi*1j/wvl*DM2surf[indBoxAS]), mp.P2.compact.dx*NboxPad1AS,wvl,-1*(mp.d_dm1_dm2 + mp.d_P2_dm1)) # back-propagate to DM1
#                dEbox = fp.ptp_inf_func(dEbox*Edm1pad[np.ix_(y_box_AS_ind,x_box_AS_ind)], mp.P2.compact.dx*NboxPad1AS,wvl, mp.d_dm1_dm2, mp.dm1.dm_spacing, mp.propMethodPTP) # forward propagate to DM2 and apply DM2 E-field
#                dEP2box = fp.ptp_inf_func(dEbox.*Edm2WFEpad[np.ix_(y_box_AS_ind,x_box_AS_ind)]*DM2stop(y_box_AS_ind,x_box_AS_ind).*exp(mirrorFac*2*np.pi*1j/wvl*DM2surf(y_box_AS_ind,x_box_AS_ind)), mp.P2.compact.dx*NboxPad1AS,wvl,-1*(mp.d_dm1_dm2 + mp.d_P2_dm1), mp.dm1.dm_spacing, mp.propMethodPTP ) # back-propagate to DM1
#
                # To simulate going forward to the next pupil plane (with the apodizer) most efficiently, 
                # First, back-propagate the apodizer (by rotating 180-degrees) to the previous pupil.
                # Second, negate the coordinates of the box used.
                dEP2boxEff = apodReimaged[indBoxAS]*dEP2box  # Apply 180deg-rotated apodizer mask.
                # dEP3box = np.rot90(dEP2box,k=2*mp.Nrelay2to3) # Forward propagate the cropped box by rotating 180 degrees mp.Nrelay2to3 times.
                # # Negate and reverse coordinate values to effectively rotate by 180 degrees. No change if 360 degree rotation.

                # Re-insert the window around the influence function back into the full beam array.
                EP2eff = np.zeros((mp.dm1.compact.NdmPad, mp.dm1.compact.NdmPad), dtype=complex)
                EP2eff[indBoxAS] = dEP2boxEff
                
                # Forward propagate from P2 (effective) to P3
                EP3 = fp.relay(EP2eff, mp.Nrelay2to3, mp.centering)
                
                # Pad pupil P3 for FFT
                EP3pad = pad_crop(EP3, Nfft1)
                
                # FFT from P3 to Fend.and apply vortex
                EF3 = fftshiftVortex*fft2(fftshift(EP3pad))/Nfft1
                
                # FFT from Vortex FPM to Lyot Plane
                EP4 = fftshift(fft2(EF3))/Nfft1
                EP4 = fp.relay(EP4, mp.Nrelay3to4-1, mp.centering)  # Add more re-imaging relays if necessary
                if(Nfft1 > mp.P4.compact.Narr):
                    EP4 = mp.P4.compact.croppedMask*pad_crop(EP4, mp.P4.compact.Narr)  # Crop EP4 and then apply Lyot stop 
                else:
                    EP4 = pad_crop(mp.P4.compact.croppedMask, Nfft1)*EP4  # Crop the Lyot stop and then apply it.
                    pass
                
                # MFT to camera
                EP4 = fp.relay(EP4, mp.NrelayFend, mp.centering)  # Rotate the final image 180 degrees if necessary
                EFend = fp.mft_p2f(EP4, mp.fl, wvl, mp.P4.compact.dx, mp.Fend.dxi, mp.Fend.Nxi, mp.Fend.deta, mp.Fend.Neta, mp.centering)
                
                Gzdl[:, Gindex] = EFend[mp.Fend.corr.maskBool]/np.sqrt(mp.Fend.compact.I00[modvar.sbpIndex])

            Gindex += 1

    """ ---------- DM2 ---------- """
    if idm == 2:
        Gzdl = np.zeros((mp.Fend.corr.Npix, mp.dm2.Nele), dtype=np.complex)
        
        # Array size for planes P3, F3, and P4
        Nfft2 = int(2**falco.util.nextpow2(np.max(np.array([mp.dm2.compact.NdmPad, minPadFacVortex*mp.dm2.compact.Nbox])))) # Don't crop--but do pad if necessary.
        
        # Generate vortex FPM with fftshift already applied
        fftshiftVortex = fftshift(falco.mask.falco_gen_vortex_mask(charge, Nfft2))
    
        # Two array sizes (at same resolution) of influence functions for MFT and angular spectrum
        NboxPad2AS = int(mp.dm2.compact.NboxAS)
        mp.dm2.compact.xy_box_lowerLeft_AS = mp.dm2.compact.xy_box_lowerLeft - (NboxPad2AS-mp.dm2.compact.Nbox)/2 # Account for the padding of the influence function boxes
        
        apodReimaged = pad_crop(apodReimaged, mp.dm2.compact.NdmPad)
        DM2stopPad = pad_crop(DM2stop, mp.dm2.compact.NdmPad)
        Edm2WFEpad = pad_crop(Edm2WFE, mp.dm2.compact.NdmPad)
    
        # Propagate full field to DM2 before back-propagating in small boxes
        Edm2inc = pad_crop(fp.ptp(Edm1out, mp.compact.NdmPad*mp.P2.compact.dx,wvl, mp.d_dm1_dm2), mp.dm2.compact.NdmPad) # E-field incident upon DM2
        Edm2inc = pad_crop(Edm2inc, mp.dm2.compact.NdmPad);
        Edm2 = DM2stopPad * Edm2WFEpad * Edm2inc * np.exp(mirrorFac*2*np.pi*1j/wvl * pad_crop(DM2surf, mp.dm2.compact.NdmPad)) # Initial E-field at DM2 including its own phase contribution
        
        # Propagate each actuator from DM2 through the rest of the optical system
        Gindex = 0  # initialize index counter
        for iact in mp.dm2.act_ele:
            # Only compute for acutators specified for use or for influence functions that are not zeroed out
            if np.sum(np.abs(mp.dm2.compact.inf_datacube[:, :, iact])) > 1e-12:
    
                # x- and y- coordinates of the padded influence function in the full padded pupil
                x_box_AS_ind = np.arange(mp.dm2.compact.xy_box_lowerLeft_AS[0, iact], mp.dm2.compact.xy_box_lowerLeft_AS[0, iact]+NboxPad2AS, dtype=np.int)  # x-indices in pupil arrays for the box
                y_box_AS_ind = np.arange(mp.dm2.compact.xy_box_lowerLeft_AS[1, iact], mp.dm2.compact.xy_box_lowerLeft_AS[1, iact]+NboxPad2AS, dtype=np.int)  # y-indices in pupil arrays for the box
                indBoxAS = np.ix_(y_box_AS_ind, x_box_AS_ind)
#               # x- and y- coordinates of the UN-padded influence function in the full padded pupil
#                x_box = mp.dm2.compact.x_pupPad[x_box_AS_ind] # full pupil x-coordinates of the box
#                y_box = mp.dm2.compact.y_pupPad[y_box_AS_ind] # full pupil y-coordinates of the box
                
                dEbox = (mp.dm2.VtoH.reshape(mp.dm2.Nact**2)[iact])*(mirrorFac*2*np.pi*1j/wvl)*pad_crop(np.squeeze(mp.dm2.compact.inf_datacube[:, :, iact]), NboxPad2AS) # the padded influence function at DM2
                dEP2box = fp.ptp(dEbox*Edm2[indBoxAS], mp.P2.compact.dx*NboxPad2AS, wvl, -1*(mp.d_dm1_dm2 + mp.d_P2_dm1)) # back-propagate to pupil P2
#                dEP2box = ptp_inf_func(dEbox.*Edm2(y_box_AS_ind,x_box_AS_ind), mp.P2.compact.dx*NboxPad2AS,wvl,-1*(mp.d_dm1_dm2 + mp.d_P2_dm1), mp.dm2.dm_spacing, mp.propMethodPTP); # back-propagate to pupil P2
                
                # To simulate going forward to the next pupil plane (with the apodizer) most efficiently,
                # First, back-propagate the apodizer (by rotating 180-degrees) to the previous pupil.
                # Second, negate the coordinates of the box used.
                dEP2boxEff = apodReimaged[indBoxAS]*dEP2box
#                dEP3box = np.rot90(dEP2box,k=2*mp.Nrelay2to3) # Forward propagate the cropped box by rotating 180 degrees mp.Nrelay2to3 times.
#                # Negate and rotate coordinates to effectively rotate by 180 degrees. No change if 360 degree rotation.
#                if np.mod(mp.Nrelay2to3,2)==1: 
#                    x_box = -1*x_box[::-1]
#                    y_box = -1*y_box[::-1]
                
                EP2eff = np.zeros((mp.dm2.compact.NdmPad, mp.dm2.compact.NdmPad), dtype=complex)
                EP2eff[indBoxAS] = dEP2boxEff

                # Forward propagate from P2 (effective) to P3
                EP3 = fp.relay(EP2eff, mp.Nrelay2to3, mp.centering)
    
                # Pad pupil P3 for FFT
                EP3pad = pad_crop(EP3, Nfft2)
                
                # FFT from P3 to Fend.and apply vortex
                EF3 = fftshiftVortex*fft2(fftshift(EP3pad))/Nfft2
    
                # FFT from Vortex FPM to Lyot Plane
                EP4 = fftshift(fft2(EF3))/Nfft2
                EP4 = fp.relay(EP4, mp.Nrelay3to4-1, mp.centering)
                
                if(Nfft2 > mp.P4.compact.Narr):
                    EP4 = mp.P4.compact.croppedMask * pad_crop(EP4, mp.P4.compact.Narr)
                else:
                    EP4 = pad_crop(mp.P4.compact.croppedMask, Nfft2) * EP4

                # MFT to detector
                EP4 = fp.relay(EP4, mp.NrelayFend, mp.centering)
                EFend = fp.mft_p2f(EP4, mp.fl, wvl, mp.P4.compact.dx, mp.Fend.dxi, mp.Fend.Nxi, mp.Fend.deta, mp.Fend.Neta, mp.centering)
                
                Gzdl[:, Gindex] = EFend[mp.Fend.corr.maskBool] / \
                    np.sqrt(mp.Fend.compact.I00[modvar.sbpIndex])

            Gindex += 1
        
    return Gzdl
