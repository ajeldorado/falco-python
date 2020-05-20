import cupy as cp
import numpy as np
import falco
import logging
import matplotlib.pyplot as plt

def lyot(mp,im,idm):
    """
    Special compact model used to compute the control Jacobian for the Lyot coronagraph.
    
    Specialized compact model used to compute the DM response matrix, aka the control 
    Jacobian for a Lyot coronagraph. Can include an apodizer, making it an apodized pupil 
    Lyot coronagraph (APLC).Does not include unknown aberrations of the full, "truth" 
    model. This model propagates the first-order Taylor expansion of the phase from the 
    poke of each actuator of the deformable mirror.

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
    
    modvar = falco.config.Object() #--Initialize the new structure
    modvar.sbpIndex = mp.jac.sbp_inds[im]
    modvar.zernIndex = mp.jac.zern_inds[im]
    
    wvl = mp.sbp_centers[modvar.sbpIndex]
    mirrorFac = 2. # Phase change is twice the DM surface height.
    NdmPad = int(mp.compact.NdmPad)

    """Input E-fields"""    
    Ein = cp.squeeze(mp.P1.compact.E[:,:,modvar.sbpIndex])  
    #--Apply a Zernike (in amplitude) at input pupil
    #--Used only for Zernike sensitivity control, which requires the perfect 
    # E-field of the differential Zernike term.
    if not (modvar.zernIndex==1):
        indsZnoll = modvar.zernIndex #--Just send in 1 Zernike mode
        zernMat = cp.squeeze(falco.zernikes.falco_gen_norm_zernike_maps(mp.P1.compact.Nbeam,mp.centering,indsZnoll)) #--2-D normalized (RMS = 1) Zernike mode
        zernMat = falco.utils.padOrCropEven(zernMat,mp.P1.compact.Narr)
        Ein = Ein*zernMat*(2*cp.pi/wvl)*mp.jac.Zcoef[mp.jac.zerns==modvar.zernIndex]
    
    """ Masks and DM surfaces """
    pupil = falco.utils.padOrCropEven(mp.P1.compact.mask,NdmPad)
    Ein = falco.utils.padOrCropEven(Ein,NdmPad)
    
    #--Re-image the apodizer from pupil P3 back to pupil P2. (Sign of mp.Nrelay2to3 doesn't matter.)
    if(mp.flagApod):
        apodReimaged = falco.utils.padOrCropEven(mp.P3.compact.mask, NdmPad)
        apodReimaged = falco.propcustom.propcustom_relay(apodReimaged,mp.Nrelay2to3,mp.centering)
    else:
        apodReimaged = cp.ones((NdmPad,NdmPad)) 
    
    #--Compute the DM surfaces for the current DM commands
    if(any(mp.dm_ind==1)): 
        DM1surf = falco.utils.padOrCropEven(mp.dm1.compact.surfM, NdmPad)
    else: 
        DM1surf = cp.zeros((NdmPad,NdmPad)) #--Pre-compute the starting DM1 surface
    if(any(mp.dm_ind==2)): 
        DM2surf = falco.utils.padOrCropEven(mp.dm2.compact.surfM, NdmPad)
    else:
        DM2surf = cp.zeros((NdmPad,NdmPad)) #--Pre-compute the starting DM2 surface

    if(mp.flagDM1stop):
        DM1stop = falco.utils.padOrCropEven(mp.dm1.compact.mask, NdmPad) 
    else: 
        DM1stop = cp.ones((NdmPad,NdmPad))
    if(mp.flagDM2stop):
        DM2stop = falco.utils.padOrCropEven(mp.dm2.compact.mask, NdmPad) 
    else: 
        DM2stop = cp.ones((NdmPad,NdmPad))

    if(mp.useGPU):
        log.warning('GPU support not yet implemented. Proceeding without GPU.')
        
    #--This block is for BMC surface error testing
    if(mp.flagDMwfe):
        if(any(mp.dm_ind==1)):
            Edm1WFE = cp.exp(2*cp.pi*1j/wvl*falco.utils.padOrCropEven(mp.dm1.compact.wfe,NdmPad,'extrapval',0)) 
        else: 
            Edm1WFE = cp.ones((NdmPad,NdmPad))
        if(any(mp.dm_ind==2)):
            Edm2WFE = cp.exp(2*cp.pi*1j/wvl*falco.utils.padOrCropEven(mp.dm2.compact.wfe,NdmPad,'extrapval',0)) 
        else: 
            Edm2WFE = cp.ones((NdmPad,NdmPad))
    else:
        Edm1WFE = cp.ones((NdmPad,NdmPad))
        Edm2WFE = cp.ones((NdmPad,NdmPad))
        
    """Propagation"""

    #--Define pupil P1 and Propagate to pupil P2
    EP1 = pupil*Ein #--E-field at pupil plane P1
    EP2 = falco.propcustom.propcustom_relay(EP1,mp.Nrelay1to2,mp.centering) #--Forward propagate to the next pupil plane (P2) by rotating 180 degrees mp.Nrelay1to2 times.

    #--Propagate from P2 to DM1, and apply DM1 surface and aperture stop
    if not (abs(mp.d_P2_dm1)==0): #--E-field arriving at DM1
        Edm1 = falco.propcustom.propcustom_PTP(EP2,mp.P2.compact.dx*NdmPad,wvl,mp.d_P2_dm1)
    else:
        Edm1 = EP2
    Edm1b = Edm1*Edm1WFE*DM1stop*cp.exp(mirrorFac*2*cp.pi*1j*DM1surf/wvl) #--E-field leaving DM1

    """ ---------- DM1 ---------- """
    if(idm==1):
        Gzdl = cp.zeros((mp.Fend.corr.Npix,mp.dm1.Nele),dtype=cp.complex)
        
        #--Two array sizes (at same resolution) of influence functions for MFT and angular spectrum
        NboxPad1AS = int(mp.dm1.compact.NboxAS) #--array size for FFT-AS propagations from DM1->DM2->DM1
        mp.dm1.compact.xy_box_lowerLeft_AS = mp.dm1.compact.xy_box_lowerLeft - (mp.dm1.compact.NboxAS-mp.dm1.compact.Nbox)/2. #--Adjust the sub-array location of the influence function for the added zero padding
    
        if(any(mp.dm_ind==2)):
            DM2surf = falco.utils.padOrCropEven(DM2surf,mp.dm1.compact.NdmPad)  
        else:
            DM2surf = cp.zeros((mp.dm1.compact.NdmPad,mp.dm1.compact.NdmPad)) 
        if(mp.flagDM2stop):
            DM2stop = falco.utils.padOrCropEven(DM2stop,mp.dm1.compact.NdmPad) 
        else:
            DM2stop = cp.ones((mp.dm1.compact.NdmPad,mp.dm1.compact.NdmPad))
        apodReimaged = falco.utils.padOrCropEven( apodReimaged, mp.dm1.compact.NdmPad)
    
        Edm1pad = falco.utils.padOrCropEven(Edm1b,mp.dm1.compact.NdmPad) #--Pad or crop for expected sub-array indexing
        Edm2WFEpad = falco.utils.padOrCropEven(Edm2WFE,mp.dm1.compact.NdmPad) #--Pad or crop for expected sub-array indexing
    
        #--Propagate each actuator from DM1 through the optical system
        Gindex = 0 #1  initialize index counter

        for iact in mp.dm1.act_ele: # cp.array([405]): # cp.array([1665]): #
            if( cp.sum(cp.abs(mp.dm1.compact.inf_datacube[:,:,iact]))>1e-12 ):  #--Compute only for influence functions that are not zeroed out

                #--x- and y- coordinate indices of the padded influence function in the full padded pupil
                x_box_AS_ind = np.arange(mp.dm1.compact.xy_box_lowerLeft_AS[0,iact],mp.dm1.compact.xy_box_lowerLeft_AS[0,iact]+NboxPad1AS,dtype=np.int) # x-indices in pupil arrays for the box
                y_box_AS_ind = np.arange(mp.dm1.compact.xy_box_lowerLeft_AS[1,iact],mp.dm1.compact.xy_box_lowerLeft_AS[1,iact]+NboxPad1AS,dtype=np.int) # y-indices in pupil arrays for the box
                #--x- and y- coordinates of the UN-padded influence function in the full padded pupil
                x_box = mp.dm1.compact.x_pupPad[x_box_AS_ind] # full pupil x-coordinates of the box 
                y_box = mp.dm1.compact.y_pupPad[y_box_AS_ind] # full pupil y-coordinates of the box
                
                #--Propagate from DM1 to DM2, and then back to P2
                dEbox = (mirrorFac*2*cp.pi*1j/wvl)*falco.utils.padOrCropEven((mp.dm1.VtoH.reshape(mp.dm1.Nact**2)[iact])*cp.squeeze(mp.dm1.compact.inf_datacube[:,:,iact]),NboxPad1AS) #--Pad influence function at DM1 for angular spectrum propagation.
                dEbox = falco.propcustom.propcustom_PTP(dEbox*Edm1pad[cp.ix_(y_box_AS_ind,x_box_AS_ind)],mp.P2.compact.dx*NboxPad1AS,wvl,mp.d_dm1_dm2) # forward propagate to DM2 and apply DM2 E-field
                dEP2box = falco.propcustom.propcustom_PTP(dEbox*Edm2WFEpad[cp.ix_(y_box_AS_ind,x_box_AS_ind)]*DM2stop[cp.ix_(y_box_AS_ind,x_box_AS_ind)]*cp.exp(mirrorFac*2*cp.pi*1j/wvl*DM2surf[cp.ix_(y_box_AS_ind,x_box_AS_ind)]),mp.P2.compact.dx*NboxPad1AS,wvl,-1*(mp.d_dm1_dm2 + mp.d_P2_dm1) ) # back-propagate to DM1
              
                #--To simulate going forward to the next pupil plane (with the apodizer) most efficiently, 
                # First, back-propagate the apodizer (by rotating 180-degrees) to the previous pupil.
                # Second, negate the coordinates of the box used.
                dEP2box = apodReimaged[cp.ix_(y_box_AS_ind,x_box_AS_ind)]*dEP2box #--Apply 180deg-rotated SP mask.
                dEP3box = cp.rot90(dEP2box,k=2*mp.Nrelay2to3) #--Forward propagate the cropped box by rotating 180 degrees mp.Nrelay2to3 times.
                #--Negate and reverse coordinate values to effectively rotate by 180 degrees. No change if 360 degree rotation.
                if cp.mod(mp.Nrelay2to3,2)==1: 
                    x_box = -1*x_box[::-1]
                    y_box = -1*y_box[::-1]
               
                #--Matrices for the MFT from the pupil P3 to the focal plane mask
                rect_mat_pre = (cp.exp(-2*cp.pi*1j*cp.outer(mp.F3.compact.etas,y_box)/(wvl*mp.fl)))*cp.sqrt(mp.P2.compact.dx*mp.P2.compact.dx)*cp.sqrt(mp.F3.compact.dxi*mp.F3.compact.deta)/(wvl*mp.fl)
                rect_mat_post  = (cp.exp(-2*cp.pi*1j*cp.outer(x_box,mp.F3.compact.xis)/(wvl*mp.fl)))
                
                EF3inc = rect_mat_pre @ dEP3box @ rect_mat_post # MFT to FPM
                
                if (mp.coro.upper()=='LC') or (mp.coro.upper()=='APLC'):
                    EF3 = (1.-mp.F3.compact.ampMask) * EF3inc #--Propagate through (1-complex FPM) for Babinet's principle
        
                    #--MFT to LS ("Sub" name for Subtrahend part of the Lyot-plane E-field)
                    EP4sub = falco.propcustom.propcustom_mft_FtoP(EF3,mp.fl,wvl,mp.F3.compact.dxi,mp.F3.compact.deta,mp.P4.compact.dx,mp.P4.compact.Narr,mp.centering)  #--Subtrahend term for the Lyot plane E-field    
                    EP4sub = falco.propcustom.propcustom_relay(EP4sub,mp.Nrelay3to4-1,mp.centering); #--Get the correct orientation
                    
                    #--Full Lyot plane pupil (for Babinet)
                    EP4noFPM = cp.zeros((mp.dm1.compact.NdmPad,mp.dm1.compact.NdmPad),dtype=cp.complex)
                    EP4noFPM[cp.ix_(y_box_AS_ind,x_box_AS_ind)] = dEP2box #--Propagating the E-field from P2 to P4 without masks gives the same E-field. 
                    EP4noFPM = falco.propcustom.propcustom_relay(EP4noFPM,mp.Nrelay2to3+mp.Nrelay3to4,mp.centering) #--Get the correct orientation 
                    EP4noFPM = falco.utils.padOrCropEven(EP4noFPM,mp.P4.compact.Narr) #--Crop down to the size of the Lyot stop opening
                    EP4 = EP4noFPM - EP4sub #--Babinet's principle to get E-field at Lyot plane
                
                elif(mp.coro.upper()=='FLC' or mp.coro.upper()=='SPLC'):
                    EF3 = mp.F3.compact.ampMask * EF3inc # Apply FPM
                    
                    #--MFT to Lyot plane
                    EP4 = falco.propcustom.propcustom_mft_FtoP(EF3,mp.fl,wvl,mp.F3.compact.dxi,mp.F3.compact.deta,mp.P4.compact.dx,mp.P4.compact.Narr,mp.centering)   
                    EP4 = falco.propcustom.propcustom_relay(EP4, mp.Nrelay3to4-1, mp.centering) #--Get the correct orientation
                    
                    
                EP4 *= mp.P4.compact.croppedMask # Apply Lyot stop
    
                #--MFT to camera
                EP4 = falco.propcustom.propcustom_relay(EP4,mp.NrelayFend,mp.centering) #--Rotate the final image 180 degrees if necessary
                EFend = falco.propcustom.propcustom_mft_PtoF(EP4,mp.fl,wvl,mp.P4.compact.dx,mp.Fend.dxi,mp.Fend.Nxi,mp.Fend.deta,mp.Fend.Neta,mp.centering)
                
                Gzdl[:,Gindex] = EFend[mp.Fend.corr.maskBool]/cp.sqrt(mp.Fend.compact.I00[modvar.sbpIndex])
            Gindex += 1

    """ ---------- DM2 ---------- """
    if(idm==2):
        Gzdl = cp.zeros((mp.Fend.corr.Npix,mp.dm2.Nele),dtype=cp.complex)
        
        #--Two array sizes (at same resolution) of influence functions for MFT and angular spectrum
        NboxPad2AS = int(mp.dm2.compact.NboxAS)
        mp.dm2.compact.xy_box_lowerLeft_AS = mp.dm2.compact.xy_box_lowerLeft - (NboxPad2AS-mp.dm2.compact.Nbox)/2 #--Account for the padding of the influence function boxes
        
        apodReimaged = falco.utils.padOrCropEven( apodReimaged, mp.dm2.compact.NdmPad)
        DM2stopPad = falco.utils.padOrCropEven(DM2stop,mp.dm2.compact.NdmPad)
        Edm2WFEpad = falco.utils.padOrCropEven(Edm2WFE,mp.dm2.compact.NdmPad)
    
        #--Propagate full field to DM2 before back-propagating in small boxes
        Edm2inc = falco.utils.padOrCropEven( falco.propcustom.propcustom_PTP(Edm1b,mp.compact.NdmPad*mp.P2.compact.dx,wvl,mp.d_dm1_dm2), mp.dm2.compact.NdmPad) # E-field incident upon DM2
        Edm2inc = falco.utils.padOrCropEven(Edm2inc,mp.dm2.compact.NdmPad);
        Edm2 = DM2stopPad*Edm2WFEpad*Edm2inc*cp.exp(mirrorFac*2*cp.pi*1j/wvl*falco.utils.padOrCropEven(DM2surf,mp.dm2.compact.NdmPad)) # Initial E-field at DM2 including its own phase contribution
        
        #--Propagate each actuator from DM2 through the rest of the optical system
        Gindex = 0 # initialize index counter
        for iact in mp.dm2.act_ele:
            if( cp.sum(cp.abs(mp.dm2.compact.inf_datacube[:,:,iact]))>1e-12 ):  #--Only compute for acutators specified for use or for influence functions that are not zeroed out
    
                #--x- and y- coordinates of the padded influence function in the full padded pupil
                x_box_AS_ind = np.arange(mp.dm2.compact.xy_box_lowerLeft_AS[0,iact],mp.dm2.compact.xy_box_lowerLeft_AS[0,iact]+NboxPad2AS,dtype=cp.int) # x-indices in pupil arrays for the box
                y_box_AS_ind = np.arange(mp.dm2.compact.xy_box_lowerLeft_AS[1,iact],mp.dm2.compact.xy_box_lowerLeft_AS[1,iact]+NboxPad2AS,dtype=cp.int) # y-indices in pupil arrays for the box
                #--x- and y- coordinates of the UN-padded influence function in the full padded pupil
                x_box = mp.dm2.compact.x_pupPad[x_box_AS_ind] # full pupil x-coordinates of the box 
                y_box = mp.dm2.compact.y_pupPad[y_box_AS_ind] # full pupil y-coordinates of the box 
                
                dEbox = (mp.dm2.VtoH.reshape(mp.dm2.Nact**2)[iact])*(mirrorFac*2*cp.pi*1j/wvl)*falco.utils.padOrCropEven(cp.squeeze(mp.dm2.compact.inf_datacube[:,:,iact]),NboxPad2AS) #--the padded influence function at DM2
                dEP2box = falco.propcustom.propcustom_PTP(dEbox*Edm2[cp.ix_(y_box_AS_ind,x_box_AS_ind)],mp.P2.compact.dx*NboxPad2AS,wvl,-1*(mp.d_dm1_dm2 + mp.d_P2_dm1)) # back-propagate to pupil P2
                
                #--To simulate going forward to the next pupil plane (with the apodizer) most efficiently, 
                # First, back-propagate the apodizer (by rotating 180-degrees) to the previous pupil.
                # Second, negate the coordinates of the box used.
                dEP2box = apodReimaged[cp.ix_(y_box_AS_ind,x_box_AS_ind)]*dEP2box; #--Apply 180deg-rotated SP mask.
                dEP3box = cp.rot90(dEP2box,k=2*mp.Nrelay2to3) #--Forward propagate the cropped box by rotating 180 degrees mp.Nrelay2to3 times.
                #--Negate and rotate coordinates to effectively rotate by 180 degrees. No change if 360 degree rotation.
                if cp.mod(mp.Nrelay2to3,2)==1: 
                    x_box = -1*x_box[::-1]
                    y_box = -1*y_box[::-1]
                
                #--Matrices for the MFT from the pupil P3 to the focal plane mask
                rect_mat_pre = (cp.exp(-2*cp.pi*1j*cp.outer(mp.F3.compact.etas,y_box)/(wvl*mp.fl)))*cp.sqrt(mp.P2.compact.dx*mp.P2.compact.dx)*cp.sqrt(mp.F3.compact.dxi*mp.F3.compact.deta)/(wvl*mp.fl)
                rect_mat_post  = (cp.exp(-2*cp.pi*1j*cp.outer(x_box,mp.F3.compact.xis)/(wvl*mp.fl)))
    
                EF3inc = rect_mat_pre @ dEP3box @ rect_mat_post # MFT to FPM
                
                if (mp.coro.upper()=='LC') or (mp.coro.upper()=='APLC'):
                    
                    EF3 = (1-mp.F3.compact.ampMask) * EF3inc #--Propagate through ( 1 - (complex FPM) ) for Babinet's principle
        
                    #--MFT to LS ("Sub" name for Subtrahend part of the Lyot-plane E-field)
                    EP4sub = falco.propcustom.propcustom_mft_FtoP(EF3,mp.fl,wvl,mp.F3.compact.dxi,mp.F3.compact.deta,mp.P4.compact.dx,mp.P4.compact.Narr,mp.centering) #--Subtrahend term for the Lyot plane E-field    
                    EP4sub = falco.propcustom.propcustom_relay(EP4sub,mp.Nrelay3to4-1,mp.centering) #--Get the correct orientation
                                    
                    EP4noFPM = cp.zeros((mp.dm2.compact.NdmPad,mp.dm2.compact.NdmPad),dtype=cp.complex)
                    EP4noFPM[cp.ix_(y_box_AS_ind,x_box_AS_ind)] = dEP2box #--Propagating the E-field from P2 to P4 without masks gives the same E-field.
                    EP4noFPM = falco.propcustom.propcustom_relay(EP4noFPM,mp.Nrelay2to3+mp.Nrelay3to4,mp.centering) #--Get the number or re-imaging relays between pupils P3 and P4. 
                    EP4noFPM = falco.utils.padOrCropEven(EP4noFPM,mp.P4.compact.Narr) #--Crop down to the size of the Lyot stop opening
                    EP4 = (EP4noFPM - EP4sub) #--Babinet's principle to get E-field at Lyot plane
                
                elif(mp.coro.upper()=='FLC' or mp.coro.upper()=='SPLC'):

                    EF3 = mp.F3.compact.ampMask * EF3inc # Apply FPM
        
                    #--MFT to LS ("Sub" name for Subtrahend part of the Lyot-plane E-field)
                    EP4 = falco.propcustom.propcustom_mft_FtoP(EF3,mp.fl,wvl,mp.F3.compact.dxi,mp.F3.compact.deta,mp.P4.compact.dx,mp.P4.compact.Narr,mp.centering)   
                    EP4 = falco.propcustom.propcustom_relay(EP4, mp.Nrelay3to4-1, mp.centering) # Get the correct orientation
                
                EP4 *= mp.P4.compact.croppedMask # Apply Lyot stop
    
                #--MFT to detector
                EP4 = falco.propcustom.propcustom_relay(EP4,mp.NrelayFend,mp.centering) #--Rotate the final image 180 degrees if necessary
                EFend = falco.propcustom.propcustom_mft_PtoF(EP4,mp.fl,wvl,mp.P4.compact.dx,mp.Fend.dxi,mp.Fend.Nxi,mp.Fend.deta,mp.Fend.Neta,mp.centering)
                
                Gzdl[:,Gindex] = EFend[mp.Fend.corr.maskBool]/cp.sqrt(mp.Fend.compact.I00[modvar.sbpIndex])

            Gindex += 1
        
    return Gzdl

    
def vortex(mp,im,idm):
    """
    Special compact model used to compute the control Jacobian for the vortex coronagraph.
    
    Specialized compact model used to compute the DM response matrix, aka the control 
    Jacobian for a vortex coronagraph. Can include an apodizer, making it an apodized 
    vortex coronagraph (AVC).Does not include unknown aberrations of the full, "truth" 
    model. This model propagates the first-order Taylor expansion of the phase from the 
    poke of each actuator of the deformable mirror.

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

    modvar = falco.config.Object() #--Initialize the new structure
    modvar.sbpIndex = mp.jac.sbp_inds[im]
    modvar.zernIndex = mp.jac.zern_inds[im]
    
    wvl = mp.sbp_centers[modvar.sbpIndex]
    mirrorFac = 2. # Phase change is twice the DM surface height.
    NdmPad = int(mp.compact.NdmPad)

    #--Minimum FPM resolution for Jacobian calculations (in pixels per lambda/D)
    minPadFacVortex = 8

    # Get FPM charge
    if( type(mp.F3.VortexCharge)==cp.ndarray ):
        # Passing an array for mp.F3.VortexCharge with
        # corresponding wavelengths mp.F3.VortexCharge_lambdas
        # represents a chromatic vortex FPM
        charge = mp.F3.VortexCharge if(mp.F3.VortexCharge.size==1) else cp.interp(wvl,mp.F3.VortexCharge_lambdas,mp.F3.VortexCharge,'linear','extrap')
    elif( type(mp.F3.VortexCharge)==int or type(mp.F3.VortexCharge)==float ):
        # single value indicates fully achromatic mask
        charge = mp.F3.VortexCharge
    else:
        raise TypeError("mp.F3.VortexCharge must be an int, float or numpy ndarray.")
        pass    

    """Input E-fields"""    
    Ein = cp.squeeze(mp.P1.compact.E[:,:,modvar.sbpIndex])  
    #--Apply a Zernike (in amplitude) at input pupil
    #--Used only for Zernike sensitivity control, which requires the perfect 
    # E-field of the differential Zernike term.
    if not (modvar.zernIndex==1):
        indsZnoll = modvar.zernIndex #--Just send in 1 Zernike mode
        zernMat = cp.squeeze(falco.zernikes.falco_gen_norm_zernike_maps(mp.P1.compact.Nbeam,mp.centering,indsZnoll)) #--2-D normalized (RMS = 1) Zernike mode
        zernMat = falco.utils.padOrCropEven(zernMat,mp.P1.compact.Narr)
        Ein = Ein*zernMat*(2*cp.pi/wvl)*mp.jac.Zcoef[mp.jac.zerns==modvar.zernIndex]
    
    """ Masks and DM surfaces """
    pupil = falco.utils.padOrCropEven(mp.P1.compact.mask,NdmPad)
    Ein = falco.utils.padOrCropEven(Ein,NdmPad)
    
    #--Re-image the apodizer from pupil P3 back to pupil P2. (Sign of mp.Nrelay2to3 doesn't matter.)
    if(mp.flagApod):
        apodReimaged = falco.utils.padOrCropEven(mp.P3.compact.mask, NdmPad)
        apodReimaged = falco.propcustom.propcustom_relay(apodReimaged,mp.Nrelay2to3,mp.centering)
    else:
        apodReimaged = cp.ones((NdmPad,NdmPad)) 
    
    #--Compute the DM surfaces for the current DM commands
    if(any(mp.dm_ind==1)): 
        DM1surf = falco.utils.padOrCropEven(mp.dm1.compact.surfM, NdmPad)
        #DM1surf = falco.dms.falco_gen_dm_surf(mp.dm1, mp.dm1.compact.dx, NdmPad) 
    else: 
        DM1surf = cp.zeros((NdmPad,NdmPad)) #--Pre-compute the starting DM1 surface
    if(any(mp.dm_ind==2)): 
        DM2surf = falco.utils.padOrCropEven(mp.dm2.compact.surfM, NdmPad)
        #DM2surf = falco.dms.falco_gen_dm_surf(mp.dm2, mp.dm2.compact.dx, NdmPad) 
    else:
        DM2surf = cp.zeros((NdmPad,NdmPad)) #--Pre-compute the starting DM2 surface

    if(mp.flagDM1stop):
        DM1stop = falco.utils.padOrCropEven(mp.dm1.compact.mask, NdmPad) 
    else: 
        DM1stop = cp.ones((NdmPad,NdmPad))
    if(mp.flagDM2stop):
        DM2stop = falco.utils.padOrCropEven(mp.dm2.compact.mask, NdmPad) 
    else: 
        DM2stop = cp.ones((NdmPad,NdmPad))
        
    #--This block is for BMC surface error testing
    if(mp.flagDMwfe): # if(mp.flagDMwfe && (mp.P1.full.Nbeam==mp.P1.compact.Nbeam))
        if(any(mp.dm_ind==1)):
            Edm1WFE = cp.exp(2*cp.pi*1j/wvl*falco.utils.padOrCropEven(mp.dm1.compact.wfe,NdmPad,'extrapval',0)) 
        else: 
            Edm1WFE = cp.ones((NdmPad,NdmPad))
        if(any(mp.dm_ind==2)):
            Edm2WFE = cp.exp(2*cp.pi*1j/wvl*falco.utils.padOrCropEven(mp.dm2.compact.wfe,NdmPad,'extrapval',0)) 
        else: 
            Edm2WFE = cp.ones((NdmPad,NdmPad))
    else:
        Edm1WFE = cp.ones((NdmPad,NdmPad))
        Edm2WFE = cp.ones((NdmPad,NdmPad))
        
    """Propagation"""

    #--Define pupil P1 and Propagate to pupil P2
    EP1 = pupil*Ein #--E-field at pupil plane P1
    EP2 = falco.propcustom.propcustom_relay(EP1,mp.Nrelay1to2,mp.centering) #--Forward propagate to the next pupil plane (P2) by rotating 180 degrees mp.Nrelay1to2 times.

    #--Propagate from P2 to DM1, and apply DM1 surface and aperture stop
    if not (abs(mp.d_P2_dm1)==0): #--E-field arriving at DM1
        Edm1 = falco.propcustom.propcustom_PTP(EP2,mp.P2.compact.dx*NdmPad,wvl,mp.d_P2_dm1)
    else:
        Edm1 = EP2
    Edm1b = Edm1*Edm1WFE*DM1stop*cp.exp(mirrorFac*2*cp.pi*1j*DM1surf/wvl) #--E-field leaving DM1

    """ ---------- DM1 ---------- """
    if(idm==1):
        Gzdl = cp.zeros((mp.Fend.corr.Npix,mp.dm1.Nele),dtype=cp.complex)
        
        #--Array size for planes P3, F3, and P4
        Nfft1 = int(2**falco.utils.nextpow2(cp.max(cp.array([mp.dm1.compact.NdmPad, minPadFacVortex*mp.dm1.compact.Nbox])))) #--Don't crop--but do pad if necessary.
        
        #--Generate vortex FPM with fftshift already applied
        fftshiftVortex = cp.fft.fftshift( falco.masks.falco_gen_vortex_mask( charge, Nfft1) )
    
        #--Two array sizes (at same resolution) of influence functions for MFT and angular spectrum
        NboxPad1AS = int(mp.dm1.compact.NboxAS) #--array size for FFT-AS propagations from DM1->DM2->DM1
        mp.dm1.compact.xy_box_lowerLeft_AS = mp.dm1.compact.xy_box_lowerLeft - (mp.dm1.compact.NboxAS-mp.dm1.compact.Nbox)/2. #--Adjust the sub-array location of the influence function for the added zero padding
    
        if(any(mp.dm_ind==2)):
            DM2surf = falco.utils.padOrCropEven(DM2surf,mp.dm1.compact.NdmPad)  
        else:
            DM2surf = cp.zeros((mp.dm1.compact.NdmPad,mp.dm1.compact.NdmPad)) 
        if(mp.flagDM2stop):
            DM2stop = falco.utils.padOrCropEven(DM2stop,mp.dm1.compact.NdmPad) 
        else:
            DM2stop = cp.ones((mp.dm1.compact.NdmPad,mp.dm1.compact.NdmPad))
        apodReimaged = falco.utils.padOrCropEven( apodReimaged, mp.dm1.compact.NdmPad)
    
        Edm1pad = falco.utils.padOrCropEven(Edm1b,mp.dm1.compact.NdmPad) #--Pad or crop for expected sub-array indexing
        Edm2WFEpad = falco.utils.padOrCropEven(Edm2WFE,mp.dm1.compact.NdmPad) #--Pad or crop for expected sub-array indexing
    
        #--Propagate each actuator from DM1 through the optical system
        Gindex = 0 #1  initialize index counter
        for iact in mp.dm1.act_ele: # cp.array([405]): # cp.array([1665]): #
            if( cp.sum(cp.abs(mp.dm1.compact.inf_datacube[:,:,iact]))>1e-12 ):  #--Compute only for influence functions that are not zeroed out
                
                #--x- and y- coordinate indices of the padded influence function in the full padded pupil
                x_box_AS_ind = np.arange(mp.dm1.compact.xy_box_lowerLeft_AS[0,iact],mp.dm1.compact.xy_box_lowerLeft_AS[0,iact]+NboxPad1AS,dtype=cp.int) # x-indices in pupil arrays for the box
                y_box_AS_ind = np.arange(mp.dm1.compact.xy_box_lowerLeft_AS[1,iact],mp.dm1.compact.xy_box_lowerLeft_AS[1,iact]+NboxPad1AS,dtype=cp.int) # y-indices in pupil arrays for the box
                #--x- and y- coordinates of the UN-padded influence function in the full padded pupil
                x_box = mp.dm1.compact.x_pupPad[x_box_AS_ind] # full pupil x-coordinates of the box 
                y_box = mp.dm1.compact.y_pupPad[y_box_AS_ind] # full pupil y-coordinates of the box
                
                #--Propagate from DM1 to DM2, and then back to P2
                dEbox = (mirrorFac*2*cp.pi*1j/wvl)*falco.utils.padOrCropEven((mp.dm1.VtoH.reshape(mp.dm1.Nact**2)[iact])*cp.squeeze(mp.dm1.compact.inf_datacube[:,:,iact]),NboxPad1AS) #--Pad influence function at DM1 for angular spectrum propagation.
                dEbox = falco.propcustom.propcustom_PTP(dEbox*Edm1pad[cp.ix_(y_box_AS_ind,x_box_AS_ind)],mp.P2.compact.dx*NboxPad1AS,wvl,mp.d_dm1_dm2) # forward propagate to DM2 and apply DM2 E-field
                dEP2box = falco.propcustom.propcustom_PTP(dEbox*Edm2WFEpad[cp.ix_(y_box_AS_ind,x_box_AS_ind)]*DM2stop[cp.ix_(y_box_AS_ind,x_box_AS_ind)]*cp.exp(mirrorFac*2*cp.pi*1j/wvl*DM2surf[cp.ix_(y_box_AS_ind,x_box_AS_ind)]),mp.P2.compact.dx*NboxPad1AS,wvl,-1*(mp.d_dm1_dm2 + mp.d_P2_dm1) ) # back-propagate to DM1
              
                #--To simulate going forward to the next pupil plane (with the apodizer) most efficiently, 
                # First, back-propagate the apodizer (by rotating 180-degrees) to the previous pupil.
                # Second, negate the coordinates of the box used.
                dEP2boxEff = apodReimaged[cp.ix_(y_box_AS_ind,x_box_AS_ind)]*dEP2box #--Apply 180deg-rotated apodizer mask.

                # #--Negate and reverse coordinate values to effectively rotate by 180 degrees. No change if 360 degree rotation.

                #--Re-insert the window around the influence function back into the full beam array.
                EP2eff = cp.zeros((mp.dm1.compact.NdmPad,mp.dm1.compact.NdmPad),dtype=complex)
                EP2eff[cp.ix_(y_box_AS_ind,x_box_AS_ind)] = dEP2boxEff
                
                #--Forward propagate from P2 (effective) to P3
                EP3 = falco.propcustom.propcustom_relay(EP2eff,mp.Nrelay2to3,mp.centering)
                
                #--Pad pupil P3 for FFT
                EP3pad = falco.utils.padOrCropEven(EP3, Nfft1)
                
                #--FFT from P3 to Fend.and apply vortex
                EF3 = fftshiftVortex*cp.fft.fft2(cp.fft.fftshift(EP3pad))/Nfft1
                
                #--FFT from Vortex FPM to Lyot Plane
                EP4 = cp.fft.fftshift(cp.fft.fft2(EF3))/Nfft1
                EP4 = falco.propcustom.propcustom_relay(EP4,mp.Nrelay3to4-1,mp.centering) #--Add more re-imaging relays if necessary
                if(Nfft1 > mp.P4.compact.Narr):
                    EP4 = mp.P4.compact.croppedMask*falco.utils.padOrCropEven(EP4,mp.P4.compact.Narr) #--Crop EP4 and then apply Lyot stop 
                else:
                    EP4 = falco.utils.padOrCropEven(mp.P4.compact.croppedMask,Nfft1)*EP4 #--Crop the Lyot stop and then apply it.
                    pass
                
                #--MFT to camera
                EP4 = falco.propcustom.propcustom_relay(EP4,mp.NrelayFend,mp.centering) #--Rotate the final image 180 degrees if necessary
                EFend = falco.propcustom.propcustom_mft_PtoF(EP4,mp.fl,wvl,mp.P4.compact.dx,mp.Fend.dxi,mp.Fend.Nxi,mp.Fend.deta,mp.Fend.Neta,mp.centering)
                
                Gzdl[:,Gindex] = EFend[mp.Fend.corr.maskBool]/cp.sqrt(mp.Fend.compact.I00[modvar.sbpIndex])

            Gindex += 1

    """ ---------- DM2 ---------- """
    if(idm==2):
        Gzdl = cp.zeros((mp.Fend.corr.Npix,mp.dm2.Nele),dtype=cp.complex)
        
        #--Array size for planes P3, F3, and P4
        Nfft2 = int(2**falco.utils.nextpow2(cp.max(cp.array([mp.dm2.compact.NdmPad, minPadFacVortex*mp.dm2.compact.Nbox])))) #--Don't crop--but do pad if necessary.
        
        #--Generate vortex FPM with fftshift already applied
        fftshiftVortex = cp.fft.fftshift( falco.masks.falco_gen_vortex_mask( charge, Nfft2) )
    
        #--Two array sizes (at same resolution) of influence functions for MFT and angular spectrum
        NboxPad2AS = int(mp.dm2.compact.NboxAS)
        mp.dm2.compact.xy_box_lowerLeft_AS = mp.dm2.compact.xy_box_lowerLeft - (NboxPad2AS-mp.dm2.compact.Nbox)/2 #--Account for the padding of the influence function boxes
        
        apodReimaged = falco.utils.padOrCropEven( apodReimaged, mp.dm2.compact.NdmPad)
        DM2stopPad = falco.utils.padOrCropEven(DM2stop,mp.dm2.compact.NdmPad)
        Edm2WFEpad = falco.utils.padOrCropEven(Edm2WFE,mp.dm2.compact.NdmPad)
    
        #--Propagate full field to DM2 before back-propagating in small boxes
        Edm2inc = falco.utils.padOrCropEven( falco.propcustom.propcustom_PTP(Edm1b,mp.compact.NdmPad*mp.P2.compact.dx,wvl,mp.d_dm1_dm2), mp.dm2.compact.NdmPad) # E-field incident upon DM2
        Edm2inc = falco.utils.padOrCropEven(Edm2inc,mp.dm2.compact.NdmPad);
        Edm2 = DM2stopPad*Edm2WFEpad*Edm2inc*cp.exp(mirrorFac*2*cp.pi*1j/wvl*falco.utils.padOrCropEven(DM2surf,mp.dm2.compact.NdmPad)) # Initial E-field at DM2 including its own phase contribution
        
        #--Propagate each actuator from DM2 through the rest of the optical system
        Gindex = 0 # initialize index counter
        for iact in mp.dm2.act_ele:
            if( cp.sum(cp.abs(mp.dm2.compact.inf_datacube[:,:,iact]))>1e-12 ):  #--Only compute for acutators specified for use or for influence functions that are not zeroed out
    
                #--x- and y- coordinates of the padded influence function in the full padded pupil
                x_box_AS_ind = cp.arange(mp.dm2.compact.xy_box_lowerLeft_AS[0,iact],mp.dm2.compact.xy_box_lowerLeft_AS[0,iact]+NboxPad2AS,dtype=cp.int) # x-indices in pupil arrays for the box
                y_box_AS_ind = cp.arange(mp.dm2.compact.xy_box_lowerLeft_AS[1,iact],mp.dm2.compact.xy_box_lowerLeft_AS[1,iact]+NboxPad2AS,dtype=cp.int) # y-indices in pupil arrays for the box
#                #--x- and y- coordinates of the UN-padded influence function in the full padded pupil 
                
                dEbox = (mp.dm2.VtoH.reshape(mp.dm2.Nact**2)[iact])*(mirrorFac*2*cp.pi*1j/wvl)*falco.utils.padOrCropEven(cp.squeeze(mp.dm2.compact.inf_datacube[:,:,iact]),NboxPad2AS) #--the padded influence function at DM2
                dEP2box = falco.propcustom.propcustom_PTP(dEbox*Edm2[cp.ix_(y_box_AS_ind,x_box_AS_ind)],mp.P2.compact.dx*NboxPad2AS,wvl,-1*(mp.d_dm1_dm2 + mp.d_P2_dm1)) # back-propagate to pupil P2
                
                #--To simulate going forward to the next pupil plane (with the apodizer) most efficiently, 
                # First, back-propagate the apodizer (by rotating 180-degrees) to the previous pupil.
                # Second, negate the coordinates of the box used.
                dEP2boxEff = apodReimaged[cp.ix_(y_box_AS_ind,x_box_AS_ind)]*dEP2box; #--Apply 180deg-rotated apodizer mask.
                
                EP2eff = cp.zeros((mp.dm2.compact.NdmPad,mp.dm2.compact.NdmPad),dtype=complex)
                EP2eff[cp.ix_(y_box_AS_ind,x_box_AS_ind)] = dEP2boxEff;

                #--Forward propagate from P2 (effective) to P3
                EP3 = falco.propcustom.propcustom_relay(EP2eff,mp.Nrelay2to3,mp.centering); 
    
                #--Pad pupil P3 for FFT
                EP3pad = falco.utils.padOrCropEven(EP3, Nfft2)
                
                #--FFT from P3 to Fend.and apply vortex
                EF3 = fftshiftVortex*cp.fft.fft2(cp.fft.fftshift(EP3pad))/Nfft2
    
                #--FFT from Vortex FPM to Lyot Plane
                EP4 = cp.fft.fftshift(cp.fft.fft2(EF3))/Nfft2
                EP4 = falco.propcustom.propcustom_relay(EP4,mp.Nrelay3to4-1,mp.centering) #--Add more re-imaging relays if necessary
                
                if(Nfft2 > mp.P4.compact.Narr):
                    EP4 = mp.P4.compact.croppedMask*falco.utils.padOrCropEven(EP4,mp.P4.compact.Narr) #--Crop EP4 and then apply Lyot stop 
                else:
                    EP4 = falco.utils.padOrCropEven(mp.P4.compact.croppedMask,Nfft2)*EP4 #--Crop the Lyot stop and then apply it.

                #--MFT to detector
                EP4 = falco.propcustom.propcustom_relay(EP4,mp.NrelayFend,mp.centering) #--Rotate the final image 180 degrees if necessary
                EFend = falco.propcustom.propcustom_mft_PtoF(EP4,mp.fl,wvl,mp.P4.compact.dx,mp.Fend.dxi,mp.Fend.Nxi,mp.Fend.deta,mp.Fend.Neta,mp.centering)
                
                Gzdl[:,Gindex] = EFend[mp.Fend.corr.maskBool]/cp.sqrt(mp.Fend.compact.I00[modvar.sbpIndex])

            Gindex += 1
        
    return Gzdl
