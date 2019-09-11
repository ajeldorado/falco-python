from falco.config import ModelParameters, ModelVariables, DeformableMirrorParameters
import numpy as np
import falco
import logging
import matplotlib.pyplot as plt #DEBUGGING
from astropy.io import fits #DEBUGGING
from mpl_toolkits.axes_grid1 import make_axes_locatable # DEBUGGING
import os #DEBUGGING

log = logging.getLogger(__name__)

def model_full(mp,modvar,**kwargs):
    """
    Truth model used to generate images. Can include aberrations/errors that are unknown 
    to the estimator and controller. This function is the wrapper for full models of any 
    coronagraph type.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    modvar : ModelVariables
        Structure containing temporary optical model variables

    Returns
    -------
    Eout : array_like
        2-D electric field in final focal plane
        
    Other Parameters
    ----------------
    GETNORM : bool
        If set, the PSF normalization factor will be computed and assigned.
    NORMOFF : bool
        If set, the PSF normalization factor is set to 1. This is useful for a standalone 
        call to model_full.

    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
        
    if(hasattr(modvar,'sbpIndex')):
        normFac = mp.Fend.full.I00[modvar.sbpIndex,modvar.wpsbpIndex] #--Value to normalize the PSF. Set to 0 when finding the normalization factor

    #--Optional Keyword arguments
    if ("GETNORM" in kwargs and kwargs["GETNORM"]):
        normFac = 0.
    
    if ("NORMOFF" in kwargs and kwargs["NORMOFF"]):
        normFac = 1.
        
#    # AJER NOTE DEBUGGING: 
#    ax = plt.subplot(111)
#    im = ax.imshow(mp.dm1.V)
#    # create an axes on the right side of ax. The width of cax will be 5%
#    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    plt.colorbar(im, cax=cax)
#    plt.pause(0.5)        
        
    #AJER NOTE: This should be in falco_config_gen_chosen_pupil
    mp.P2.full.xsDL = np.linspace(-mp.P1.full.Narr/2,mp.P1.full.Narr/2-1,mp.P1.full.Narr)*mp.P2.full.dx/mp.P2.D
    [mp.P2.full.XsDL,mp.P2.full.YsDL] = np.meshgrid(mp.P2.full.xsDL,mp.P2.full.xsDL)
    
    #AJER NOTE DEBUGGING
    mp.full.lambdasMat = np.zeros((1,1))
    mp.full.lambdasMat[0,0] = mp.lambda0
    mp.P4.full.Narr = 196
    mp.P4.full.dx = mp.P2.D/mp.P1.full.Nbeam
    dir_path = os.path.join( os.path.dirname(os.path.realpath(__file__)), "debug" )
    mp.P4.full.croppedMask = fits.getdata(os.path.join(dir_path, 'mpP4fullcroppedMask.fits'),ext=0)
    mp.P1.full.mask = fits.getdata(os.path.join(dir_path, 'mpP1fullmask.fits'),ext=0)
    mp.F3.full.mask.amp = fits.getdata(os.path.join(dir_path, 'mpF3fullmaskamp.fits'),ext=0)
#    mp.P4.full.croppedMask = fits.getdata('/Users/ajriggs/Downloads/mpP4fullcroppedMask.fits', ext=0)
#    mp.P1.full.mask = fits.getdata('/Users/ajriggs/Downloads/mpP1fullmask.fits', ext=0)    
#    mp.F3.full.mask.amp = fits.getdata('/Users/ajriggs/Downloads/mpF3fullmaskamp.fits', ext=0)
        
    #--Set the wavelength
    if(hasattr(modvar,'wvl')): #--For FALCO or for standalone use of full model
        wvl = modvar.wvl;
    elif(hasattr(modvar,'sbpIndex')): #--For use in FALCO
        wvl = mp.full.lambdasMat[modvar.sbpIndex,modvar.wpsbpIndex]
    else:
        log.warning('model_full: Need to specify a value or indices for a wavelength.')  
 
    #""" Input E-fields """ 
    #--Set the point source as the exoplanet or the star
    if modvar.whichSource.lower()=='exoplanet': #--Don't include tip/tilt jitter for planet wavefront since the effect is minor
        #--The planet does not move in sky angle, so the actual tip/tilt angle needs to scale inversely with wavelength.
        #planetAmp = np.sqrt(mp.c_planet);  # Scale the E field to the correct contrast
        #planetPhase = (-1)*(2*np.pi*(mp.x_planet*mp.P2.full.XsDL + mp.y_planet*mp.P2.full.YsDL));
        #Ein = planetAmp*exp(1j*planetPhase*mp.lambda0/wvl)*mp.P1.full.E[:,:,modvar.wpsbpIndex,modvar.sbpIndex];
        pass
    elif modvar.whichSource.lower()=='offaxis': #--Use for throughput calculations 
        TTphase = (-1.)*(2*np.pi*(modvar.x_offset*mp.P2.full.XsDL + modvar.y_offset*mp.P2.full.YsDL));
        Ett = np.exp(1j*TTphase*mp.lambda0/wvl)
        Ein = Ett*np.squeeze(mp.P1.full.E[:,:,modvar.wpsbpIndex,modvar.sbpIndex]) 
        
    else: # Default to using the starlight
        Ein = np.squeeze(mp.P1.full.E[:,:,modvar.wpsbpIndex,modvar.sbpIndex])  

    #--Shift the source off-axis to compute the intensity normalization value.
    #  This replaces the previous way of taking the FPM out in the optical model.
    if(normFac==0):
        source_x_offset = mp.source_x_offset_norm #--source offset in lambda0/D for normalization
        source_y_offset = mp.source_y_offset_norm #--source offset in lambda0/D for normalization
        TTphase = (-1)*(2*np.pi*(source_x_offset*mp.P2.full.XsDL + source_y_offset*mp.P2.full.YsDL));
        Ett = np.exp(1j*TTphase*mp.lambda0/wvl)
        Ein = Ett*np.squeeze(mp.P1.full.E[:,:,modvar.sbpIndex]) 

    #--Apply a Zernike (in amplitude) at input pupil if specified
    if not (hasattr(modvar,'zernIndex')): modvar.zernIndex = 1
    if not modvar.zernIndex==1:
        indsZnoll = modvar.zernIndex #--Just send in 1 Zernike mode
        zernMat = falco_gen_norm_zernike_maps(mp.P1.full.Nbeam,mp.centering,indsZnoll) #--Cube of normalized (RMS = 1) Zernike modes.
        zernMat = falco.utils.padOrCropEven(zernMat,mp.P1.full.Narr)
        Ein = Ein*zernMat*(2*np.pi/wvl)*mp.jac.Zcoef[modvar.zernIndex]

    #--Pre-compute the FPM first for HLC as mp.FPM.mask
    if mp.layout.lower() == 'fourier':
        pass
    elif mp.layout.lower() == 'fpm_scale':
        pass
    
    
    #--Select which optical layout's full model to use.
    if mp.layout.lower() == 'fourier':
        Eout = model_full_Fourier(mp, wvl, Ein, normFac)
    elif mp.layout.lower() == 'fpm_scale': #--FPM scales with wavelength
        Eout = model_full_scale(mp, wvl, Ein, normFac)

    return Eout
 
def model_full_Fourier(mp, wvl, Ein, normFac):
    """
    Truth model used to generate images. Can include aberrations/errors that are unknown 
    to the estimator and controller. This function uses the simplest model (FTs except 
    for angular spectrum between DMs) for several coronagraph types.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    wvl : Wavelength
        Scalar value for the wavelength of the light in meters
    Ein : Electric field input
        2-D electric field in the input pupil
    normFac : Normalization factor
        Value of the PSF peak normalization factor to apply to the whole image

    Returns
    -------
    Eout : array_like
        2-D electric field in final focal plane

    """

    mirrorFac = 2  # Phase change is twice the DM surface height in reflection mode
    NdmPad = int(mp.full.NdmPad)

    """ Masks and DM surfaces """
    if any(mp.dm_ind == 1):
        DM1surf = falco.dms.falco_gen_dm_surf(mp.dm1, mp.dm1.dx, NdmPad)
#        try:
#            DM1surf = falco.utils.padOrCropEven(mp.dm1.surfM, NdmPad)
#        except AttributeError:  # No surfM parameter exists, create DM surface
#            DM1surf = falco.dms.falco_gen_dm_surf(mp.dm1, mp.dm1.dx, NdmPad)
    else:
        DM1surf = np.zeros((NdmPad,NdmPad))

    if any(mp.dm_ind == 2):
        DM2surf = falco.dms.falco_gen_dm_surf(mp.dm2, mp.dm2.dx, NdmPad)
#        try:
#            DM2surf = falco.utils.padOrCropEven(mp.dm2.surfM, NdmPad)
#        except AttributeError:  # No surfM parameter exists, create DM surface
#            DM2surf = falco.dms.falco_gen_dm_surf(mp.dm2, mp.dm2.dx, NdmPad)
    else:
        DM2surf = np.zeros((NdmPad,NdmPad))

    pupil = falco.utils.padOrCropEven(mp.P1.full.mask, NdmPad)
    Ein = falco.utils.padOrCropEven(Ein, NdmPad)

    if mp.flagDM1stop:
        DM1stop = falco.utils.padOrCropEven(mp.dm1.full.mask, NdmPad)
    else:
        DM1stop = np.ones((NdmPad,NdmPad))

    if mp.flagDM2stop:
        DM2stop = falco.utils.padOrCropEven(mp.dm2.full.mask, NdmPad)
    else:
        DM2stop = np.ones((NdmPad,NdmPad))
    
    if(mp.flagDMwfe):
        pass
        #if(any(mp.dm_ind==1));  Edm1WFE = exp(2*np.pi*1j/wvl*padOrCropEven(mp.dm1.wfe,NdmPad,'extrapval',0)); else; Edm1WFE = ones(NdmPad); end
        #if(any(mp.dm_ind==2));  Edm2WFE = exp(2*np.pi*1j/wvl*padOrCropEven(mp.dm2.wfe,NdmPad,'extrapval',0)); else; Edm2WFE = ones(NdmPad); end
    else:
        Edm1WFE = np.ones((NdmPad,NdmPad))
        Edm2WFE = np.ones((NdmPad,NdmPad))
    
    
    """ Propagation: entrance pupil, 2 DMs, (optional) apodizer, vortex FPM, LS, and final focal plane """

    #--Define pupil P1 and Propagate to pupil P2
    EP1 = pupil*Ein #--E-field at pupil plane P1
    EP2 = falco.propcustom.propcustom_relay(EP1,mp.Nrelay1to2,mp.centering) #--Forward propagate to the next pupil plane (P2) by rotating 180 degrees mp.Nrelay1to2 times.

    #--Propagate from P2 to DM1, and apply DM1 surface and aperture stop
    if not ( abs(mp.d_P2_dm1)==0 ): 
        Edm1 = falco.propcustom.propcustom_PTP(EP2,mp.P2.full.dx*NdmPad,wvl,mp.d_P2_dm1) 
    else: 
        Edm1 = EP2   #--E-field arriving at DM1
    Edm1b = Edm1*Edm1WFE*DM1stop*np.exp(mirrorFac*2*np.pi*1j*DM1surf/wvl) #--E-field leaving DM1

    #--Propagate from DM1 to DM2, and apply DM2 surface and aperture stop
    Edm2 = falco.propcustom.propcustom_PTP(Edm1b,mp.P2.full.dx*NdmPad,wvl,mp.d_dm1_dm2)
    Edm2 *= Edm2WFE*DM2stop*np.exp(mirrorFac*2*np.pi*1j*DM2surf/wvl)

    #--Back-propagate to pupil P2
    if(mp.d_P2_dm1 + mp.d_dm1_dm2 == 0):
        EP2eff = Edm2 #--Do nothing if zero distance
    else:
        EP2eff = falco.propcustom.propcustom_PTP(Edm2,mp.P2.full.dx*NdmPad,wvl,-1*(mp.d_dm1_dm2 + mp.d_P2_dm1)) #--Back propagate to pupil P2

    #--Re-image to pupil P3
    EP3 = falco.propcustom.propcustom_relay(EP2eff,mp.Nrelay2to3,mp.centering);

    #--Apply the apodizer mask (if there is one)
    if(mp.flagApod):
        EP3 = mp.P3.full.mask*falco.utils.padOrCropEven(EP3, mp.P3.full.Narr); 
    
    #--Propagations Specific to the Coronagraph Type
    if(mp.coro.upper()=='LC' or mp.coro.upper()=='APLC' or mp.coro.upper()=='RODDIER'):
        #--MFT from apodizer plane to FPM (i.e., P3 to F3)
        EF3inc = falco.propcustom.propcustom_mft_PtoF(EP3, mp.fl,wvl,mp.P2.full.dx,mp.F3.full.dxi,mp.F3.full.Nxi,mp.F3.full.deta,mp.F3.full.Neta,mp.centering)
        # Apply (1-FPM) for Babinet's principle later
        if(mp.coro.upper()=='RODDIER'):
            FPM = mp.F3.full.mask.amp*np.exp(1j*2*np.pi/wvl*(mp.F3.n(wvl)-1)*mp.F3.t*mp.F3.full.mask.phzSupport)
            EF3 = (1.-FPM)*EF3inc #--Apply (1-FPM) for Babinet's principle later
        else:
            EF3 = (1.-mp.F3.full.mask.amp)*EF3inc;
        # Use Babinet's principle at the Lyot plane. This is the term without the FPM.
        EP4noFPM = falco.propcustom.propcustom_relay(EP3,mp.Nrelay3to4,mp.centering) #--Propagate forward another pupil plane 
        #--MFT from FPM to Lyot Plane (i.e., F3 to P4)
        EP4subtrahend = falco.propcustom.propcustom_mft_FtoP(EF3,mp.fl,wvl,mp.F3.full.dxi,mp.F3.full.deta,mp.P4.full.dx,mp.P4.full.Narr,mp.centering) # Subtrahend term for Babinet's principle     
        #--Babinet's principle at P4
        EP4 = falco.utils.padOrCropEven(EP4noFPM,mp.P4.full.Narr) - EP4subtrahend
    else:
        log.warning('The chosen coronagraph type is not included yet.')

    """ Back to common propagation any coronagraph type """
    #--Apply the (cropped-down) Lyot stop
    EP4 *= mp.P4.full.croppedMask

    #--MFT from Lyot Stop to final focal plane (i.e., P4 to Fend)
    EP4 = falco.propcustom.propcustom_relay(EP4,mp.NrelayFend,mp.centering) #--Rotate the final image 180 degrees if necessary
    EFend = falco.propcustom.propcustom_mft_PtoF(EP4,mp.fl,wvl,mp.P4.full.dx,mp.Fend.dxi,mp.Fend.Nxi,mp.Fend.deta,mp.Fend.Neta)

    #--Don't apply FPM if normalization value is being found
    if(normFac==0):
        Eout = EFend #--Don't normalize if normalization value is being found
    else:
        Eout = EFend/np.sqrt(normFac) #--Apply normalization

#    # NOTE: DEBUGGING
#    hduFPM = fits.PrimaryHDU(mp.F3.full.mask.amp)
#    hduFPM.writeto('/Users/ajriggs/Downloads/FPM_full.fits', overwrite=True)

    return Eout
    
     
def model_full_scale(mp, wvl, Ein, normFac):   
    pass  
     
        
def model_compact(mp,modvar,**kwargs):
    """
    Simplified (aka compact) mdoel used by estimator and controller. Does not include 
    unknown aberrations of the full, "truth" model. This function is the wrapper for 
    compact models of any coronagraph type.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    modvar : ModelVariables
        Structure containing temporary optical model variables

    Returns
    -------
    Eout : array_like
        2-D electric field in final focal plane
        
    Other Parameters
    ----------------
    GETNORM : bool
        If set, the PSF normalization factor will be computed and assigned.
    NORMOFF :  bool
        If set, the PSF normalization factor is set to 1. This is useful for a standalone 
        call to model_full.
    EVAL : bool
        If set, uses a higher resolution in the focal plane for measuring performance.

    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
        
    modvar.wpsbpIndex = 1000 #--Dummy index since not needed in compact model

    # Set default values of input parameters
    normFac = mp.Fend.compact.I00[modvar.sbpIndex] # Value to normalize the PSF. Set to 0 when finding the normalization factor
    flagEval = False # flag to use a different (usually higher) resolution at final focal plane for evaluation
    flagNewNorm = False

    #--Optional Keyword arguments
    if ("GETNORM" in kwargs and kwargs["GETNORM"]):
        normFac = 0.
        flagNewNorm = True
    if ("NORMOFF" in kwargs and kwargs["NORMOFF"]):
        normFac = 1.
    if ("EVAL" in kwargs and kwargs["EVAL"]):
        flagEval = True
        

    #--Normalization factor for compact evaluation model
    if( (flagNewNorm==False) and (flagEval==True) ):
        normFac = mp.Fend.eval.I00[modvar.sbpIndex] # Value to normalize the PSF. Set to 0 when finding the normalization factor

    #--Set the wavelength
    if(hasattr(modvar,'wvl')):
        wvl = modvar.wvl
    else:
        wvl = mp.sbp_centers[modvar.sbpIndex]
        
    """ Input E-fields """ 

    #--Include the tip/tilt in the input wavefront
#    if(hasattr(mp,'ttx')):
#         %--Scale by wvl/lambda0 because ttx and tty are in lambda0/D
#         x_offset = mp.ttx(modvar.ttIndex)*(mp.lambda0/wvl);
#         y_offset = mp.tty(modvar.ttIndex)*(mp.lambda0/wvl);
# 
#         TTphase = (-1)*(2*np.pi*(x_offset*mp.P2.compact.XsDL + y_offset*mp.P2.compact.YsDL));
#         Ett = exp(1j*TTphase*mp.lambda0/wvl);
#         Ein = Ett.*mp.P1.compact.E(:,:,modvar.sbpIndex)
    if modvar.whichSource.lower()=='offaxis': #--Use for throughput calculations 
        TTphase = (-1)*(2*np.pi*(modvar.x_offset*mp.P2.compact.XsDL + modvar.y_offset*mp.P2.compact.YsDL))
        Ett = np.exp(1j*TTphase*mp.lambda0/wvl)
        Ein = Ett*mp.P1.compact.E[:,:,modvar.sbpIndex]
    else: #--Backward compatible with code without tip/tilt offsets in the Jacobian
        Ein = mp.P1.compact.E[:,:,modvar.sbpIndex]  
        
    #AJER NOTE: This should be in falco_config_gen_chosen_pupil
    mp.P2.compact.xsDL = np.linspace(-mp.P1.compact.Narr/2,mp.P1.compact.Narr/2-1,mp.P1.compact.Narr)*mp.P2.compact.dx/mp.P2.D
    [mp.P2.compact.XsDL,mp.P2.compact.YsDL] = np.meshgrid(mp.P2.compact.xsDL,mp.P2.compact.xsDL)
    #mp.P2.compact.xsDL = (-mp.P1.compact.Narr/2:(mp.P1.compact.Narr/2-1))*mp.P2.compact.dx/mp.P2.D;
    #[mp.P2.compact.XsDL,mp.P2.compact.YsDL] = meshgrid(mp.P2.compact.xsDL)

    #--Shift the source off-axis to compute the intensity normalization value.
    #  This replaces the previous way of taking the FPM out in the optical model.
    if(normFac==0):
        source_x_offset = mp.source_x_offset_norm #--source offset in lambda0/D for normalization
        source_y_offset = mp.source_y_offset_norm #--source offset in lambda0/D for normalization
        TTphase = (-1.)*(2*np.pi*(source_x_offset*mp.P2.compact.XsDL + source_y_offset*mp.P2.compact.YsDL))
        Ett = np.exp(1j*TTphase*mp.lambda0/wvl);
        Ein = Ett*mp.P1.compact.E[:,:,modvar.sbpIndex]; 


    #--Apply a Zernike (in amplitude) at input pupil if specified
    if not hasattr(modvar,'zernIndex'):
        modvar.zernIndex = 1

    #--Only used for Zernike sensitivity control, which requires the perfect 
    # E-field of the differential Zernike term.
    if not (modvar.zernIndex==1):
        indsZnoll = modvar.zernIndex #--Just send in 1 Zernike mode
        zernMat = falco_gen_norm_zernike_maps(mp.P1.compact.Nbeam,mp.centering,indsZnoll) #--Cube of normalized (RMS = 1) Zernike modes.
        zernMat = falco.utils.padOrCropEven(zernMat,mp.P1.compact.Narr)
        indArr = mp.jac.zerns==modvar.zernIndex
        ind = indArr[0].item()
        Ein = Ein*zernMat*(2*np.pi*1j/wvl)*mp.jac.Zcoef[ind]
      
    #--Select which optical layout's compact model to use and get the output E-field
    if mp.layout.lower()=='fourier':
        Eout = model_compact_general(mp, wvl, Ein, normFac, flagEval);
    elif mp.layout.lower()=='fpm_scale':
        if mp.coro.upper()=='HLC':
            Eout = model_compact_scale(mp, wvl, Ein, normFac, flagEval)

    return Eout


def model_compact_general(mp, wvl, Ein, normFac, flagEval):
    """
    Simplified (aka compact) mdoel used by estimator and controller. Does not 
    include unknown aberrations of the full, "truth" model. This has a general 
    optical layout that should work for most applications

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    wvl : float
        Wavelength of light [meters]
    Ein : array_like
        2-D input electric field
    normFac : float
        Intensity normalization factor
    flagEval : bool
        Flag whether to use a higher resolution in final image plane for evaluation

    Returns
    -------
    Eout : array_like
        2-D electric field in final focal plane

    """
    
    #AJER NOTE DEBUGGING
    mp.P4.compact.Narr = 196
    mp.P4.compact.dx = mp.P2.D/mp.P1.compact.Nbeam
    dir_path = os.path.join( os.path.dirname(os.path.realpath(__file__)), "debug" )
    mp.P4.compact.croppedMask = fits.getdata(os.path.join(dir_path, 'mpP4compactcroppedMask.fits'),ext=0)
    mp.P1.compact.mask = fits.getdata(os.path.join(dir_path, 'mpP1compactmask.fits'),ext=0)
    mp.F3.compact.mask.amp = fits.getdata(os.path.join(dir_path, 'mpF3compactmaskamp.fits'),ext=0)
#    mp.P4.compact.croppedMask = fits.getdata('/Users/ajriggs/Downloads/mpP4compactcroppedMask.fits', ext=0)
#    mp.P1.compact.mask = fits.getdata('/Users/ajriggs/Downloads/mpP1compactmask.fits', ext=0)
#    mp.F3.compact.mask.amp = fits.getdata('/Users/ajriggs/Downloads/mpF3compactmaskamp.fits', ext=0)
    
    mirrorFac = 2. # Phase change is twice the DM surface height.
    NdmPad = int(mp.compact.NdmPad)

    if(flagEval): #--Higher resolution at final focal plane for computing stats such as throughput
        dxi = mp.Fend.eval.dxi
        Nxi = mp.Fend.eval.Nxi
        deta = mp.Fend.eval.deta
        Neta = mp.Fend.eval.Neta 
    else: #--Otherwise use the detector resolution
        dxi = mp.Fend.dxi
        Nxi = mp.Fend.Nxi
        deta = mp.Fend.deta
        Neta = mp.Fend.Neta 

    """ Masks and DM surfaces """
    #--Compute the DM surfaces for the current DM commands
    if(any(mp.dm_ind==1)): 
        DM1surf = falco.dms.falco_gen_dm_surf(mp.dm1, mp.dm1.compact.dx, NdmPad) 
    else: 
        DM1surf = np.zeros((NdmPad,NdmPad)) #--Pre-compute the starting DM1 surface
    if(any(mp.dm_ind==2)): 
        DM2surf = falco.dms.falco_gen_dm_surf(mp.dm2, mp.dm2.compact.dx, NdmPad) 
    else:
        DM2surf = np.zeros((NdmPad,NdmPad)) #--Pre-compute the starting DM2 surface

    pupil = falco.utils.padOrCropEven(mp.P1.compact.mask,NdmPad)
    Ein = falco.utils.padOrCropEven(Ein,NdmPad)

    if(mp.flagDM1stop):
        DM1stop = falco.utils.padOrCropEven(mp.dm1.compact.mask, NdmPad) 
    else: 
        DM1stop = np.ones((NdmPad,NdmPad))
    if(mp.flagDM2stop):
        DM2stop = falco.utils.padOrCropEven(mp.dm2.compact.mask, NdmPad) 
    else: 
        DM2stop = np.ones((NdmPad,NdmPad))

#     DM1stop = falco.utils.padOrCropEven(mp.dm1.compact.mask, NdmPad) if(mp.flagDM1stop) else DM1stop = ones((NdmPad,NdmPad))
#     DM2stop = falco.utils.padOrCropEven(mp.dm2.compact.mask, NdmPad) if(mp.flagDM2stop) else DM2stop = ones((NdmPad,NdmPad))

    if(mp.useGPU):
        log.warning('GPU support not yet implemented. Proceeding without GPU.')
        
    #--This block is for BMC surface error testing
    if(mp.flagDMwfe): # if(mp.flagDMwfe && (mp.P1.full.Nbeam==mp.P1.compact.Nbeam))
        if(any(mp.dm_ind==1)):
            Edm1WFE = np.exp(2*np.pi*1j/wvl*falco.utils.padOrCropEven(mp.dm1.compact.wfe,NdmPad,'extrapval',0)) 
        else: 
            Edm1WFE = np.ones((NdmPad,NdmPad))
        if(any(mp.dm_ind==2)):
            Edm2WFE = np.exp(2*np.pi*1j/wvl*falco.utils.padOrCropEven(mp.dm2.compact.wfe,NdmPad,'extrapval',0)) 
        else: 
            Edm2WFE = np.ones((NdmPad,NdmPad))
    else:
        Edm1WFE = np.ones((NdmPad,NdmPad))
        Edm2WFE = np.ones((NdmPad,NdmPad))
        
    """Propagation"""

    #--Define pupil P1 and Propagate to pupil P2
    EP1 = pupil*Ein #--E-field at pupil plane P1
    EP2 = falco.propcustom.propcustom_relay(EP1,mp.Nrelay1to2,mp.centering) #--Forward propagate to the next pupil plane (P2) by rotating 180 degrees mp.Nrelay1to2 times.

    #--Propagate from P2 to DM1, and apply DM1 surface and aperture stop
    if not (abs(mp.d_P2_dm1)==0): #--E-field arriving at DM1
        Edm1 = falco.propcustom.propcustom_PTP(EP2,mp.P2.compact.dx*NdmPad,wvl,mp.d_P2_dm1)
    else:
        Edm1 = EP2
    Edm1b = Edm1*Edm1WFE*DM1stop*np.exp(mirrorFac*2*np.pi*1j*DM1surf/wvl) #--E-field leaving DM1

    #--Propagate from DM1 to DM2, and apply DM2 surface and aperture stop
    Edm2 = falco.propcustom.propcustom_PTP(Edm1b,mp.P2.compact.dx*NdmPad,wvl,mp.d_dm1_dm2); 
    Edm2 *= Edm2WFE*DM2stop*np.exp(mirrorFac*2*np.pi*1j*DM2surf/wvl)

    #--Back-propagate to pupil P2
    if(mp.d_P2_dm1 + mp.d_dm1_dm2 == 0):
        EP2eff = Edm2
    else:
        EP2eff = falco.propcustom.propcustom_PTP(Edm2,mp.P2.compact.dx*NdmPad,wvl,-1*(mp.d_dm1_dm2 + mp.d_P2_dm1))

    #--Re-image to pupil P3
    EP3 = falco.propcustom.propcustom_relay(EP2eff,mp.Nrelay2to3,mp.centering)

    #--Apply apodizer mask.
    if(mp.flagApod):
        EP3 = mp.P3.compact.mask*falco.utils.padOrCropEven(EP3, mp.P3.compact.Narr); 

    """  Select propagation based on coronagraph type   """
    if mp.coro.upper()=='LC' or mp.coro.upper()=='APLC' or mp.coro.upper()=='RODDIER':
        #--MFT from SP to FPM (i.e., P3 to F3)
        EF3inc = falco.propcustom.propcustom_mft_PtoF(EP3, mp.fl,wvl,mp.P2.compact.dx,mp.F3.compact.dxi,mp.F3.compact.Nxi,mp.F3.compact.deta,mp.F3.compact.Neta,mp.centering) #--E-field incident upon the FPM
        #--Apply (1-FPM) for Babinet's principle later
        if(mp.coro.upper()=='RODDIER'):
            pass
            #FPM = mp.F3.compact.mask.amp*exp(1j*2.*np.pi/wvl*(mp.F3.n(wvl)-1)*mp.F3.t*mp.F3.compact.mask.phzSupport);
            #EF3 = (1.-FPM).*EF3inc; #--Apply (1-FPM) for Babinet's principle later
        else:
            EF3 = (1. - mp.F3.compact.mask.amp)*EF3inc;
        #--Use Babinet's principle at the Lyot plane.
        EP4noFPM = falco.propcustom.propcustom_relay(EP3,mp.Nrelay3to4,mp.centering) #--Propagate forward another pupil plane 
        EP4noFPM = falco.utils.padOrCropEven(EP4noFPM,mp.P4.compact.Narr) #--Crop down to the size of the Lyot stop opening
        #--MFT from FPM to Lyot Plane (i.e., F3 to P4)
        EP4sub = falco.propcustom.propcustom_mft_FtoP(EF3,mp.fl,wvl,mp.F3.compact.dxi,mp.F3.compact.deta,mp.P4.compact.dx,mp.P4.compact.Narr,mp.centering) # Subtrahend term for Babinet's principle     
        EP4subRelay = falco.propcustom.propcustom_relay(EP4sub,mp.Nrelay3to4-1,mp.centering) #--Propagate forward more pupil planes if necessary.
        #--Babinet's principle at P4
        EP4 = (EP4noFPM-EP4subRelay);

    """  Back to common propagation any coronagraph type   """
    #--Apply the Lyot stop
    EP4 = mp.P4.compact.croppedMask*EP4

    #--MFT to camera
    EP4 = falco.propcustom.propcustom_relay(EP4,mp.NrelayFend,mp.centering) #--Rotate the final image 180 degrees if necessary
    EFend = falco.propcustom.propcustom_mft_PtoF(EP4,mp.fl,wvl,mp.P4.compact.dx,dxi,Nxi,deta,Neta,mp.centering);

    #--Don't apply FPM if normalization value is being found
    if(normFac==0):
        Eout = EFend #--Don't normalize if normalization value is being found
    else:
        Eout = EFend/np.sqrt(normFac) #--Apply normalization

#    # NOTE: DEBUGGING
#    hduR = fits.PrimaryHDU(np.real(EFend))
#    hduI = fits.PrimaryHDU(np.imag(EFend))
#    hduR.writeto('/Users/ajriggs/Downloads/EFend_real_compact.fits', overwrite=True)
#    hduI.writeto('/Users/ajriggs/Downloads/EFend_imag_compact.fits', overwrite=True)
    
#    # NOTE: DEBUGGING
#    hdu = fits.PrimaryHDU(Mrot)
#    hdu.writeto('/Users/ajriggs/Downloads/Mrot_python.fits', overwrite=True)
    
    return Eout


def model_Jacobian(mp):
    
    jacStruct = falco.config.EmptyObject() #--Initialize the new structure
    
    #--Pre-compute the DM surfaces to save time
    NdmPad = int(mp.compact.NdmPad)
    if(any(mp.dm_ind==1)): 
        mp.dm1.compact.surfM = falco.dms.falco_gen_dm_surf(mp.dm1, mp.dm1.compact.dx, NdmPad) 
    else: 
        mp.dm1.compact.surfM = np.zeros((NdmPad,NdmPad)) #--Pre-compute the starting DM1 surface
    if(any(mp.dm_ind==2)): 
        mp.dm2.compact.surfM = falco.dms.falco_gen_dm_surf(mp.dm2, mp.dm2.compact.dx, NdmPad) 
    else:
        mp.dm2.compact.surfM = np.zeros((NdmPad,NdmPad)) #--Pre-compute the starting DM2 surface

    #--Initialize the Jacobians for each DM
    if(any(mp.dm_ind==1)): jacStruct.G1 = np.zeros((mp.Fend.corr.Npix,mp.dm1.Nele,mp.jac.Nmode),dtype=np.complex)
    if(any(mp.dm_ind==2)): jacStruct.G2 = np.zeros((mp.Fend.corr.Npix,mp.dm2.Nele,mp.jac.Nmode),dtype=np.complex)
    if(any(mp.dm_ind==8)): jacStruct.G8 = np.zeros((mp.Fend.corr.Npix,mp.dm8.Nele,mp.jac.Nmode),dtype=np.complex)  
    if(any(mp.dm_ind==9)): jacStruct.G9 = np.zeros((mp.Fend.corr.Npix,mp.dm9.Nele,mp.jac.Nmode),dtype=np.complex)

    print('Computing control Jacobian matrices:\n  ',end='')
    with falco.utils.TicToc():
        for im in range(mp.jac.Nmode):
            if(any(mp.dm_ind==1)):
                print('mode%ddm%d...' % (im,1),end='')
                jacStruct.G1[:,:,im] =  model_Jacobian_middle_layer(mp, im, 1)
            if(any(mp.dm_ind==2)):
                print('mode%ddm%d...' % (im,2),end='')
                jacStruct.G2[:,:,im] =  model_Jacobian_middle_layer(mp, im, 2)
            print('done.')
    return jacStruct


def model_Jacobian_middle_layer(mp,im,idm):
    
    #%--Select which optical layout's Jacobian model to use and get the output E-field
    if(mp.layout.lower()=='fourier'):
        if (mp.coro.upper()=='LC') or (mp.coro.upper()=='APLC'): #--DMs, optional apodizer, occulting spot FPM, and LS.
            jacMode = model_Jacobian_LC(mp, im, idm) 
            
    return jacMode


def model_Jacobian_LC(mp,im,idm):
    """
    Simplified (aka compact) mdoel used by estimator and controller. Does not 
    include unknown aberrations of the full, "truth" model. This has a general 
    optical layout that should work for most applications

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    wvl : float
        Wavelength of light [meters]
    Ein : array_like
        2-D input electric field
    normFac : float
        Intensity normalization factor
    flagEval : bool
        Flag whether to use a higher resolution in final image plane for evaluation

    Returns
    -------
    Eout : array_like
        2-D electric field in final focal plane

    """
    
    #AJER NOTE DEBUGGING: Hard-coded values until these inputs are initialized correctly
    mp.P4.compact.Narr = 196
    mp.P4.compact.dx = mp.P2.D/mp.P1.compact.Nbeam
    dir_path = os.path.join( os.path.dirname(os.path.realpath(__file__)), "debug" )
    mp.P4.compact.croppedMask = fits.getdata(os.path.join(dir_path, 'mpP4compactcroppedMask.fits'),ext=0)
    mp.P1.compact.mask = fits.getdata(os.path.join(dir_path, 'mpP1compactmask.fits'),ext=0)
    mp.F3.compact.mask.amp = fits.getdata(os.path.join(dir_path, 'mpF3compactmaskamp.fits'),ext=0)
#    mp.P4.compact.croppedMask = fits.getdata('/Users/ajriggs/Downloads/mpP4compactcroppedMask.fits', ext=0)
#    mp.P1.compact.mask = fits.getdata('/Users/ajriggs/Downloads/mpP1compactmask.fits', ext=0)
#    mp.F3.compact.mask.amp = fits.getdata('/Users/ajriggs/Downloads/mpF3compactmaskamp.fits', ext=0)
    
    
    modvar = falco.config.EmptyObject() #--Initialize the new structure
    modvar.sbpIndex = mp.jac.sbp_inds[im]
    modvar.zernIndex = mp.jac.zern_inds[im]
    
    wvl = mp.sbp_centers[modvar.sbpIndex]
    mirrorFac = 2. # Phase change is twice the DM surface height.
    NdmPad = int(mp.compact.NdmPad)

    """Input E-fields"""    
    Ein = np.squeeze(mp.P1.compact.E[:,:,modvar.sbpIndex])  
    #--Apply a Zernike (in amplitude) at input pupil
    #--Used only for Zernike sensitivity control, which requires the perfect 
    # E-field of the differential Zernike term.
    if not (modvar.zernIndex==1):
        indsZnoll = modvar.zernIndex #--Just send in 1 Zernike mode
        zernMat = falco_gen_norm_zernike_maps(mp.P1.compact.Nbeam,mp.centering,indsZnoll) #--Cube of normalized (RMS = 1) Zernike modes.
        zernMat = falco.utils.padOrCropEven(zernMat,mp.P1.compact.Narr)
        Ein = Ein*zernMat*(2*np.pi/wvl)*mp.jac.Zcoef(mp.jac.zerns==modvar.zernIndex)
    
    """ Masks and DM surfaces """
    pupil = falco.utils.padOrCropEven(mp.P1.compact.mask,NdmPad)
    Ein = falco.utils.padOrCropEven(Ein,NdmPad)
    
    #--Re-image the apodizer from pupil P3 back to pupil P2. (Sign of mp.Nrelay2to3 doesn't matter.)
    if(mp.flagApod):
        apodReimaged = falco.utils.padOrCropEven(mp.P3.compact.mask, NdmPad)
        apodReimaged = falco.propcustom.propcustom_relay(apodReimaged,mp.Nrelay2to3,mp.centering)
    else:
        apodReimaged = np.ones((NdmPad,NdmPad)) 
    
    #--Compute the DM surfaces for the current DM commands
    if(any(mp.dm_ind==1)): 
        DM1surf = falco.utils.padOrCropEven(mp.dm1.compact.surfM, NdmPad)
        #DM1surf = falco.dms.falco_gen_dm_surf(mp.dm1, mp.dm1.compact.dx, NdmPad) 
    else: 
        DM1surf = np.zeros((NdmPad,NdmPad)) #--Pre-compute the starting DM1 surface
    if(any(mp.dm_ind==2)): 
        DM2surf = falco.utils.padOrCropEven(mp.dm2.compact.surfM, NdmPad)
        #DM2surf = falco.dms.falco_gen_dm_surf(mp.dm2, mp.dm2.compact.dx, NdmPad) 
    else:
        DM2surf = np.zeros((NdmPad,NdmPad)) #--Pre-compute the starting DM2 surface

    if(mp.flagDM1stop):
        DM1stop = falco.utils.padOrCropEven(mp.dm1.compact.mask, NdmPad) 
    else: 
        DM1stop = np.ones((NdmPad,NdmPad))
    if(mp.flagDM2stop):
        DM2stop = falco.utils.padOrCropEven(mp.dm2.compact.mask, NdmPad) 
    else: 
        DM2stop = np.ones((NdmPad,NdmPad))

    if(mp.useGPU):
        log.warning('GPU support not yet implemented. Proceeding without GPU.')
        
    #--This block is for BMC surface error testing
    if(mp.flagDMwfe): # if(mp.flagDMwfe && (mp.P1.full.Nbeam==mp.P1.compact.Nbeam))
        if(any(mp.dm_ind==1)):
            Edm1WFE = np.exp(2*np.pi*1j/wvl*falco.utils.padOrCropEven(mp.dm1.compact.wfe,NdmPad,'extrapval',0)) 
        else: 
            Edm1WFE = np.ones((NdmPad,NdmPad))
        if(any(mp.dm_ind==2)):
            Edm2WFE = np.exp(2*np.pi*1j/wvl*falco.utils.padOrCropEven(mp.dm2.compact.wfe,NdmPad,'extrapval',0)) 
        else: 
            Edm2WFE = np.ones((NdmPad,NdmPad))
    else:
        Edm1WFE = np.ones((NdmPad,NdmPad))
        Edm2WFE = np.ones((NdmPad,NdmPad))
        
    """Propagation"""

    #--Define pupil P1 and Propagate to pupil P2
    EP1 = pupil*Ein #--E-field at pupil plane P1
    EP2 = falco.propcustom.propcustom_relay(EP1,mp.Nrelay1to2,mp.centering) #--Forward propagate to the next pupil plane (P2) by rotating 180 degrees mp.Nrelay1to2 times.

    #--Propagate from P2 to DM1, and apply DM1 surface and aperture stop
    if not (abs(mp.d_P2_dm1)==0): #--E-field arriving at DM1
        Edm1 = falco.propcustom.propcustom_PTP(EP2,mp.P2.compact.dx*NdmPad,wvl,mp.d_P2_dm1)
    else:
        Edm1 = EP2
    Edm1b = Edm1*Edm1WFE*DM1stop*np.exp(mirrorFac*2*np.pi*1j*DM1surf/wvl) #--E-field leaving DM1

    """ ---------- DM1 ---------- """
    if(idm==1):
        Gzdl = np.zeros((mp.Fend.corr.Npix,mp.dm1.Nele),dtype=np.complex)
        
        #--Two array sizes (at same resolution) of influence functions for MFT and angular spectrum
        NboxPad1AS = int(mp.dm1.compact.NboxAS) #--array size for FFT-AS propagations from DM1->DM2->DM1
        mp.dm1.compact.xy_box_lowerLeft_AS = mp.dm1.compact.xy_box_lowerLeft - (mp.dm1.compact.NboxAS-mp.dm1.compact.Nbox)/2. #--Adjust the sub-array location of the influence function for the added zero padding
    
        if(any(mp.dm_ind==2)):
            DM2surf = falco.utils.padOrCropEven(DM2surf,mp.dm1.compact.NdmPad)  
        else:
            DM2surf = np.zeros((mp.dm1.compact.NdmPad,mp.dm1.compact.NdmPad)) 
        if(mp.flagDM2stop):
            DM2stop = falco.utils.padOrCropEven(DM2stop,mp.dm1.compact.NdmPad) 
        else:
            DM2stop = np.ones((mp.dm1.compact.NdmPad))
        apodReimaged = falco.utils.padOrCropEven( apodReimaged, mp.dm1.compact.NdmPad)
    
        Edm1pad = falco.utils.padOrCropEven(Edm1b,mp.dm1.compact.NdmPad) #--Pad or crop for expected sub-array indexing
        Edm2WFEpad = falco.utils.padOrCropEven(Edm2WFE,mp.dm1.compact.NdmPad) #--Pad or crop for expected sub-array indexing
    
        #--Propagate each actuator from DM1 through the optical system
        Gindex = 0 #1  initialize index counter
        for iact in mp.dm1.act_ele: # np.array([405]): # np.array([1665]): #
            if( np.sum(np.abs(mp.dm1.compact.inf_datacube[:,:,iact]))>1e-12 ):  #--Compute only for influence functions that are not zeroed out
                
                #--x- and y- coordinate indices of the padded influence function in the full padded pupil
                x_box_AS_ind = np.arange(mp.dm1.compact.xy_box_lowerLeft_AS[0,iact],mp.dm1.compact.xy_box_lowerLeft_AS[0,iact]+NboxPad1AS,dtype=np.int) # x-indices in pupil arrays for the box
                y_box_AS_ind = np.arange(mp.dm1.compact.xy_box_lowerLeft_AS[1,iact],mp.dm1.compact.xy_box_lowerLeft_AS[1,iact]+NboxPad1AS,dtype=np.int) # y-indices in pupil arrays for the box
                #--x- and y- coordinates of the UN-padded influence function in the full padded pupil
                x_box = mp.dm1.compact.x_pupPad[x_box_AS_ind] # full pupil x-coordinates of the box 
                y_box = mp.dm1.compact.y_pupPad[y_box_AS_ind] # full pupil y-coordinates of the box
                
                #--Propagate from DM1 to DM2, and then back to P2
                dEbox = (mirrorFac*2*np.pi*1j/wvl)*falco.utils.padOrCropEven((mp.dm1.VtoH.reshape(mp.dm1.Nact**2)[iact])*np.squeeze(mp.dm1.compact.inf_datacube[:,:,iact]),NboxPad1AS) #--Pad influence function at DM1 for angular spectrum propagation.
                dEbox = falco.propcustom.propcustom_PTP(dEbox*Edm1pad[np.ix_(y_box_AS_ind,x_box_AS_ind)],mp.P2.compact.dx*NboxPad1AS,wvl,mp.d_dm1_dm2) # forward propagate to DM2 and apply DM2 E-field
                dEP2box = falco.propcustom.propcustom_PTP(dEbox*Edm2WFEpad[np.ix_(y_box_AS_ind,x_box_AS_ind)]*DM2stop[np.ix_(y_box_AS_ind,x_box_AS_ind)]*np.exp(mirrorFac*2*np.pi*1j/wvl*DM2surf[np.ix_(y_box_AS_ind,x_box_AS_ind)]),mp.P2.compact.dx*NboxPad1AS,wvl,-1*(mp.d_dm1_dm2 + mp.d_P2_dm1) ) # back-propagate to DM1
#                dEbox = falco.propcustom.propcustom_PTP_inf_func(dEbox*Edm1pad[np.ix_(y_box_AS_ind,x_box_AS_ind)],mp.P2.compact.dx*NboxPad1AS,wvl,mp.d_dm1_dm2,mp.dm1.dm_spacing,mp.propMethodPTP) # forward propagate to DM2 and apply DM2 E-field
#                dEP2box = falco.propcustom.propcustom_PTP_inf_func(dEbox.*Edm2WFEpad[np.ix_(y_box_AS_ind,x_box_AS_ind)]*DM2stop(y_box_AS_ind,x_box_AS_ind).*exp(mirrorFac*2*np.pi*1j/wvl*DM2surf(y_box_AS_ind,x_box_AS_ind)),mp.P2.compact.dx*NboxPad1AS,wvl,-1*(mp.d_dm1_dm2 + mp.d_P2_dm1),mp.dm1.dm_spacing,mp.propMethodPTP ) # back-propagate to DM1
#                
                #--To simulate going forward to the next pupil plane (with the apodizer) most efficiently, 
                # First, back-propagate the apodizer (by rotating 180-degrees) to the previous pupil.
                # Second, negate the coordinates of the box used.
                dEP2box = apodReimaged[np.ix_(y_box_AS_ind,x_box_AS_ind)]*dEP2box #--Apply 180deg-rotated SP mask.
                dEP3box = np.rot90(dEP2box,k=2*mp.Nrelay2to3) #--Forward propagate the cropped box by rotating 180 degrees mp.Nrelay2to3 times.
                #--Negate and reverse coordinate values to effectively rotate by 180 degrees. No change if 360 degree rotation.
                if np.mod(mp.Nrelay2to3,2)==1: 
                    x_box = -1*x_box[::-1]
                    y_box = -1*y_box[::-1]
               
                #--Matrices for the MFT from the pupil P3 to the focal plane mask
                rect_mat_pre = (np.exp(-2*np.pi*1j*(mp.F3.compact.etas.reshape(mp.F3.compact.etas.size,1)@y_box.reshape(1,y_box.size))/(wvl*mp.fl)))*np.sqrt(mp.P2.compact.dx*mp.P2.compact.dx)*np.sqrt(mp.F3.compact.dxi*mp.F3.compact.deta)/(wvl*mp.fl)
                rect_mat_post  = (np.exp(-2*np.pi*1j*(x_box.reshape(x_box.size,1)@mp.F3.compact.xis.reshape(1,mp.F3.compact.xis.size))/(wvl*mp.fl)))
    
                #--MFT from pupil P3 to FPM
                EF3 = rect_mat_pre@dEP3box@rect_mat_post; # MFT to FPM
                EF3 = (1.-mp.F3.compact.mask.amp)*EF3; #--Propagate through (1-complex FPM) for Babinet's principle
    
                #--MFT to LS ("Sub" name for Subtrahend part of the Lyot-plane E-field)
                EP4sub = falco.propcustom.propcustom_mft_FtoP(EF3,mp.fl,wvl,mp.F3.compact.dxi,mp.F3.compact.deta,mp.P4.compact.dx,mp.P4.compact.Narr,mp.centering)  #--Subtrahend term for the Lyot plane E-field    
                EP4sub = falco.propcustom.propcustom_relay(EP4sub,mp.Nrelay3to4-1,mp.centering); #--Get the correct orientation
                
                
                #--Full Lyot plane pupil (for Babinet)
                EP4noFPM = np.zeros((mp.dm1.compact.NdmPad,mp.dm1.compact.NdmPad),dtype=np.complex)
                EP4noFPM[np.ix_(y_box_AS_ind,x_box_AS_ind)] = dEP2box #--Propagating the E-field from P2 to P4 without masks gives the same E-field. 
                EP4noFPM = falco.propcustom.propcustom_relay(EP4noFPM,mp.Nrelay2to3+mp.Nrelay3to4,mp.centering) #--Get the correct orientation 
                EP4noFPM = falco.utils.padOrCropEven(EP4noFPM,mp.P4.compact.Narr) #--Crop down to the size of the Lyot stop opening
                EP4 = mp.P4.compact.croppedMask*(EP4noFPM - EP4sub) #--Babinet's principle to get E-field at Lyot plane
    
                #--MFT to camera
                EP4 = falco.propcustom.propcustom_relay(EP4,mp.NrelayFend,mp.centering) #--Rotate the final image 180 degrees if necessary
                EFend = falco.propcustom.propcustom_mft_PtoF(EP4,mp.fl,wvl,mp.P4.compact.dx,mp.Fend.dxi,mp.Fend.Nxi,mp.Fend.deta,mp.Fend.Neta,mp.centering)
                
                Gzdl[:,Gindex] = EFend[mp.Fend.corr.maskBool]/np.sqrt(mp.Fend.compact.I00[modvar.sbpIndex])

            Gindex += 1

    """ ---------- DM2 ---------- """
    if(idm==2):
        Gzdl = np.zeros((mp.Fend.corr.Npix,mp.dm2.Nele),dtype=np.complex)
        
        #--Two array sizes (at same resolution) of influence functions for MFT and angular spectrum
        NboxPad2AS = int(mp.dm2.compact.NboxAS)
        mp.dm2.compact.xy_box_lowerLeft_AS = mp.dm2.compact.xy_box_lowerLeft - (NboxPad2AS-mp.dm2.compact.Nbox)/2 #--Account for the padding of the influence function boxes
        
        apodReimaged = falco.utils.padOrCropEven( apodReimaged, mp.dm2.compact.NdmPad)
        DM2stopPad = falco.utils.padOrCropEven(DM2stop,mp.dm2.compact.NdmPad)
        Edm2WFEpad = falco.utils.padOrCropEven(Edm2WFE,mp.dm2.compact.NdmPad)
    
        #--Propagate full field to DM2 before back-propagating in small boxes
        Edm2inc = falco.utils.padOrCropEven( falco.propcustom.propcustom_PTP(Edm1b,mp.compact.NdmPad*mp.P2.compact.dx,wvl,mp.d_dm1_dm2), mp.dm2.compact.NdmPad) # E-field incident upon DM2
        Edm2inc = falco.utils.padOrCropEven(Edm2inc,mp.dm2.compact.NdmPad);
        Edm2 = DM2stopPad*Edm2WFEpad*Edm2inc*np.exp(mirrorFac*2*np.pi*1j/wvl*falco.utils.padOrCropEven(DM2surf,mp.dm2.compact.NdmPad)) # Initial E-field at DM2 including its own phase contribution
        
        #--Propagate each actuator from DM2 through the rest of the optical system
        Gindex = 0 # initialize index counter
        for iact in mp.dm2.act_ele:
            if( np.sum(np.abs(mp.dm2.compact.inf_datacube[:,:,iact]))>1e-12 ):  #--Only compute for acutators specified for use or for influence functions that are not zeroed out
    
                #--x- and y- coordinates of the padded influence function in the full padded pupil
                x_box_AS_ind = np.arange(mp.dm2.compact.xy_box_lowerLeft_AS[0,iact],mp.dm2.compact.xy_box_lowerLeft_AS[0,iact]+NboxPad2AS,dtype=np.int) # x-indices in pupil arrays for the box
                y_box_AS_ind = np.arange(mp.dm2.compact.xy_box_lowerLeft_AS[1,iact],mp.dm2.compact.xy_box_lowerLeft_AS[1,iact]+NboxPad2AS,dtype=np.int) # y-indices in pupil arrays for the box
                #--x- and y- coordinates of the UN-padded influence function in the full padded pupil
                x_box = mp.dm2.compact.x_pupPad[x_box_AS_ind] # full pupil x-coordinates of the box 
                y_box = mp.dm2.compact.y_pupPad[y_box_AS_ind] # full pupil y-coordinates of the box 
                
                dEbox = (mp.dm2.VtoH.reshape(mp.dm2.Nact**2)[iact])*(mirrorFac*2*np.pi*1j/wvl)*falco.utils.padOrCropEven(np.squeeze(mp.dm2.compact.inf_datacube[:,:,iact]),NboxPad2AS) #--the padded influence function at DM2
                dEP2box = falco.propcustom.propcustom_PTP(dEbox*Edm2[np.ix_(y_box_AS_ind,x_box_AS_ind)],mp.P2.compact.dx*NboxPad2AS,wvl,-1*(mp.d_dm1_dm2 + mp.d_P2_dm1)) # back-propagate to pupil P2
#                dEP2box = propcustom_PTP_inf_func(dEbox.*Edm2(y_box_AS_ind,x_box_AS_ind),mp.P2.compact.dx*NboxPad2AS,wvl,-1*(mp.d_dm1_dm2 + mp.d_P2_dm1),mp.dm2.dm_spacing,mp.propMethodPTP); # back-propagate to pupil P2
                
                #--To simulate going forward to the next pupil plane (with the apodizer) most efficiently, 
                # First, back-propagate the apodizer (by rotating 180-degrees) to the previous pupil.
                # Second, negate the coordinates of the box used.
                dEP2box = apodReimaged[np.ix_(y_box_AS_ind,x_box_AS_ind)]*dEP2box; #--Apply 180deg-rotated SP mask.
                dEP3box = np.rot90(dEP2box,k=2*mp.Nrelay2to3) #--Forward propagate the cropped box by rotating 180 degrees mp.Nrelay2to3 times.
                #--Negate and rotate coordinates to effectively rotate by 180 degrees. No change if 360 degree rotation.
                if np.mod(mp.Nrelay2to3,2)==1: 
                    x_box = -1*x_box[::-1]
                    y_box = -1*y_box[::-1]
#                x_box = (-1)^mp.Nrelay2to3*rot90(x_box,2*mp.Nrelay2to3); 
#                y_box = (-1)^mp.Nrelay2to3*rot90(y_box,2*mp.Nrelay2to3); #--Negate and rotate coordinates to effectively rotate by 180 degrees. No change if 360 degree rotation.
                
                #--Matrices for the MFT from the pupil P3 to the focal plane mask
                rect_mat_pre = (np.exp(-2*np.pi*1j*(mp.F3.compact.etas.reshape(mp.F3.compact.etas.size,1)@y_box.reshape(1,y_box.size))/(wvl*mp.fl)))*np.sqrt(mp.P2.compact.dx*mp.P2.compact.dx)*np.sqrt(mp.F3.compact.dxi*mp.F3.compact.deta)/(wvl*mp.fl)
                rect_mat_post  = (np.exp(-2*np.pi*1j*(x_box.reshape(x_box.size,1)@mp.F3.compact.xis.reshape(1,mp.F3.compact.xis.size))/(wvl*mp.fl)))
    
                #--MFT from pupil P3 to FPM
                EF3 = rect_mat_pre@dEP3box@rect_mat_post # MFT to FPM
                EF3 = (1-mp.F3.compact.mask.amp)*EF3 #--Propagate through ( 1 - (complex FPM) ) for Babinet's principle
    
                #--MFT to LS ("Sub" name for Subtrahend part of the Lyot-plane E-field)
                EP4sub = falco.propcustom.propcustom_mft_FtoP(EF3,mp.fl,wvl,mp.F3.compact.dxi,mp.F3.compact.deta,mp.P4.compact.dx,mp.P4.compact.Narr,mp.centering) #--Subtrahend term for the Lyot plane E-field    
                EP4sub = falco.propcustom.propcustom_relay(EP4sub,mp.Nrelay3to4-1,mp.centering) #--Get the correct orientation
                                
                EP4noFPM = np.zeros((mp.dm2.compact.NdmPad,mp.dm2.compact.NdmPad),dtype=np.complex)
                EP4noFPM[np.ix_(y_box_AS_ind,x_box_AS_ind)] = dEP2box #--Propagating the E-field from P2 to P4 without masks gives the same E-field.
                EP4noFPM = falco.propcustom.propcustom_relay(EP4noFPM,mp.Nrelay2to3+mp.Nrelay3to4,mp.centering) #--Get the number or re-imaging relays between pupils P3 and P4. 
                EP4noFPM = falco.utils.padOrCropEven(EP4noFPM,mp.P4.compact.Narr) #--Crop down to the size of the Lyot stop opening
                EP4 = mp.P4.compact.croppedMask*(EP4noFPM - EP4sub) #--Babinet's principle to get E-field at Lyot plane
    
                #--MFT to detector
                EP4 = falco.propcustom.propcustom_relay(EP4,mp.NrelayFend,mp.centering) #--Rotate the final image 180 degrees if necessary
                EFend = falco.propcustom.propcustom_mft_PtoF(EP4,mp.fl,wvl,mp.P4.compact.dx,mp.Fend.dxi,mp.Fend.Nxi,mp.Fend.deta,mp.Fend.Neta,mp.centering)
                
                Gzdl[:,Gindex] = EFend[mp.Fend.corr.maskBool]/np.sqrt(mp.Fend.compact.I00[modvar.sbpIndex])

            Gindex += 1
        
    
    return Gzdl