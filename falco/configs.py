import falco
import numpy as np

def falco_get_PSF_norm_factor(mp):
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    pass

    
def falco_config_gen_FPM_FOHLC(mp):
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
def falco_config_gen_FPM_HLC(mp):
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
def falco_config_gen_FPM_LC(mp):
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
def falco_config_gen_FPM_Roddier(mp):
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
def falco_config_gen_FPM_SPLC(mp):
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
def falco_config_gen_chosen_LS(mp):
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
def falco_config_gen_chosen_apodizer(mp):
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

def falco_config_gen_chosen_pupil(mp):
    """
    Function to generate the apodizer representation based on configuration settings.

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

#    ## Input pupil plane resolution, masks, and coordinates
#    #--Resolution at input pupil and DM1 and DM2
#    mp.P2.full.dx = mp.P2.D/mp.P1.full.Nbeam;
#    mp.P2.compact.dx = mp.P2.D/mp.P1.compact.Nbeam;
#
#    whichPupil = mp.whichPupil.upper()
#    if whichPupil in ('SIMPLE', 'SIMPLEPROPER'):
#        print('whichPupil = %s'%(whichPupil))
#        if whichPupil == 'SIMPLEPROPER':
#            inputs.flagPROPER = True;
#
#        inputs.Nbeam = mp.P1.full.Nbeam; # number of points across the pupil diameter
#        inputs.OD = mp.P1.ODnorm;
#        inputs.ID = mp.P1.IDnorm;
#        inputs.Nstrut = mp.P1.Nstrut;
#        inputs.angStrut = mp.P1.angStrut; # Angles of the struts 
#        inputs.wStrut = mp.P1.wStrut; # spider width (fraction of the pupil diameter)
#        inputs.Npad = 2^(nextpow2(mp.P1.full.Nbeam));
#        inputs.stretch = mp.P1.stretch;
#
#        mp.P1.full.mask = falco_gen_pupil_Simple(inputs);
#
#        #--Generate low-res input pupil for the 'compact' model
#        inputs.Nbeam = mp.P1.compact.Nbeam; # number of points across usable pupil   
#        inputs.Npad = 2^(nextpow2(mp.P1.compact.Nbeam)); # number of points across usable pupil 
#        mp.P1.compact.mask = falco_gen_pupil_Simple(inputs);
#        pass
#    elif whichPupil == 'WFIRST180718':
#        print('whichPupil = %s'%(whichPupil))
#        #--Generate low-res input pupil for the 'compact' model
#        if hasattr(mp, 'P1'):
#            if hasattr(mp.P1,'full'):
#                if not hasattr(mp.P1.full,'mask'):
#                    mp.P1.full.mask = falco_gen_pupil_WFIRST_CGI_180718(mp.P1.full.Nbeam, mp.centering);
#            if hasattr(mp.P1,'compact'):
#                if hasattr(mp.P1.compact,'mask'):
#                    mp.P1.compact.mask = falco_gen_pupil_WFIRST_CGI_180718(mp.P1.compact.Nbeam, mp.centering);
#
#        pass
#    elif whichPupil == 'WFIRST20180103':
#        print('whichPupil = %s'%(whichPupil))
#        #--Generate high-res input pupil for the 'full' model
#        mp.P1.full.mask = falco.masks.falco_gen_pupil_WFIRST_20180103(mp.P1.full.Nbeam, mp.centering);
#
#        #--Generate low-res input pupil for the 'compact' model
#        mp.P1.compact.mask = falco.masks.falco_gen_pupil_WFIRST_20180103(mp.P1.compact.Nbeam, mp.centering);
#
#        pass
#    elif whichPupil == 'WFIRST_ONAXIS':
#        print('whichPupil = %s'%(whichPupil))
#        pass
#    elif whichPupil == 'LUVOIRAFINAL':
#        print('whichPupil = %s'%(whichPupil))
#        pass
#    elif whichPupil == 'LUVOIRA5':
#        print('whichPupil = %s'%(whichPupil))
#        pass
#    elif whichPupil == 'LUVOIRA0':
#        print('whichPupil = %s'%(whichPupil))
#        pass
#    elif whichPupil == 'LUVOIR_B_OFFAXIS':
#        print('whichPupil = %s'%(whichPupil))
#        pass
#    elif whichPupil == 'DST_LUVOIRB':
#        print('whichPupil = %s'%(whichPupil))
#        pass
#    elif whichPupil == 'HABEX_B_OFFAXIS':
#        print('whichPupil = %s'%(whichPupil))
#        pass
#    elif whichPupil == 'ISAT':
#        print('whichPupil = %s'%(whichPupil))
#        pass
#    else:
#        print('whichPupil = %s'%(whichPupil))
#        pass
#
#    mp.P1.compact.Narr = len(mp.P1.compact.mask); #--Number of pixels across the array containing the input pupil in the compact model
#    
#    #--NORMALIZED (in pupil diameter) coordinate grids in the input pupil for making the tip/tilted input wavefront within the compact model
#    if mp.centering.lower() == 'interpixel':
#        mp.P2.compact.xsDL = np.arange(-(mp.P1.compact.Narr-1)/2, (mp.P1.compact.Narr-1)/2) * mp.P2.compact.dx/mp.P2.D;
#    else:
#        mp.P2.compact.xsDL = np.arange(-mp.P1.compact.Narr/2, (mp.P1.compact.Narr/2-1)) * mp.P2.compact.dx/mp.P2.D;
#    
#    ### STOPED HERE
#    #[mp.P2.compact.XsDL,mp.P2.compact.YsDL] = np.meshgrid(mp.P2.compact.xsDL);
#    mp.P2.compact.YsDL = np.meshgrid(mp.P2.compact.xsDL);
#    
#    if mp.layout.lower() in ('wfirst_phaseb_simple','wfirst_phaseb_proper'):
#        if mp.centering.lower() == 'interpixel':
#            mp.P1.full.Narr = ceil_even(mp.P1.full.Nbeam);
#        else:
#            mp.P1.full.Narr = ceil_even(mp.P1.full.Nbeam+1);
#    else:
#        mp.P1.full.Narr = len(mp.P1.full.mask);  #--Total number of pixels across array containing the pupil in the full model. Add 2 pixels to Nbeam when the beam is pixel-centered.
#
#    #--NORMALIZED (in pupil diameter) coordinate grids in the input pupil for making the tip/tilted input wavefront within the full model
#    if mp.centering.lower() == 'interpixel':
#        mp.P2.full.xsDL = np.arange(- (mp.P1.full.Narr-1)/2,(mp.P1.full.Narr-1)/2)*mp.P2.full.dx/mp.P2.D;
#    else:
#        mp.P2.full.xsDL = np.arange( -mp.P1.full.Narr/2,(mp.P1.full.Narr/2 -1) )*mp.P2.full.dx/mp.P2.D;
#    
#    #[mp.P2.full.XsDL,mp.P2.full.YsDL] = meshgrid(mp.P2.full.xsDL);

    pass

def falco_config_jac_weights(mp):
    """
    Function to set the relative weights for the Jacobian modes based on wavelength and 
    Zernike mode.

    Function to set the relative weights for the Jacobian modes. The weights are 
    formulated first in a 2-D array with rows for wavelengths and columns for Zernike 
    modes. The weights are then normalized in each column. The weight matrix is then 
    vectorized, with all zero weights removed.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    Returns
    -------
    nothing
        Values are added by reference into the mp structure.
    """

    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
                
    #--Initialize mp.jac if it doesn't exist
    if not hasattr(mp, 'jac'):
        mp.jac = falco.config.EmptyClass()
    
    #--Which Zernike modes to include in Jacobian. Given as a vector of Noll indices. 1 is the on-axis piston mode.
    if not hasattr(mp.jac, 'zerns'):
        mp.jac.zerns = np.array([1])
        mp.jac.Zcoef = np.array([1.])
        
    mp.jac.Nzern = np.size(mp.jac.zerns)
    mp.jac.Zcoef[np.nonzero(mp.jac.zerns==1)][0] = 1.; #--Reset coefficient for piston term to 1
    
    #--Initialize weighting matrix of each Zernike-wavelength mode for the controller
    mp.jac.weightMat = np.zeros((mp.Nsbp,mp.jac.Nzern)); 
    for izern in range(0,mp.jac.Nzern):
        whichZern = mp.jac.zerns[izern];
        if whichZern==1:
            mp.jac.weightMat[:,0] = np.ones(mp.Nsbp) #--Include all wavelengths for piston Zernike mode
        else: #--Include just middle and end wavelengths for Zernike mode 2 and up
            mp.jac.weightMat[0,izern] = 1
            mp.jac.weightMat[mp.si_ref,izern] = 1
            mp.jac.weightMat[mp.Nsbp-1,izern] = 1
        
    #--Half-weighting if endpoint wavelengths are used
    if mp.estimator.lower()=='perfect': #--For design or modeling without estimation: Choose ctrl wvls evenly between endpoints of the total bandpass
        mp.jac.weightMat[0,:] = 0.5*mp.jac.weightMat[0,:];
        mp.jac.weightMat[mp.Nsbp-1,:] = 0.5*mp.jac.weightMat[mp.Nsbp-1,:];
    
    #--Normalize the summed weights of each column separately
    for izern in range(mp.jac.Nzern):
        colSum = np.double(sum(mp.jac.weightMat[:,izern]))
        mp.jac.weightMat[:,izern] = mp.jac.weightMat[:,izern]/colSum

    #--Zero out columns for which the RMS Zernike value is zero
    for izern in range(mp.jac.Nzern):
        if mp.jac.Zcoef[izern]==0:
            mp.jac.weightMat[:,izern] = 0*mp.jac.weightMat[:,izern]

    mp.jac.weightMat_ele = np.nonzero(mp.jac.weightMat>0) #--Indices of the non-zero control Jacobian modes in the weighting matrix
    mp.jac.weights = mp.jac.weightMat[mp.jac.weightMat_ele] #--Vector of control Jacobian mode weights
    mp.jac.Nmode = np.size(mp.jac.weights) #--Number of (Zernike-wavelength pair) modes in the control Jacobian

    #--Get the wavelength indices for the nonzero values in the weight matrix. 
    tempMat = np.tile( np.arange(mp.Nsbp).reshape((mp.Nsbp,1)), (1,mp.jac.Nzern) )
    mp.jac.sbp_inds = tempMat[mp.jac.weightMat_ele];

    #--Get the Zernike indices for the nonzero elements in the weight matrix. 
    tempMat = np.tile(mp.jac.zerns,(mp.Nsbp,1));
    mp.jac.zern_inds = tempMat[mp.jac.weightMat_ele];

    pass
    
def falco_config_spatial_weights(mp):
    """
    Set up spatially-based weighting of the dark hole intensity.

    Set up spatially-based weighting of the dark hole intensity in annular zones centered 
    on the star. Zones are specified with rows of three values: zone inner radius [l/D],
    zone outer radius [l/D], and intensity weight. As many rows can be used as desired.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    Returns
    -------
    nothing
        Values are added by reference into the mp structure.
    """

    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')

    #--Define 2-D coordinate grid
    [XISLAMD,ETASLAMD] = np.meshgrid(mp.Fend.xisDL, mp.Fend.etasDL)
    RHOS = np.sqrt(XISLAMD**2+ETASLAMD**2)
    mp.Wspatial = mp.Fend.corr.mask #--Convert from boolean to float
    if hasattr(mp, 'WspatialDef'): #--Do only if spatial weights are defined
        if(np.size(mp.WspatialDef)>0): #--Do only if variable is not empty
            for kk in range(0,mp.WspatialDef.shape[0]): #--Increment through the rows
                Wannulus = 1. + (np.sqrt(mp.WspatialDef[kk,2])-1.)*((RHOS>=mp.WspatialDef[kk,0]) & (RHOS<mp.WspatialDef[kk,1]))
                mp.Wspatial = mp.Wspatial*Wannulus

    pass
