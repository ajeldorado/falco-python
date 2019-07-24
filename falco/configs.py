import falco

def falco_get_PSF_norm_factor(mp):
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    pass

def falco_gen_dm_poke_cube(dm, mp, dx_dm, flagGenCube=True, **kwds):
    # SFF NOTE:  This function exists in falco/lib/dm/falco_gen_dm_poke_cube.py but not sure if this is good
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    pass

    return mp.dm1.compact
    
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
def falco_config_spatial_weights(mp):
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
