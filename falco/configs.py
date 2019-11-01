import falco
import numpy as np
    
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
    Make or read in focal plane mask (FPM) amplitude for the full model.

    Detailed description here

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    Returns
    -------
    mp: falco.config.ModelParameters
        Structure of model parameters
    """

    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    

#    class FPMgenIn(object):
#        pass
    
    FPMgenInputs = {} #FPMgenIn()
    
    #--Make or read in focal plane mask (FPM) amplitude for the full model
    #FPMgenInputs.flagRot180deg
    FPMgenInputs["pixresFPM"] = mp.F3.full.res #--pixels per lambda_c/D
    FPMgenInputs["rhoInner"] = mp.F3.Rin # radius of inner FPM amplitude spot (in lambda_c/D)
    FPMgenInputs["rhoOuter"] = mp.F3.Rout # radius of outer opaque FPM ring (in lambda_c/D)
    FPMgenInputs["FPMampFac"] = mp.FPMampFac # amplitude transmission of inner FPM spot
    FPMgenInputs["centering"] = mp.centering
    #kwargs = FPMgenInputs.__dict__
    
    if not hasattr(mp.F3.full,'mask'):
        mp.F3.full.mask = falco.config.Object()
        
    mp.F3.full.mask.amp = falco.masks.falco_gen_annular_FPM(FPMgenInputs)

    mp.F3.full.Nxi = mp.F3.full.mask.amp.shape[1]
    mp.F3.full.Neta= mp.F3.full.mask.amp.shape[0]  
    
    #--Number of points across the FPM in the compact model
    if(mp.F3.Rout==np.inf):
        if mp.centering == 'pixel':
            mp.F3.compact.Nxi = falco.utils.ceil_even((2*(mp.F3.Rin*mp.F3.compact.res + 1/2)))
        else:
            mp.F3.compact.Nxi = falco.utils.ceil_even((2*mp.F3.Rin*mp.F3.compact.res))
            
    else:
        if mp.centering == 'pixel':
            mp.F3.compact.Nxi = falco.utils.ceil_even((2*(mp.F3.Rout*mp.F3.compact.res + 1/2)))
        else: #case 'interpixel'
            mp.F3.compact.Nxi = falco.utils.ceil_even((2*mp.F3.Rout*mp.F3.compact.res))

    mp.F3.compact.Neta = mp.F3.compact.Nxi
    
    #--Make or read in focal plane mask (FPM) amplitude for the compact model
    FPMgenInputs["pixresFPM"] = mp.F3.compact.res #--pixels per lambda_c/D
    #kwargs=FPMgenInputs.__dict__
    
    if not hasattr(mp.F3.compact,'mask'):
        mp.F3.compact.mask = falco.config.Object()
        
    mp.F3.compact.mask.amp = falco.masks.falco_gen_annular_FPM(FPMgenInputs)
    
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
    Function to generate the Lyot stop representation based on configuration 
    settings.

    Detailed description here

    Created on 2018-05-29 by A.J. Riggs.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
        
    Returns
    -------
    NA
    """

    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    
    #--Resolution at Lyot Plane
    if(mp.full.flagPROPER==False):
        mp.P4.full.dx = mp.P4.D/mp.P4.full.Nbeam


    ### Changes to the pupil
    changes = {}
#    class Changes(object):
#        pass
#    
#    changes = Changes()
    
    
    """
    % switch mp.layout
    %     case{'wfirst_phaseb_simple','wfirst_phaseb_proper'}
    %         
    %     otherwise
    %         mp.P4.full.dx = mp.P4.D/mp.P4.full.Nbeam;
    % end
    """
    mp.P4.compact.dx = mp.P4.D/mp.P4.compact.Nbeam

    whichPupil = mp.whichPupil.upper()
    if whichPupil in ('SIMPLE','SIMPLEPROPER','DST_LUVOIRB','ISAT'):
        """
        if whichPupil in ('SIMPLEPROPER'):  
            inputs.flagPROPER = true
        
        inputs.Nbeam = mp.P4.full.Nbeam # number of points across incoming beam 
        inputs.Npad = 2^(falco.utils.nextpow2nextpow2(mp.P4.full.Nbeam))
        inputs.OD = mp.P4.ODnorm
        inputs.ID = mp.P4.IDnorm
        inputs.Nstrut = mp.P4.Nstrut
        inputs.angStrut = mp.P4.angStrut # Angles of the struts 
        inputs.wStrut = mp.P4.wStrut # spider width (fraction of the pupil diameter)

        mp.P4.full.mask = falco_gen_pupil_Simple(inputs)
        
        inputs.Nbeam = mp.P4.compact.Nbeam #--Number of pixels across the aperture or beam (independent of beam centering)
        inputs.Npad = 2^(nextpow2(mp.P4.compact.Nbeam))
        
        mp.P4.compact.mask = falco_gen_pupil_Simple(inputs)
        """
        pass
    elif whichPupil == 'WFIRST180718':
        #--Define Lyot stop generator function inputs for the 'full' optical model
        if mp.compact.flagGenLS or mp.full.flagGenLS:
            changes["ID"] = mp.P4.IDnorm
            changes["OD"] = mp.P4.ODnorm
            changes["wStrut"] = mp.P4.wStrut
            changes["flagRot180"] = True
        
        #kwargs = changes.__dict__ #convert changes to dictionary to use as input to gen_pupil routine
        if(mp.full.flagGenLS):
            mp.P4.full.mask = falco.masks.falco_gen_pupil_WFIRST_CGI_180718(mp.P4.full.Nbeam,mp.centering,changes)
        
        ##--Make or read in Lyot stop (LS) for the 'compact' model
        if(mp.compact.flagGenLS):
            mp.P4.compact.mask = falco.masks.falco_gen_pupil_WFIRST_CGI_180718(mp.P4.compact.Nbeam,mp.centering,changes)
        
        if hasattr(mp, 'LSshape'):
            LSshape = mp.LSshape.lower()
            if LSshape in ('bowtie'):
                #--Define Lyot stop generator function inputs in a structure
                inputs.Dbeam = mp.P4.D # meters;
                inputs.ID = mp.P4.IDnorm # (pupil diameters)
                inputs.OD = mp.P4.ODnorm # (pupil diameters)
                inputs.ang = mp.P4.ang # (degrees)
                inputs.centering = mp.centering # 'interpixel' or 'pixel'

                if(mp.full.flagGenLS):
                    inputs.Nbeam = mp.P4.full.Nbeam
                    mp.P4.full.mask = falco_gen_bowtie_LS(inputs)
                
                #--Make bowtie Lyot stop (LS) for the 'compact' model
                if(mp.compact.flagGenLS):
                    inputs.Nbeam = mp.P4.compact.Nbeam
                    mp.P4.compact.mask = falco_gen_bowtie_LS(inputs)
       
    elif whichPupil in ('WFIRST_ONAXIS'):
        pass
        
        """
        #--Define Lyot stop generator function inputs for the 'full' optical model
        inputs.Nbeam = mp.P4.full.Nbeam     # number of points across incoming beam  
        inputs.Dbeam = mp.P4.D #--diameter of the beam at the mask (meters)
        inputs.ID = mp.P4.IDnorm
        inputs.OD = mp.P4.ODnorm
        inputs.wStrut = mp.P4.wStrut
        inputs.centering = mp.centering

        #--Make or read in Lyot stop (LS) for the 'full' model
        mp.P4.full.mask = falco_gen_pupil_WFIRSTcycle6_LS(inputs,'ROT180')

        #--Make or read in Lyot stop (LS) for the 'compact' model
        inputs.Nbeam = mp.P4.compact.Nbeam; % number of points across incoming beam           
        mp.P4.compact.mask = falco_gen_pupil_WFIRSTcycle6_LS(inputs,'ROT180')
        
        if hasattr(mp,'LSshape'):
            if mp.LSshape.lower() in ('bowtie'):
                #--Define Lyot stop generator function inputs in a structure
                inputs.Dbeam = mp.P4.D # meters
                inputs.Nbeam = mp.P4.full.Nbeam 
                inputs.ID = mp.P4.IDnorm # (pupil diameters)
                inputs.OD = mp.P4.ODnorm # (pupil diameters)
                inputs.ang = mp.P4.ang # (degrees)
                inputs.centering = mp.centering # 'interpixel' or 'pixel'

                #--Make bowtie Lyot stop (LS) for the 'full' model
                mp.P4.full.mask = falco_gen_bowtie_LS(inputs)

                #--Make bowtie Lyot stop (LS) for the 'compact' model
                inputs.Nbeam = mp.P4.compact.Nbeam 
                mp.P4.compact.mask = falco_gen_bowtie_LS(inputs)              
        """
    elif whichPupil in ('LUVOIRAFINAL'):
        pass
        """
        #--Define Lyot stop generator function inputs for the 'full' optical model
        inputs.Nbeam = mp.P4.full.Nbeam # number of points across incoming beam  
        inputs.Dbeam = mp.P1.D
        inputs.ID = mp.P4.IDnorm
        inputs.OD = mp.P4.ODnorm
        inputs.wStrut = mp.P4.wStrut
        inputs.centering = mp.centering
        #--Make or read in Lyot stop (LS) for the 'full' model
        mp.P4.full.mask = falco_gen_pupil_LUVOIR_A_final_Lyot(inputs,'ROT180');
        
        #--Make or read in Lyot stop (LS) for the 'compact' model
        inputs.Nbeam = mp.P4.compact.Nbeam;     # number of points across incoming beam           
        mp.P4.compact.mask = falco_gen_pupil_LUVOIR_A_final_Lyot(inputs,'ROT180')
        """
    elif whichPupil in ('LUVOIRA5','LUVOIRA0'):
        #--Define Lyot stop generator function inputs for the 'full' optical model
        pass
        """
        inputs.Nbeam = mp.P4.full.Nbeam; % number of points across incoming beam  
        inputs.Dbeam = mp.P1.D;
        inputs.ID = mp.P4.IDnorm;
        inputs.OD = mp.P4.ODnorm;
        inputs.wStrut = mp.P4.wStrut;
        inputs.centering = mp.centering;
        %--Make or read in Lyot stop (LS) for the 'full' model
        mp.P4.full.mask = falco_gen_pupil_LUVOIR_A_5_Lyot_struts(inputs,'ROT180');
        
        %--Make or read in Lyot stop (LS) for the 'compact' model
        inputs.Nbeam = mp.P4.compact.Nbeam;     % number of points across incoming beam           
        mp.P4.compact.mask = falco_gen_pupil_LUVOIR_A_5_Lyot_struts(inputs,'ROT180');
        """
    elif whichPupil in ('LUVOIR_B_OFFAXIS','HABEX_B_OFFAXIS'):
        
        inputs = {} # initialize
        inputs["ID"] = mp.P4.IDnorm #- Outer diameter (fraction of Nbeam)
        inputs["OD"] = mp.P4.ODnorm#- Inner diameter (fraction of Nbeam)
#        inputs["Nstrut"] = 0 #- Number of struts
#        inputs["angStrut"] = np.array([])#- Array of struct angles (deg)
#        inputs["wStrut"] = np.array([]) #- Strut widths (fraction of Nbeam)
#        inputs["stretch"] = 1.#- Create an elliptical aperture by changing Nbeam along
        #                  the horizontal direction by a factor of stretch (PROPER
        #                  version isn't implemented as of March 2019).

        inputs["Nbeam"] = mp.P4.compact.Nbeam #- Number of samples across the beam 
        inputs["Npad"] = int(2**falco.utils.nextpow2( falco.utils.ceil_even(mp.P4.compact.Nbeam )))
        mp.P4.compact.mask = falco.masks.falco_gen_pupil_Simple( inputs )

        inputs["Nbeam"] = mp.P4.full.Nbeam #- Number of samples across the beam 
        inputs["Npad"] = int(2**falco.utils.nextpow2( falco.utils.ceil_even(mp.P4.full.Nbeam )))
        mp.P4.full.mask = falco.masks.falco_gen_pupil_Simple( inputs )


        pass
        """  
        #--Full model
        inputs.Nbeam = mp.P4.full.Nbeam; % number of points across incoming beam 
        inputs.Npad = 2^(nextpow2(mp.P4.full.Nbeam));
        inputs.OD = mp.P4.ODnorm;
        inputs.ID = mp.P4.IDnorm;
        inputs.Nstrut = 0;
        inputs.angStrut = []; % Angles of the struts 
        inputs.wStrut = 0; % spider width (fraction of the pupil diameter)

        mp.P4.full.mask = falco_gen_pupil_Simple(inputs);
        
        pad_pct = mp.P4.padFacPct;
        if(pad_pct>0) %--Also apply an eroded/padded version of the segment gaps

            pupil0 = mp.P1.full.mask;
            Nbeam = inputs.Nbeam;
            Npad = inputs.Npad;

            xsD = (-Npad/2:(Npad/2-1))/Nbeam; %--coordinates, normalized to the pupil diameter
            [XS,YS] = meshgrid(xsD);
            RS = sqrt(XS.^2 + YS.^2);
        
            pupil1 = 1-pupil0;

            spot = zeros(Npad);
            spot(RS <= pad_pct/100) = 1;

            pupil4 = ifftshift(ifft2(fft2(fftshift(pupil1)).*fft2(fftshift(spot))));
            pupil4 = abs(pupil4);
            pupil4 = pupil4/max(pupil4(:));

            pupil5 = 1-pupil4;

            thresh = 0.99;
            pupil5(pupil5<thresh) = 0;
            pupil5(pupil5>=thresh) = 1;

            mp.P4.full.mask = mp.P4.full.mask.*pupil5;            
        end
        
        #--Compact model
        inputs.Nbeam = mp.P4.compact.Nbeam #--Number of pixels across the aperture or beam (independent of beam centering)
        inputs.Npad = 2^(nextpow2(mp.P4.compact.Nbeam))
        
        mp.P4.compact.mask = falco_gen_pupil_Simple(inputs)
        
        if(pad_pct>0): #--Also apply an eroded/padded version of the segment gaps
            pupil0 = mp.P1.compact.mask
            Nbeam = inputs.Nbeam
            Npad = inputs.Npad

            xsD = (-Npad/2:(Npad/2-1))/Nbeam; %--coordinates, normalized to the pupil diameter
            [XS,YS] = meshgrid(xsD)
            RS = sqrt(XS.^2 + YS.^2)

            pupil1 = 1-pupil0

            spot = zeros(Npad)
            spot(RS <= pad_pct/100) = 1

            pupil4 = ifftshift(ifft2(fft2(fftshift(pupil1)).*fft2(fftshift(spot))))
            pupil4 = abs(pupil4)
            pupil4 = pupil4/max(pupil4(:))

            pupil5 = 1-pupil4

            thresh = 0.99
            pupil5(pupil5<thresh) = 0
            pupil5(pupil5>=thresh) = 1

            mp.P4.compact.mask = mp.P4.compact.mask.*pupil5
    """


    ## Crop down the Lyot stop(s) to get rid of extra zero padding for the full model
    if(False): # mp.coro.upper() in ('VORTEX','VC','AVC'):
        mp.P4.full.Narr = mp.P4.full.mask.shape[0]
        mp.P4.full.croppedMask = mp.P4.full.mask
        mp.P4.compact.Narr = mp.P4.compact.mask.shape[0]
        mp.P4.compact.croppedMask = mp.P4.compact.mask
    else:
        if(mp.full.flagPROPER==False):
            #--Crop down the high-resolution Lyot stop to get rid of extra zero padding
            LSsum = np.sum(mp.P4.full.mask)
            LSdiff = 0 
            counter = 2
            while(np.abs(LSdiff) <= 1e-7):
                mp.P4.full.Narr = len(mp.P4.full.mask)-counter
                LSdiff = LSsum - np.sum(falco.utils.padOrCropEven(mp.P4.full.mask, mp.P4.full.Narr-2)) #--Subtract an extra 2 to negate the extra step that overshoots.
                counter = counter + 2
            
            mp.P4.full.croppedMask = falco.utils.padOrCropEven(mp.P4.full.mask,mp.P4.full.Narr) #--The cropped-down Lyot stop for the full model. 
        
        ## --Crop down the low-resolution Lyot stop to get rid of extra zero padding. Speeds up the compact model.
        LSsum = np.sum(mp.P4.compact.mask)
        LSdiff= 0
        counter = 2
        while(abs(LSdiff) <= 1e-7):
            mp.P4.compact.Narr = len(mp.P4.compact.mask)-counter #--Number of points across the cropped-down Lyot stop
            LSdiff = LSsum - np.sum(falco.utils.padOrCropEven(mp.P4.compact.mask, mp.P4.compact.Narr-2))  #--Subtract an extra 2 to negate the extra step that overshoots.
            counter = counter + 2

        mp.P4.compact.croppedMask = falco.utils.padOrCropEven(mp.P4.compact.mask,mp.P4.compact.Narr) #--The cropped-down Lyot stop for the compact model
 
    #--(METERS) Lyot plane coordinates (over the cropped down to Lyot stop mask) for MFTs in the compact model from the FPM to the LS.
    if mp.centering == 'interpixel':
        #mp.P4.compact.xs = (-(mp.P4.compact.Narr-1)/2:(mp.P4.compact.Narr-1)/2)*mp.P4.compact.dx
        mp.P4.compact.xs = np.linspace(-(mp.P4.compact.Narr-1)/2, (mp.P4.compact.Narr-1)/2,mp.P4.compact.Narr)*mp.P4.compact.dx
    else:
        #mp.P4.compact.xs = (-mp.P4.compact.Narr/2:(mp.P4.compact.Narr/2-1))*mp.P4.compact.dx
        mp.P4.compact.xs = np.linspace(-mp.P4.compact.Narr/2, (mp.P4.compact.Narr/2-1),mp.P4.compact.Narr)*mp.P4.compact.dx
    
    mp.P4.compact.ys = np.transpose(mp.P4.compact.xs) #transpose of array (196x1)


def falco_config_gen_chosen_apodizer(mp):
    """
    Function to generate the apodizer representation based on configuration 
    settings.

    Parameters
    ----------
    mp: falco.config.ModelParameters
        Structure of model parameters
    Returns
    -------
    NA
    """

    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    pass

    if not hasattr(mp.P3,'full'):
        mp.P3.full = falco.config.Object()
        
    if not hasattr(mp.P3,'compact'):
        mp.P3.compact = falco.config.Object()   

    
    if mp.flagApod:
        #--mp.apodType is used only when generating certain types of analytical apodizers
        if mp.apodType.lower() in ('simple'): #--A simple, circular aperture stop
            #--Inputs common to both the compact and full models
            inputs = {}
            inputs["ID"] = mp.P3.IDnorm
            inputs["OD"] = mp.P3.ODnorm

            inputs["Nstrut"] = mp.P3.Nstrut
            inputs["angStrut"] = mp.P3.angStrut # Angles of the struts 
            inputs["wStrut"] = mp.P3.wStrut # spider width (fraction of the pupil diameter)
            inputs["stretch"] = mp.P3.stretch

            #--Full model only
            inputs["Nbeam"] = mp.P1.full.Nbeam # number of points across incoming beam 
            inputs["Npad"] = 2**(falco.utils.nextpow2(mp.P1.full.Nbeam)) 
            
            if(mp.full.flagGenApod):
                mp.P3.full.mask = falco.masks.falco_gen_pupil_Simple( inputs );
            else:
                disp('*** Simple aperture stop to be loaded instead of generated for full model. ***')
        
            # Compact model only 
            inputs["Nbeam"] = mp.P1.compact.Nbeam #--Number of pixels across the aperture or beam (independent of beam centering)
            inputs["Npad"] = 2**(falco.utils.nextpow2(mp.P1.compact.Nbeam))
            
            if(mp.compact.flagGenApod):
                mp.P3.compact.mask = falco.masks.falco_gen_pupil_Simple( inputs );
            else:
                disp('*** Simple aperture stop to be loaded instead of generated for compact model. ***')
            
            """
            if mp.apodType.lower() in ('ring') #--Concentric ring apodizer
                #--Full model
                if(mp.full.flagGenApod):
                    mp.P3.full.mask = falco_gen_multi_ring_SP(mp.rEdgesLeft,mp.rEdgesRight,mp.P2.full.dx,mp.P2.D,mp.centering); %--Generate binary-amplitude ring apodizer for the full model
                else:
                    disp('*** Concentric ring apodizer loaded instead of generated for full model. ***')
                
                #--Compact model
                if(mp.compact.flagGenApod)
                    mp.P3.compact.mask = falco_gen_multi_ring_SP(mp.rEdgesLeft,mp.rEdgesRight,mp.P2.compact.dx,mp.P2.D,mp.centering); %--Generate binary-amplitude ring apodizer for the compact model
                else
                    disp('*** Concentric ring apodizer loaded instead of generated for compact model. ***')
            
                if(mp.flagPlot):  
                    figure(504); 
                    imagesc(padOrCropEven(mp.P3.full.mask,length(mp.P1.full.mask)) + mp.P1.full.mask); 
                    axis xy equal tight; 
                    colorbar 
                    drawnow

            if mp.apodType.lower() in ('grayscale','traditional','classical') #--A grayscale aperture generated with a Gerchberg-Saxton type algorithm

                if mp.centering not in ('pixel'):  
                    error('Use pixel centering for APLC')
                    
                if(mp.P1.full.Nbeam ~= mp.P1.compact.Nbeam):  
                    error('Tradional apodizer generation currently requires Nbeam for the full and compact (in order to use the same apodizer).')

                #--Generate the grayscale apodizer
                if(mp.full.flagGenApod && mp.compact.flagGenApod): 
                    mp.P3.full.mask = falco_gen_tradApodizer(mp.P1.full.mask,mp.P1.full.Nbeam,mp.F3.Rin,(1+mp.fracBW/2)*mp.F3.Rout,mp.useGPU);
                    mp.P3.full.Narr = length(mp.P3.full.mask);
                    mp.P3.compact.mask = mp.P3.full.mask;
                    mp.P3.compact.Narr = length(mp.P3.compact.mask);
                else:
                    disp('*** Grayscale apodizer loaded instead of generated for full and compact models. ***')
            
                if(mp.flagPlot):  
                    figure(504) 
                    imagesc(padOrCropEven(mp.P3.full.mask,length(mp.P1.full.mask)).*mp.P1.full.mask) 
                    axis xy equal tight 
                    colorbar 
                    drawnow
            else:
                disp('No apodizer type specified for generation.')

"""
    mp.P3.full.dummy = 1
    if hasattr(mp.P3.full,'mask'):   #==false || isfield(mp.P3.compact,'mask')==false)
        mp.P3.full.Narr = mp.P3.full.mask.shape[0]
    else:
        print('*** If not generated or loaded in a PROPER model, the apodizer must be loaded \n    in the main script or config file into the variable mp.P3.full.mask ***')

    
    mp.P3.compact.dummy = 1
    if hasattr(mp.P3.compact,'mask'):    #    ==false || isfield(mp.P3.compact,'mask')==false)
        mp.P3.compact.Narr = mp.P3.compact.mask.shape[0]
    else:
        print('*** If not generated, the apodizer must be loaded in the main script or config \n    file into the variable mp.P3.compact.mask ***')
    
   
    ##--Set the pixel width [meters]
    mp.P3.full.dx = mp.P2.full.dx
    mp.P3.compact.dx = mp.P2.compact.dx
    

def falco_config_gen_chosen_pupil(mp):
    """
    Function to generate the apodizer representation based on configuration 
    settings.

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

    # Input pupil plane resolution, masks, and coordinates
    #--Resolution at input pupil and DM1 and DM2
    if not hasattr(mp.P2,'full'):
        mp.P2.full = falco.config.Object()
        
    mp.P2.full.dx = mp.P2.D/mp.P1.full.Nbeam
    
    if not hasattr(mp.P2,'compact'):
        mp.P2.compact = falco.config.Object()
        
    mp.P2.compact.dx = mp.P2.D/mp.P1.compact.Nbeam

    whichPupil = mp.whichPupil.upper()
    if whichPupil in ('SIMPLE', 'SIMPLEPROPER'):
        pass
    elif whichPupil == 'WFIRST180718':
        if mp.full.flagGenPupil:
            mp.P1.full.mask = falco.masks.falco_gen_pupil_WFIRST_CGI_180718(mp.P1.full.Nbeam, mp.centering)

        if mp.compact.flagGenPupil:
            mp.P1.compact.mask = falco.masks.falco_gen_pupil_WFIRST_CGI_180718(mp.P1.compact.Nbeam, mp.centering)
            
    elif whichPupil == 'WFIRST20180103':
        pass
#        print('whichPupil = %s'%(whichPupil))
#        #--Generate high-res input pupil for the 'full' model
#        mp.P1.full.mask = falco.masks.falco_gen_pupil_WFIRST_20180103(mp.P1.full.Nbeam, mp.centering);
#
#        #--Generate low-res input pupil for the 'compact' model
#        mp.P1.compact.mask = falco.masks.falco_gen_pupil_WFIRST_20180103(mp.P1.compact.Nbeam, mp.centering);
    elif whichPupil == 'WFIRST_ONAXIS':
#        print('whichPupil = %s'%(whichPupil))
        pass
    elif whichPupil == 'LUVOIRAFINAL':
#        print('whichPupil = %s'%(whichPupil))
        pass
    elif whichPupil == 'LUVOIRA5':
#        print('whichPupil = %s'%(whichPupil))
        pass
    elif whichPupil == 'LUVOIRA0':
#        print('whichPupil = %s'%(whichPupil))
        pass
    elif whichPupil == 'LUVOIR_B_OFFAXIS':
#        print('whichPupil = %s'%(whichPupil))
        if mp.full.flagGenPupil:
            mp.P1.full.mask = falco.masks.falco_gen_pupil_LUVOIR_B(mp.P1.full.Nbeam)

        if mp.compact.flagGenPupil:
            mp.P1.compact.mask = falco.masks.falco_gen_pupil_LUVOIR_B(mp.P1.compact.Nbeam)
    
        pass
    elif whichPupil == 'DST_LUVOIRB':
#        print('whichPupil = %s'%(whichPupil))
        pass
    elif whichPupil == 'HABEX_B_OFFAXIS':
#        print('whichPupil = %s'%(whichPupil))
        pass
    elif whichPupil == 'ISAT':
#        print('whichPupil = %s'%(whichPupil))
        pass
    else:
#        print('whichPupil = %s'%(whichPupil))
        pass

    mp.P1.compact.Narr = len(mp.P1.compact.mask) #--Number of pixels across the array containing the input pupil in the compact model
    
    ##--NORMALIZED (in pupil diameter) coordinate grids in the input pupil for making the tip/tilted input wavefront within the compact model
    if mp.centering.lower() == ('interpixel'):
        mp.P2.compact.xsDL = np.linspace(-(mp.P1.compact.Narr-1)/2, (mp.P1.compact.Narr-1)/2,mp.P1.compact.Narr)*mp.P2.compact.dx/mp.P2.D
    else:
        mp.P2.compact.xsDL = np.linspace(-mp.P1.compact.Narr/2, (mp.P1.compact.Narr/2-1),mp.P1.compact.Narr)*mp.P2.compact.dx/mp.P2.D


    [mp.P2.compact.XsDL,mp.P2.compact.YsDL] = np.meshgrid(mp.P2.compact.xsDL,mp.P2.compact.xsDL)
    
    if(mp.full.flagPROPER):
        if mp.centering.lower() == ('interpixel'):
            mp.P1.full.Narr = falco.utils.ceil_even(mp.P1.full.Nbeam)
        else:
            mp.P1.full.Narr = falco.utils.ceil_even(mp.P1.full.Nbeam+1)
    else:
        mp.P1.full.Narr = len(mp.P1.full.mask)  ##--Total number of pixels across array containing the pupil in the full model. Add 2 pixels to Nbeam when the beam is pixel-centered.


    #--NORMALIZED (in pupil diameter) coordinate grids in the input pupil for making the tip/tilted input wavefront within the full model
    if mp.centering.lower() == ('interpixel'):
        mp.P2.full.xsDL = np.linspace(-(mp.P1.full.Narr-1)/2, (mp.P1.full.Narr-1)/2,mp.P1.full.Narr)*mp.P2.full.dx/mp.P2.D
    else:
        mp.P2.full.xsDL = np.linspace(-mp.P1.full.Narr/2, (mp.P1.full.Narr/2-1),mp.P1.full.Narr)*mp.P2.full.dx/mp.P2.D

    [mp.P2.full.XsDL,mp.P2.full.YsDL] = np.meshgrid(mp.P2.full.xsDL,mp.P2.full.xsDL)

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
