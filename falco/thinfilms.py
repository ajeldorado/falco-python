"""Module for thin film functions."""
import numpy as np
from os.path import isfile


def discretize_fpm_surf(FPMsurf, t_nm_vec, dt_nm):
    """
    Discretize the surface profiles of the FPM materials.

    Parameters
    ----------
    FPMsurf : numpy ndarray
        2-D array of the FPM material layer
    t_nm_vec : numpy ndarray
        The vector of allowed thickness values for the material layer
    dt_nm : float
        Scalar value of the step size between allowed thickness values
    

    Returns
    -------
    DMtransInd : numpy ndarray, dtype = int
        Array index of the metal/dielectric layer for each DM8/9 surface value in the complex transmission matrix 

    """
    #--Convert surface profiles from meters to nanometers
    FPMsurf = 1e9*FPMsurf

    #--Stay within the thickness range since material properties are not defined outside it
    FPMsurf[FPMsurf<np.min(t_nm_vec)] = np.min(t_nm_vec)
    FPMsurf[FPMsurf>np.max(t_nm_vec)] = np.max(t_nm_vec)

    #--Discretize to find the index the the complex transmission array
    DMtransInd = 0 + np.round(1/dt_nm*(FPMsurf - np.min(t_nm_vec)));

    return DMtransInd.astype(int)
   
    
def solver(n, d0, theta, lam, tetm=0):
    """
    Solve the thin film equations for the given materials.

    Parameters
    ----------
    n : array_like
        index of refraction for each layer.
        n(1) = index of incident medium
        n(N) = index of transmission medium
        then length(n) must be >= 2
    d0 : array_like
        thickness of each layer, not counting incident medium or transmission
        medium. length(d) = length(n)-2.
    theta : float
        angle of incidence [radians].
    lam : float
        wavelength. units of lam must be same as d0.
    tetm : bool, optional
        0 => TE, 1 => TM. The default is 0.

    Returns
    -------
    R : numpy ndarray
        normalized reflected intensity coefficient
    T : numpy ndarray
        normalized transmitted intensity coefficient
    rr : numpy ndarray
        complex field reflection coefficient
    tt : numpy ndarray
        complex field transmission coefficient

    """
    N = len(n)
    if not (len(d0) == N-2):
        raise ValueError('n and d size mismatch')
        pass
    
    np.hstac
    d = np.hstack((0,d0.reshape(len(d0,)), 0))
        
    kx = 2*np.pi*n[0]*np.sin(theta)/lam
    kz = -np.sqrt( (2*np.pi*n/lam)**2 - kx**2 ) # sign agrees with measurement convention
    
    if (tetm == 1):
       kzz = kz/(n**2)
    else:
       kzz = kz
       pass
    

    eep = np.exp(-1j*kz*d)
    eem = np.exp(1j*kz*d)
    
    i1  = np.arange(N-1) #1:N-1; 
    i2  = np.arange(1,N) #2:N;
    tin = 0.5*(kzz[i1] + kzz[i2])/kzz[i1]
    ri  = (kzz[i1] - kzz[i2])/(kzz[i1] + kzz[i2])
    
    A = np.eye(2)
    for i in range(N-1):
    	A = A * np.array( [tin[i]*[eep[i], ri[i]*eep[i]], [ri[i]*eem[i], eem[i]]])
    
    rr = A[1,0]/A[0,0] #A(2,1)/A(1,1);
    tt = 1/A[0,0]
    
    # transmitted power flux (Poynting vector . surface) depends on index of the
    # substrate and angle
    R = np.abs(rr)**2
    if tetm == 1:
    	Pn = np.real( (kz[N-1]/(n[N-1]**2)) / (kz[0]/(n[0]**2)) );
    else:
        Pn = np.real((kz(N)/kz[0]))
        pass
    
    T = Pn*abs(tt)**2
    tt= np.sqrt[Pn]*tt
    
    return [R, T, rr, tt]


def define_materials(lam, aoi, t_Ti_base, t_Ni_vec, t_PMGI_vec, \
                                 d0, pol, flagOPD=False, SUBSTRATE='FS'): 
    """
    Calculate the thin-film complex transmission and reflectance.
    
    Calculates the thin-film complex transmission and reflectance for the
    provided combinations of metal and dielectric thicknesses and list of
    wavelengths.

    Parameters
    ----------
    lam : float
        Wavelength in meters.
    aoi : flaot
        Angle of incidence in degrees.
    t_Ti_base : float
        Thickness of the titanium layer between the glass and metal in meters.
    t_Ni_vec : array_like
        1-D array of nickel thicknesses in meters.
    t_PMGI_vec : array_like
        1-D array of PMGI thicknesses in meters.
    d0 : TYPE
        DESCRIPTION.
    pol : {0, 1, 2}
        Polarization state to compute values for. 
        0 for TE(s) polarization,
        1 for TM(p) polarization,
        2 for mean of s and p polarizations
    flagOPD : bool, optional
        Flag to use the OPD convention. The default is False.
    SUBSTRATE : str, optional
        Material to use as the substrate. The default is 'FS'.

    Returns
    -------
    tCoef : numpy ndarray
        2-D array of complex transmission amplitude values.
    rCoef : numpy ndarray
        2-D array of complex reflection amplitude values.
    """
    lam_nm = lam * 1.0e9 # m --> nm
    lam_u = lam*1.0e6; # m --> microns
    theta  = aoi*(np.pi/180.) #     % deg --> rad
    
    # Define Material Properties    
    # ---------------------------------------------
    #--Substrate properties
    if 'FS' in SUBSTRATE.upper():
        # ----------- Fused Silica from Dwight Moody------------------
        lamFS = 1e9*np.array([0.4e-6, 0.5e-6, .51e-6, .52e-6, .53e-6, .54e-6, .55e-6, .56e-6, .57e-6, .58e-6, .59e-6, .6e-6, .72e-6, .76e-6, 0.8e-6, 0.88e-6, 0.90e-6, 1.04e-6])    
        nFS = np.array([ 1.47012, 1.462, 1.462,  1.461,  1.461,  1.460,  1.460,  1.460,  1.459,  1.459,  1.458,  1.458, 1.45485, 1.45404, 1.45332, 1.45204, 1.45175, 1.44992])
        #vsilica = np.hstack((lamm, nx))
        #lam_silica = vsilica[:,0]  # nm
        #n_silica   = vsilica[:,1]
        n_substrate = np.interp(lam_nm,lamFS,nFS) #interp1(lam_silica, n_silica, lam_nm, 'linear');        
        pass 
    elif 'BK7' in SUBSTRATE.upper():
        B1 = 1.03961212
        B2 = 0.231792344
        B3 = 1.01046945
        C1 = 0.00600069867
        C2 = 0.0200179144
        C3 = 103.560653
    
        wvl_um = lam_u
        n_substrate = np.sqrt(1 + (B1*(wvl_um)**2/((wvl_um)**2 - C1)) + (B2*(wvl_um)**2/((wvl_um)**2 - C2)) + (B3*(wvl_um)**2/((wvl_um)**2 - C3)))
        pass
    
    #---------------------------------------------
    #--Dielectric properties
    npmgi = 1.524 + 5.176e-03/lam_u**2 + 2.105e-4/lam_u**4
    Ndiel  = len(t_PMGI_vec)
    
    #---------------------------------------------
    #--Metal layer properties
    #--New logic: Titanium layer goes beneath Nickel only. Always include them
    #together. Subtract off the thickness of the Ti layer from the intended Ni
    #layer thickness.
    
    Nmetal = len(t_Ni_vec);
    t_Ti_vec = np.zeros(Nmetal,)
    
    for ii in range(Nmetal):
        if(t_Ni_vec[ii] > t_Ti_base): #--For thicker layers
            t_Ni_vec[ii] = t_Ni_vec[ii] - t_Ti_base
            t_Ti_vec[ii] = t_Ti_base
        else: #--For very thin layers.
            t_Ti_vec[ii] = t_Ni_vec[ii]
            t_Ni_vec[ii] = 0
            pass
    
#    % % GUIDE:
#    % if(t_Ni > t_Ti) #--For thicker layers
#    %     t_Ni = t_Ni - t_Ti;
#    % else #--For very thin layers.
#    %     t_Ti = t_Ni;
#    %     t_Ni = 0;
    
    vnickel = np.loadtxt("nickel_data_from_Palik_via_Bala_wvlNM_n_k.txt",delimiter="\t", unpack=False,comments="#") 
    lam_nickel = vnickel[:,0]  # nm
    n_nickel   = vnickel[:,1]
    k_nickel   = vnickel[:,2]
    nnickel = np.interp(lam_nm,lam_nickel,n_nickel)
    knickel = np.interp(lam_nm,lam_nickel,k_nickel)
#    nnickel    = interp1(lam_nickel, n_nickel, lam_nm, 'linear');
#    knickel    = interp1(lam_nickel, k_nickel, lam_nm, 'linear');
    
    # ---------------------------------------------
    # from D Moody
    titanium =np.array([ 
        [397,          2.08,          2.95],
        [413,          2.14,          2.98],
        [431,          2.21,          3.01],
        [451,          2.27,          3.04],
        [471,           2.3,           3.1],
        [496,          2.36,          3.19],
        [521,          2.44,           3.2],
        [549,          2.54,          3.43],
        [582,           2.6,          3.58],
        [617,          2.67,          3.74],
        [659,          2.76,          3.84],
        [704,          2.86,          3.96],
        [756,             3,          4.01],
        [821,          3.21,          4.01],
        [892,          3.29,          3.96],
        [984,          3.35,          3.97],
        [1088,           3.5,          4.02],
        [1216,          3.62,          4.15]
        ])
        
    lam_ti = titanium[:,0] # nm
    n_ti   = titanium[:,1]
    k_ti   = titanium[:,2]
    nti = np.interp(lam_nm, lam_ti ,n_ti)
    kti = np.interp(lam_nm, lam_ti, k_ti)
#    nti    = interp1(lam_ti, n_ti, lam_nm, 'linear');
#    kti    = interp1(lam_ti, k_ti, lam_nm, 'linear');
    # ---------------------------------------------
    
    # Compute the complex transmission
    tCoef = np.zeros((Ndiel, Nmetal)) #--initialize
    rCoef = np.zeros((Ndiel, Nmetal)) #--initialize
    for jj in range(Ndiel):
        dpm = t_PMGI_vec[jj]
        
        for ii in range(Nmetal):
            dni = t_Ni_vec[ii]
            dti = t_Ti_vec[ii]
            
            nvec = np.array([1, 1, npmgi, nnickel-1j*knickel, nti-1j*kti, n_substrate],dtype=complex)
            dvec = np.array([d0-dpm-dni-dti, dpm, dni, dti])
            
            #--Choose polarization
            if(pol==2): #--Mean of the two
                [dummy1, dummy2, rr0, tt0] = solver(nvec, dvec, theta, lam, 0)
                [dummy1, dummy2, rr1, tt1] = solver(nvec, dvec, theta, lam, 1)
                rr = (rr0+rr1)/2.;
                tt = (tt0+tt1)/2.;
            elif(pol==0 or pol==1):
                [dumm1, dummy2, rr, tt] = solver(nvec, dvec, theta, lam, pol)
            else:
                #error('define_materials.m: Wrong input value for polarization.')
                pass

            #--Choose phase convention
            if not flagOPD:
                tCoef[jj,ii] = np.conj(tt) #--Complex field transmission coeffient
                rCoef[jj,ii] = np.conj(rr) #--Complex field reflection coeffient
            else: #--OPD phase convention is negative of oppositive convention
                tCoef[jj,ii] = tt #--Complex field transmission coeffient
                rCoef[jj,ii] = rr #--Complex field reflection coeffient
    
    return tCoef, rCoef


def gen_complex_trans_table(mp, flagRefl=False, SUBSTRATE='FS'):
    """
    Calculate 3-D look-up table for thin film transmission data.
    
    Calculate thin-film complex transmission data cube. The three dimensions
    are for metal thickness, dielectric thickness, and wavelength.
    
    Parameters
    ----------
    mp : ModelParameters
        Model parameters object.
    flagRefl : TYPE, optional
        Compute the thin film properties in reflection. The default is False.
    SUBSTRATE : TYPE, optional
        Change the substrate material. The default is 'FS'. The only other
        option is 'BK7'.

    Returns
    -------
    complexTransCompact : numpy ndarray
        Complex transmission datacube for FALCO's compact model.
    complexTransFull : numpy ndarray
        Complex transmission datacube for FALCO's full model.
    """
    mp.F3.metal = 'Ni'
    mp.F3.diel = 'PMGI'
    
    fn_cube_compact = print('%s/data/material/ct_cube_%s_Ti%.1fnm_%s_%.1fto%.1fby%.2f_%s_%.1fto%.1fby%.2f_wvl%dnm_BW%.1fN%d_%.1fdeg_compact.npy' %
        (mp.path.falco,SUBSTRATE,mp.t_Ti_nm,mp.F3.metal,min(mp.t_metal_nm_vec), max(mp.t_metal_nm_vec), mp.dt_metal_nm,
        mp.F3.diel, min(mp.t_diel_nm_vec),  max(mp.t_diel_nm_vec),  mp.dt_diel_nm,
        (1e9*mp.lambda0),100*mp.fracBW,mp.Nsbp,mp.aoi))
    
    fn_cube_full = print('%s/data/material/ct_cube_%s_Ti%.1fnm_%s_%.1fto%.1fby%.2f_%s_%.1fto%.1fby%.2f_wvl%dnm_BW%.1f_%dN%d_%.1fdeg_full.npy' %
        (mp.path.falco,SUBSTRATE,mp.t_Ti_nm,mp.F3.metal,min(mp.t_metal_nm_vec), max(mp.t_metal_nm_vec), mp.dt_metal_nm, 
        mp.F3.diel, min(mp.t_diel_nm_vec),  max(mp.t_diel_nm_vec),  mp.dt_diel_nm,
        (1e9*mp.lambda0),100*mp.fracBW,mp.Nsbp,mp.Nwpsbp,mp.aoi))
    
    if(flagRefl):
        fn_cube_compact = fn_cube_compact[0:-4] + '_refl.mat'
        fn_cube_full = fn_cube_full[0:-4] + '_refl.mat'
    
    t_Ti_m = 1e-9*mp.t_Ti_nm #--Static base layer of titanium beneath any nickel.
    aoi = mp.aoi
    Nsbp = mp.Nsbp
    t_diel_m_vec = 1e-9*mp.t_diel_nm_vec #--PMGI thickness range
    t_metal_m_vec = 1e-9*mp.t_metal_nm_vec #--nickel thickness range
    
    Nmetal = len(mp.t_metal_nm_vec)
    Ndiel  = len(mp.t_diel_nm_vec)
    
    #--Compact Model: Load the data if it has been generated before; otherwise generate it.
    if(isfile(fn_cube_compact)):
        complexTransCompact = np.load(fn_cube_compact)
        print('Loaded complex transmission datacube for compact model: %s' % fn_cube_compact)    
    else:
    
        print('Computing thin film equations for compact model:')
        complexTransCompact = np.zeros((Ndiel,Nmetal,mp.Nsbp))
        sbp_centers = mp.sbp_centers
        
        #--Parallel/distributed computing
        # To be completed later
        
        #--Regular (serial) computing
        for si in range(Nsbp):
            lam = sbp_centers[si]
            d0 = lam * mp.FPM.d0fac # Max thickness of PMGI + Ni
            [tCoef,rCoef] = define_materials(lam, aoi, t_Ti_m, t_metal_m_vec, t_diel_m_vec, d0, 2) 
            if(flagRefl):
                complexTransCompact[:,:,si] = rCoef     
            else:
                complexTransCompact[:,:,si] = tCoef
                pass
            print('\tDone computing wavelength %d of %d.\n' % (si,Nsbp))

        #--Save out for future use
        np.save(fn_cube_compact,complexTransCompact)
        print('Saved complex transmission datacube: %s' % fn_cube_compact)
        pass
    
    #--Full Model: Load the data if it has been generated before; otherwise generate it.
    if(isfile(fn_cube_full)):
        complexTransFull = np.load(fn_cube_full)
        print('Loaded complex transmission datacube for full model: %s' % fn_cube_full)    
    else:
        print('Computing thin film equations for full model:')
        if(mp.Nwpsbp==1):
            complexTransFull = complexTransCompact
        else:
            complexTransFull = np.zeros((Ndiel,Nmetal,mp.Nsbp*mp.Nwpsbp))
            lambdas = mp.full.lambdas
            
            #--Parallel/distributed computing
            # To be completed later
            
            #--Regular (serial) computing
            for li in range(len(lambdas)):
                lam = lambdas[li]
                d0 = lam * mp.FPM.d0fac # Max thickness of PMGI + Ni
                [tCoef,rCoef] = define_materials(lam, aoi, t_Ti_m, t_metal_m_vec, t_diel_m_vec, d0, 2)
                if(flagRefl):
                    complexTransFull[:,:,li] = rCoef
                else:
                    complexTransFull[:,:,li] = tCoef
                    pass
                print('\tDone computing wavelength %d of %d.\n' % (li,len(lambdas)))
        
        #--Save out for future use
        np.save(fn_cube_full,complexTransFull)
        print('Saved complex transmission datacube: %s\n' % fn_cube_full)
    
    return complexTransCompact, complexTransFull
