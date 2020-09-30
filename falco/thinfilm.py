"""Module for thin film functions to make complex-valued HLC occulters."""
import numpy as np
from os.path import isfile
import os
import numpy as np

import falco
from falco.check import real_scalar, real_positive_scalar,\
                            real_nonnegative_scalar, scalar_integer,\
                            positive_scalar_integer, real_array,\
                            oneD_array, twoD_array, twoD_square_array


def calc_complex_occulter(lam, aoi, t_Ti, t_Ni_vec, t_PMGI_vec,
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
    t_Ti : float
        Titanium thickness in meters. Titanium goes only between
        fused silica and nickel layers.
    t_Ni_vec : array_like
        1-D array of nickel thicknesses in meters. Nickel goes between
        titanium and PMGI layers.
    t_PMGI_vec : array_like
        1-D array of PMGI thicknesses in meters.
    d0 : float
        Reference height for all phase offsets. Must be larger than the stack
        of materials, not including the substrate. Units of meters.
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
    real_positive_scalar(lam, 'lam', TypeError)
    real_nonnegative_scalar(aoi, 'theta', TypeError)
    real_nonnegative_scalar(t_Ti, 't_Ti', TypeError)
    oneD_array(t_Ni_vec, 't_Ni_vec', ValueError)
    oneD_array(t_PMGI_vec, 't_PMGI_vec', ValueError)
    # if len(t_Ti) != len(t_Ni_vec) or len(t_Ni_vec) != len(t_PMGI_vec):
    #     raise ValueError('Ti, Ni, and PMGI thickness vectors must all ' +
    #                      'have same length.')
    scalar_integer(pol, 'pol', TypeError)
    
    lam_nm = lam * 1.0e9  # m --> nm
    lam_um = lam * 1.0e6  # m --> microns
    lam_um2 = lam_um * lam_um
    theta = aoi * (np.pi/180.)  # deg --> rad
    
    # Define Material Properties
    # ---------------------------------------------
    # Substrate properties
    if SUBSTRATE.upper() in ('FS', 'FUSEDSILICA'):
        A1 = 0.68374049400
        A2 = 0.42032361300
        A3 = 0.58502748000
        B1 = 0.00460352869
        B2 = 0.01339688560
        B3 = 64.49327320000
        n_substrate = np.sqrt(1 + A1*lam_um2/(lam_um2 - B1) +
                           A2*lam_um2/(lam_um2 - B2) +
                           A3*lam_um2/(lam_um2 - B3))

    elif SUBSTRATE.upper() in ('N-BK7', 'NBK7', 'BK7'):
        B1 = 1.03961212
        B2 = 0.231792344
        B3 = 1.01046945
        C1 = 0.00600069867
        C2 = 0.0200179144
        C3 = 103.560653
        n_substrate = np.sqrt(1 + (B1*lam_um2/(lam_um2 - C1)) +
                              (B2*lam_um2/(lam_um2 - C2)) +
                              (B3*lam_um2/(lam_um2 - C3)))
    
    # Dielectric properties
    npmgi = 1.524 + 5.176e-03/lam_um**2 + 2.105e-4/lam_um**4
    Ndiel = len(t_PMGI_vec)
    
    # Metal layer properties
    # Titanium base layer under the nickel
    Nmetal = len(t_Ni_vec)
    t_Ti_vec = t_Ti * np.ones(Nmetal)
    t_Ti_vec[np.asarray(t_Ni_vec) < 1e-10] = 0  # no Ti where no Ni
    # from D Moody
    titanium = np.array([
                        [397, 2.08, 2.95],
                        [413, 2.14, 2.98],
                        [431, 2.21, 3.01],
                        [451, 2.27, 3.04],
                        [471, 2.3, 3.1],
                        [496, 2.36, 3.19],
                        [521, 2.44, 3.2],
                        [549, 2.54, 3.43],
                        [582, 2.6, 3.58],
                        [617, 2.67, 3.74],
                        [659, 2.76, 3.84],
                        [704, 2.86, 3.96],
                        [756, 3.00, 4.01],
                        [821, 3.21, 4.01],
                        [892, 3.29, 3.96],
                        [984, 3.35, 3.97],
                        [1088, 3.5, 4.02],
                        [1216, 3.62, 4.15]
                        ])
    lam_ti = titanium[:, 0]  # nm
    n_ti = titanium[:, 1]
    k_ti = titanium[:, 2]
    nti = np.interp(lam_nm, lam_ti, n_ti)
    kti = np.interp(lam_nm, lam_ti, k_ti)
    
    # Nickel
    localpath = os.path.dirname(os.path.abspath(__file__))
    fnNickel = os.path.join(localpath, 'data',
                            'nickel_data_from_Palik_via_Bala_wvlNM_n_k.txt')
    vnickel = np.loadtxt(fnNickel, delimiter="\t", unpack=False, comments="#")
    lam_nickel = vnickel[:, 0]  # nm
    n_nickel = vnickel[:, 1]
    k_nickel = vnickel[:, 2]
    nnickel = np.interp(lam_nm, lam_nickel, n_nickel)
    knickel = np.interp(lam_nm, lam_nickel, k_nickel)
    
    # Compute the complex transmission
    # tCoef = np.zeros((Nmetal, ), dtype=complex)  # initialize
    # rCoef = np.zeros((Nmetal, ), dtype=complex)  # initialize

    # for ii in range(Nmetal):
    #     dni = t_Ni_vec[ii]
    #     dti = t_Ti_vec[ii]
    #     dpm = t_PMGI_vec[ii]
        
    #     nvec = np.array([1, 1, npmgi, nnickel-1j*knickel, nti-1j*kti,
    #                      n_substrate], dtype=complex)
    #     dvec = np.array([d0-dpm-dni-dti, dpm, dni, dti])
        
    #     # Choose polarization
    #     if(pol == 2):  # Mean of the two
    #         [dummy1, dummy2, rr0, tt0] = solver(nvec, dvec, theta,
    #                                             lam, False)
    #         [dummy1, dummy2, rr1, tt1] = solver(nvec, dvec, theta,
    #                                             lam, True)
    #         rr = (rr0+rr1)/2.
    #         tt = (tt0+tt1)/2.
    #     elif(pol == 0 or pol == 1):
    #         [dumm1, dummy2, rr, tt] = solver(nvec, dvec, theta, lam,
    #                                          bool(pol))
    #     else:
    #         raise ValueError('Wrong input value for polarization.')

    #     # Choose phase convention
    #     if not flagOPD:
    #         tCoef[ii] = np.conj(tt)  # Complex field transmission coef
    #         rCoef[ii] = np.conj(rr)  # Complex field reflection coef
    #     else:  # OPD phase convention
    #         tCoef[ii] = tt  # Complex field transmission coeffient
    #         rCoef[ii] = rr  # Complex field reflection coeffient
                
    # Compute the complex transmission
    tCoef = np.zeros((Ndiel, Nmetal), dtype=complex)  # initialize
    rCoef = np.zeros((Ndiel, Nmetal), dtype=complex)  # initialize
    for jj in range(Ndiel):
        dpm = t_PMGI_vec[jj]
        
        for ii in range(Nmetal):
            dni = t_Ni_vec[ii]
            dti = t_Ti_vec[ii]
            
            nvec = np.array([1, 1, npmgi, nnickel-1j*knickel, nti-1j*kti,
                              n_substrate], dtype=complex)
            dvec = np.array([d0-dpm-dni-dti, dpm, dni, dti])
            
            # Choose polarization
            if(pol == 2):  # Mean of the two
                [dummy1, dummy2, rr0, tt0] = solver(nvec, dvec, theta,
                                                    lam, False)
                [dummy1, dummy2, rr1, tt1] = solver(nvec, dvec, theta,
                                                    lam, True)
                rr = (rr0+rr1)/2.
                tt = (tt0+tt1)/2.
            elif(pol == 0 or pol == 1):
                [dumm1, dummy2, rr, tt] = solver(nvec, dvec, theta, lam,
                                                  bool(pol))
            else:
                raise ValueError('Wrong input value for polarization.')

            # Choose phase convention
            if not flagOPD:
                tCoef[jj, ii] = np.conj(tt)  # Complex field transmission coef
                rCoef[jj, ii] = np.conj(rr)  # Complex field reflection coef
            else:  # OPD phase convention
                tCoef[jj, ii] = tt  # Complex field transmission coeffient
                rCoef[jj, ii] = rr  # Complex field reflection coeffient
    
    return tCoef, rCoef


def solver(n, d0, theta, lam, tetm=False):
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
        False => TE, True => TM. The default is False.

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
    oneD_array(n, 'n', ValueError)
    oneD_array(d0, 'd0', ValueError)
    N = len(n)
    if not (len(d0) == N-2):
        raise ValueError('n and d size mismatch')
        pass
    real_nonnegative_scalar(theta, 'theta', TypeError)
    real_positive_scalar(lam, 'lam', TypeError)
    if not type(tetm) == bool:
        raise TypeError('tetm must be a boolean.')
    
    # np.hstac
    d = np.hstack((0, d0.reshape(len(d0, )), 0))
        
    kx = 2*np.pi*n[0]*np.sin(theta)/lam
    # sign agrees with measurement convention:
    kz = -np.sqrt((2*np.pi*n/lam)**2 - kx*kx)
    
    if tetm:
        kzz = kz/(n*n)
    else:
        kzz = kz
    
    eep = np.exp(-1j*kz*d)
    eem = np.exp(1j*kz*d)
    
    i1 = np.arange(N-1)
    i2 = np.arange(1, N)
    tin = 0.5*(kzz[i1] + kzz[i2])/kzz[i1]
    ri = (kzz[i1] - kzz[i2])/(kzz[i1] + kzz[i2])
    
    A = np.eye(2, dtype=complex)
    for i in range(N-1):
        A = A @ np.array(tin[i]*np.array([[eep[i], ri[i]*eep[i]],
                                          [ri[i]*eem[i], eem[i]]]))
    
    rr = A[1, 0] / A[0, 0]
    tt = 1 / A[0, 0]
    
    # transmitted power flux (Poynting vector . surface) depends on index of
    # the substrate and angle
    R = np.abs(rr)**2
    if tetm:
        Pn = np.real((kz[-1]/(n[-1]**2)) / (kz[0]/(n[0]**2)))
    else:
        Pn = np.real((kz[-1]/kz[0]))
        pass
    
    T = Pn*np.abs(tt)**2
    tt = np.sqrt(Pn)*tt
    
    return [R, T, rr, tt]
   

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
    localpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    mp.F3.metal = 'Ni'
    mp.F3.diel = 'PMGI'
    
    fn_compact = ('ct_cube_%s_Ti%.1fnm_%s_%.1fto%.1fby%.2f_%s_%.1fto%.1fby%.2f_wvl%dnm_BW%.1fN%d_%.1fdeg_compact.npy' %
        (SUBSTRATE, mp.t_Ti_nm, mp.F3.metal, np.min(mp.t_metal_nm_vec),
         np.max(mp.t_metal_nm_vec), mp.dt_metal_nm, mp.F3.diel,
         np.min(mp.t_diel_nm_vec), np.max(mp.t_diel_nm_vec), mp.dt_diel_nm,
         1e9*mp.lambda0, 100*mp.fracBW, mp.Nsbp, mp.aoi))
    fn_cube_compact = os.path.join(localpath, 'data', 'material', fn_compact)
    
    fn_full = ('ct_cube_%s_Ti%.1fnm_%s_%.1fto%.1fby%.2f_%s_%.1fto%.1fby%.2f_wvl%dnm_BW%.1f_%dN%d_%.1fdeg_full.npy' %
        (SUBSTRATE, mp.t_Ti_nm, mp.F3.metal, np.min(mp.t_metal_nm_vec), np.max(mp.t_metal_nm_vec), mp.dt_metal_nm,
        mp.F3.diel, np.min(mp.t_diel_nm_vec), np.max(mp.t_diel_nm_vec), mp.dt_diel_nm,
        (1e9*mp.lambda0), 100*mp.fracBW, mp.Nsbp, mp.Nwpsbp, mp.aoi))
    fn_cube_full = os.path.join(localpath, 'data', 'material', fn_full)

    if(flagRefl):
        fn_cube_compact = fn_cube_compact[0:-4] + '_refl.mat'
        fn_cube_full = fn_cube_full[0:-4] + '_refl.mat'
    
    t_Ti_m = 1e-9*mp.t_Ti_nm  # Static base layer of titanium beneath nickel.
    aoi = mp.aoi
    Nsbp = mp.Nsbp
    t_diel_m_vec = 1e-9*mp.t_diel_nm_vec  # PMGI thickness range
    t_metal_m_vec = 1e-9*mp.t_metal_nm_vec  # nickel thickness range
    
    Nmetal = len(mp.t_metal_nm_vec)
    Ndiel = len(mp.t_diel_nm_vec)
    
    # Compact Model: Load the data if it has been generated before; otherwise generate it.
    if(isfile(fn_cube_compact)):
        complexTransCompact = np.load(fn_cube_compact)
        print('Loaded complex transmission datacube for compact model: %s' % fn_cube_compact)
    else:
    
        print('Computing thin film equations for compact model:')
        complexTransCompact = np.zeros((Ndiel, Nmetal, mp.Nsbp), dtype=complex)
        sbp_centers = mp.sbp_centers
        
        # Parallel/distributed computing
        # To be completed later
        
        # Regular (serial) computing
        for si in range(Nsbp):
            lam = sbp_centers[si]
            d0 = lam * mp.F3.d0fac  # Max thickness of PMGI + Ni
            [tCoef, rCoef] = calc_complex_occulter(lam, aoi, t_Ti_m,
                                            t_metal_m_vec, t_diel_m_vec, d0, 2)
            if(flagRefl):
                complexTransCompact[:, :, si] = rCoef
            else:
                complexTransCompact[:, :, si] = tCoef
                pass
            print('\tDone computing wavelength %d of %d.\n' % (si, Nsbp))

        # Save out for future use
        np.save(fn_cube_compact, complexTransCompact)
        print('Saved complex transmission datacube: %s' % fn_cube_compact)
        pass
    
    # Full Model: Load the data if it has been generated before; otherwise generate it.
    if isfile(fn_cube_full):
        complexTransFull = np.load(fn_cube_full)
        print('Loaded complex transmission datacube for full model: %s' % fn_cube_full)
    else:
        print('Computing thin film equations for full model:')
        if mp.Nwpsbp == 1:
            complexTransFull = complexTransCompact
        else:
            complexTransFull = np.zeros((Ndiel, Nmetal, mp.Nsbp*mp.Nwpsbp),
                                        dtype=complex)
            lambdas = mp.full.lambdas
            
            # Parallel/distributed computing
            # To be completed later
            
            # Regular (serial) computing
            for li in range(len(lambdas)):
                lam = lambdas[li]
                d0 = lam * mp.F3.d0fac  # Max thickness of PMGI + Ni
                [tCoef, rCoef] = calc_complex_occulter(lam, aoi, t_Ti_m,
                                            t_metal_m_vec, t_diel_m_vec, d0, 2)
                if(flagRefl):
                    complexTransFull[:, :, li] = rCoef
                else:
                    complexTransFull[:, :, li] = tCoef
                    pass
                print('\tDone computing wavelength %d of %d.\n' %
                      (li, len(lambdas)))
        
        # Save out for future use
        np.save(fn_cube_full, complexTransFull)
        print('Saved complex transmission datacube: %s\n' % fn_cube_full)
    
    return complexTransCompact, complexTransFull
