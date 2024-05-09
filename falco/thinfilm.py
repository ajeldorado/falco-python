"""Module for thin film functions to make complex-valued HLC occulters."""
import numpy as np
from os.path import isfile
import os

from falco.check import real_scalar, real_positive_scalar,\
                            real_nonnegative_scalar, scalar_integer,\
                            positive_scalar_integer, real_array,\
                            oneD_array, twoD_array, twoD_square_array


def calc_complex_occulter(substrate, metal, dielectric, lam, aoi, t_Ti,
                          t_metal_map, t_diel_map, d0, pol, flagOPD=False):
    """
    Calculate the complex-valued transmission of a 2-D mask.

    Calculates the thin-film complex transmission for the provided 2-D maps
    of metal and dielectric thicknesses for a single wavelength.

    This function is a wrapper around calc_complex_trans_matrix().

    Parameters
    ----------
    substrate : str
        Name of the substrate material.
    metal : str
        Name of the metal used in the mask.
    dielectric : str
        Name of the dielectric used in the mask.
    lam : float
        Wavelength in meters.
    aoi : flaot
        Angle of incidence in degrees.
    t_Ti : float
        Titanium layer thickness in meters. Titanium is used in a uniform
        thickness only between the substrate and the main metal to help
        adhesion.
    t_metal_map : array_like
        2-D array of metal thicknesses in meters. This metal goes between the
        titanium and dielectric layers.
    t_diel_map : array_like
        2-D array of dielectric thicknesses in meters.
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

    Returns
    -------
    out_map : numpy ndarray
        2-D complex transmission map for the provided layer thicknesses.
    """
    real_nonnegative_scalar(t_Ti, 't_Ti', TypeError)
    twoD_array(t_metal_map, 't_metal_map', TypeError)
    twoD_array(t_diel_map, 't_diel_map', TypeError)

    out_map = np.zeros_like(t_metal_map, dtype=complex)

    t_Ti_map = np.zeros_like(t_metal_map)
    t_Ti_map[t_metal_map > 10*np.finfo(float).eps] = t_Ti
    # Put each vector as a column in a matrix
    t_unique_mat = np.unique(np.stack((t_diel_map.flatten(),
                                      t_metal_map.flatten(),
                                      t_Ti_map.flatten()),
                                      ),
                             axis=1)

    t_diel_vec_short = t_unique_mat[0, :]
    t_metal_vec_short = t_unique_mat[1, :]
    t_Ti_vec_short = t_unique_mat[2, :]

    Nshort = t_unique_mat.shape[1]
    # tCoefShort = np.zeros(Nshort)
    # rCoefShort = np.zeros(Nshort)

    for ii in range(Nshort):

        t_diel = t_diel_vec_short[ii]
        t_metal = t_metal_vec_short[ii]
        t_Ti_here = t_Ti_vec_short[ii]

        tCoef, rCoef = calc_complex_trans_matrix(
            substrate, metal, dielectric, lam, aoi, t_Ti_here,
            [t_metal, ], [t_diel, ], d0, pol, flagOPD)

        thisRegion = np.logical_and(
            np.logical_and(t_metal_map == t_metal, t_diel_map == t_diel),
            t_Ti_map == t_Ti_here)

        out_map[thisRegion] = tCoef[0]

    return out_map


def calc_complex_trans_matrix(substrate, metal, dielectric, lam, aoi, t_Ti,
                              t_metal_vec, t_diel_vec, d0, pol, flagOPD=False):
    """
    Calculate the thin-film complex transmission and reflectance.

    Calculates the thin-film complex transmission and reflectance for the
    provided combinations of metal and dielectric thicknesses at the given
    wavelength.

    Parameters
    ----------
    substrate : str
        Name of the substrate material.
    metal : str
        Name of the metal used in the mask.
    dielectric : str
        Name of the dielectric used in the mask.
    lam : float
        Wavelength in meters.
    aoi : flaot
        Angle of incidence in degrees.
    t_Ti : float
        Titanium layer thickness in meters. Titanium is used in a uniform
        thickness only between the substrate and the main metal to help
        adhesion.
    t_metal_vec : array_like
        1-D array of metal thicknesses in meters. This metal goes between the
        titanium and dielectric layers.
    t_diel_vec : array_like
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
    oneD_array(t_metal_vec, 't_metal_vec', ValueError)
    oneD_array(t_diel_vec, 't_diel_vec', ValueError)
    # if len(t_Ti) != len(t_metal_vec) or len(t_metal_vec) != len(t_diel_vec):
    #     raise ValueError('Ti, Ni, and PMGI thickness vectors must all ' +
    #                      'have same length.')
    scalar_integer(pol, 'pol', TypeError)

    lam_nm = lam * 1.0e9  # m --> nm
    lam_um = lam * 1.0e6  # m --> microns
    lam_um2 = lam_um * lam_um
    theta = aoi * (np.pi/180.)  # deg --> rad
    localpath = os.path.dirname(os.path.abspath(__file__))

    # Define Material Properties
    # ---------------------------------------------
    # Substrate properties
    if substrate.upper() in ('FS', 'FUSEDSILICA, FUSED_SILICA'):
        A1 = 0.68374049400
        A2 = 0.42032361300
        A3 = 0.58502748000
        B1 = 0.00460352869
        B2 = 0.01339688560
        B3 = 64.49327320000
        n_substrate = np.sqrt(1 + A1*lam_um2/(lam_um2 - B1) +
                              A2*lam_um2/(lam_um2 - B2) +
                              A3*lam_um2/(lam_um2 - B3))

    elif substrate.upper() in ('N-BK7', 'NBK7', 'BK7'):
        B1 = 1.03961212
        B2 = 0.231792344
        B3 = 1.01046945
        C1 = 0.00600069867
        C2 = 0.0200179144
        C3 = 103.560653
        n_substrate = np.sqrt(1 + (B1*lam_um2/(lam_um2 - C1)) +
                              (B2*lam_um2/(lam_um2 - C2)) +
                              (B3*lam_um2/(lam_um2 - C3)))

    else:
        raise ValueError('Invalid value of substrate for complex mask.')

    # Dielectric properties
    lenDiel = len(t_diel_vec)

    if dielectric.lower() in ('pmgi',):

        n_diel = 1.524 + 5.176e-03/lam_um**2 + 2.105e-4/lam_um**4
        k_diel = np.zeros_like(n_diel)

    elif dielectric.lower() in ('mgf2',):

        fn_mgf2 = os.path.join(
            localpath, 'data',
            'MgF2_data_from_Rodriguez-deMarcos_wvlUM_n_k.txt')
        dataMgF2 = np.loadtxt(fn_mgf2, delimiter="\t", unpack=False,
                              comments="#")
        lamUM_mgf2_0 = dataMgF2[:, 0]  # nm
        lam_mgf2_0 = lamUM_mgf2_0 * 1e3  # [nm]
        n_mgf2_0 = dataMgF2[:, 1]
        k_mgf2_0 = dataMgF2[:, 2]
        n_diel = np.interp(lam_nm, lam_mgf2_0, n_mgf2_0)
        k_diel = np.interp(lam_nm, lam_mgf2_0, k_mgf2_0)

    else:
        raise ValueError('Invalid value of dielectric for complex mask.')

    # Titanium base layer under the main metal layer
    lenMetal = len(t_metal_vec)
    t_Ti_vec = t_Ti * np.ones(lenMetal)
    t_Ti_vec[np.asarray(t_metal_vec) < 1e-10] = 0  # no Ti where no Ni
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

    # (Main) Metal layer properties
    if metal.lower() in ('nickel', 'ni'):

        fnNickel = os.path.join(
            localpath, 'data', 'nickel_data_from_Palik_via_Bala_wvlNM_n_k.txt')
        vnickel = np.loadtxt(fnNickel, delimiter="\t", unpack=False,
                             comments="#")
        lam_nickel = vnickel[:, 0]  # nm
        n_nickel_0 = vnickel[:, 1]
        k_nickel_0 = vnickel[:, 2]
        nnickel = np.interp(lam_nm, lam_nickel, n_nickel_0)
        knickel = np.interp(lam_nm, lam_nickel, k_nickel_0)

    else:
        raise ValueError('Invalid value of metal for complex mask.')

    # Compute the complex transmission
    # tCoef = np.zeros((lenMetal, ), dtype=complex)  # initialize
    # rCoef = np.zeros((lenMetal, ), dtype=complex)  # initialize

    # for ii in range(lenMetal):
    #     dni = t_metal_vec[ii]
    #     dti = t_Ti_vec[ii]
    #     dpm = t_diel_vec[ii]

    #     nvec = np.array([1, 1, n_diel, nnickel-1j*knickel, nti-1j*kti,
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
    tCoef = np.zeros((lenDiel, lenMetal), dtype=complex)  # initialize
    rCoef = np.zeros((lenDiel, lenMetal), dtype=complex)  # initialize
    for jj in range(lenDiel):
        dpm = t_diel_vec[jj]

        for ii in range(lenMetal):
            dni = t_metal_vec[ii]
            dti = t_Ti_vec[ii]

            nvec = np.array([1, 1, n_diel-1j*k_diel, nnickel-1j*knickel, nti-1j*kti,
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


def gen_complex_trans_table(mp, flagRefl=False):
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

    Returns
    -------
    complexTransCompact : numpy ndarray
        Complex transmission datacube for FALCO's compact model.
    complexTransFull : numpy ndarray
        Complex transmission datacube for FALCO's full model.
    """
    localpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # mp.F3.metal = 'Ni'
    # mp.F3.diel = 'PMGI'

    fn_compact = ('ct_cube_%s_Ti%.1fnm_%s_%.1fto%.1fby%.2f_%s_%.1fto%.1fby%.2f_wvl%dnm_BW%.1fN%d_%.1fdeg_compact.npy' %
        (mp.F3.substrate, mp.t_Ti_nm, mp.F3.metal, np.min(mp.t_metal_nm_vec),
         np.max(mp.t_metal_nm_vec), mp.dt_metal_nm, mp.F3.dielectric,
         np.min(mp.t_diel_nm_vec), np.max(mp.t_diel_nm_vec), mp.dt_diel_nm,
         1e9*mp.lambda0, 100*mp.fracBW, mp.Nsbp, mp.aoi))
    fn_cube_compact = os.path.join(localpath, 'data', 'material', fn_compact)

    fn_full = ('ct_cube_%s_Ti%.1fnm_%s_%.1fto%.1fby%.2f_%s_%.1fto%.1fby%.2f_wvl%dnm_BW%.1f_%dN%d_%.1fdeg_full.npy' %
        (mp.F3.substrate, mp.t_Ti_nm, mp.F3.metal, np.min(mp.t_metal_nm_vec), np.max(mp.t_metal_nm_vec), mp.dt_metal_nm,
         mp.F3.dielectric, np.min(mp.t_diel_nm_vec), np.max(mp.t_diel_nm_vec),
         mp.dt_diel_nm, (1e9*mp.lambda0), 100*mp.fracBW, mp.Nsbp, mp.Nwpsbp,
         mp.aoi))
    fn_cube_full = os.path.join(localpath, 'data', 'material', fn_full)

    if flagRefl:
        fn_cube_compact = fn_cube_compact[0:-4] + '_refl.npy'
        fn_cube_full = fn_cube_full[0:-4] + '_refl.npy'

    t_Ti_m = 1e-9*mp.t_Ti_nm  # Static base layer of titanium beneath nickel.
    aoi = mp.aoi
    Nsbp = mp.Nsbp
    t_diel_m_vec = 1e-9*mp.t_diel_nm_vec  # dielectric thickness range
    t_metal_m_vec = 1e-9*mp.t_metal_nm_vec  # nickel thickness range

    lenMetal = len(mp.t_metal_nm_vec)
    lenDiel = len(mp.t_diel_nm_vec)

    # Compact Model: Load pre-generated data, or else generate it.
    if(isfile(fn_cube_compact)):
        complexTransCompact = np.load(fn_cube_compact)
        print('Loaded complex transmission datacube for compact model: %s' %
              fn_cube_compact)
    else:

        print('Computing thin film equations for compact model:')
        complexTransCompact = np.zeros((lenDiel, lenMetal, mp.Nsbp), dtype=complex)
        sbp_centers = mp.sbp_centers

        # Parallel/distributed computing
        # To be completed later

        # Regular (serial) computing
        for si in range(Nsbp):
            lam = sbp_centers[si]
            d0 = lam * mp.F3.d0fac  # Max thickness of PMGI + Ni
            [tCoef, rCoef] = calc_complex_trans_matrix(
                mp.F3.substrate, mp.F3.metal, mp.F3.dielectric, lam, aoi,
                t_Ti_m, t_metal_m_vec, t_diel_m_vec, d0, 2)
            if flagRefl:
                complexTransCompact[:, :, si] = rCoef
            else:
                complexTransCompact[:, :, si] = tCoef
                pass
            print('\tDone computing wavelength %d of %d.\n' % (si, Nsbp))

        # Save out for future use
        np.save(fn_cube_compact, complexTransCompact)
        print('Saved complex transmission datacube: %s' % fn_cube_compact)
        pass

    # Full Model: Load pre-generated data or else generate it.
    if isfile(fn_cube_full):
        complexTransFull = np.load(fn_cube_full)
        print('Loaded complex transmission datacube for full model: %s' %
              fn_cube_full)
    else:
        print('Computing thin film equations for full model:')
        if mp.Nwpsbp == 1:
            complexTransFull = complexTransCompact
        else:
            complexTransFull = np.zeros((lenDiel, lenMetal, mp.Nsbp*mp.Nwpsbp),
                                        dtype=complex)
            lambdas = mp.full.lambdas

            # Parallel/distributed computing
            # To be completed later

            # Regular (serial) computing
            for li in range(len(lambdas)):
                lam = lambdas[li]
                d0 = lam * mp.F3.d0fac  # Max thickness of PMGI + Ni
                [tCoef, rCoef] = calc_complex_trans_matrix(
                    mp.F3.substrate, mp.F3.metal, mp.F3.dielectric, lam, aoi,
                    t_Ti_m, t_metal_m_vec, t_diel_m_vec, d0, 2)
                if flagRefl:
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
