"""Functions for Zernike generation and sensitivity calculations."""
import falco
import numpy as np
import multiprocessing
import copy
import math

import proper

_VALID_CENTERING = ['pixel', 'interpixel']
_CENTERING_ERR = 'Invalid centering specification. Options: {}'.format(_VALID_CENTERING)


def calc_zern_sens(mp):
    """
    Ccompute the Zernike aberration sensitivities of the coronagraph.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters

    Returns
    -------
    dE2mat : numpy ndarray
        A 2-D array of Zernike sensitivities for different radial zones and Zernike modes.
    """
    indsZnoll = mp.eval.indsZnoll
    Rsens = mp.eval.Rsens  # Radii ranges. Can overlap.
    Nannuli = Rsens.shape[0]
    Nzern = indsZnoll.size
 
    # Make scoring masks
    maskCube = np.zeros((mp.Fend.Neta, mp.Fend.Nxi, Nannuli))
    for ni in range(Nannuli):
        # Make scoring masks for the annular regions
        # Set Inputs:
        maskDict = {}
        maskDict["pixresFP"] = mp.Fend.res
        maskDict["rhoInner"] = Rsens[ni, 0]  # [lambda0/D]
        maskDict["rhoOuter"] = Rsens[ni, 1]  # [lambda0/D]
        maskDict["angDeg"] = mp.Fend.corr.ang  # [degrees]
        maskDict["centering"] = mp.centering
        maskDict["FOV"] = mp.Fend.FOV
        maskDict["whichSide"] = mp.Fend.sides  # which sides the dark hole exists in
        if hasattr(mp.Fend, 'shape'):
            maskDict.shape = mp.Fend.shape
        maskCube[:, :, ni], xisDL, etasDL = falco.mask.falco_gen_SW_mask(maskDict)

    if not mp.full.flagPROPER:  # When using full models completely made with PROPER
        # Generate cube of normalized (RMS = 1) Zernike modes.
        ZmapCube = gen_norm_zern_maps(mp.P1.full.Nbeam, mp.centering, indsZnoll)
        # Make sure ZmapCube is padded or cropped to the right array size
        if not ZmapCube.shape[0] == mp.P1.full.Narr:
            ZmapCubeTemp = np.zeros((mp.P1.full.Narr, mp.P1.full.Narr, Nzern))
            for zi in range(Nzern):
                ZmapCubeTemp[:, :, zi] = falco.util.pad_crop(np.squeeze(ZmapCube[:, :, zi]),
                                                             mp.P1.full.Narr)
            ZmapCube = ZmapCubeTemp
            del ZmapCubeTemp

    # Number of polarization states used
    if not hasattr(mp, 'full'):
        mp.full = falco.config.Object()  # Initialize if this doesn't exist
    if hasattr(mp.full, 'pol_conds'):
        Npol = len(mp.full.pol_conds)
    else:
        Npol = 1

    # Get unaberrated E-fields
    # Loop over all wavelengths and polarizations
    inds_list = [(x, y) for x in np.arange(mp.full.NlamUnique) for y in np.arange(Npol)]
    Nvals = mp.full.NlamUnique*Npol
    
    # Get nominal, unaberrated final E-field at each wavelength and polarization
    E0array = np.zeros((mp.Fend.Neta, mp.Fend.Nxi, mp.full.NlamUnique, Npol), dtype=complex)
    Eunab = np.zeros((mp.Fend.Neta, mp.Fend.Nxi, Nvals), dtype=complex)  # Temporary array

    print('Computing unaberrated E-fields for Zernike sensitivities...\t', end='')
    if mp.flagMultiproc:
        pool = multiprocessing.Pool(processes=mp.Nthreads)
        resultsRaw = [pool.apply_async(falco_get_single_sim_Efield_LamPol,
                                       args=(iv, inds_list, mp)) for iv in range(Nvals)]
        results = [p.get() for p in resultsRaw]  # All the E-fields in a list
        pool.close()
        pool.join()
        for iv in range(Nvals):
            Eunab[:, :, iv] = results[iv]
    else:
        for iv in range(Nvals):
            Eunab[:, :, iv] = falco_get_single_sim_Efield_LamPol(iv, inds_list, mp)
    print('done.')

    # Reorganize the output
    for iv in range(Nvals):
        ilam = inds_list[iv][0]
        ipol = inds_list[iv][1]
        E0array[:, :, ilam, ipol] = Eunab[:, :, iv]
    del Eunab
    
    # Get E-fields with Zernike aberrations
    # Loop over all wavelengths, polarizations, and Zernike modes
    inds_list_zern = [(x, y, z) for x in np.arange(mp.full.NlamUnique)
                      for y in np.arange(Npol) for z in np.arange(Nzern)]
    NvalsZern = mp.full.NlamUnique*Npol*Nzern

    # Get nominal, unaberrated final E-field at each wavelength and polarization
    dEZarray = np.zeros((mp.Fend.Neta, mp.Fend.Nxi, mp.full.NlamUnique, Npol, Nzern),
                        dtype=complex)
    Eab = np.zeros((mp.Fend.Neta, mp.Fend.Nxi, NvalsZern), dtype=complex)
    
    print('Computing aberrated E-fields for Zernike sensitivities...\t', end='')
    if mp.flagMultiproc:
        pool = multiprocessing.Pool(processes=mp.Nthreads)
        resultsRaw = [pool.apply_async(falco_get_single_sim_Efield_LamPolZern,
                                       args=(iv, inds_list_zern, mp)) for iv in range(NvalsZern)]
        results = [p.get() for p in resultsRaw]  # All the E-fields in a list
        pool.close()
        pool.join()
        for iv in range(NvalsZern):
            Eab[:, :, iv] = results[iv]
        pass
    else:
        for iv in range(NvalsZern):
            Eab[:, :, iv] = falco_get_single_sim_Efield_LamPolZern(iv, inds_list_zern, mp)
    print('done.')
    
    # Reorganize the output and compute delta E-field
    for ni in range(NvalsZern):
        ilam = inds_list_zern[ni][0]
        ipol = inds_list_zern[ni][1]
        izern = inds_list_zern[ni][2]
        dEZarray[:, :, ilam, ipol, izern] = Eab[:, :, ni] - E0array[:, :, ilam, ipol]
    del Eab
    
    # Compute Zernike sensitivity values averaged across each annulus in dark hole
    # |dE|^2 averaged over wavelength and polarization state
    dE2cube = np.squeeze(np.mean(np.mean(np.abs(dEZarray)**2, axis=3), axis=2))
    dE2mat = np.zeros((Nzern, Nannuli))
    for iz in range(Nzern):
        dEtemp = np.squeeze(dE2cube[:, :, iz])
        for ia in range(Nannuli):
            dE2mat[iz, ia] = np.mean(dEtemp[(np.squeeze(maskCube[:, :, ia]) == 1)])

    # Print Zernike sensitivity results to command line
    for iz in range(Nzern):
        print('|dE|^2 at %dnm with %dnm RMS of    Z%d =' %
              (np.round(mp.lambda0*1e9), np.round(1e9*mp.full.ZrmsVal), indsZnoll[iz]), end='')
        for ia in range(Nannuli):
            print('\t%.2e (%.1f-%.1f l/D)' % (dE2mat[iz, ia], Rsens[ia, 0], Rsens[ia, 1]), end='')
        print('\n', end='')

    return dE2mat


def falco_get_single_sim_Efield_LamPol(ni, inds_list, mp):
    """
    Calculate the E-field for a given wavelength and polarization state.

    Used only by the function calc_zern_sens.

    Parameters
    ----------
    ni : int
        index for the set of possible combinations of wavelengths and polarization states
    inds_list : list
        the set of possible index combinations for wavelengths and polarization states
    mp : ModelParameters
        Structure containing optical model parameters

    Returns
    -------
    Estar : numpy ndarray
        2-D array of the stellar E-field for the given wavelength and polarization state.
    """
    # This function is used only by calc_zern_sens
    
    ilam = inds_list[ni][0]
    ipol = inds_list[ni][1]
    
    # Get the stellar E-field
    modvar = falco.config.Object()
    modvar.sbpIndex = mp.full.indsLambdaMat[mp.full.indsLambdaUnique[ilam], 0]
    modvar.wpsbpIndex = mp.full.indsLambdaMat[mp.full.indsLambdaUnique[ilam], 1]
    mp.full.polaxis = mp.full.pol_conds[ipol]
    modvar.whichSource = 'star'
    Estar = falco.model.full(mp, modvar)
    
    return Estar


def falco_get_single_sim_Efield_LamPolZern(ni, inds_list_zern, mp):
    """
    Calculate the E-field for a given wavelength, polarization, and Zernike.
    
    Used only by the function calc_zern_sens.

    Parameters
    ----------
    ni : int
        index for the set of possible combinations of wavelengths, polarization
        states, and Zernike modes
    inds_list : list
        the set of possible index combinations for wavelengths, polarization
        states, and Zernike modes
    mp : ModelParameters
        Structure containing optical model parameters

    Returns
    -------
    Estar : numpy ndarray
        2-D array of the stellar E-field for the given wavelength, polarization
        state, and Zernike mode.
    """
    # This function is used only by calc_zern_sens
    
    ilam = inds_list_zern[ni][0]
    ipol = inds_list_zern[ni][1]
    izern = inds_list_zern[ni][2]
    
    indsZnoll = mp.eval.indsZnoll
    
    # Get the stellar E-field
    si = mp.full.indsLambdaMat[mp.full.indsLambdaUnique[ilam], 0]
    wi = mp.full.indsLambdaMat[mp.full.indsLambdaUnique[ilam], 1]
    modvar = falco.config.Object()
    modvar.sbpIndex = si
    modvar.wpsbpIndex = wi
    mp.full.polaxis = mp.full.pol_conds[ipol]
    modvar.whichSource = 'star'
    
    # Save the original input E-field as E0 to reset it later
    E = mp.P1.full.E
    E0 = E.copy()
    
    # Initialize the Zernike modes to include as empty if the variable doesn't exist already
    if not hasattr(mp.full, 'zindex'):
        mp.full.zindex = np.array([])
        mp.full.zval_m = np.array([])
    zindex0 = copy.copy(mp.full.zindex)  # Save the original
    zval_m0 = copy.copy(mp.full.zval_m)  # Save the original
    
    if(mp.full.flagPROPER):  # WARNING: THIS OPTION HAS NOT BEEN TESTED

        # Put the Zernike index and coefficent in the vectors used by the PROPER full model
        if(any(zindex0 == indsZnoll[izern])):  # Add the delta to an existing entry
            zind = np.nonzero(zindex0 == indsZnoll[izern])[0]
            mp.full.zval_m[zind] = mp.full.zval_m[zind] + mp.full.ZrmsVal
        else:  # Concatenate the Zenike modes to the vector if it isn't included already
            mp.full.zindex = np.concatenate((zindex0, np.array([indsZnoll[izern]]))).astype(int)
            mp.full.zval_m = np.concatenate((zval_m0, np.array([mp.full.ZrmsVal])))  # [meters]
        mp.full.zval = mp.full.zval_m  # for PROPER models defined differently
        
    else:  # Include the Zernike map at the input pupil for the FALCO full model
        ZernMap = np.squeeze(gen_norm_zern_maps(mp.P1.full.Nbeam, mp.centering,
                                                np.array([indsZnoll[izern]])))
        ZernMap = falco.util.pad_crop(ZernMap, mp.P1.full.Narr)  # Adjust zero padding if necessary
        mp.P1.full.E[:, :, wi, si] = np.exp(1j*2*np.pi/mp.full.lambdasMat[si, wi] *
                                            mp.full.ZrmsVal*ZernMap) *\
            np.squeeze(mp.P1.full.E[:, :, wi, si])
        
    Estar = falco.model.full(mp, modvar)
    mp.P1.full.E = E0  # Reset to original value
    mp.full.zindex = zindex0
    mp.full.zval_m = zval_m0
    
    return Estar


def gen_norm_zern_maps(Nbeam, centering, indsZnoll):
    """
    Compute normalized 2-D maps of the specified Zernike modes.

    Parameters
    ----------
    Nbeam : int
        The number of pixels across the circle over which to compute the Zernike
    centering : str
        The centering of the array. Either 'pixel' or 'interpixel'.
    indsZnoll : numpy ndarray
        The iterable set of Zernike modes for which to compute maps.
    

    Returns
    -------
    ZmapCube : numpy ndarray
        A datacube in which each slice is a normalized Zernike mode.
    """
    if centering not in _VALID_CENTERING:
        raise ValueError(_CENTERING_ERR)
        
    # If a scalar integer, convert indsZnoll to an array so that it is indexable
    if not (type(indsZnoll) == np.ndarray):
        indsZnoll = np.array([indsZnoll])

    # Set array size as minimum width to contain the beam.
    if 'interpixel' in centering:
        Narray = falco.util.ceil_even(Nbeam)
    else:
        Narray = falco.util.ceil_even(Nbeam+1)

    # PROPER setup values
    Dbeam = 1.  # Diameter of aperture, normalized to itself
    wl = 1e-6  # wavelength (m); Dummy value--no propagation here, so not used.
    beam_diam_frac = Narray/Nbeam

    # Initialize wavefront structure in PROPER
    bm = proper.prop_begin(Dbeam, wl, Narray, beam_diam_frac)

    # Use modified PROPER function to generate the Zernike datacube
    Nzern = indsZnoll.size
    ZmapCube = np.zeros((Narray, Narray, Nzern))
    
    bm.centering = centering
    for iz in range(Nzern):
        ZmapCube[:, :, iz] = propcustom_zernikes(bm, np.array([indsZnoll[iz]]),
                np.array([1.]), NO_APPLY=True, CENTERING=centering)
        
    return ZmapCube

     
def propcustom_zernikes(a, zernike_num, zernike_val, eps=0., **kwargs):
    """
    Add Zernike-polynomial wavefront errors to current wavefront.
    
    Noll ordering is used and a circular system is assumed. An arbitrary number
    of Zernikes normalized for an unobscured circular region can be computed,
    but only the first 22 Zernikes can be computed normalized for a
    centrally-obscured region.
    
    
    Parameters
    ----------
    a : object
        WaveFront class object
        
    zernike_num : numpy ndarray
        Scalar or 1D array specifying which Zernike polynomials to include
        
    zernike_val : numpy ndarray
        Scalar or 1D array containing Zernike coefficients (in meters of RMS
        wavefront phase error or dimensionless RMS amplitude error) for Zernike
        polynomials indexed by "zernike_num".
        
    eps : float
        Central obscuration ratio (0.0-1.0); default is 0.0
        
    
    Returns
    -------
        None
        Adds wavefront errors to current wavefront array
        
    dmap : numpy ndarray
        Aberration map
        
    
    Other Parameters
    ----------------
    AMPLITUDE : bool
        Optional keyword that specifies that the Zernike values in "zernike_val"
        represent the wavefront RMS amplitude (rather than phase) variation.
        The current wavefront will be multipled by the generated map.
        
    NAME : str
        String containing name of surface that will be printed when executed.
        
    NO_APPLY : bool
        If set, the aberration map will be generated but will not be applied to
        the wavefront. This is useful if you just want to generate a map for
        your own use and modification (e.g. to create an error map for a multi-
        segmented system, each with its own aberration map).
        
    RADIUS : float
        Optional keyword specifying the radius to which the Zernike polynomials
        are normalized. If not specified, the pilot beam radius is used.
    
    CENTERING : str
        String containing the centering of the array. "interpixel" centers the
        array between pixels. Any other value gives pixel centering.
    
    
    Raises
    ------
    ValueError:
        Maximum index for an obscured Zernike polynomial is 22
    
    
    Notes
    -----
    The user specifies 1D arrays containing the Zernike polynomial coefficient
    indicies, the respective coefficients, and if an obstructed circular aperture
    the central obscuration ratio. A wavefront error map will be generated and
    added to the current wavefront.
    
    Zernike index and corresponding aberration for 1st 22 zernikes
    1 : Piston
    2 : X tilt
    3 : Y tilt
    4 : Focus
    5 : 45 degree astigmatism
    6 : 0 degree astigmatism
    7 : Y coma
    8 : X coma
    9 : Y clover (trefoil)
    10 : X clover (trefoil)
    11 : 3rd order spherical
    12 : 5th order 0 degree astig
    13 : 5th order 45 degree astig
    14 : X quadrafoil
    15 : Y quadrafoil
    16 : 5th order X coma
    17 : 5th order Y coma
    18 : 5th order X clover
    19 : 5th order Y clover
    20 : X pentafoil
    21 : Y pentafoil
    22 : 5th order spherical
    
    Update - JEK - Fixed use of ** instead of ** in obscured Zernikes
    Update - AR - Added CENTERING as keyword
    """
    zernike_num = np.asarray(zernike_num)
    zernike_val = np.asarray(zernike_val)
    n = proper.n
    
    if proper.print_it and not ("NO_APPLY" in kwargs and kwargs["NO_APPLY"]):
        if "NAME" in kwargs:
            print("Applying aberrations at %s" % kwargs["NAME"])
        else:
            print("Applying aberrations")
            
    max_z = zernike_num.max()
    
    if eps != 0. and max_z > 22:
        raise ValueError("PROP_ZERNIKES: Maximum index for an obscured Zernike polynomial is 22.")
        
    dmap = np.zeros([n, n], dtype=np.float64)
    
    if "RADIUS" not in kwargs:
        beam_radius = proper.prop_get_beamradius(a)
    else:
        beam_radius = kwargs["RADIUS"]
        
    dx = proper.prop_get_sampling(a) / beam_radius
    x_offset = 0.
    y_offset = 0.
    if("CENTERING" in kwargs):
        if kwargs["CENTERING"] not in _VALID_CENTERING:
            raise ValueError(_CENTERING_ERR)
    
        # Shift by half pixel
        if('interpixel' in kwargs["CENTERING"]):
            x_offset = dx/2.
            y_offset = dx/2.

    x = (np.arange(n, dtype=np.float64) - n//2) * dx + x_offset
    # x_pow_2 = x**2
    
    if (eps == 0.):
        # get list of executable equations defining Zernike polynomials
        zlist, maxrp, maxtc = proper.prop_noll_zernikes(max_z, COMPACT=True,
                                                        EXTRA_VALUES=True)
        
        for j in range(n):
            ab = np.zeros(n, dtype=np.float64)
            y = (j - n//2) * dx + y_offset
            r = np.sqrt(x**2 + y**2)
            t = np.arctan2(y, x)
        
            # predefine r**power, cos(const*theta), sin(const*theta) vectors
            for i in range(2, maxrp+1):
                rps = str(i).strip()
                cmd = "r_pow_" + rps + " = r**i"
                exec(cmd)
            
            for i in range(1, maxtc+1):
                tcs = str(i).strip()
                cmd = "cos" + tcs + "t = np.cos(i*t)"
                exec(cmd)
                cmd = "sin" + tcs + "t = np.sin(i*t)"
                exec(cmd)
           
            # assemble aberrations
            for iz in range(zernike_num.size):
                tmp = eval(zlist[zernike_num[iz]])
                ab += zernike_val[iz] * tmp
            
            dmap[j, :] += ab
            
    else:
        
        for j in range(n):
            y = (j-n//2) * dx + y_offset
            r = np.sqrt(x**2 + y**2)
            r2 = r**2
            r3 = r**3
            r4 = r**4
            r5 = r**5
            t = np.arctan2(y, x)
            
            for iz in range(len(zernike_num)):
                if zernike_num[iz] == 1:
                    ab = 1.
                elif zernike_num[iz] == 2:
                    ab = (2*r*np.cos(t))/np.sqrt(1 + eps**2)
                elif zernike_num[iz] == 3:
                    ab = (2*r*np.sin(t))/np.sqrt(1 + eps**2)
                elif zernike_num[iz] == 4:
                    ab = (np.sqrt(3)*(1 + eps**2 - 2*r2))/(-1 + eps**2)
                elif zernike_num[iz] == 5:
                    ab = (np.sqrt(6)*r2*np.sin(2*t))/np.sqrt(1 + eps**2 + eps**4)
                elif zernike_num[iz] == 6:
                    ab = (np.sqrt(6)*r2*np.cos(2*t))/np.sqrt(1 + eps**2 + eps**4)
                elif zernike_num[iz] == 7:
                    ab = (2*np.sqrt(2)*r*(2 + 2*eps**4 - 3*r2 + eps**2*(2 - 3*r2))*np.sin(t))/((-1 + eps**2)*np.sqrt(1 + 5*eps**2 + 5*eps**4 + eps**6))
                elif zernike_num[iz] == 8:
                    ab = (2*np.sqrt(2)*r*(2 + 2*eps**4 - 3*r2 + eps**2*(2 - 3*r2))*np.cos(t))/((-1 + eps**2)*np.sqrt(1 + 5*eps**2 + 5*eps**4 + eps**6))
                elif zernike_num[iz] == 9:
                    ab = (2*np.sqrt(2)*r3*np.sin(3*t))/np.sqrt(1 + eps**2 + eps**4 + eps**6)
                elif zernike_num[iz] == 10:
                    ab = (2*np.sqrt(2)*r3*np.cos(3*t))/np.sqrt(1 + eps**2 + eps**4 + eps**6)
                elif zernike_num[iz] == 11:
                    ab = (np.sqrt(5)*(1 + eps**4 - 6*r2 + 6*r4 + eps**2*(4 - 6*r2)))/ (-1 + eps**2)**2
                elif zernike_num[iz] == 12:
                    ab = (np.sqrt(10)*r2*(3 + 3*eps**6 - 4*r2 + eps**2*(3 - 4*r2) + eps**4*(3 - 4*r2))*np.cos(2*t))/ ((-1 + eps**2)*np.sqrt((1 + eps**2 + eps**4)*(1 + 4*eps**2 + 10*eps**4 + 4*eps**6 + eps**8)))
                elif zernike_num[iz] == 13:
                    ab = (np.sqrt(10)*r2*(3 + 3*eps**6 - 4*r2 + eps**2*(3 - 4*r2) + eps**4*(3 - 4*r2))*np.sin(2*t))/ ((-1 + eps**2)*np.sqrt((1 + eps**2 + eps**4)*(1 + 4*eps**2 + 10*eps**4 + 4*eps**6 + eps**8)))
                elif zernike_num[iz] == 14:
                    ab = (np.sqrt(10)*r4*np.cos(4*t))/np.sqrt(1 + eps**2 + eps**4 + eps**6 + eps**8)
                elif zernike_num[iz] == 15:
                    ab = (np.sqrt(10)*r4*np.sin(4*t))/np.sqrt(1 + eps**2 + eps**4 + eps**6 + eps**8)
                elif zernike_num[iz] == 16:
                    ab = (2*np.sqrt(3)*r*(3 + 3*eps**8 - 12*r2 + 10*r4 - 12*eps**6*(-1 + r2) + 2*eps**4*(15 - 24*r2 + 5*r4) + 4*eps**2*(3 - 12*r2 + 10*r4))*np.cos(t))/((-1 + eps**2)**2*np.sqrt((1 + 4*eps**2 + eps**4)* (1 + 9*eps**2 + 9*eps**4 + eps**6)))
                elif zernike_num[iz] == 17:
                    ab = (2*np.sqrt(3)*r*(3 + 3*eps**8 - 12*r2 + 10*r4 - 12*eps**6*(-1 + r2) + 2*eps**4*(15 - 24*r2 + 5*r4) + 4*eps**2*(3 - 12*r2 + 10*r4))*np.sin(t))/((-1 + eps**2)**2*np.sqrt((1 + 4*eps**2 + eps**4)* (1 + 9*eps**2 + 9*eps**4 + eps**6)))
                elif zernike_num[iz] == 18:
                    ab = (2*np.sqrt(3)*r3*(4 + 4*eps**8 - 5*r2 + eps**2*(4 - 5*r2) + eps**4*(4 - 5*r2) + eps**6*(4 - 5*r2))*np.cos(3*t))/ ((-1 + eps**2)*np.sqrt((1 + eps**2 + eps**4 + eps**6)*(1 + 4*eps**2 + 10*eps**4 + 20*eps**6 + 10*eps**8 + 4*eps**10 + eps**12)))
                elif zernike_num[iz] == 19:
                    ab = (2*np.sqrt(3)*r3*(4 + 4*eps**8 - 5*r2 + eps**2*(4 - 5*r2) + eps**4*(4 - 5*r2) + eps**6*(4 - 5*r2))*np.sin(3*t))/ ((-1 + eps**2)*np.sqrt((1 + eps**2 + eps**4 + eps**6)*(1 + 4*eps**2 + 10*eps**4 + 20*eps**6 + 10*eps**8 + 4*eps**10 + eps**12)))
                elif zernike_num[iz] == 20:
                    ab = (2*np.sqrt(3)*r5*np.cos(5*t))/ np.sqrt(1 + eps**2 + eps**4 + eps**6 + eps**8 + eps**10)
                elif zernike_num[iz] == 21:
                    ab = (2*np.sqrt(3)*r5*np.sin(5*t))/ np.sqrt(1 + eps**2 + eps**4 + eps**6 + eps**8 + eps**10)
                elif zernike_num[iz] == 22:
                    ab = (np.sqrt(7)*(1 + eps**6 - 12*r2 + 30*r4 - 20*r**6 + eps**4*(9 - 12*r2) + eps**2*(9 - 36*r2 + 30*r4)))/ (-1 + eps**2)**3
                    
            dmap[j, :] += zernike_val[iz] * ab
            
    if not ("NO_APPLY" in kwargs and kwargs["NO_APPLY"]):
        if ("AMPLITUDE" in kwargs and kwargs["AMPLITUDE"]):
            a.wfarr *= proper.prop_shift_center(dmap)
        else:
            i = complex(0, 1)
            a.wfarr *= np.exp(i*2*np.pi/a.lamda*proper.prop_shift_center(dmap))
            
    return dmap
