"""Estimation functions for WFSC."""

import numpy as np
import multiprocessing
# from astropy.io import fits 
# import matplotlib.pyplot as plt 
import falco

def perfect(mp):
    """
    Return the perfect-knowledge E-field from the full model.
    
    Optionally add Zernikes at the input pupil.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
        
    Returns
    -------
    Emat : numpy ndarray
        2-D array with the vectorized, complex E-field of the dark hole pixels
        for each mode included in the control Jacobian.
    """  
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    
    if mp.flagMultiproc:
        
        Emat = np.zeros((mp.Fend.corr.Npix, mp.jac.Nmode), dtype=complex)
        
        #--Loop over all modes and wavelengths
        inds_list = [(x,y) for x in range(mp.jac.Nmode) for y in range(mp.Nwpsbp)] #--Make all combinations of the values  
        Nvals = mp.jac.Nmode*mp.Nwpsbp

        pool = multiprocessing.Pool(processes=mp.Nthreads)
        resultsRaw = [pool.apply_async(_est_perfect_Efield_with_Zernikes_in_parallel, args=(mp, ilist, inds_list)) for ilist in range(Nvals) ]
        results = [p.get() for p in resultsRaw] #--All the images in a list
        pool.close()
        pool.join()  
        
#        parfor ni=1:Nval
#            Evecs{ni} = falco_est_perfect_Efield_with_Zernikes_parfor(ni,ind_list,mp)
#        end
        
        #--Re-order for easier indexing
        Ecube = np.zeros((mp.Fend.corr.Npix, mp.jac.Nmode, mp.Nwpsbp), dtype=complex)
        for iv in range(Nvals):
            im = inds_list[iv][0]  #--Index of the Jacobian mode
            wi = inds_list[iv][1]   #--Index of the wavelength in the sub-bandpass
            Ecube[:, im, wi] = results[iv]
        Emat = np.mean(Ecube, axis=2) # Average over wavelengths in the subband
  
#        EmatAll = np.zeros((mp.Fend.corr.Npix, Nval))
#        for iv in range(Nval):
#            EmatAll[:, iv] = results[iv]
#
#        counter = 0;
#        for im=1:mp.jac.Nmode
#            EsbpMean = 0;
#            for wi=1:mp.Nwpsbp
#                counter = counter + 1;
#                EsbpMean = EsbpMean + EmatAll(:,counter)*mp.full.lambda_weights(wi);
#            end
#            Emat(:,im) = EsbpMean;
#        end
    
    else:
    
        Emat = np.zeros((mp.Fend.corr.Npix, mp.jac.Nmode), dtype=complex)
        modvar = falco.config.Object() #--Initialize
        
        for im in range(mp.jac.Nmode):
            modvar.sbpIndex = mp.jac.sbp_inds[im]
            modvar.zernIndex = mp.jac.zern_inds[im]
            modvar.whichSource = 'star'
            
            #--Take the mean over the wavelengths within the sub-bandpass
            EmatSbp = np.zeros((mp.Fend.corr.Npix, mp.Nwpsbp),dtype=complex)
            for wi in range(mp.Nwpsbp):
                modvar.wpsbpIndex = wi
                E2D = falco.model.full(mp, modvar)
                EmatSbp[:,wi] = mp.full.lambda_weights[wi]*E2D[mp.Fend.corr.maskBool] #--Actual field in estimation area. Apply spectral weight within the sub-bandpass
            Emat[:,im] = np.sum(EmatSbp,axis=1)
            
    
    return Emat
    

#%--Extra function needed to use parfor (because parfor can have only a
#%  single changing input argument).
def _est_perfect_Efield_with_Zernikes_in_parallel(mp, ilist, inds_list):

    im = inds_list[ilist][0]  #--Index of the Jacobian mode
    wi = inds_list[ilist][1]   #--Index of the wavelength in the sub-bandpass
    
    modvar = falco.config.Object()
    modvar.sbpIndex = mp.jac.sbp_inds[im]
    modvar.zernIndex = mp.jac.zern_inds[im]
    modvar.wpsbpIndex = wi
    modvar.whichSource = 'star'
    
    E2D = falco.model.full(mp, modvar)
    
    return E2D[mp.Fend.corr.maskBool] # Actual field in estimation area. Don't apply spectral weight here.


def pairwise_probing(mp, jacStruct={}):
    """
    Estimate the dark hole E-field with pair-wise probing.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    jacStruct : TYPE, optional
        DESCRIPTION. The default is {}.

    Returns
    -------
    None.

    """
    return None