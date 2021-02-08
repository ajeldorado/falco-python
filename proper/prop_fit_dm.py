#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import numpy as np
from scipy.ndimage.filters import convolve
import proper

def prop_fit_dm(dm_z, inf_kernel):
    """Determine deformable mirror actuator piston values that generate a desired 
    DM surface, accounting for the effect of the actuator influence function.
    
    Parameters
    ----------
    dm_z : numpy ndarray
        DM surface to match (2D array, with each element representing the 
        desired surface amplitude for the corresponding actuator)
        
    inf_kernel : numpy ndarray
        Influence function kernel (2D image sampled at actuator spacing)
        
    Returns
    -------
    dm_z_command : numpy ndarray
        DM actuator positions that create the desired surface when the influence 
        function is included (2D image) 
        
    dm_surface : numpy ndarray
        dm_z_command array convolved with inf_kernel

    Notes
    -----
        Intended for use by the dm function and not for general users
    """
    e0 = 1000000.
    
    dm_z_command = np.copy(dm_z)
    last_good_dm = np.copy(dm_z_command)
    dm_surface = convolve(dm_z_command, inf_kernel)
    diff = dm_z - dm_surface
    e = np.sqrt(np.sum(diff**2))
    
    # Iterate for a solution
    while (e < e0 and (e0-e)/e > 0.01):
        last_good_dm = np.copy(dm_z_command)
        e0 = e
        dm_z_command = dm_z_command + diff
        dm_surface = convolve(dm_z_command, inf_kernel)
        diff = dm_z - dm_surface
        if np.max(np.abs(diff)) < 1.e-15:
            break
        e = np.sqrt(np.sum(diff**2))
        
    # If fit diverged, use the result from the previous iteration
    if e > e0:
        dm_z_command = last_good_dm
        
    return (dm_z_command, dm_surface)
        
    
    
