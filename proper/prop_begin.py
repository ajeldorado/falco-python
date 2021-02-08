#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#  Modified by J. Krist - 19 April 2019 - moved search for FFTW wisdom file
#  from prop_fftw to here to avoid redundant searches

import proper
import numpy as np

def prop_begin(beam_diameter, lamda, grid_n, beam_diam_fraction = 0.5):
    """Initialize variables for PROPER routines. 
    
    This routine must be called before any other PROPER routines in order to 
    initialize required variables.
    
    Parameters
    ----------
    beam_diameter : float
        Initial diameter of beam in meters
        
    lamda : float
        Wavelength in meters
        
    grid_n : int
        Wavefront gridsize in pixels (n by n)
    
    beam_diam_fraction : float
        Fraction of the grid width corresponding to the beam diameter. If not 
        specified, it is assumed to be 0.5.
    
    Returns
    -------
    wf : numpy ndarray
        Initialized wavefront array structure created by this routine
    """
    grid_n = int(grid_n)
    proper.n = grid_n 
    
    ndiam = grid_n * beam_diam_fraction
    proper.ndiam = ndiam
    
    diam = float(beam_diameter)
    
    w0 = diam / 2.
    z_ray = np.pi * w0**2 / lamda
    
    proper.rayleigh_factor = 2.
    proper.old_opd = 0.
    
    nlist = 1500

    if proper.use_fftw == True and proper.use_ffti == False:
        proper.prop_load_fftw_wisdom( grid_n, proper.fft_nthreads )
 
    # Create WaveFront object 
    wf = proper.WaveFront(diam, ndiam, lamda, grid_n, w0, z_ray)
    
    if proper.do_table:
        proper.lens_fl_list = np.zeros(nlist, dtype = np.float64)          # list of lens focal lengths
        proper.lens_eff_fratio_list = np.zeros(nlist, dtype = np.float64)  # list of effective fratios after each lens 
        proper.beam_diam_list = np.zeros(nlist, dtype = np.float64)        # list of beam diameters at each lens
        proper.distance_list = np.zeros(nlist, dtype = np.float64)         # list of propagation distances 
        proper.surface_name_list = np.zeros(nlist, dtype = "S25")          # list of surface names 
        proper.sampling_list = np.zeros(nlist, dtype = np.float64)         # list of sampling at each surface
        proper.action_num = 0
                
    return wf
