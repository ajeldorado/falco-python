#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import numpy as np
import proper

def prop_select_propagator(wf, dz):
    """Used by propagator to decide which in which propagation regime the next
    surface will be (to decide which propagatoion method to use (spherical-to-
    planar, planar-to-spherical, or planar-to-planar))

        
    Parameters
    ----------
    wf : obj
        WaveFront class object
        
    dz : float
        Distance to propagate from current z location

                
    Returns
    -------
    dzw : float
        Distance to new focus position from new position
    """
    # w0 = min possible waist radius for current beam
    # z_w0 = location of minimum beam waist
    # z_Rayleigh = Rayleigh distance from minimum beam waist 
    # w_at_surface = waist radius at current surface 
    # R_beam = beam radius of curvature 
    rayleigh_factor = 2.
    
    dzw = wf.z_w0 - wf.z
    newz = wf.z + dz
    
    if np.abs(wf.z_w0 - newz) < rayleigh_factor * wf.z_Rayleigh:
        beam_type_new = "INSIDE_"
    else:
        beam_type_new = "OUTSIDE"
        
    wf.propagator_type = wf.beam_type_old + "_to_" + beam_type_new
    
    wf.beam_type_old = beam_type_new
    
    return dzw
