#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#  Modified 18 April 2019 by J. Krist - force garbage collection


import proper
import numpy as np
import gc

def prop_propagate(wf, dz, surface_name = "", **kwargs):
    """Determine which propagator to use to propagate the current wavefront by a 
    specified distance and do it.
    
    Parameters
    ----------
    wf : obj
        WaveFront class object
        
    dz : float
        Distance in meters to propagate wavefront
        
    surface_name : str
        String containing name of surface to which to propagate
    
    
    Returns
    -------
        None 
        Replaces the wavefront with a new one.

    
    Other Parameters
    ----------------
    TO_PLANE : bool
    """
    if proper.print_it:
        if surface_name == "":
            print("Propagating")
        else:
            print("Propagating to %s" %(surface_name))
        
    dzw = proper.prop_select_propagator(wf, dz)
    
    z1 = wf.z
    z2 = z1 + dz
    
    if proper.switch_set("TO_PLANE",**kwargs):
        wf.propagator_type = wf.propagator_type[:11] + "INSIDE_"
    
    if proper.verbose:
        print("  PROPAGATOR: propagator_type = %s" %(wf.propagator_type))
    
    if wf.propagator_type == "INSIDE__to_INSIDE_":
         proper.prop_ptp(wf, dz)
    elif wf.propagator_type == "INSIDE__to_OUTSIDE": 
        proper.prop_ptp(wf, wf.z_w0 - z1)
        proper.prop_wts(wf, z2 - wf.z_w0)
    elif wf.propagator_type == "OUTSIDE_to_INSIDE_":
        proper.prop_stw(wf, wf.z_w0 - z1)
        proper.prop_ptp(wf, z2 - wf.z_w0)
    elif wf.propagator_type == "OUTSIDE_to_OUTSIDE":
        proper.prop_stw(wf, wf.z_w0 - z1)
        proper.prop_wts(wf, z2 - wf.z_w0)
    
    if proper.print_total_intensity:
        intensity = np.sum(np.abs(wf.wfarr)**2, dtype = np.float64)
        if surface_name == "":
            print("Total intensity = ", intensity)
        else:
            print("Total intensity at surface ", surface_name, " = ", intensity)

    if proper.do_table:
        proper.sampling_list[proper.action_num] = wf.dx
        proper.distance_list[proper.action_num] = dz
        proper.beam_diam_list[proper.action_num] = 1 * proper.prop_get_beamradius(wf)
        
        if surface_name != "":
           proper.surface_name_list[proper.action_num] = surface_name
        else:
           proper.surface_name_list[proper.action_num] = "(SURFACE)"        
        
        proper.action_num += 1

    gc.collect()
   
    return
