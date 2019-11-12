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

def prop_lens(wf, lens_fl, surface_name = ""):
    """Alter the current wavefront as a perfect lens would. 
    
    This routine computes the phase change cause by a perfect lens that has a 
    focal length specified by the user. A positive focal length corresponds to 
    a convex lens or concave mirror; a negative length corresponds to a concave 
    lens or convex mirror. This routine updates the new beam waist position.
    
    Parameters
    ----------
    wf : obj
        WaveFront class object
        
    lens_fl : float
        Focal length of lens in meters
        
    surface_name : str
        String containing name of surface; used when printing out that a lens 
        is being applied
        
    Returns
    -------
        None
        Modifies wavefront array in wf object.
    """
    rayleigh_factor = proper.rayleigh_factor
    
    if proper.print_it:
        if surface_name == "":
            print( "Applying lens" ) 
        else:
            print( "Applying lens at " + surface_name ) 
        
    # calculate waist radius at current surface
    wf.z_Rayleigh = np.pi * wf.w0**2 / wf.lamda
    w_at_surface = wf.w0 * np.sqrt(1. + ((wf.z - wf.z_w0)/wf.z_Rayleigh)**2)
    
    if (wf.z - wf.z_w0) != 0.0:     # lens is not at focus or entrance pupil
        gR_beam_old = (wf.z - wf.z_w0) + wf.z_Rayleigh**2 / (wf.z - wf.z_w0)
        gR_beam_old_inf = 0.
        if gR_beam_old != lens_fl:
            gR_beam = 1./(1./gR_beam_old - 1./lens_fl)
            gR_beam_inf = 0
            if proper.verbose:
                print("  LENS: Gaussian R_beam_old = ", gR_beam_old, "  R_beam = ", gR_beam )
        else:
            gR_beam_inf = 1
            if proper.verbose:
                print( "  LENS: Gaussian R_beam_old = ", gR_beam_old, "  R_beam = Infinite" )
    else:
        gR_beam_old_inf = 1         # at focus or entrance pupil, input beam is planar
        gR_beam = -lens_fl
        gR_beam_inf = 0             # output beam is spherical
        if proper.verbose:
            print( "  LENS: Gaussian R_beam_old = Infinite    R_beam = ", gR_beam )
        
        
    if wf.beam_type_old == "INSIDE_" or wf.reference_surface == "PLANAR":
        R_beam_old = 0.0
    else:
        R_beam_old = wf.z - wf.z_w0
    
    if not gR_beam_inf:
        wf.z_w0 = -gR_beam / (1. + (wf.lamda * gR_beam/(np.pi*w_at_surface**2))**2) + wf.z
        wf.w0 = w_at_surface / np.sqrt(1. + (np.pi*w_at_surface**2/(wf.lamda*gR_beam))**2)
    else:
        wf.z_w0 = wf.z
        wf.w0 = w_at_surface        # output beam is planar
    
    # determine new Rayleigh distance from focus; if currently inside this, 
    # then output beam is planar
    wf.z_Rayleigh = np.pi * wf.w0**2 / wf.lamda
    if np.abs(wf.z_w0 - wf.z) < rayleigh_factor * wf.z_Rayleigh:
        beam_type_new = "INSIDE_"
        R_beam = 0.
    else:
        beam_type_new = "OUTSIDE"
        R_beam = wf.z - wf.z_w0
        
    wf.propagator_type = wf.beam_type_old + "_to_" + beam_type_new
    
    # Apply phase changes as needed, but don't apply if the phase is going to be 
    # similarly altered during propagation
    if proper.verbose:
        print ("  LENS: propagator_type = ", wf.propagator_type )
        if wf.beam_type_old == "INSIDE_":
            sR_beam_old = "Infinite"
        else:
            sR_beam_old = str(R_beam_old).strip()
        
        if beam_type_new == "INSIDE_":
            sR_beam = "Infinite"
        else:
            sR_beam = str(R_beam).strip()  
    
        print( "  LENS: R beam old = %s  R_beam = %s  lens_fl = %s" %(sR_beam_old, sR_beam, lens_fl) )
        print( "  LENS: Beam diameter at lens = %4.3f" %(w_at_surface * 2) )
    
    
    # For different propagator types
    if wf.propagator_type == "INSIDE__to_INSIDE_":
        lens_phase = 1./lens_fl
    elif wf.propagator_type == "INSIDE__to_OUTSIDE":
        lens_phase = 1./lens_fl + 1./R_beam
    elif wf.propagator_type == "OUTSIDE_to_INSIDE_":
        lens_phase = 1./lens_fl - 1./R_beam_old        
    elif  wf.propagator_type == "OUTSIDE_to_OUTSIDE":
        if R_beam_old == 0.0:
            lens_phase = 1./lens_fl + 1./R_beam
        elif R_beam == 0.0:
            lens_phase = 1./lens_fl - 1./R_beam_old
        else:
            lens_phase = 1./lens_fl - 1./R_beam_old + 1./R_beam
            if proper.verbose:
                print( "  LENS: 1/lens_fl = ", 1./lens_fl )
                print( "  LENS: 1/R_beam_old = ", 1./R_beam_old )
                print( "  LENS: 1/R_beam = ", 1./R_beam )
                print( "  LENS: lens_phase = ", lens_phase )

    rho = proper.prop_radius(wf)
    proper.prop_add_phase(wf, -rho**2 * (lens_phase/2.))
    rho = 0
        
    if beam_type_new == "INSIDE_":
        wf.reference_surface = "PLANAR"
    else:
        wf.reference_surface = "SPHERI"
        
    wf.beam_type_old = beam_type_new
    
    wf.current_fratio = np.abs(wf.z_w0 - wf.z) / (2. * w_at_surface)
    
    
    # save stuff for layout plots
    if proper.do_table:
        proper.lens_fl_list[proper.action_num] = lens_fl
        proper.lens_eff_fratio_list[proper.action_num] = wf.current_fratio
        proper.beam_diam_list[proper.action_num] = 2 * w_at_surface
        proper.sampling_list[proper.action_num] = wf.dx
        
        if surface_name != "":
           proper.surface_name_list[proper.action_num] = surface_name
        else:
           proper.surface_name_list[proper.action_num] = "(LENS)"
        
        proper.action_num += 1
    
    if proper.verbose:
        print( "  LENS: Rayleigh distance = %4.12f" %(wf.z_Rayleigh) )

    gc.collect()
     
    return
