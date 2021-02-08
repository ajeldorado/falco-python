#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#   Modified by J. Krist - 19 April 2019 - Got it to work


import proper


def prop_table():
    """Print the beam dimensions and sampling at each surface in a prescription.
    
    Parameters
    ----------
        None
    
    Returns
    -------
        None
    """
    action_num = proper.action_num
    lens_fl_list = proper.lens_fl_list
    surface_name_list = proper.surface_name_list
    distance_list = proper.distance_list
    sampling_list = proper.sampling_list
    beam_diam_list = proper.beam_diam_list
    lens_eff_fratio_list = proper.lens_eff_fratio_list
    
    print("")
    print("                         Dist from                     Paraxial")
    print("                          previous      Sampling/        beam           Grid        Lens paraxial     Paraxial")
    print("Surface    Surface         surface         pixel       diameter         width       focal length       focal")
    print("  type       name            (m)           (m)            (m)            (m)            (m)           ratio")
    print("--------  ------------  -------------  ------------   ------------   ------------   -------------  ------------")
    
    for i in range(action_num):
        if lens_fl_list[i] != 0:
            print( "   LENS   %-12s  %-14.6e %-14.6e %-14.6e %-14.6e %-14.6e %-14.6e" % 
                (surface_name_list[i], distance_list[i], sampling_list[i], beam_diam_list[i], proper.n*sampling_list[i], lens_fl_list[i], lens_eff_fratio_list[i]) )
        else:
            print( "SURFACE   %-12s  %-14.6e %-14.6e %-14.6e %-14.6e" % (surface_name_list[i], distance_list[i], sampling_list[i], beam_diam_list[i], proper.n*sampling_list[i]) )
            
    print("")
    
    return
