#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper


def telescope(wfo, f_lens, use_errors, use_dm = False):

    if use_errors:
        rms_error = 10.e-9       # RMS wavefront error in meters
        c_freq = 15.             # correlation frequency (cycles/meter)
        high_power = 3.          # high frewquency falloff (r^-high_power)
        
        proper.prop_psd_errormap(wfo, rms_error, c_freq, high_power, RMS = True, MAP = "obj_map", FILE = "telescope_obj.fits")
        
    proper.prop_lens(wfo, f_lens, "objective")
    
    # propagate through focus to pupil
    proper.prop_propagate(wfo, f_lens*2, "telescope pupil imaging lens")
    
    proper.prop_lens(wfo, f_lens, "telescope pupil imaging lens")
    
    # propagate to a deformable mirror (to be inserted later)
    proper.prop_propagate(wfo, f_lens, "DM")
    
    proper.prop_propagate(wfo, f_lens, "coronagraph lens")
    
    return
