#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np


def telescope_dm(wfo, f_lens, use_errors, use_dm):

    if use_errors:
        rms_error = 10.e-9       # RMS wavefront error in meters
        c_freq = 15.             # correlation frequency (cycles/meter)
        high_power = 3.          # high frewquency falloff (r^-high_power)
        
        obj_map = proper.prop_psd_errormap(wfo, rms_error, c_freq, high_power, RMS = True, MAP = "obj_map", FILE = "telescope_obj.fits")
        
    proper.prop_lens(wfo, f_lens, "objective")
    
    # propagate through focus to pupil
    proper.prop_propagate(wfo, f_lens*2, "telescope pupil imaging lens")
    proper.prop_lens(wfo, f_lens, "telescope pupil imaging lens")
    proper.prop_propagate(wfo, f_lens, "DM")
    
    if use_dm:
        nact = 49                       # number of DM actuators along one axis
        nact_across_pupil = 47          # number of DM actuators across pupil
        dm_xc = nact // 2
        dm_yc = nact // 2
        d_beam = 2 * proper.prop_get_beamradius(wfo)        # beam diameter
        act_spacing = d_beam / nact_across_pupil     # actuator spacing
        map_spacing = proper.prop_get_sampling(wfo)         # map sampling
     
        # have passed through focus, so pupil has rotated 180 deg;
	# need to rotate error map (also need to shift due to the way
	# the rotate() function operates to recenter map)    
        obj_map = np.roll(np.roll(np.rot90(obj_map, 2), 1, 0), 1, 1)
        
        # interpolate map to match number of DM actuators
        dm_map = proper.prop_magnify(obj_map, map_spacing/act_spacing, nact, QUICK = True)
        
        # Need to put on opposite pattern; convert wavefront error to surface height
        proper.prop_dm(wfo, -dm_map/2, dm_xc, dm_yc, act_spacing, FIT = True)
                   
    proper.prop_propagate(wfo, f_lens, "coronagraph lens")
    
    return
