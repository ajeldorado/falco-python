#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np

def multi_example(lambda_m, n, PASSVALUE = {'use_dm': False, 'dm': np.zeros([48,48], dtype = np.float64)}):

    diam = 0.048
    pupil_ratio = 0.25
    fl_lens = 0.48
    n_actuators = 48     # number of DM actuators in each dimension

    wfo = proper.prop_begin(diam, lambda_m, n, pupil_ratio)
    proper.prop_circular_aperture(wfo, diam/2)
    proper.prop_define_entrance(wfo)

    if PASSVALUE['use_dm']:
        dm_xc = n_actuators/2.
        dm_yc = n_actuators/2.
        dm_spacing = 1.e-3

        proper.prop_dm(wfo, PASSVALUE['dm'], dm_xc, dm_yc, dm_spacing)

    proper.prop_lens(wfo, fl_lens)

    proper.prop_propagate(wfo, fl_lens)

    (wfo, sampling) = proper.prop_end(wfo, NOABS = True)

    return (wfo, sampling)
