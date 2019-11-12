#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
from telescope import telescope
from coronagraph import coronagraph


def run_coronagraph(wavelength, grid_size, PASSVALUE = {'use_errors': False, 'occulter_type': 'GAUSSIAN'}):
    
    diam = 0.1                 # telescope diameter in meters
    f_lens = 24 * diam
    
    beam_ratio = 0.3
    
    # Define the wavefront
    wfo = proper.prop_begin(diam, wavelength, grid_size, beam_ratio)
    
    # Circular aperture
    proper.prop_circular_aperture(wfo, diam/2)
    proper.prop_define_entrance(wfo)
    
    # Define telescope optical assembly
    telescope(wfo, f_lens, PASSVALUE['use_errors'])
    
    # Coronagraph
    coronagraph(wfo, f_lens, PASSVALUE['occulter_type'], diam)
    
    # End
    (wfo, sampling) = proper.prop_end(wfo)
    
    return (wfo, sampling)
