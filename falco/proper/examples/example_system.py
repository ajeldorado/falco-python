#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper

def example_system(wavelength, gridsize):

    diam = 1.
    lens_fl = 20.
    beam_ratio = 0.5

    # Define the wavefront
    wfo = proper.prop_begin(diam, wavelength, gridsize, beam_ratio)

    if not proper.prop_is_statesaved(wfo):
        proper.prop_circular_aperture(wfo, diam/2)
        proper.prop_define_entrance(wfo)
        proper.prop_lens(wfo, lens_fl, '1st lens')
        proper.prop_propagate(wfo, lens_fl, 'intermediate focus')

    proper.prop_state(wfo)
        
    # we are now at the intermediate focus, so pretend that
    # we do something to the wavefront here and continue on
    proper.prop_propagate(wfo, lens_fl, 'second lens')
    proper.prop_lens(wfo, lens_fl, 'second lens')
    (wfo, sampling) = proper.prop_end(wfo)

    return (wfo, sampling)
