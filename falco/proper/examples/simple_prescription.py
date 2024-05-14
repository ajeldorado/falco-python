#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper


def simple_prescription(wavelength, gridsize):

    # Define entrance aperture diameter and other quantities
    diam = 1.0
    focal_ratio = 15.0
    focal_length = diam * focal_ratio
    beam_ratio = 0.5 

    # Define the wavefront
    wfo = proper.prop_begin(diam, wavelength, gridsize, beam_ratio)

    # Define a circular aperture
    proper.prop_circular_aperture(wfo, diam/2)

    # Define entrance
    proper.prop_define_entrance(wfo)

    # Define a lens
    proper.prop_lens(wfo, focal_length)

    # Propagate the wavefront
    proper.prop_propagate(wfo, focal_length)

    # End
    (wfo, sampling) = proper.prop_end(wfo)

    return (wfo, sampling)
