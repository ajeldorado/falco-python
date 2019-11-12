#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper


def simple_telescope(wavelength, gridsize):
    
    # Define entrance aperture diameter and other quantities
    d_objective = 0.060                        # objective diameter in meters
    fl_objective = 15.0 * d_objective          # objective focal length in meters
    fl_eyepiece = 0.021                        # eyepiece focal length
    fl_eye = 0.022                             # human eye focal length
    beam_ratio = 0.5                           # initial beam width/grid width

    # Define the wavefront
    wfo = proper.prop_begin(d_objective, wavelength, gridsize, beam_ratio)

    # Define a circular aperture
    proper.prop_circular_aperture(wfo, d_objective/2)

    # Define entrance
    proper.prop_define_entrance(wfo)

    # Define a lens
    proper.prop_lens(wfo, fl_objective, "objective")

    # Propagate the wavefront
    proper.prop_propagate(wfo, fl_objective+fl_eyepiece, "eyepiece")

    # Define another lens
    proper.prop_lens(wfo, fl_eyepiece, "eyepiece")

    exit_pupil_distance = fl_eyepiece / (1 - fl_eyepiece/(fl_objective+fl_eyepiece))
    proper.prop_propagate(wfo, exit_pupil_distance, "exit pupil at eye lens")

    proper.prop_lens(wfo, fl_eye, "eye")
    proper.prop_propagate(wfo, fl_eye, "retina")

    # End
    (wfo, sampling) = proper.prop_end(wfo)

    return (wfo, sampling)
