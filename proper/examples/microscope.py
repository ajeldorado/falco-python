#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper


def microscope(wavelength, gridsize, PASSVALUE = {'focus_offset': 0.}):
    
    # Define entrance aperture diameter and other quantities
    d_objective = 0.005			# objective diameter in meters
    fl_objective = 0.010			# objective focal length in meters
    fl_eyepiece = 0.020			# eyepiece focal length
    fl_eye = 0.022		        # human eye focal length

    beam_ratio = 0.4

    # Define the wavefront
    wfo = proper.prop_begin(d_objective, wavelength, gridsize, beam_ratio)
    
    d1 = 0.160                           # standard tube length
    d_intermediate_image = fl_objective + d1

    # Compute in-focus distance of object from objective
    d_object = 1 /(1/fl_objective - 1/d_intermediate_image)

    # Define a circular aperture
    proper.prop_circular_aperture(wfo, d_objective/2.)

    # Define entrance
    proper.prop_define_entrance(wfo)

    # simulate the diverging wavefront emitted from a point source placed 
    # "d_object" in front of the objective by using a negative lens (focal 
    # length = -d_object) placed at the location of the objective

    focus_offset = PASSVALUE['focus_offset']

    # Define a lens
    proper.prop_lens(wfo, -(d_object + focus_offset))
    proper.prop_lens(wfo, fl_objective, "objective")

    # Propagate the wavefront
    proper.prop_propagate(wfo, d_intermediate_image, "intermediate image")
    proper.prop_propagate(wfo, fl_eyepiece, "eyepiece")
    proper.prop_lens(wfo, fl_eyepiece, "eyepiece")
    exit_pupil_distance = fl_eyepiece / (1 - fl_eyepiece/(d_intermediate_image+fl_eyepiece))
    proper.prop_propagate(wfo, exit_pupil_distance, "exit pupil/eye")

    proper.prop_lens(wfo, fl_eye, "eye")
    proper.prop_propagate(wfo, fl_eye, "retina")

    # End
    (wfo, sampling) = proper.prop_end(wfo)

    return (wfo, sampling)
