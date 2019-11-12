#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper


def hubble_simple(wavelength, gridsize, PASSVALUE = {'delta_sec': 0.}):
    # Define entrance aperture diameter and other quantities
    diam = 2.4                                 # telescope diameter in meters
    fl_pri = 5.52085                           # HST primary focal length (m)
    d_pri_sec = 4.907028205                    # primary to secondary separation (m)
    fl_sec = -0.6790325                        # HST secondary focal length (m)
    d_sec_to_focus = 6.3919974                 # nominal distance from secondary to focus
    beam_ratio = 0.5                           # initial beam width/grid width

    # delta_sec = additional primary-to-secondary separation offset (m)
    delta_sec = PASSVALUE['delta_sec']

    # Define the wavefront
    wfo = proper.prop_begin(diam, wavelength, gridsize, beam_ratio)

    # Define a circular aperture
    proper.prop_circular_aperture(wfo, diam/2)                        # HST aperture (primary mirror)
    proper.prop_circular_obscuration(wfo, 0.396)                      # secondary mirror obscuration
    proper.prop_rectangular_obscuration(wfo, 0.0264, 2.5)             # secondary vane (vertical)
    proper.prop_rectangular_obscuration(wfo, 2.5, 0.0264)             # secondary vane (horizontal)
    proper.prop_circular_obscuration(wfo, 0.078, -0.9066, -0.5538)    # primary mirror pad 1
    proper.prop_circular_obscuration(wfo, 0.078, 0., 1.0705)          # primary mirror pad 2
    proper.prop_circular_obscuration(wfo, 0.078, 0.9127, -0.5477)     # primary mirror pad 3 

    # Define entrance
    proper.prop_define_entrance(wfo)

    # Define a lens
    proper.prop_lens(wfo, fl_pri, "primary")                          # primary mirror

    # Propagate the wavefront
    proper.prop_propagate(wfo, d_pri_sec+delta_sec, "secondary")

    proper.prop_lens(wfo, fl_sec, "secondary")

    proper.prop_propagate(wfo, d_sec_to_focus+delta_sec, "HST focus", TO_PLANE = False)

    # End
    (wfo, sampling) = proper.prop_end(wfo)

    return (wfo, sampling)
