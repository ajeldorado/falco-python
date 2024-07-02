#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper


def psdtest(wavelength, gridsize, PASSVALUE = {'usepsdmap': True}):
    
    lens_diam = 0.212            # 0.212 meter lean diameter
    lens_fl = 24. * lens_diam    # focal length (f/24 focal ratio)
    beam_width_ratio = 0.5
    
    wfo = proper.prop_begin(lens_diam, wavelength, gridsize, beam_width_ratio)
    
    # Create circular entrance aperture
    proper.prop_circular_aperture(wfo, lens_diam/2)
    proper.prop_define_entrance(wfo)
    
    #-- If the variable usepsdmap is not defined (via optional passed value), 
    #-- read in and use the map which represents the wavefront error (in this case, 
    #-- it's in nanometers, hence we need to multiply it by 1e9 to convert it to meters)
    #-- and has a sampling of 0.4 mm/pixel.  If usepsdmap is defined, then generate 
    #-- and use a PSD-defined map.  The maps have an RMS of about 1.0 nm.
    if PASSVALUE['usepsdmap']:
        a = 3.29e-23     # low-freq power in m^4
        b = 212.26       # correlation length (cycles/m)
        c = 7.8          # high-freq falloff (r^-c)
        
        proper.prop_psd_errormap(wfo, a, b, c)
    else:
        proper.prop_errormap(wfo, 'errormap.fits', SAMPLING = 0.0004, 
                MULTIPLY = 1e-9, WAVEFRONT = True)
                
                
    proper.prop_lens(wfo, lens_fl, 'telescope lens')
    proper.prop_propagate(wfo, proper.prop_get_distancetofocus(wfo), 'intermediate focus')
    
    # multiply field by occulting mask with 4*lam/D HWHM transmission
    mask = proper.prop_8th_order_mask(wfo, 4, CIRCULAR = True, MASK = True)
    
    proper.prop_propagate(wfo, lens_fl, 'pupil imaging lens')
    proper.prop_lens(wfo, lens_fl, 'pupil imaging lens')
    proper.prop_propagate(wfo, lens_fl, 'lyot stop')
    proper.prop_circular_aperture(wfo, 0.53, NORM = True)  # Lyot stop
    
    proper.prop_propagate(wfo, lens_fl, 'reimaging lens')
    proper.prop_lens(wfo, lens_fl, 'reimaging lens')
    proper.prop_propagate(wfo, proper.prop_get_distancetofocus(wfo), 'final focus')
    
    (wfo, sampling) = proper.prop_end(wfo)
    
    return (wfo, sampling)
