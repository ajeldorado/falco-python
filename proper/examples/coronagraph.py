#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np
import matplotlib.pylab as plt


def coronagraph(wfo, f_lens, occulter_type, diam):
    
    proper.prop_lens(wfo, f_lens, "coronagraph imaging lens")
    proper.prop_propagate(wfo, f_lens, "occulter")
    
    # occulter sizes are specified here in units of lambda/diameter;
    # convert lambda/diam to radians then to meters
    lamda = proper.prop_get_wavelength(wfo)
    occrad = 4.                           # occulter radius in lam/D
    occrad_rad = occrad * lamda / diam    # occulter radius in radians
    dx_m = proper.prop_get_sampling(wfo)
    dx_rad = proper.prop_get_sampling_radians(wfo)    
    occrad_m = occrad_rad * dx_m / dx_rad  # occulter radius in meters

    plt.figure(figsize=(12,8))
        
    if occulter_type == "GAUSSIAN":
        r = proper.prop_radius(wfo)
        h = np.sqrt(-0.5 * occrad_m**2 / np.log(1 - np.sqrt(0.5)))
        gauss_spot = 1 - np.exp(-0.5 * (r/h)**2)
        proper.prop_multiply(wfo, gauss_spot)
        plt.suptitle("Gaussian spot", fontsize = 18)
    elif occulter_type == "SOLID":
        proper.prop_circular_obscuration(wfo, occrad_m)
        plt.suptitle("Solid spot", fontsize = 18)
    elif occulter_type == "8TH_ORDER":
        proper.prop_8th_order_mask(wfo, occrad, CIRCULAR = True)
        plt.suptitle("8th order band limited spot", fontsize = 18)
        
    # After occulter
    plt.subplot(1,2,1)
    plt.imshow(np.sqrt(proper.prop_get_amplitude(wfo)), origin = "lower", cmap = plt.cm.gray)
    plt.text(200, 10, "After Occulter", color = "w")
        
    proper.prop_propagate(wfo, f_lens, "pupil reimaging lens")  
    proper.prop_lens(wfo, f_lens, "pupil reimaging lens")
    
    proper.prop_propagate(wfo, 2*f_lens, "lyot stop")

    plt.subplot(1,2,2)        
    plt.imshow(proper.prop_get_amplitude(wfo)**0.2, origin = "lower", cmap = plt.cm.gray)
    plt.text(200, 10, "Before Lyot Stop", color = "w")
    plt.show()   
    
    if occulter_type == "GAUSSIAN":
        proper.prop_circular_aperture(wfo, 0.25, NORM = True)
    elif occulter_type == "SOLID":
        proper.prop_circular_aperture(wfo, 0.84, NORM = True)
    elif occulter_type == "8TH_ORDER":
        proper.prop_circular_aperture(wfo, 0.50, NORM = True)
    
    proper.prop_propagate(wfo, f_lens, "reimaging lens")
    proper.prop_lens(wfo, f_lens, "reimaging lens")
    
    proper.prop_propagate(wfo, f_lens, "final focus")
    
    return
