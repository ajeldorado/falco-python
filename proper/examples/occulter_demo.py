#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper

def occulter_demo():
    
    n = 512           # grid size
    lamda = 0.55      # wavelength (microns)
    
    (solid, sampl_solid) = proper.prop_run('run_occulter', lamda, n, PASSVALUE = {"occulter_type": "SOLID"})
    
    (gaussian, sampl_gauss) = proper.prop_run('run_occulter', lamda, n, PASSVALUE = {"occulter_type": "GAUSSIAN"})    
    
    (eighth_order, sampl_eighth_order) = proper.prop_run('run_occulter', lamda, n, PASSVALUE = {"occulter_type": "8TH_ORDER"})
    
    return
    
if __name__ == '__main__':
    occulter_demo()
