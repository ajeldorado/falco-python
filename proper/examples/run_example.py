#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper

def run_example(wavelength, gridsize):
    
    proper.prop_init_savestate()
    
    for i in range(11):
        (psf, sampling) = proper.prop_run('example_system', wavelength, gridsize)
        
        #-- let us pretend that we now do something useful with
	#-- this iteration's PSF and then compute another
	
    proper.prop_end_savestate()
    
if __name__ == '__main__':
    run_example(0.5, 512)
