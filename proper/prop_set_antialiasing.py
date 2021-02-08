#   Copyright 2020 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Created 28 Jan 2020 - JEK

import proper

def prop_set_antialiasing( nsub=11 ):
    """This function sets the pixel subsampling factor (nsub by nsub) used to 
       antialias the edges of shapes. NOTE: This should be called after prop_run.
    
    Parameters
    ----------
    nsub : 
        Subsampling factor (must be odd-valued integer)
        
    """

    s = int(nsub)
    if (s % 2) == 0:
        raise ValueError("PROP_SET_ANTIALIASING: subsampling factor must be odd-valued integer.")
    else:
        proper.antialias_subsampling = s

