#   Copyright 2019 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri

import proper

def switch_set( name, **kwargs ):
    if name in kwargs and kwargs[name]:
        return True
    else:
        return False

