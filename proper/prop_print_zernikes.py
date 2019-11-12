#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper


def prop_print_zernikes(maxz):
    """Print to the screen the first N number of Noll-ordered Zernike 
    polynomials for an unobscured circular aperture, up to a specified index.
    
    Parameters
    ----------
    maxz : int
        The number of zernike polynomials to print, from 1 to numz
        
    Returns
    -------
    Zernike polynomials : float
        Print Zernike polynomials to screen
    """
    zernlist = proper.prop_noll_zernikes(maxz)
    
    for i in range(1,maxz+1):
        print("%d  =  %s" %(i, zernlist[i]))
        
    return
