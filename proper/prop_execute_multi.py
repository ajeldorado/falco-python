#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#   Modified by J. Krist - 19 April 2019 - added IS_MULTI=1 to prop_run to
#   tell it that it is running run prop_run_multi.


import proper


def prop_execute_multi(params):
    """Execute a PROPER prescription from within an prop_run_multi.
    
    This internal routine is called by prop_run_multi and is not intended for
    general users.
    
    Parameters
    ----------
    params : list
        List of input parameters
    
    Returns
    -------
    psf : numpy ndarray
        Image containing result of the propagation pf the prescription routine.
        
    sampling : float
        Returns sampling of "result" in meters per element.
    """
    routine_name, lamda, gridsize, passvalue, quiet, phase_offset = params

    if passvalue != {}:
        psf, sampling = proper.prop_run(routine_name, lamda, gridsize, PASSVALUE = passvalue, QUIET = quiet, PHASE_OFFSET = phase_offset, IS_MULTI=True)
    else:
        psf, sampling = proper.prop_run(routine_name, lamda, gridsize, QUIET = quiet, PHASE_OFFSET = phase_offset, IS_MULTI=True)
   
    return (psf, sampling)
