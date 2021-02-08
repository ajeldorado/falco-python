#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri


import os
import proper


def prop_use_fftw(**kwargs):
    """Instructs PROPER to use the FFTW FFT routines rather than the built-in
    numpy FFT.

    See the manual for how to use FFTW.  FFTW will be used by PROPER for all
    future runs (including after exiting python shell), unless changed by using
    the DISABLE switch.


    Parameters
    ----------
        None


    Returns
    -------
        None


    Other Parameters
    ----------------
    DISABLE : bool
        If set, the Numpy FFT routine will be used by pyPROPER routines in
        future.
    """

    fftw_dummy_file = os.path.join( os.path.expanduser('~'), '.proper_use_fftw' )

    # Check if pyFFTW is available on user's machine
    try:
        import pyfftw
    except ImportError:
        print('pyFFTW not found, using Numpy FT.')
        fftw_flag = False
    else:
        fftw_flag = True

    if proper.switch_set("DISABLE",**kwargs) or not fftw_flag:
        if os.path.isfile(fftw_dummy_file):
            os.remove(fftw_dummy_file)
        proper.use_fftw = False
    else:
        with open(fftw_dummy_file, 'w') as fd:
            fd.write('Using FFTW routines\n')
        proper.use_fftw = True

    return
