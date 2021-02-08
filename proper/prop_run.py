#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#   Modified by N. Saini on 19 April 2019 - changed umpack to unpack, added code 
#   to allow relative module path import as suggested by Kristina Davis
#   Modified by J. Krist - 19 April 2019 - added call to prop_table; added
#   IS_MULTI option so that maximum number of FFT threads are appropriately
#   set for FFTI/FFTW.
#   Modified by J. Krist - 28 Jan 2020 - Added OMP_NUM_THREADS settings

import os
import proper
import importlib
import numpy as np
from time import time


def prop_run(routine_name, lambda0, gridsize, **kwargs):
    """Execute a prescription.

    Parameters
    ----------
    routine_name : str
        Filename (excluding extension) of the python routine containing the
        prescription.

    lambda0 : float
        Either the wavelength in microns at which to propagate the wavefront,
        or the name of a text file containing a list of wavelength and weight
        pairs. In the latter case, the prescription is run for each wavelength
        and the results added together with the respective weights.

    gridsize : int
        Size of the computational grid (arrays are gridsize by gridsize). Must
        be power of 2.


    Returns
    -------
    psf : numpy ndarray
        Image containing result of the propagation pf the prescription routine.

    pixscale : float
        Returns sampling of "result" in meters per element.  It is the
        responsibility of the prescription to return this value (which is
        returned from function end).


    Other Parameters
    ----------------
    QUIET : bool
        If set, intermediate messages and surface labels will not be printed.

    PHASE_OFFSET : bool
        If set, a phase offset is added as the wavefront is propagated. For
        instance, if a wavefront is propagated over a distance of 1/4 wavelength,
        a phase offset of pi/2 radians will be added. This is useful in cases
        where the offset between separate beams that may be combined later may
        be important (e.g. the individual arms of an interferometer). By default,
        a phase offset is not applied.

    VERBOSE : bool
        If set, informational messages will be printed.

    PASSVALUE : dict
        Points to a value (which could be a constant or a variable) that is
        passed to the prescription for use as the prescription desires.

    TABLE : bool
        If set, prints out a table of sampling and beam size for each surface.

    IS_MULTI: bool
        If set, signifies that this instance of prop_run is running as an instance
        of multiple prop_run processes.  This is only meant to be used by
        prop_execute_multi.

    PRINT_INTENSITY : bool
        If set, print intensity values
    """
    if (int(gridsize) & int(gridsize-1)) != 0:
        raise Exception( "ERROR: grid size must be a power of 2" )

    proper.do_table = proper.switch_set("TABLE",**kwargs)

    if proper.switch_set("IS_MULTI",**kwargs):
        os.environ["OMP_NUM_THREADS"] = str(proper.ffti_multi_nthreads)    # for Intel MKL routines, if used
        if proper.use_ffti:
            proper.fft_nthreads = proper.ffti_multi_nthreads
        elif proper.use_fftw:
            proper.fft_nthreads = proper.fftw_multi_nthreads
    else:
        os.environ["OMP_NUM_THREADS"] = str(proper.ffti_single_nthreads)   # for Intel MKL routines, if used
        if proper.use_ffti:
            proper.fft_nthreads = proper.ffti_single_nthreads
        elif proper.use_fftw:
            proper.fft_nthreads = proper.fftw_single_nthreads

    proper.print_total_intensity = proper.switch_set("PRINT_INTENSITY",**kwargs)

    proper.n = gridsize
    proper.layout_only = 0

    proper.verbose = proper.switch_set("VERBOSE",**kwargs)

    proper.print_it = not proper.switch_set("QUIET",**kwargs)

    proper.phase_offset = proper.switch_set("PHASE_OFFSET",**kwargs)

    if type(lambda0) == str:
        try:
            lam, throughput = np.loadtxt(lambda0, unpack = True, usecols = (0,1))
        except IOError:
            raise IOError("Unable to open wavelength file %s" %(lambda0))

        lam *= 1.e-6
    else:
        lam = np.array([lambda0 * 1.e-6])
        throughput = np.ones(shape = 1, dtype = np.float64)

    start_time = time()

    proper.first_pass = 0
    proper.action_num = 0

    nlams = len(lam)
    for ilam in range(nlams):
        if proper.print_it:
            print("Lambda = %2.4E   Throughput = %3.2f" %(lam[ilam], throughput[ilam]))

        try:
            module = importlib.import_module(routine_name)
        except ImportError:
            raise ImportError("Unable to run %s prescription. Stopping." %routine_name)

        f1 = os.path.basename(module.__file__)
        f2 = os.path.splitext(f1)[0]
        func = getattr(module, routine_name)

        if "PASSVALUE" in kwargs:
            psf_ilam, pixscale = func(lam[ilam], gridsize, kwargs["PASSVALUE"])
        else:
            psf_ilam, pixscale = func(lam[ilam], gridsize)

        psf_ilam *= throughput[ilam]

        if ilam == 0:
            psf = psf_ilam
        else:
            psf += psf_ilam

        psf_ilam = 0

        if proper.do_table:
            proper.prop_table()

    end_time = time()

    if proper.print_it:
        print("Total elapsed time (seconds) = %8.4f" %(end_time - start_time))

    return (psf, pixscale)
