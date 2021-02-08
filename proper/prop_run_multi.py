#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#   Modified by J. Krist - 19 April 2019 - fixed bug that caused this to
#   crash if a PASSVALUE was not provided.
#   Modified by J. Krist - 12 Sept 2019 - pools were not being closed, causing
#   a memory leak

import proper
import numpy as np
import multiprocessing as mp


def prop_run_multi(routine_name, lamda0, gridsize, **kwargs):
    """Execute multiple instances of a prescription simulatenously, either for
    multiple wavelengths or multiple optional parameters (e.g., DM settings).


    Parameters
    ----------
    routine_name : str
        Filename (excluding extension) of the python routine containing the
        prescription.

    lamda0 : float or numpy ndarray
        The simulation wavelength in microns, either a scalar (float) or numpy
        1D array. If an array, each wavelength will be run in parallel and the
        resulting field will be three-dimensional, with the 3rd dimension being
        the field at the corresponding wavlength. NOTE: Unlike prop_run, this
        entry CANNOT be a file pointing to a table to wavelengths and weights.

     gridsize : int
       Size of rhe computational grid (arrays are gridsize by gridsize). Must
       be power of 2. The same grid size is used for all instances of the
       prescription being executed.


    Returns
    -------
    psf : numpy ndarray
        Image containing result of the propagation of the prescription routine.

    pixscale : float or numpy ndarray
        Returns sampling of "psf" in meters per element. It is the
        responsibility of the prescription to return this value (which is
        returned from prop_end). If either "lamda0" or the PASSVALUE variables
        are arrays, then "pixscale" will be an array with each element corresponding
        to the respective entry in that/those arrays.


    Other Parameters
    ----------------
    QUIET : bool
       If set, intermediate messages will not be printed.

    PHASE_OFFSET : bool
       If set, a phase offset is added as the wavefront is propagated. For
       instance, if a wavefront is propagated over a distance of 1/4 wavelength
       a phase offset of pi/2 radians will be added. This is useful in cases
       where the offset between separate beams that may be combined later may
       be important (e.g. the individual arms of an interferometer). By default,
       a phase offset is not applied.

   PASSVALUE : dict or list of dict or array of dict
       Points to a value or variable that is passed to the prescription for
       use as the prescription desires. This may be a dict, or list of dict.
       If this is a list of dict, the prescription will be run in parallel for
       each entry. If both this and the wavelength ("lamda0" variable) are arrays,
       they must have the same number of entries, and each element here will
       match to the corresponding wavelength.

   NCPUS : int
       Number of processor (or cores) to use in parallel mode. By default it is
       set of number of wavelengths or PASSVALUE parameters.
    """
    if (int(gridsize) & int(gridsize-1)) != 0:
        raise Exception( "Grid size must be a power of 2" )

    # the DO_TABLE, PRINT_INTENSITY, and VERBOSE switches to PROP_RUN
    # are not supported by this routine
    if type(lamda0) == np.ndarray or type(lamda0) == list:
        lamda0 = np.asarray(lamda0)
        nlam = len(lamda0)
    else:
        nlam = 1
        lamda0 = np.array([lamda0])

    if "PASSVALUE" in kwargs:
        if type(kwargs["PASSVALUE"]) == list or type(kwargs["PASSVALUE"]) == np.ndarray:
            pvtype = list
            npass = len(kwargs["PASSVALUE"])
        elif type(kwargs["PASSVALUE"]) == dict:
            pvtype = dict
            npass = 1
        else:
            raise ValueError("PASSVALUE can be either dict or list of dict. Stopping")
    else:
        npass = 0

    quiet = proper.switch_set("QUIET",**kwargs)
    phase_offset = proper.switch_set("PHASE_OFFSET",**kwargs)

    if nlam > 1 and (npass > nlam):
        print("PROP_RUN_MULTI: ERROR:")
        print("  Number of wavlengths > 1 AND ")
        print("  number of passed optional values > number of wavelengths")
        raise Exception( "Stopping." )

    if nlam > npass:
        nel = nlam
    else:
        nel = npass 

    params = []
    for i in range(nel):
        if nlam > 1 or nlam == npass:
            lamda = lamda0[i]
        else:
            lamda = lamda0[0]

        if npass > 1:
            passvalue = kwargs["PASSVALUE"][i]
        elif npass == 1:
            if pvtype == dict:
                passvalue = kwargs["PASSVALUE"]
            else:
                passvalue = kwargs["PASSVALUE"][0]
        else:
            passvalue = {}

        params.append((routine_name, lamda, gridsize, passvalue, quiet, phase_offset))

    # Create pool of workers
    ncpus = mp.cpu_count()
    if "NCPUS" in kwargs:
        ncpus = kwargs["NCPUS"]
    elif nel < ncpus:
        ncpus = nel

    pool = mp.Pool(ncpus)
    res = pool.map(proper.prop_execute_multi, params)
    pool.close()
    pool.join()

    ## May 1 2018 - Navtej - Bug report #5 and code snippet by Bryn Jeffries
    ## When NOABS in prop_end is False, only real part (intensity) is used.
    ## Rather than using complex128 array, store PSF's in a list and then
    ## stack them.
    psfs = []
    pixscale = np.zeros(nel)

    for i in range(nel):
        psfs.append(res[i][0])
        pixscale[i] = res[i][1]

    return (np.stack(psfs), pixscale)
