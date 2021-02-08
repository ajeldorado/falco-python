#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np


def prop_end(wf, **kwargs):
    """Set variables needed to properly conclude a propagation run.

    Parameters
    ----------
    wf : obj
        The current WaveFront class object


    Returns
    -------
    wf.wfarr : numpy ndarray
        Wavefront array

    sampling : float
        Sampling in meters


    Other Parameters
    ----------------
    EXTRACT : int
        Returns the dx by dx pixel central portion of the wavefront.

    NOABS : bool
        If set, the complex-values wavefront field is returned. By default, the
        intensity (modulus squared) of the field is returned.
    """
    sampling = proper.prop_get_sampling(wf)

    if proper.switch_set("NOABS",**kwargs):
        wf.wfarr = proper.prop_shift_center(wf.wfarr)
    else:
        wf.wfarr = proper.prop_shift_center(np.abs(wf.wfarr)**2)

    if "EXTRACT" in kwargs:
        EXTRACT = kwargs["EXTRACT"]
        ny, nx = wf.wfarr.shape
        wf.wfarr = wf.wfarr[ny/2-EXTRACT/2:ny/2+EXTRACT/2,nx/2-EXTRACT/2:nx/2+EXTRACT/2]

    return (wf.wfarr, sampling)
