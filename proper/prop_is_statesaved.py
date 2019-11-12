#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper


def prop_is_statesaved(wf):
    """Determine if a previously saved state exists for the current wavelength.

    Parameters
    ----------
    wf : obj
        WaveFront class object

    Returns
    -------
        None
    """
    save_state_lam = proper.save_state_lam

    # save_state_lam is a python list
    nlam =len(save_state_lam)
    if nlam == 0:
        return False
    else:
        for i in range(nlam):
            # does a state exist for the current wavelength?
            if save_state_lam[i] == wf.lamda:
                return True

        return False
