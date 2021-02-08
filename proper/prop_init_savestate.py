#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
from time import time


def prop_init_savestate():
    """
    Initialize the save state system. This must be called before any calls to
    prop_state or prop_is_statesaved.

    Parameters
    ----------
    None

    Returns
    -------
    statefile : string
        Name of state file
    """
    proper.save_state = 1
    proper.save_state_lam = []

    # creating a random id number is better this way because the random number
    # generator uses seeds based on the current time in integer seconds, and
    # that can return the same id number if two processes are started very
    # close to one another

    num = time()
    num = int(num * 500000. - int(num) * 500000.)

    proper.statefile = "_" + str(num).strip() + "_prop_savestate"

    return
