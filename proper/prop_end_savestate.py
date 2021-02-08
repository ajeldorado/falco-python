#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import os
import proper
from glob import glob


def prop_end_savestate():
    """Terminate the current save state system.

    This deletes the files created by prop_state/prop_savestate.

    Parameters
    ----------
        None

    Returns
    -------
        None
    """
    proper.save_state = 0
    proper.save_state_lam = []

    statefile = proper.statefile

    sfiles = glob('*' + statefile)
    if len(sfiles) > 0:
        for sfile in sfiles:
            os.remove(sfile)

    return
