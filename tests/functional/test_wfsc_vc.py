"""FALCO regression test of WFSC with a vortex coronagraph."""
from copy import deepcopy
import numpy as np
import os
from math import isclose

import falco

import config_wfsc_vc as CONFIG


def test_wfsc_vc():

    mp = deepcopy(CONFIG.mp)
    mp.flagPlot = False
    LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
    mp.path.falco = os.path.dirname(os.path.dirname(LOCAL_PATH))

    # Generate the label associated with this trial
    mp.runLabel = 'testing_wfsc_vc'

    # Perform the Wavefront Sensing and Control
    out = falco.setup.flesh_out_workspace(mp)
    falco.wfsc.loop(mp, out)

    # print(out.IrawCorrHist[-1])
    # print(out.dm1.Spv[-1])
    # print(out.thput[-1])
    # print(out.log10regHist)

    # Tests:
    Iend = out.IrawCorrHist[-1]  # 1.38e-10 in matlab, 1.2958e-10 in python
    assert isclose(Iend, 1.2958e-10, abs_tol=1e-11)

    dm1pv = out.dm1.Spv[-1]  # 1.5057e-08 in matlab, 1.4218e-08 in python
    assert isclose(dm1pv, 1.42e-08, abs_tol=1e-9)

    thput = out.thput[-1]  # 28.52%
    assert isclose(thput, 0.2852, abs_tol=1e-3)

    assert np.allclose(out.log10regHist, np.array([-4.5, -4, -4]), rtol=1e-2)


if __name__ == '__main__':
    test_wfsc_vc()
