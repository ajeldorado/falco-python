"""FALCO regression test of WFSC with a Lyot coronagraph."""
from copy import deepcopy
import numpy as np
import os
from math import isclose

import falco

import config_wfsc_lc as CONFIG


def test_wfsc_lc():

    mp = deepcopy(CONFIG.mp)
    mp.flagPlot = False
    LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
    mp.path.falco = os.path.dirname(os.path.dirname(LOCAL_PATH))

    # Generate the label associated with this trial
    mp.runLabel = 'testing_wfsc_lc'

    # Perform the Wavefront Sensing and Control
    out = falco.setup.flesh_out_workspace(mp)
    falco.wfsc.loop(mp, out)

    print(out.IrawScoreHist[-1])

    print(out.IrawCorrHist[-1])
    print(out.IestScoreHist[-1])
    print(out.dm1.Spv[-1])
    print(out.thput[-1])
    print(out.log10regHist)

    # Tests:
    Iend = out.IrawCorrHist[-1]  # 1.32e-7 in matlab, 1.34e-7 in python
    assert isclose(Iend, 1.34e-7, abs_tol=1e-8)

    Iest = out.IestScoreHist[-1]  # 8.8397e-07 in matlab, 8.934e-7 in python
    assert isclose(Iest, 8.9e-7, abs_tol=1e-8)

    dm1pv = out.dm1.Spv[-1]  # 8.267e-8, 8.271e-8 in python
    assert isclose(dm1pv, 8.27e-8, abs_tol=1e-9)

    thput = out.thput[-1]  # 0.0740
    assert isclose(thput, 0.074, abs_tol=1e-3)

    assert np.allclose(out.log10regHist, np.array([-2, -2, -3]), rtol=1e-2)


if __name__ == '__main__':
    test_wfsc_lc()
