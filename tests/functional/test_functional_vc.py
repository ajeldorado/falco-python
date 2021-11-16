"""Simple functional example used to verify that FALCO is set up correctly."""
import numpy as np
import os

import falco

import config_wfsc_vc as CONFIG


def test_wfsc_vc():

    mp = CONFIG.mp
    mp.flagPlot = False  # DEBUGGING
    LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
    mp.path.falco = os.path.dirname(os.path.dirname(LOCAL_PATH))

    # Generate the label associated with this trial
    mp.runLabel = 'testing_wfsc_vc'

    # Perform the Wavefront Sensing and Control
    out = falco.setup.flesh_out_workspace(mp)
    falco.wfsc.loop(mp, out)

    print(out.IrawCorrHist[-1])
    print(out.dm1.Spv[-1])
    print(out.thput[-1])
    print(out.log10regHist)

    # Tests:
    Iend = out.IrawCorrHist[-1]  # 1.38e-10 in matlab,   1.15e-11 in python
    # assert Iend > 1.0e-11
    assert Iend < 1.5e-11

    dm1pv = out.dm1.Spv[-1]  # 1.377e-08
    assert dm1pv > 1.3e-8
    assert dm1pv < 1.5e-8

    thput = out.thput[-1]  # 26.32%
    assert thput > 0.26
    assert thput < 0.27

    assert np.allclose(out.log10regHist, np.array([-4.5, -5.5, -6]), rtol=1e-2)


if __name__ == '__main__':
    test_wfsc_vc()
