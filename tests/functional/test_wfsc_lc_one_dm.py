"""FALCO regression test of WFSC with a Lyot coronagraph."""
from copy import deepcopy
import numpy as np
import os
from math import isclose

import falco

import config_wfsc_lc_quick_one_dm as CONFIG


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

    # print(out.IrawScoreHist[-1])

    # print(out.IrawCorrHist[-1])
    # print(out.IestScoreHist[-1])
    # print(out.dm1.Spv[-1])
    # print(out.thput[-1])
    # print(out.log10regHist)

    # Tests:
    print(f'out.IrawCorrHist[-1]  = {out.IrawCorrHist[-1]}')
    print(f'out.IestScoreHist[-1]  = {out.IestScoreHist[-1]}')
    print(f'out.dm1[-1]  = {out.dm1.Spv[-1]}')
    print(f'out.thput[-1]  = {out.thput[-1]}')
    print(f'out.log10regHist  = {out.log10regHist}')
    # out.IrawCorrHist[-1]  = 5.049915478269254e-07
    # out.IestScoreHist[-1]  = 8.070818118695458e-07
    # out.dm1[-1]  = 7.684291769284934e-08
    # out.thput[-1]  = 0.0768364858394983
    # out.log10regHist  = [-2. -3. -3.]


    Iend = out.IrawCorrHist[-1]  
    assert isclose(Iend, 5.049915478269254e-07, abs_tol=1e-8)

    Iest = out.IestScoreHist[-1] 
    assert isclose(Iest, 8.070818118695458e-07, abs_tol=1e-8)

    dm1pv = out.dm1.Spv[-1]  
    assert isclose(dm1pv, 7.684291769284934e-08, abs_tol=1e-9)

    thput = out.thput[-1]  
    assert isclose(thput, 0.0768364858394983, abs_tol=1e-3)

    assert np.allclose(out.log10regHist, np.array([-2, -3, -3]), rtol=1e-2)


if __name__ == '__main__':
    test_wfsc_lc()
