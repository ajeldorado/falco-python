"""Unit test suite for pairwise probing."""
from copy import deepcopy
import numpy as np
import os
import unittest

import falco

import config_wfsc_vc as CONFIG


class TestPairwiseProbing(unittest.TestCase):

    def setUp(self):
        mp = deepcopy(CONFIG.mp)

        mp.runLabel = 'testing_probing'
        mp.jac.mftToVortex = False
        mp.flagPlot = False

        mp.fracBW = 0.01
        mp.Nsbp = 1
        mp.Nwpsbp = 1
        mp.Nitr = 3

        # Overwrite the LUVOIR-B pupil with an open circle to get
        # better contrast for this test.
        # Inputs common to both the compact and full models
        inputs = {}
        inputs['ID'] = 0
        inputs['OD'] = 1.0
        # Full model
        inputs['Nbeam'] = mp.P1.full.Nbeam
        inputs['Npad'] = falco.util.ceil_even(mp.P1.full.Nbeam)
        mp.P1.full.mask = falco.mask.falco_gen_pupil_Simple(inputs)
        # Compact model
        inputs['Nbeam'] = mp.P1.compact.Nbeam
        inputs['Npad'] = falco.util.ceil_even(mp.P1.compact.Nbeam)
        mp.P1.compact.mask = falco.mask.falco_gen_pupil_Simple(inputs)

        mp.path = falco.config.Object()
        LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
        mp.path.falco = os.path.dirname(os.path.dirname(LOCAL_PATH))

        out = falco.setup.flesh_out_workspace(mp)
        N = mp.P1.full.E.shape[0]
        alpha = 2.5
        mirror_figure = 1e-10
        errormap = falco.util.gen_simple_psd_errormap(N, alpha, mirror_figure)
        temp = np.exp(2*np.pi*1j/mp.lambda0*errormap)
        mp.P1.full.E = np.zeros((N, N, mp.Nsbp, mp.Nwpsbp), dtype=complex)
        mp.P1.full.E[:, :, 0, 0] = temp

        N = mp.P1.compact.E.shape[0]
        mp.P1.compact.E = np.ones((N, N, mp.Nsbp), dtype=complex)
        # mp.P1.compact.E[:, :, 0] = temp
        Im = falco.imaging.get_summed_image(mp)
        falco.imaging.calc_psf_norm_factor(mp)

        # Get exact E-field for comparison:
        mp.estimator = 'perfect'
        ev = falco.config.Object()
        falco.est.perfect(mp, ev)
        Etrue = ev.Eest

        self.mp = mp
        self.ev = ev
        self.Etrue = Etrue

    def test_square_region(self):
        """Test pairwise probing in a square region centered on the star."""
        mp = self.mp
        ev = self.ev
        Etrue = self.Etrue

        # Estimate E-field with square-defined probing region
        mp.estimator = 'pairwise'
        mp.est.probe = falco.config.Probe()
        mp.est.probe.Npairs = 3  # Number of pair-wise probe PAIRS to use.
        mp.est.probe.whichDM = 1  # Which DM # to use for probing. 1 or 2. Default is 1
        mp.est.probe.radius = 12  # Max x/y extent of probed region [lambda/D].
        mp.est.probe.xOffset = 0  # offset of probe center in x [actuators]. Use to avoid central obscurations.
        mp.est.probe.yOffset = 0  # offset of probe center in y [actuators]. Use to avoid central obscurations.
        mp.est.probe.axis = 'alternate'  #  which axis to have the phase discontinuity along [x or y or xy/alt/alternate]
        mp.est.probe.gainFudge = 1  #  empirical fudge factor to make average probe amplitude match desired value.
        ev.Itr = 1
        falco.est.pairwise_probing(mp, ev)
        Eest = ev.Eest
        Eest2D = np.zeros((mp.Fend.Neta, mp.Fend.Nxi), dtype=complex)
        Eest2D[mp.Fend.corr.maskBool] = Eest[:, 0]
        Etrue2D = np.zeros((mp.Fend.Neta, mp.Fend.Nxi), dtype=complex)
        Etrue2D[mp.Fend.corr.maskBool] = Etrue[:, 0]
        Etrue2D[Eest2D == 0] = 0
        meanI = np.mean(np.abs(Etrue)**2)
        meanIdiff = np.mean(np.abs(Etrue2D[mp.Fend.corr.maskBool] -
                                   Eest2D[mp.Fend.corr.maskBool])**2)
        percentEstError = meanIdiff/meanI*100

        self.assertTrue(percentEstError < 4.0)

    def test_rectangular_region(self):
        """Test pairwise probing in a rectangular focal plane region."""
        mp = self.mp
        ev = self.ev
        Etrue = self.Etrue

        # Estimate E-field with rectangle-defined probing region
        mp.estimator = 'pairwise-rect';
        # mp.est = rmfield(mp.est, 'probe');
        mp.est.probe = falco.config.Probe()
        mp.est.probe.Npairs = 3  # Number of pair-wise probe PAIRS to use.
        mp.est.probe.whichDM = 1  # Which DM # to use for probing. 1 or 2. Default is 1
        mp.est.probe.radius = 12  # Max x/y extent of probed region [lambda/D].
        mp.est.probe.xOffset = 0  # offset of probe center in x [actuators]. Use to avoid central obscurations.
        mp.est.probe.yOffset = 0  # offset of probe center in y [actuators]. Use to avoid central obscurations.
        # mp.est.probe.axis = 'alternate'  #  which axis to have the phase discontinuity along [x or y or xy/alt/alternate]
        mp.est.probe.gainFudge = 1  #  empirical fudge factor to make average probe amplitude match desired value.
        mp.est.probe.xiOffset = 6
        mp.est.probe.etaOffset = 0
        mp.est.probe.width = 12
        mp.est.probe.height = 24
        out = falco.setup.flesh_out_workspace(mp)

        ev.Itr = 1
        falco.est.pairwise_probing(mp, ev)
        Eest = ev.Eest
        Eest2D = np.zeros((mp.Fend.Neta, mp.Fend.Nxi), dtype=complex)
        Eest2D[mp.Fend.corr.maskBool] = Eest[:, 0]
        Etrue2D = np.zeros((mp.Fend.Neta, mp.Fend.Nxi), dtype=complex)
        Etrue2D[mp.Fend.corr.maskBool] = Etrue[:, 0]
        Etrue2D[Eest2D == 0] = 0
        # Block out the unprobed strip in the middle.
        xMiddle = int(np.floor((mp.Fend.Neta+1)/2))
        Eest2D[:, xMiddle-1:xMiddle+2] = 0
        Etrue2D[:, xMiddle-1:xMiddle+2] = 0

        meanI = np.mean(np.abs(Etrue)**2)
        meanIdiff = np.mean(np.abs(Etrue2D[mp.Fend.corr.maskBool] -
                                   Eest2D[mp.Fend.corr.maskBool])**2)
        percentEstError = meanIdiff/meanI*100

        self.assertTrue(percentEstError < 4.0)


if __name__ == '__main__':
    unittest.main()
