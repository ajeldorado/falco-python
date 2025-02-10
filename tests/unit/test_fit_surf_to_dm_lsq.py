"""Unit tests for fitting DM shapes with influence functions taken into account."""
import os
import unittest

import astropy.io.fits as pyfits
import numpy as np
from scipy.ndimage import convolve
import scipy.sparse

from falco.dm import resample_inf_func, build_prefilter, fit_surf_with_dm_lsq
from falco.util import pad_crop
import falco.zern


class TestResampleInfFunc(unittest.TestCase):
    """Tests for resampling influence function."""

    def test_success(self):
        """Verify nominal input doesn't fail"""
        inf_func = np.ones((41, 41))
        ppa_in = 10
        ppa_out = 5
        resample_inf_func(inf_func, ppa_in, ppa_out)
        pass


    def test_invalid_inf_func(self):
        """Check invalid inputs rejected as expected"""
        ppa_in = 10
        ppa_out = 5

        badlist = [np.ones((5,)), np.ones((5, 5, 2)), 1, [1], '1']

        for perr in badlist:
            with self.assertRaises(TypeError):
                resample_inf_func(perr, ppa_in, ppa_out)
                pass
            pass
        pass


    def test_invalid_ppa_in(self):
        """Check invalid inputs rejected as expected"""
        inf_func = np.ones((41, 41))
        ppa_out = 5

        badlist = [np.ones((5,)), np.ones((5, 5, 2)), [1], '1', 1j,
                   0, -1]

        for perr in badlist:
            with self.assertRaises(TypeError):
                resample_inf_func(inf_func, perr, ppa_out)
                pass
            pass
        pass


    def test_invalid_ppa_out(self):
        """Check invalid inputs rejected as expected"""
        inf_func = np.ones((41, 41))
        ppa_in = 10

        badlist = [np.ones((5,)), np.ones((5, 5, 2)), [1], '1', 1j,
                   0, -1]

        for perr in badlist:
            with self.assertRaises(TypeError):
                resample_inf_func(inf_func, ppa_in, perr)
                pass
            pass
        pass


    def test_rectangle_inf_func(self):
        """Check we can rescale a rectangular influence function"""
        inf_func = np.ones((41, 43))
        ppa_in = 10
        ppa_out = 5

        resample_inf_func(inf_func, ppa_in, ppa_out)
        pass


    def test_known_resample(self):
        """
        Test a resample on a simple low-order polynomial case

        Run a few of these to look at edge cases

        Note for future readers: the nr1/nc1 regrid is now using floor instead
        of ceil because otherwise the spline gets everything right except the
        edge extrapolation if there is extrapolation
        """
        tol = 1e-13

        # n, ppa_in, ppa_out
        npplist = [[51, 10, 1], # both integers
                   [51, 10.1, 1], # noninteger ppa_in
                   [51, 10, 1.1], # noninteger ppa_out
                   [51, 10.1, 1.1], # both nonintegers
                   [23, 10.1, 1.1], # smaller n
                   [51, 1.1, 10.1]] # other direction

        for n, ppa_in, ppa_out in npplist:

            # Make a starting high-res function
            xi = np.linspace(-(n-1.)/2., (n-1.)/2., n)/ppa_in
            Xi, Yi = np.meshgrid(xi, xi)
            inf_func = 1-(Xi**2 + Yi**2)

            # Build same function at different sampling
            xo = np.linspace(-(n-1.)/2., (n-1.)/2., n)/ppa_out
            Xo, Yo = np.meshgrid(xo, xo)
            inf_func_targ = 1-(Xo**2 + Yo**2)

            # Run actual resampler
            inf_func_actres = resample_inf_func(inf_func, ppa_in, ppa_out)

            # Since sizes may be different, choose min along both axes
            sa = inf_func_actres.shape
            st = inf_func_targ.shape
            minsat = [min(sa[0], st[0]), min(sa[1], st[1])]

            self.assertTrue(np.max(np.abs(pad_crop(inf_func_targ, minsat) -
                                          pad_crop(inf_func_actres, minsat)))
                                          < tol)
            pass
        pass


class TestFitSurfToDm(unittest.TestCase):
    """
    Tests for the main fitting method

    x Test inputs
    x Test with/without defaults
    x Check output is same size as input
    Convolve a setting with a (compact) function and see if we can back it out
    """

    def setUp(self):
        self.nrow = 48
        self.ncol = 48
        self.surf = np.eye(self.nrow)
        self.inf_func = np.ones((71, 71))
        self.ppa_in = 10.
        self.act_effect = build_prefilter(self.nrow,
                                          self.ncol,
                                          self.inf_func,
                                          self.ppa_in)
        pass


    def test_success(self):
        """verify base algorithm does not fail on good input"""
        fit_surf_with_dm_lsq(self.surf, self.act_effect)
        pass


    def test_same_size_input_output(self):
        """Verify output array is same size as input"""

        shapelist = [(48, 48),
                     (47, 47),
                     (48, 47),
                     (47, 48)]

        for shape in shapelist:
            surf = np.ones(shape)
            act_effect = build_prefilter(shape[0],
                                         shape[1],
                                         self.inf_func,
                                         self.ppa_in)
            out = fit_surf_with_dm_lsq(surf, act_effect)
            self.assertTrue(surf.shape == out.shape)
        pass


    def test_invalid_surf(self):
        """verify bad inputs caught"""

        for perr in [np.ones((48,)), np.ones((48, 48, 2)),
                     'txt', [1, 2], 0, 1j, -3.5]:
            with self.assertRaises(TypeError):
                fit_surf_with_dm_lsq(perr, self.act_effect)
                pass
            pass
        pass


    def test_invalid_act_effect(self):
        """verify bad inputs caught"""
        # expecting sparse array
        badsparse = scipy.sparse.csr_matrix((2304, 2303))

        for perr in [np.ones((48,)), np.ones((48, 48, 2)),
                     np.ones((2304, 2304)),
                     badsparse,
                     'txt', [1, 2], 0, 1j, -3.5]:
            with self.assertRaises(TypeError):
                fit_surf_with_dm_lsq(perr, self.act_effect)
                pass
            pass
        pass


    def test_convolution_recovery(self):
        """
        Convolve a setting with an influence function and see if it can be
        successfully deconvolved

        Use one already at the right sampling

        Run with and without inf_func normalized to
         integral_{all x, y} inf_func dx dy = 1
         - this is intended to catch issue seen previously where it would only
          recover original if input influence function was normalized to 1
        """
        assert_tol = 1e-12

        # Zernike DM shape
        Nact = 48
        Nbeam = 46.3
        centering = 'pixel'
        indsZnoll = 10  # some trefoil
        zernCube = falco.zern.gen_norm_zern_maps(Nbeam, centering, indsZnoll)
        dm0 = np.sum(zernCube, 2)
        dm0 = pad_crop(dm0, Nact)
        ppa_in = 1.

        # Prep arrays to make influence functions on the fly
        n = 15
        x = np.linspace(-(n-1.)/2., (n-1.)/2., n)/ppa_in
        X, Y = np.meshgrid(x, x)
        R2 = X**2 + Y**2

        for inf_func in [np.exp(-R2)/np.pi, # normalize to unity under integral
                         np.exp(-R2)/7.8, # unnormalized but same shape
                         np.sinc(X)*np.sinc(Y), # match sinc to grid
                         0.99*np.sinc(2.1*X)*np.sinc(2.2*Y), # desynchronize
                         pad_crop(0.9*np.ones((3, 3))/9., (n, n)),
                         ]:
            act_ef = build_prefilter(48, 48, inf_func, ppa_in)

            # Build final setting, try to back out
            surf = convolve(dm0, inf_func, mode='constant', cval=0.0)
            dm1 = fit_surf_with_dm_lsq(surf, act_ef)

            self.assertTrue(np.max(np.abs(dm1 - dm0)) < assert_tol)
            pass

        pass


    def test_howfs_inf(self):
        """Verify works with the HOWFS test influence function"""
        assert_tol = 1e-13

        # Zernike DM shape
        Nact = 48
        Nbeam = 46.3
        centering = 'pixel'
        indsZnoll = 10  # some trefoil
        zernCube = falco.zern.gen_norm_zern_maps(Nbeam, centering, indsZnoll)
        dm0 = np.sum(zernCube, 2)
        dm0 = pad_crop(dm0, Nact)
        ppa_in = 10. # testbed design ppa

        # Load in design influence function from HOWFS
        localpath = os.path.dirname(os.path.abspath(__file__))
        design_inf = pyfits.getdata(os.path.join(localpath, 'testdata',
                                         'ut_influence_dm5v2_inffix.fits'))

        # as written
        dinf1 = resample_inf_func(design_inf, ppa_in, 1.)
        surf = convolve(dm0, dinf1, mode='constant', cval=0.0)
        act_ef = build_prefilter(48, 48, design_inf, ppa_in)
        dm1 = fit_surf_with_dm_lsq(surf, act_ef)
        self.assertTrue(np.max(np.abs(dm1 - dm0)) < assert_tol)

        # With arbitrary normalization
        dinf1 = resample_inf_func(design_inf/2.65, ppa_in, 1.)
        surf = convolve(dm0, dinf1, mode='constant', cval=0.0)
        act_ef = build_prefilter(48, 48, design_inf/2.65, ppa_in)
        dm1 = fit_surf_with_dm_lsq(surf, act_ef)

        self.assertTrue(np.max(np.abs(dm1 - dm0)) < assert_tol)
        pass


class TestBuildPrefilter(unittest.TestCase):
    """
    Test the precomputation is done right

    x success
    fails on bad inputs
    builds a known shape (1 poke? 3?)
    x accepts square and rect

    """

    def setUp(self):
        self.nrow = 48
        self.ncol = 48
        self.inf_func = np.ones((71, 71))
        self.ppa_in = 10.


    def test_success(self):
        """Verify completes without incident when given good inputs"""
        build_prefilter(self.nrow, self.ncol, self.inf_func, self.ppa_in)
        pass


    def test_squares_and_rect(self):
        """Verify square and rectangle inputs have outputs of the right size"""

        rclist = [(48, 48),
                  (47, 49),
                  (1, 300),
                  (300, 2)]

        for rc in rclist:
            out = build_prefilter(rc[0], rc[1], self.inf_func, self.ppa_in)
            self.assertTrue(out.shape == (rc[0]*rc[1], rc[0]*rc[1]))
            pass
        pass


    def test_invalid_nrow(self):
        """Verify bad inputs caught"""

        for perr in [0, -1, 48.5, 1j, 'txt', (48,), np.eye(48)]:
            with self.assertRaises(TypeError):
                build_prefilter(perr, self.ncol, self.inf_func, self.ppa_in)
                pass
            pass
        pass


    def test_invalid_ncol(self):
        """Verify bad inputs caught"""

        for perr in [0, -1, 48.5, 1j, 'txt', (48,), np.eye(48)]:
            with self.assertRaises(TypeError):
                build_prefilter(self.nrow, perr, self.inf_func, self.ppa_in)
                pass
            pass
        pass


    def test_invalid_inf(self):
        """verify bad inputs caught"""

        for perr in [np.ones((48,)), np.ones((48, 48, 2)),
                     'txt', [1, 2], 0, 1j, -3.5]:
            with self.assertRaises(TypeError):
                build_prefilter(self.nrow, self.ncol, perr, self.ppa_in)
                pass
            pass
        pass


    def test_invalid_ppa_in(self):
        """Verify bad inputs caught"""

        for perr in [0, -1, 1j, 'txt', (48,), np.eye(48)]:
            with self.assertRaises(TypeError):
                build_prefilter(self.nrow, self.ncol, self.inf_func, perr)
                pass
            pass
        pass


    def test_exact_prefilter(self):
        """Check that we get what we expect given a precisely-known shape"""

        tol = 1e-13

        nrow = 3
        ncol = 3
        r0c0 = [[1, 1, 0],
                [1, 1, 0],
                [0, 0, 0]]
        r0c1 = [[1, 1, 1],
                [1, 1, 1],
                [0, 0, 0]]
        r0c2 = [[0, 1, 1],
                [0, 1, 1],
                [0, 0, 0]]
        r1c0 = [[1, 1, 0],
                [1, 1, 0],
                [1, 1, 0]]
        r1c1 = [[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]]
        r1c2 = [[0, 1, 1],
                [0, 1, 1],
                [0, 1, 1]]
        r2c0 = [[0, 0, 0],
                [1, 1, 0],
                [1, 1, 0]]
        r2c1 = [[0, 0, 0],
                [1, 1, 1],
                [1, 1, 1]]
        r2c2 = [[0, 0, 0],
                [0, 1, 1],
                [0, 1, 1]]
        rclist = [r0c0, r0c1, r0c2, r1c0, r1c1, r1c2, r2c0, r2c1, r2c2]

        target = scipy.sparse.lil_matrix((9, 9))
        for index, rc in enumerate(rclist):
            target[index, :] = np.asarray(rc).ravel()
            pass

        out = build_prefilter(nrow, ncol, np.ones((3, 3)), 1)
        self.assertTrue(np.max(np.abs(out-target)) < tol)


if __name__ == '__main__':
    unittest.main()
