# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for pad_crop.py
"""

import unittest

import numpy as np

import numpy as np

from falco.util import pad_crop


class TestOffcenterCrop(unittest.TestCase):
    """
    Unit test suite for pad_crop()

    This will have lots of special cases to handle odd/even sizing and
    truncating vs. non-truncating
    """

    # Success tests (non-truncating)
    def test_insert_all_size_odd(self):
        """Check insert behavior, (o,o) --> (o,o)"""
        out = pad_crop(np.ones((3, 3)), (5, 5))
        test = np.array([[0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0]])
        self.assertTrue((out == test).all())

    def test_insert_all_size_even(self):
        """Check insert behavior, (e,e) --> (e,e)"""
        out = pad_crop(np.ones((2, 2)), (4, 4))
        test = np.array([[0, 0, 0, 0],
                         [0, 1, 1, 0],
                         [0, 1, 1, 0],
                         [0, 0, 0, 0]])
        self.assertTrue((out == test).all())

    def test_insert_first_index_odd_size_even(self):
        """Check insert behavior, (o,e) --> (e,e)"""
        out = pad_crop(np.ones((3, 2)), (4, 4))
        test = np.array([[0, 0, 0, 0],
                         [0, 1, 1, 0],
                         [0, 1, 1, 0],
                         [0, 1, 1, 0]])
        self.assertTrue((out == test).all())

    def test_insert_second_index_odd_size_even(self):
        """Check insert behavior, (e,o) --> (e,e)"""
        out = pad_crop(np.ones((2, 3)), (4, 4))
        test = np.array([[0, 0, 0, 0],
                         [0, 1, 1, 1],
                         [0, 1, 1, 1],
                         [0, 0, 0, 0]])
        self.assertTrue((out == test).all())

    def test_insert_first_index_even_size_odd(self):
        """Check insert behavior, (e,o) --> (o,o)"""
        out = pad_crop(np.ones((4, 3)), (5, 5))
        test = np.array([[0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0]])
        self.assertTrue((out == test).all())

    def test_insert_second_index_even_size_odd(self):
        """Check insert behavior, (o,e) --> (o,o)"""
        out = pad_crop(np.ones((3, 4)), (5, 5))
        test = np.array([[0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0]])
        self.assertTrue((out == test).all())

    # Truncating
    def test_insert_all_size_odd_trunc(self):
        """Check insert behavior, (o,o) --> (o,o), truncating"""
        tmat = np.outer(np.arange(0, 10, 2), np.arange(0, 15, 3))
        out = pad_crop(tmat, (3, 3))
        test = np.array([[6, 12, 18],
                         [12, 24, 36],
                         [18, 36, 54]])
        self.assertTrue((out == test).all())

    def test_insert_all_size_even_trunc(self):
        """Check insert behavior, (e,e) --> (e,e), truncating"""
        tmat = np.outer(np.arange(0, 8, 2), np.arange(0, 12, 3))
        out = pad_crop(tmat, (2, 2))
        test = np.array([[6, 12],
                         [12, 24]])
        self.assertTrue((out == test).all())

    def test_insert_first_index_odd_size_even_trunc(self):
        """Check insert behavior, (e,e) --> (o,e), truncating"""
        tmat = np.outer(np.arange(0, 8, 2), np.arange(0, 12, 3))
        out = pad_crop(tmat, (3, 2))
        test = np.array([[6, 12],
                         [12, 24],
                         [18, 36]])
        self.assertTrue((out == test).all())

    def test_insert_second_index_odd_size_even_trunc(self):
        """Check insert behavior, (e,e) --> (e,o), truncating"""
        tmat = np.outer(np.arange(0, 8, 2), np.arange(0, 12, 3))
        out = pad_crop(tmat, (2, 3))
        test = np.array([[6, 12, 18],
                         [12, 24, 36]])
        self.assertTrue((out == test).all())

    def test_insert_first_index_even_size_odd_trunc(self):
        """Check insert behavior, (e,o) --> (o,o), truncating"""
        tmat = np.outer(np.arange(0, 10, 2), np.arange(0, 15, 3))
        out = pad_crop(tmat, (4, 3))
        test = np.array([[0, 0, 0],
                         [6, 12, 18],
                         [12, 24, 36],
                         [18, 36, 54]])
        self.assertTrue((out == test).all())

    def test_insert_second_index_even_size_odd_trunc(self):
        """Check insert behavior, (o,e) --> (o,o), truncating"""
        tmat = np.outer(np.arange(0, 10, 2), np.arange(0, 15, 3))
        out = pad_crop(tmat, (3, 4))
        test = np.array([[0, 6, 12, 18],
                         [0, 12, 24, 36],
                         [0, 18, 36, 54]])
        self.assertTrue((out == test).all())

    # Mixed
    def test_insert_first_large_second_small(self):
        """Check insert behavior, truncating second axis only"""
        out = pad_crop(np.ones((4, 4)), (6, 2))
        test = np.array([[0, 0],
                         [1, 1],
                         [1, 1],
                         [1, 1],
                         [1, 1],
                         [0, 0]])
        self.assertTrue((out == test).all())

    def test_insert_first_small_second_large(self):
        """Check insert behavior, truncating first axis only"""
        out = pad_crop(np.ones((4, 4)), (2, 6))
        test = np.array([[0, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 0]])
        self.assertTrue((out == test).all())

    # Other success
    def test_transitivity(self):
        """
        Verify that successive pad_crops always end up at the same location.
        Should be no path dependence
        """
        for midsize in [(3, 3), (3, 4), (8, 7), (6, 6), (2, 2), (8, 8)]:
            out1a = pad_crop(np.ones((2, 2)), midsize)
            out1b = pad_crop(out1a, (8, 8))
            out2 = pad_crop(np.ones((2, 2)), (8, 8))
            self.assertTrue((out1b == out2).all())

    # def test_dtype_passed(self):
    #     """Check pad_crop maintains data type"""
    #     for dt in [bool, np.int32, np.float32, np.float64,
    #                np.complex64, np.complex128]:
    #         inm = np.ones((2, 2), dtype=dt)
    #         out = pad_crop(inm, (4, 4))
    #         self.assertTrue(out.dtype == inm.dtype)

    # Failure tests
    def test_arr0_2darray(self):
        """Check input array type valid"""
        for arr0 in [(2, 2), np.zeros((2,)), np.zeros((2, 2, 2))]:
            with self.assertRaises(TypeError):
                pad_crop(arr0, (4, 4))

    def test_outsize_1or2elem_list(self):
        """Check outsize is 2-element list"""
        for outsize in [(4, 4, 4), [], None]:
            with self.assertRaises(TypeError):
                pad_crop(np.zeros((2, 2)), outsize)

    def test_outsize_has_non_positive_int(self):
        """Check outsize elements are positive integers"""
        for outsize in [(0, 4), (-5, 4), (6.3, 4), (4.0, 4)]:
            with self.assertRaises(TypeError):
                pad_crop(np.zeros((2, 2)), outsize)


if __name__ == '__main__':
    unittest.main()
