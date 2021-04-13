"""Unit tests for check.py."""
import unittest
import numpy as np
import sys

sys.path.insert(0,"../")
from falco import check

class TestCheckException(Exception):
    pass


class TestCheck(unittest.TestCase):
    """
    For each check, test with valid and invalid inputs for all three inputs.

    Test valid here as well since most other functions rely on these for
    error checking
    """

    # real_positive_scalar
    def test_real_positive_scalar_good(self):
        """
        Verify checker works correctly for valid input.

        Type: real positive scalar
        """
        try:
            check.real_positive_scalar(1, 'rps', TestCheckException)
        except check.CheckException:
            self.fail('real_positive_scalar failed on valid input')

    def test_real_positive_scalar_bad_var(self):
        """
        Fail on invalid variable type.

        Type: real positive scalar
        """
        for v0 in [-1, 1j, (1.,), [5, 5], 'v0']:
            with self.assertRaises(TestCheckException):
                check.real_positive_scalar(v0, 'rps', TestCheckException)

    def test_real_positive_scalar_bad_vname(self):
        """Fail on invalid input name for user output."""
        with self.assertRaises(check.CheckException):
            check.real_positive_scalar(1, (1,), TestCheckException)

    def test_real_positive_scalar_bad_vexc(self):
        """Fail on input vexc not an Exception."""
        with self.assertRaises(check.CheckException):
            check.real_positive_scalar(1, 'rps', 'TestCheckException')

    # real_nonnegative_scalar
    def test_real_nonnegative_scalar_good(self):
        """
        Verify checker works correctly for valid input.

        Type: real nonnegative scalar
        """
        try:
            check.real_nonnegative_scalar(0, 'rps', TestCheckException)
        except check.CheckException:
            self.fail('real_nonnegative_scalar failed on valid input')

    def test_real_nonnegative_scalar_bad_var(self):
        """
        Fail on invalid variable type.

        Type: real nonnegative scalar
        """
        for v0 in [-1, 1j, (1.,), [5, 5], 'v0']:
            with self.assertRaises(TestCheckException):
                check.real_nonnegative_scalar(v0, 'rps', TestCheckException)

    def test_real_nonnegative_scalar_bad_vname(self):
        """Fail on invalid input name for user output."""
        with self.assertRaises(check.CheckException):
            check.real_nonnegative_scalar(0, (1,), TestCheckException)

    def test_real_nonnegative_scalar_bad_vexc(self):
        """Fail on input vexc not an Exception."""
        with self.assertRaises(check.CheckException):
            check.real_nonnegative_scalar(0, 'rps', 'TestCheckException')

    # real_array
    def test_real_array_good(self):
        """
        Verify checker works correctly for valid input.

        Type: real array
        """
        try:
            check.real_array(np.ones((5, 5)), 'real', TestCheckException)
        except check.CheckException:
            self.fail('real_array failed on valid input')

    def test_real_array_bad_var(self):
        """
        Fail on invalid variable type.

        Type: real array
        """
        for v0 in [1j*np.ones((5, 4)), (1+1j)*np.ones((5, 5, 5)), 'foo']:
            with self.assertRaises(TestCheckException):
                check.real_array(v0, '1D', TestCheckException)

    def test_real_array_bad_vname(self):
        """Fail on invalid input name for user output."""
        with self.assertRaises(check.CheckException):
            check.real_array(np.ones((5, 5)), (1,), TestCheckException)

    def test_real_array_bad_vexc(self):
        """Fail on input vexc not an Exception."""
        with self.assertRaises(check.CheckException):
            check.real_array(np.ones((5, )), 'rps', 'TestCheckException')

    # oneD_array
    def test_oneD_array_good(self):
        """
        Verify checker works correctly for valid input.

        Type: 1D array
        """
        try:
            check.oneD_array(np.ones((5, )), '1D', TestCheckException)
        except check.CheckException:
            self.fail('oneD_array failed on valid input')

    def test_oneD_array_bad_var(self):
        """
        Fail on invalid variable type.

        Type: 1D array
        """
        for v0 in [np.ones((5, 4)), np.ones((5, 5, 5)), 'foo']:
            with self.assertRaises(TestCheckException):
                check.oneD_array(v0, '1D', TestCheckException)

    def test_oneD_array_bad_vname(self):
        """Fail on invalid input name for user output."""
        with self.assertRaises(check.CheckException):
            check.oneD_array(np.ones((5, )), (1,), TestCheckException)

    def test_oneD_array_bad_vexc(self):
        """Fail on input vexc not an Exception."""
        with self.assertRaises(check.CheckException):
            check.oneD_array(np.ones((5, )), 'rps', 'TestCheckException')

    # twoD_array
    def test_twoD_array_good(self):
        """
        Verify checker works correctly for valid input.

        Type: 2D array
        """
        try:
            check.twoD_array(np.ones((5, 5)), '2d', TestCheckException)
        except check.CheckException:
            self.fail('twoD_array failed on valid input')

    def test_twoD_array_bad_var(self):
        """
        Fail on invalid variable type.

        Type: 2D array
        """
        for v0 in [np.ones((5,)), np.ones((5, 5, 5)), [], 'foo']:
            with self.assertRaises(TestCheckException):
                check.twoD_array(v0, '2d', TestCheckException)

    def test_twoD_array_bad_vname(self):
        """Fail on invalid input name for user output."""
        with self.assertRaises(check.CheckException):
            check.twoD_array(np.ones((5, 5)), (1,), TestCheckException)

    def test_twoD_array_bad_vexc(self):
        """Fail on input vexc not an Exception."""
        with self.assertRaises(check.CheckException):
            check.twoD_array(np.ones((5, 5)), 'rps', 'TestCheckException')

    # twoD_square_array
    def test_twoD_square_array_good(self):
        """
        Verify checker works correctly for valid input.

        Type: 2D array
        """
        try:
            check.twoD_array(np.ones((5, 5)), '2d', TestCheckException)
        except check.CheckException:
            self.fail('twoD_square_array failed on valid input')

    def test_twoD_square_array_bad_var(self):
        """
        Fail on invalid variable type.

        Type: 2D array
        """
        for v0 in [np.ones((5,)), np.ones((5, 5, 5)), [], 'foo']:
            with self.assertRaises(TestCheckException):
                check.twoD_square_array(v0, '2d', TestCheckException)

    def test_twoD_square_array_bad_var_shape(self):
        """
        Fail on invalid variable type.

        Type: 2D square array
        """
        for v0 in [np.ones((5, 4)), np.ones((4, 6))]:
            with self.assertRaises(TestCheckException):
                check.twoD_square_array(v0, '2d', TestCheckException)

    def test_twoD_square_array_bad_vname(self):
        """Fail on invalid input name for user output."""
        with self.assertRaises(check.CheckException):
            check.twoD_square_array(np.ones((5, 5)), (1,), TestCheckException)

    def test_twoD_square_array_bad_vexc(self):
        """Fail on input vexc not an Exception."""
        with self.assertRaises(check.CheckException):
            check.twoD_square_array(np.ones((5, 5)), 'rps',
                                    'TestCheckException')

    # threeD_array
    def test_threeD_array_good(self):
        """
        Verify checker works correctly for valid input.

        Type: 3D array
        """
        try:
            check.threeD_array(np.ones((5, 5, 2)), '3d', TestCheckException)
        except check.CheckException:
            self.fail('threeD_array failed on valid input')

    def test_threeD_array_bad_var(self):
        """
        Fail on invalid variable type.

        Type: 3D array
        """
        for v0 in [np.ones((5,)), np.ones((5, 5)), np.ones((2, 2, 2, 2)),
                   [], 'foo']:
            with self.assertRaises(TestCheckException):
                check.threeD_array(v0, '3d', TestCheckException)

    def test_threeD_array_bad_vname(self):
        """Fail on invalid input name for user output."""
        with self.assertRaises(check.CheckException):
            check.threeD_array(np.ones((5, 5, 2)), (1,), TestCheckException)

    def test_threeD_array_bad_vexc(self):
        """Fail on input vexc not an Exception."""
        with self.assertRaises(check.CheckException):
            check.threeD_array(np.ones((5, 5, 2)), 'rps', 'TestCheckException')

    # real_scalar
    def test_real_scalar_good(self):
        """
        Verify checker works correctly for valid input.

        Type: real scalar
        """
        try:
            check.real_scalar(1, 'rs', TestCheckException)
        except check.CheckException:
            self.fail('real_scalar failed on valid input')

    def test_real_scalar_bad_var(self):
        """
        Fail on invalid variable type.

        Type: real scalar
        """
        for v0 in [1j, (1.,), [5, 5], 'rs']:
            with self.assertRaises(TestCheckException):
                check.real_scalar(v0, 'rs', TestCheckException)

    def test_real_scalar_bad_vname(self):
        """Fail on invalid input name for user output."""
        with self.assertRaises(check.CheckException):
            check.real_scalar(1, (1,), TestCheckException)

    def test_real_scalar_bad_vexc(self):
        """Fail on input vexc not an Exception."""
        with self.assertRaises(check.CheckException):
            check.real_scalar(1, 'rs', 'TestCheckException')

    # positive_scalar_integer
    def test_positive_scalar_integer_good(self):
        """
        Verify checker works correctly for valid input.

        Type: positive scalar integer
        """
        try:
            check.positive_scalar_integer(1, 'psi', TestCheckException)
        except check.CheckException:
            self.fail('positive_scalar_integer failed on valid input')

    def test_positive_scalar_integer_bad_var(self):
        """
        Fail on invalid variable type.

        Type: positive scalar integer
        """
        for v0 in [1.0, -1, 0, 1j, (1.,), [5, 5], 'psi']:
            with self.assertRaises(TestCheckException):
                check.positive_scalar_integer(v0, 'psi', TestCheckException)

    def test_positive_scalar_integer_bad_vname(self):
        """Fail on invalid input name for user output."""
        with self.assertRaises(check.CheckException):
            check.positive_scalar_integer(1, (1,), TestCheckException)

    def test_positive_scalar_integer_bad_vexc(self):
        """Fail on input vexc not an Exception."""
        with self.assertRaises(check.CheckException):
            check.positive_scalar_integer(1, 'psi', 'TestCheckException')

    # nonnegative_scalar_integer
    def test_nonnegative_scalar_integer_good(self):
        """
        Verify checker works correctly for valid input.

        Type: nonnegative scalar integer
        """
        for j in [0, 1, 2]:
            try:
                check.nonnegative_scalar_integer(j, 'nsi', TestCheckException)
            except check.CheckException:
                self.fail('nonnegative_scalar_integer failed on valid input')

    def test_nonnegative_scalar_integer_bad_var(self):
        """
        Fail on invalid variable type.

        Type: nonnegative scalar integer
        """
        for v0 in [1.0, -1, 1j, (1.,), [5, 5], 'nsi']:
            with self.assertRaises(TestCheckException):
                check.nonnegative_scalar_integer(v0, 'nsi', TestCheckException)

    def test_nonnegative_scalar_integer_bad_vname(self):
        """Fail on invalid input name for user output."""
        with self.assertRaises(check.CheckException):
            check.nonnegative_scalar_integer(1, (1,), TestCheckException)

    def test_nonnegative_scalar_integer_bad_vexc(self):
        """Fail on input vexc not an Exception."""
        with self.assertRaises(check.CheckException):
            check.nonnegative_scalar_integer(1, 'nsi', 'TestCheckException')

    # scalar_integer
    def test_scalar_integer_good(self):
        """
        Verify checker works correctly for valid input.

        Type: scalar integer
        """
        for j in [-2, -1, 0, 1, 2]:
            try:
                check.scalar_integer(j, 'si', TestCheckException)
            except check.CheckException:
                self.fail('scalar_integer failed on valid input')

    def test_scalar_integer_bad_var(self):
        """
        Fail on invalid variable type.

        Type: scalar integer
        """
        for v0 in [1.0, 1j, (1.,), [5, 5], 'si']:
            with self.assertRaises(TestCheckException):
                check.scalar_integer(v0, 'si', TestCheckException)

    def test_scalar_integer_bad_vname(self):
        """Fail on invalid input name for user output."""
        with self.assertRaises(check.CheckException):
            check.scalar_integer(1, (1,), TestCheckException)

    def test_scalar_integer_bad_vexc(self):
        """Fail on input vexc not an Exception."""
        with self.assertRaises(check.CheckException):
            check.scalar_integer(1, 'si', 'TestCheckException')


if __name__ == '__main__':
    unittest.main()
