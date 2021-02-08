"""Module to hold input-checking functions to minimize repetition."""

import numpy as np
import numbers

class CheckException(Exception):
    pass

# String check support
string_types = (str, bytes)

# Int check support
int_types = (int, np.integer)

def _checkname(vname):
    """Check internally that we can use vname as a string for printing."""
    if not isinstance(vname, str):
        raise CheckException('vname must be a string when fed to check ' +
                             'functions')
    pass


def _checkexc(vexc):
    """Check interanally that we can raise from the vexc object."""
    if not isinstance(vexc, type):  # pre-check it is class-like
        raise CheckException('vexc must be a Exception, or an object ' +
                             'descended from one when fed to check functions')
    if not issubclass(vexc, Exception):
        raise CheckException('vexc must be a Exception, or an object ' +
                             'descended from one when fed to check functions')
    pass


def centering(var):
    """
    Check whether an object is in the values ['pixel', 'interpixel'].

    Parameters
    ----------
    var
        Variable to check

    Returns
    -------
    var
        Same value as input

    """
    _VALID_CENTERING = ['pixel', 'interpixel']
    _CENTERING_ERR = ('Invalid centering specification. Options: \{}'.format(_VALID_CENTERING))

    if not isinstance(var, str):
        raise TypeError("'centering' value must be a string'")
    if not (var in _VALID_CENTERING):
        raise ValueError(_CENTERING_ERR)
    return var


def is_dict(var, vname):
    """
    Check whether an object is a dictionary.

    Parameters
    ----------
    var: dict
        variable to check
    vname: str
        string to output in case of error for debugging
    """
    _checkname(vname)

    if not isinstance(var, dict):
        raise TypeError(vname + 'must be a dictionary')
    return var


def is_bool(var, vname):
    """
    Check whether an object is a boolean.

    Parameters
    ----------
    var : bool
        variable to check
    vname : str
        string to output in case of error for debugging
    """
    _checkname(vname)

    if not isinstance(var, bool):
        raise TypeError(vname + 'must be a bool')
    return var


def real_positive_scalar(var, vname, vexc):
    """
    Check whether an object is a real positive scalar.

    Parameters
    ----------
    var : float
        Variable to check.
    vname : str
        string to output in case of error for debugging.
    vexc : Exception
        Exception to raise in case of error for debugging.

    Returns
    -------
    var : float
        Same as input.

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, numbers.Number):
        raise vexc(vname + ' must be scalar')
    if not np.isrealobj(var):
        raise vexc(vname + ' must be real')
    if var <= 0:
        raise vexc(vname + ' must be positive')
    return var


def real_nonnegative_scalar(var, vname, vexc):
    """
    Check whether an object is a real nonnegative scalar.

    Parameters
    ----------
    var: variable to check
    vname: string to output in case of error for debugging
    vexc: Exception to raise in case of error for debugging

    Returns
    -------
     var:
         Same as input.

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, numbers.Number):
        raise vexc(vname + ' must be scalar')
    if not np.isrealobj(var):
        raise vexc(vname + ' must be real')
    if var < 0:
        raise vexc(vname + ' must be nonnegative')
    return var


def real_array(var, vname, vexc):
    """
    Check whether an object is a real numpy array, or castable to one.

    Parameters
    ----------
    var: variable to check
    vname: string to output in case of error for debugging
    vexc: Exception to raise in case of error for debugging

    Returns
    -------
     var:
         Same as input.

    """
    _checkname(vname)
    _checkexc(vexc)

    var = np.array(var)  # cast to array
    if len(var.shape) == 0:
        raise vexc(vname + ' must have length > 0')
    if not np.isrealobj(var):
        raise vexc(vname + ' must be a real array')
    return var


def oneD_array(var, vname, vexc):
    """
    Check whether an object is a 1D numpy array, or castable to one.

    Parameters
    ----------
    var: variable to check
    vname: string to output in case of error for debugging
    vexc: Exception to raise in case of error for debugging

    Returns
    -------
     var:
         Same as input.

    """
    _checkname(vname)
    _checkexc(vexc)

    var = np.array(var) # cast to array
    if len(var.shape) != 1:
        raise vexc(vname + ' must be a 1D array')
    if (not np.isrealobj(var)) and (not np.iscomplexobj(var)):
        raise vexc(vname + ' must be a real or complex 1D array')
    return var


def twoD_array(var, vname, vexc):
    """
    Check whether an object is a 2D numpy array, or castable to one.

    Parameters
    ----------
    var: variable to check
    vname: string to output in case of error for debugging
    vexc: Exception to raise in case of error for debugging

    Returns
    -------
    var

    """
    _checkname(vname)
    _checkexc(vexc)

    var = np.array(var) # cast to array
    if len(var.shape) != 2:
        raise vexc(vname + ' must be a 2D array')
    if (not np.isrealobj(var)) and (not np.iscomplexobj(var)):
        raise vexc(vname + ' must be a real or complex 2D array')
    return var


def twoD_square_array(var, vname, vexc):
    """
    Check whether an object is a 2D square array_like.

    Parameters
    ----------
    var: variable to check
    vname: string to output in case of error for debugging
    vexc: Exception to raise in case of error for debugging

    Returns
    -------
     var:
         Same as input.

    """
    _checkname(vname)
    _checkexc(vexc)

    var = np.array(var) # cast to array
    if len(var.shape) != 2:
        raise vexc(vname + ' must be a 2D array')
    else: # is 2-D
        if not var.shape[0] == var.shape[1]:
            raise vexc(vname + ' must be a square 2D array')
    if (not np.isrealobj(var)) and (not np.iscomplexobj(var)):
        raise vexc(vname + ' must be a real or complex square 2D array')
    return var


def threeD_array(var, vname, vexc):
    """
    Check whether an object is a 3D numpy array, or castable to one.

    Parameters
    ----------
    var: variable to check
    vname: string to output in case of error for debugging
    vexc: Exception to raise in case of error for debugging

    Returns
    -------
    var

    """
    _checkname(vname)
    _checkexc(vexc)

    var = np.array(var) # cast to array
    if len(var.shape) != 3:
        raise vexc(vname + ' must be a 3D array')
    if (not np.isrealobj(var)) and (not np.iscomplexobj(var)):
        raise vexc(vname + ' must be a real or complex 3D array')
    return var


def real_scalar(var, vname, vexc):
    """
    Check whether an object is a real scalar.

    Parameters
    ----------
    var: variable to check
    vname: string to output in case of error for debugging
    vexc: Exception to raise in case of error for debugging

    Returns
    -------
    var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, numbers.Number):
        raise vexc(vname + ' must be scalar')
    if not np.isrealobj(var):
        raise vexc(vname + ' must be real')
    return var


def real_nonnegative_scalar(var, vname, vexc):
    """
    Checks whether an object is a real nonnegative scalar.

    Parameters
    ----------
    var: variable to check
    vname: string to output in case of error for debugging
    vexc: Exception to raise in case of error for debugging

    Returns
    -------
    var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, numbers.Number):
        raise vexc(vname + ' must be scalar')
    if not np.isrealobj(var):
        raise vexc(vname + ' must be real')
    if var < 0:
        raise vexc(vname + ' must be nonnegative')
    return var
    

def positive_scalar_integer(var, vname, vexc):
    """
    Check whether an object is a positive scalar integer.

    Parameters
    ----------
    var: variable to check
    vname: string to output in case of error for debugging
    vexc: Exception to raise in case of error for debugging

    Returns
    -------
    var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, numbers.Number):
        raise vexc(vname + ' must be scalar')
    if not isinstance(var, int_types):
        raise vexc(vname + ' must be integer')
    if var <= 0:
        raise vexc(vname + ' must be positive')
    return var


def nonnegative_scalar_integer(var, vname, vexc):
    """
    Check whether an object is a nonnegative scalar integer.

    Parameters
    ----------
    var: variable to check
    vname: string to output in case of error for debugging
    vexc: Exception to raise in case of error for debugging

    Returns
    -------
    var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, numbers.Number):
        raise vexc(vname + ' must be scalar')
    if not isinstance(var, int_types):
        raise vexc(vname + ' must be integer')
    if var < 0:
        raise vexc(vname + ' must be nonnegative')
    return var


def scalar_integer(var, vname, vexc):
    """
    Check whether an object is a scalar integer (no sign dependence).

    Parameters
    ----------
    var: variable to check
    vname: string to output in case of error for debugging
    vexc: Exception to raise in case of error for debugging

    Returns
    -------
    var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, numbers.Number):
        raise vexc(vname + ' must be scalar')
    if not isinstance(var, int_types):
        raise vexc(vname + ' must be integer')
    return var
