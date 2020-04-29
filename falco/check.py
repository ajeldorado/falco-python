"""Module to hold input-checking functions to minimize repetition."""

import numpy as np

class CheckException(Exception):
    pass


def _checkname(vname):
    """Check interanally that we can use vname as a string for printing."""
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
    _CENTERING_ERR = 'Invalid centering specification. Options: \{}'.format(_VALID_CENTERING)

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

    if not np.isscalar(var):
        raise vexc(vname + ' must be real positive scalar')
    if not np.isreal(var):
        raise vexc(vname + ' must be real positive scalar')
    if var <= 0:
        raise vexc(vname + ' must be real positive scalar')
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

    if not np.isscalar(var):
        raise vexc(vname + ' must be real nonnegative scalar')
    if not np.isreal(var):
        raise vexc(vname + ' must be real nonnegative scalar')
    if var < 0:
        raise vexc(vname + ' must be real nonnegative scalar')
    return var


def oneD_array(var, vname, vexc):
    """
    Check whether an object is a 1D numpy array, or castable to one.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    var = np.array(var) # cast to array
    if len(var.shape) != 1:
        raise vexc(vname + ' must be a real or complex 1D array')
    if (not np.isrealobj(var)) and (not np.iscomplexobj(var)):
        raise vexc(vname + ' must be a real or complex 1D array')
    return var


def twoD_array(var, vname, vexc):
    """
    Checks whether an object is a 2D numpy array, or castable to one

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    var = np.array(var) # cast to array
    if len(var.shape) != 2:
        raise vexc(vname + ' must be a real or complex 2D array')
    if (not np.isrealobj(var)) and (not np.iscomplexobj(var)):
        raise vexc(vname + ' must be a real or complex 2D array')
    return var


def threeD_array(var, vname, vexc):
    """
    Checks whether an object is a 3D numpy array, or castable to one

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    var = np.array(var) # cast to array
    if len(var.shape) != 3:
        raise vexc(vname + ' must be a real or complex 2D array')
    if (not np.isrealobj(var)) and (not np.iscomplexobj(var)):
        raise vexc(vname + ' must be a real or complex 2D array')
    return var


def real_scalar(var, vname, vexc):
    """
    Checks whether an object is a real scalar

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not np.isscalar(var):
        raise vexc(vname + ' must be real scalar')
    if not np.isreal(var):
        raise vexc(vname + ' must be real scalar')
    return var


def positive_scalar_integer(var, vname, vexc):
    """
    Checks whether an object is a positive scalar integer

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not np.isscalar(var):
        raise vexc(vname + ' must be positive scalar integer')
    if not isinstance(var, (int, np.integer)):
        raise vexc(vname + ' must be positive scalar integer')
    if var <= 0:
        raise vexc(vname + ' must be positive scalar integer')
    return var


def nonnegative_scalar_integer(var, vname, vexc):
    """
    Checks whether an object is a nonnegative scalar integer

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not np.isscalar(var):
        raise vexc(vname + ' must be nonnegative scalar integer')
    if not isinstance(var, (int, np.integer)):
        raise vexc(vname + ' must be nonnegative scalar integer')
    if var < 0:
        raise vexc(vname + ' must be nonnegative scalar integer')
    return var


def scalar_integer(var, vname, vexc):
    """
    Checks whether an object is a scalar integer (no sign dependence)

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not np.isscalar(var):
        raise vexc(vname + ' must be scalar integer')
    if not isinstance(var, (int, np.integer)):
        raise vexc(vname + ' must be scalar integer')
    return var
