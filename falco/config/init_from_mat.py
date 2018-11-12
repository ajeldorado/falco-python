import scipy.io
import numpy as np
from numpy import inf


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    See
    https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    for more info
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], scipy.io.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, scipy.io.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def add_ident(s):
    return "    " + s


def _spec_arg(k, kwargs, v):
    if k in kwargs:
        return kwargs[k]
    elif "mat_struct" in kwargs:
        return eval(kwargs["mat_struct"] + "." + k)
    else:
        return v


def reconstruct_classes(d, keys=None, outter_scope=False):
    """
    Recursively generate code nested classes (similar to MATLAB's nested structs) from a nested dictionary.
    Returns Python code lines to be used to define the classes and initialize their instances.
    """
    if keys is None:
        keys = d.keys()

    l_vars = []
    l_subclasses = []
    for k in keys:
        if isinstance(d[k], dict):
            l_subclasses.append("class _base_%s:" % (k))
            l_subclasses += map(add_ident, reconstruct_classes(d[k]))
            if outter_scope:
                l_vars.append("global " + k)
                l_vars.append("%s = _base_%s()" % (k, k))
            else:
                l_vars.append((k, "self._base_%s()" % (k)))
        elif isinstance(d[k], str):
            l_vars.append((k, "\"" + d[k] + "\""))
        else:
            l_vars.append((k, d[k]))

    # TODO: give this a more descriptive name
    l = l_subclasses
    if outter_scope:
        l += l_vars
    else:
        # TODO: update to python3 string formatting
        l += ["def __init__(self,**kwargs):"] + \
            [add_ident("self.%s = _spec_arg(\"%s\", kwargs, %s)" % (k, k, v)) for k, v in l_vars]

    return l
