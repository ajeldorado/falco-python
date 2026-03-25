"""Secure serialization for FALCO objects using HDF5 and JSON."""
import json
import numpy as np
import h5py
from pathlib import Path
from falco.config import Object


def _object_to_dict(obj, hdf5_file, hdf5_path=""):
    """
    Recursively convert Object to dict, saving numpy arrays to HDF5.

    Parameters
    ----------
    obj : Object, dict, list, or other
        Object to serialize
    hdf5_file : h5py.File
        Open HDF5 file handle
    hdf5_path : str
        Current path in HDF5 hierarchy

    Returns
    -------
    dict or list or primitive
        JSON-serializable structure with references to HDF5 datasets
    """
    if isinstance(obj, Object):
        # Access the underlying data dict
        return _object_to_dict(obj.data, hdf5_file, hdf5_path)

    elif isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            # Create safe HDF5 key (avoid special chars)
            safe_key = str(key).replace('/', '_')
            new_path = f"{hdf5_path}/{safe_key}" if hdf5_path else safe_key
            result[key] = _object_to_dict(value, hdf5_file, new_path)
        return result

    elif isinstance(obj, (list, tuple)):
        result = []
        for idx, value in enumerate(obj):
            new_path = f"{hdf5_path}/{idx}"
            result.append(_object_to_dict(value, hdf5_file, new_path))
        return result

    elif isinstance(obj, np.ndarray):
        # Save array to HDF5 and return reference
        try:
            hdf5_file.create_dataset(hdf5_path, data=obj, compression="gzip")
            return {"__ndarray__": hdf5_path, "dtype": str(obj.dtype), "shape": obj.shape}
        except Exception as e:
            # If compression fails (e.g., for complex types), try without
            hdf5_file.create_dataset(hdf5_path, data=obj)
            return {"__ndarray__": hdf5_path, "dtype": str(obj.dtype), "shape": obj.shape}

    elif isinstance(obj, (np.integer, np.floating)):
        # Convert numpy scalars to Python types
        return obj.item()

    elif isinstance(obj, np.complexfloating):
        # Store complex as dict with real and imaginary parts
        return {"__complex__": True, "real": obj.real.item(), "imag": obj.imag.item()}

    elif isinstance(obj, (str, int, float, bool, type(None))):
        # JSON-serializable primitive types
        return obj

    else:
        # For other types, try to get a string representation
        # This handles types that can't be easily serialized
        return {"__type__": type(obj).__name__, "__repr__": repr(obj)}


def _dict_to_object(data, hdf5_file, reconstruct_objects=True):
    """
    Recursively convert dict back to Object, loading numpy arrays from HDF5.

    Parameters
    ----------
    data : dict, list, or other
        Deserialized JSON structure
    hdf5_file : h5py.File
        Open HDF5 file handle
    reconstruct_objects : bool
        Whether to reconstruct Object instances (default True)

    Returns
    -------
    Object, dict, list, or other
        Reconstructed Python object
    """
    if isinstance(data, dict):
        # Check for special markers
        if "__ndarray__" in data:
            # Load numpy array from HDF5
            path = data["__ndarray__"]
            return np.array(hdf5_file[path])

        elif "__complex__" in data:
            # Reconstruct complex number
            return complex(data["real"], data["imag"])

        elif "__type__" in data:
            # Can't reconstruct, return dict with type info
            return data

        else:
            # Regular dict - recurse on values
            result = {}
            for key, value in data.items():
                result[key] = _dict_to_object(value, hdf5_file, reconstruct_objects)

            # Convert to Object if requested
            if reconstruct_objects:
                obj = Object()
                obj.data = result
                return obj
            else:
                return result

    elif isinstance(data, list):
        return [_dict_to_object(item, hdf5_file, reconstruct_objects) for item in data]

    else:
        # Primitive type, return as-is
        return data


def save(obj, filename):
    """
    Save a FALCO Object to HDF5 + JSON format.

    Parameters
    ----------
    obj : Object
        FALCO Object to save (typically `out` or `mp`)
    filename : str
        Output filename (will create .h5 and .json files)

    Returns
    -------
    None
    """
    filename = Path(filename)

    # Create HDF5 and JSON filenames
    h5_file = filename.with_suffix('.h5')
    json_file = filename.with_suffix('.json')

    # Serialize to HDF5 + dict
    with h5py.File(h5_file, 'w') as hf:
        serialized = _object_to_dict(obj, hf)

    # Save dict structure to JSON
    with open(json_file, 'w') as jf:
        json.dump(serialized, jf, indent=2)


def load(filename, reconstruct_objects=True):
    """
    Load a FALCO Object from HDF5 + JSON format.

    Parameters
    ----------
    filename : str
        Input filename (can be .h5, .json, or base name)
    reconstruct_objects : bool
        Whether to reconstruct Object instances (default True)
        If False, returns plain dicts instead

    Returns
    -------
    Object or dict
        Reconstructed FALCO Object
    """
    filename = Path(filename)

    # Determine HDF5 and JSON filenames
    if filename.suffix == '.h5':
        h5_file = filename
        json_file = filename.with_suffix('.json')
    elif filename.suffix == '.json':
        h5_file = filename.with_suffix('.h5')
        json_file = filename
    else:
        h5_file = filename.with_suffix('.h5')
        json_file = filename.with_suffix('.json')

    # Load JSON structure
    with open(json_file, 'r') as jf:
        serialized = json.load(jf)

    # Reconstruct object from HDF5 + dict
    with h5py.File(h5_file, 'r') as hf:
        obj = _dict_to_object(serialized, hf, reconstruct_objects)

    return obj
