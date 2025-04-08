import deepmerge
from dataclasses import dataclass, field

from falco.config.Eval import Eval


class Object:
    """
    An object wrapper around a dictionary.

    Allows use of field syntax (`obj.field`) as well as index syntax (`obj['field']`).
    Automatically evaluates `Eval` lazy parameters.
    """

    def __init__(self, **kwargs):
        data = kwargs if kwargs is not None else {}
        if self.__dict__ is not None:
            data = deepmerge.conservative_merger.merge(data, self.__dict__)
        self.__dict__ = {"data": data}

    def merge(self, **kwargs):
        deepmerge.always_merger.merge(self.data, kwargs)

    def __getattr__(self, item):
        # Normally `self.data` exists in `self.__dict__`, so referring to `self.data`
        # doesn't result in a `__getattr__` call.
        # But if `__init__` has not been called yet, then `data` does not exist
        # and any `__getattr__` call for any attribute will recurse later in the function.
        # So we just create the dictionary now because it needs to exist anyway.
        if item == 'data':
            self.data = {}
            return self.data

        if item not in self.data:
            raise AttributeError

        if isinstance(self.data[item], Eval):
            return self.data[item].evaluate()
        return self.data[item]

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __eq__(self, other):
        if not isinstance(other, Object):
            return False
        return self.data == other.data and self.__dict__ == other.__dict__

    def show(self):
        """Print self as a dictionary"""
        print(self.data)
