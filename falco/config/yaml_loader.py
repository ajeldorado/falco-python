import yaml
from falco.config import Eval, Object

def load_from_str(yaml_str, eval_globals, eval_locals, ctors_dict):
    """
    Loads a yaml string into a config Object.
    Automatically provides a tag constructor for the Eval class, and
    all dictionaries are automatically converted to the config Object class.

    See Eval for more details on the globals and locals arguments.

    :param yaml_str: a yaml string
    :param eval_globals: globals to expose to eval code, in a dictionary
    :param eval_locals: locals to expose to eval code, in a dictionary
    :param ctors_dict: a dictionary of extra constructors, such as `{'!Probe', object_constructor(Probe)}`
    :return a python object
    """
    result_obj = yaml.load(yaml_str, Loader=_get_loader(eval_globals, eval_locals, ctors_dict))
    return result_obj.data

def object_constructor(noarg_constructor):
    """
    Makes a yaml class loader from a class constructor that takes no arguments.
    :param noarg_constructor: a constructor with no arguments
    """
    def _result(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode):
        obj = noarg_constructor(**loader.construct_mapping(node))
        return obj
    return _result

def _eval_constructor(eval_globals, eval_locals):
    def _result(loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode):
        s = loader.construct_scalar(node)
        if not isinstance(s, str):
            raise ValueError(f"Cannot eval anything other than a string. Found type {type(s)}: {s}")
        return Eval(eval_globals, eval_locals, loader.construct_yaml_str(node))
    return _result

def _get_loader(eval_globals, eval_locals, ctors_dict):
    """Add constructors to PyYAML loader."""
    loader = yaml.SafeLoader
    loader.add_constructor(u'tag:yaml.org,2002:map', object_constructor(Object))
    loader.add_constructor("!eval", _eval_constructor(eval_globals, eval_locals))

    for tag in ctors_dict:
        loader.add_constructor(tag, ctors_dict[tag])
    return loader



