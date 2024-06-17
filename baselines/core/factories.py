import functools
import glob
import importlib
import os
import sys
import time
from dataclasses import dataclass

from .factory_utils import initialize_mapper


@dataclass
class ProfilingData:
    execution_time: float


_BASELINES_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_BASELINES_PATH)


def _normalize_path_from_root(pth):
    if os.path.isabs(pth):
        raise ValueError(f"Absolute are not supported")
    return os.path.join(_BASELINES_PATH, pth)


def _fix_builtin_path_to_relative(file_path):
    relative_path = os.path.relpath(file_path, _BASELINES_PATH)  # Get the relative path
    module_path = os.path.splitext(relative_path)[0]  # Remove .py extension
    module_path = module_path.replace('..', '.').replace(os.sep, '.')  # Replace path separator with a dot
    return module_path


def _get_package_modules(package_root_path):
    if '.' in package_root_path:
        package_root_path = package_root_path.replace('.', os.sep)  # Replace dots with path separator
    package_root_path = os.path.realpath(package_root_path)  # Normalize the path
    py_files = glob.glob(package_root_path + '/**/*.py', recursive=True)

    modules = []
    for file_path in py_files:
        if os.path.basename(file_path) == '__init__.py':
            continue  # Skip __init__.py files
        module_path = _fix_builtin_path_to_relative(file_path)
        modules.append(module_path)

    return modules


_MAPPERS_MODULES = _get_package_modules(_normalize_path_from_root('mappers'))  # assumes working dir is root dir!
_AGGREGATORS_MODULES = [_fix_builtin_path_to_relative(_normalize_path_from_root('aggregators'))]
# currently the transforms used for aggreagtors sit within the same module
_TRANSFORMS_MODULES = [_fix_builtin_path_to_relative(_normalize_path_from_root('aggregators'))]


def _import_module_from_path(path):
    try:
        spec = importlib.util.spec_from_file_location("module_name", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        raise FileNotFoundError(f"Failed to import module {path}. Make sure the given custom function is in the "
                                f"correct path, given as a relative to the working directory ({os.getcwd()})")
    return module


def _create_custom_partial_function(func_name, module_name, arguments, prep_func=None):
    module = _import_module_from_path(module_name)
    function = getattr(module, func_name)
    if prep_func is not None:
        function = prep_func(function, **arguments)
    return lambda *args, **kwargs: function(*args, **kwargs, **arguments)


def _load_function(func, default_modules, prep_func=None, **kwargs):
    """
    Factory function for creating a partial function with the given arguments.
    :param func: The function to be called.
    :param kwargs: The arguments to be passed to the function.
    :return: A partial function with the given arguments.
    """
    arguments = {k: v for k, v in kwargs.items() if k not in ["module", "func"] and not k.startswith('_')}

    if '.' not in func:
        # a builtin function
        for module_name in default_modules:
            module_name = importlib.import_module(module_name, package=_BASELINES_PATH)
            if (function := getattr(module_name, func, None)) is not None:
                prep_func = (lambda f, **_: f) if prep_func is None else prep_func
                return functools.partial(prep_func(function, **arguments), **arguments)
        raise ValueError(f"Function {func} not found in modules {default_modules}")

    else:
        # a custom function from a module
        module_name, func = func.rsplit('.', 1)
        if module_name.startswith('.'):
            raise ValueError(f"Relative imports are not supported, please specify absolute path or "
                             f"use a higher level working directory")
        module_name = module_name.replace('.', os.path.sep) + '.py'
        if not os.path.isabs(module_name):
            module_name = os.path.join(os.getcwd(), module_name)
        module = _import_module_from_path(module_name)
        assert func in dir(module), f"Function {func} not found in module {module_name}"
        return _create_custom_partial_function(func, module_name, arguments, prep_func)


def _mapper_prep_func(func, **kwargs):
    orig_func = func
    init_func = initialize_mapper(func, **kwargs)
    if init_func == orig_func:
        return orig_func
    return lambda x, **_: init_func(x)


def get_mapper(func, _profile=False, _safe=False, **kwargs):
    """
    Factory function for creating a partial function with the given arguments.
    :param func: The function to be called.
    :param _profile: If True, the partial function will be wrapped to return a 2-tuple with the results,
        and the profiling (Pinfo instead of just the results.
    :param _safe: If True, the partial function will be wrapped to return return a raised error if one is thrown, rather than raise it
    :param kwargs: The arguments to be passed to the function.
    :return: A partial function with the given arguments.
    """
    partial_func = _load_function(func, _MAPPERS_MODULES, prep_func=_mapper_prep_func, **kwargs)
    if _safe:
        _func = partial_func

        def safe_partial_func(*args, **kwargs):
            try:
                return _func(*args, **kwargs)
            except Exception as e:
                return repr(e)

        partial_func = safe_partial_func
    if _profile:
        _func2 = partial_func

        def profiled_partial_func(*args, **kwargs):
            start = time.time()
            results = _func2(*args, **kwargs)
            end = time.time()
            return results, ProfilingData(end - start)

        partial_func = profiled_partial_func
    return partial_func


def get_aggregator(func, **kwargs):
    """
    Factory function for creating a partial function with the given arguments.
    :param func: The function to be called.
    :param kwargs: The arguments to be passed to the function.
    :return: A partial function with the given arguments.
    """
    return _load_function(func, _AGGREGATORS_MODULES, **kwargs)


def get_transform(func, **kwargs):
    """
    Factory function for creating a partial function with the given arguments, for a transform of a value to
    be sent to aggregation.
    :param func: The function to be called.
    :param kwargs: The arguments to be passed to the function.
    :return: A partial function with the given arguments.
    """
    return _load_function(func, _TRANSFORMS_MODULES, **kwargs)
