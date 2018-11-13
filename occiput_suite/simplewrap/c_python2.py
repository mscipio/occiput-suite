# SimpleWrap - Simple wrapper for C libraries based on ctypes
# Stefano Pedemonte 
# - Aalto University, School of Science, Helsinki 
# - Martinos Center, MGH, Harvard, Boston 
# 2013

from __future__ import absolute_import, print_function

__all__ = ['load_c_library', 'localpath', 'filepath', 'wrap_c_library', 'wrap_c_function', 'int32', 'uint32', 'uint16',
           'load_c_library', 'float32']

from ctypes import *
from numpy import *
import os, sys, inspect
from .exceptions import *
from json import loads
import platform

MAX_STR_LEN = 16384

if platform.system() == 'Linux':
    extensions = ['so', 'SO']
elif platform.system() == 'Darwin':
    extensions = ['dylib', 'DYLIB']
elif platform.system() == 'Windows':
    extensions = ['dll', 'DLL']
else:
    extensions = ['so', 'SO', 'dylib', 'DYLIB', 'dll', 'DLL']


def load_c_library(library_name, library_path):
    """Load the dynamic library with the given name (with path). """
    library_found = False
    for extension in extensions:
        for prefix in ['', 'lib', 'lib_']:
            filename = library_path + os.path.sep + prefix + library_name + "." + extension
            if os.path.exists(filename):
                library_found = True
                break
        if library_found:
            break
    if not library_found:
        raise InstallationError(
            "The library %s could not be found in %s. Please specify the correct location or add location to the system path." % (
            library_name, library_path))
    else:
        try:
            L = CDLL(filename)
        except OSError:
            raise InstallationError(
                "The library %s was found but could not be loaded. It is likely due to a linking error, missing libraries. " % library_name)
        else:
            return L


def string_buffer():
    s = create_string_buffer(MAX_STR_LEN)
    return s


def list_functions(c_library):
    c_library.swrap_list_functions.restype = c_int
    c_library.swrap_list_functions.argtypes = [c_char_p]
    list = string_buffer()
    if c_library.swrap_list_functions(list):
        raise ("The library does not support simplewrap. ")
    list = loads(list.value.replace("'", '"'))
    return list


def descriptor_c_library(library_name, library_path):
    """Call a C function in a dynamic library. The descriptor is a dictionary 
    that contains that parameters and describes how to use them. """
    # Load the library with ctypes
    L = load_c_library(library_name, library_path)

    # Obtain the list of functions to be wrapped 
    functions = list_functions(L)

    # Make wrapper for each function 
    lib_descriptor = []
    for f in functions:
        iface_name = 'swrap_' + str(f)
        iface = getattr(L, iface_name)
        iface.restype = c_int
        iface.argtypes = [c_char_p]
        fun_descriptor = string_buffer()
        iface(fun_descriptor)
        fun_descriptor = loads(fun_descriptor.value.replace("'", '"'))
        lib_descriptor.append({'name': f, 'params': fun_descriptor})
    return lib_descriptor


class Matrix(Structure):
    _fields_ = [("ndim", c_int),
                ("dim", POINTER(c_int)),
                ("data", c_void_p), ]


class TypeMap():
    def type_to_ctype(self, type):
        if type == 'matrix':
            return POINTER(Matrix)

    def arg_to_ctype(self, arg, type):
        # FIXME: check if ndarray, also possibly handle other matrix types here
        if type == 'matrix':
            return Matrix()
            # return Matrix(arg.ndim, 0, 0)


def wrap_c_function(library, fun_descriptor):
    class Function():
        def __init__(self, fun_descriptor):
            self._descriptor = fun_descriptor
            self._name = fun_descriptor['name']
            self._params = fun_descriptor['params']
            self._n_args_c = len(self._params)
            self._param_names = []
            self._param_types = []
            self._param_directions = []
            self._input_param_names = []
            self._input_param_types = []
            self._argtypes = []
            self._type_map = TypeMap()
            for param in self._params:
                name = param['name']
                type = param['type']
                direction = param['direction']
                self._param_names.append(name)
                self._param_types.append(type)
                self._param_directions.append(direction)
                if direction == 'in':
                    self._input_param_names.append(name)
                    self._input_param_types.append(type)
                self._argtypes.append(self._type_map.type_to_ctype(type))
            self._n_args_python = len(self._input_param_names)

            # define restype and argtypes 
            self._function = getattr(library, self._name)
            self._function.restype = c_int
            self._function.argtypes = self._argtypes

        def list_names(self):
            return self._input_param_names

        def list_types(self):
            return self._input_param_types

        def __check_input_parameters(self, args):
            # 1) check if the number of arguments is correct 
            # 2) check if the types of the arguments are correct 
            return 0

        def __call__(self, *args):
            self.__check_input_parameters(args)

            arguments = []
            results = []
            index_py = 0
            for index_c in range(self._n_args_c):
                type = self._param_types[index_c]
                direction = self._param_directions[index_c]
                if direction == 'in':
                    # Input: extract pointers and other info from Python objects 
                    arg = args[index_py]
                    arg = self._type_map.arg_to_ctype(arg, type)
                    index_py += 1
                else:
                    # Output: make empty object 
                    arg = self._type_map.arg_to_ctype(arg, type)
                arguments.append(arg)

            # call the C function 
            self._function(*arguments)
            return args[0] + args[1]

    return Function(fun_descriptor)


def wrap_c_library(library_name, library_path):
    L = load_c_library(library_name, library_path)
    lib_descriptor = descriptor_c_library(library_name, library_path)

    class Wrapper():
        pass

    wrapper = Wrapper()

    for fun_descriptor in lib_descriptor:
        fun = wrap_c_function(L, fun_descriptor)
        fun_name = fun_descriptor['name']
        setattr(wrapper, fun_name, fun)
    return wrapper


def localpath():
    return os.path.dirname(os.path.realpath(inspect.getfile(sys._getframe(1))))


def filepath(fullfilename):
    return os.path.dirname(os.path.realpath(fullfilename))
