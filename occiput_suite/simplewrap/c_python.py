# SimpleWrap - Simple wrapper for C libraries based on ctypes
# Stefano Pedemonte 
# Aalto University, School of Science, Helsinki 
# Oct. 2013, Helsinki 
# Harvard University, Martinos Center for Biomedical Imaging 
# Dec. 2013, Boston 

from __future__ import absolute_import, print_function

__all__ = ['exists_c_library', 'find_c_library', 'load_c_library', 'call_c_function', 'export_dl_library_path',
           'LibraryNotFound', 'FOUND', 'NOT_FOUND', 'FOUND_NOT_LOADABLE', 'localpath', 'filepath', 'int32', 'uint32',
           'uint16', 'float32']
from ctypes import *
from numpy import *
import os, sys, inspect
from .exceptions import *
import platform
import copy

prefixes = ['', 'lib', 'lib_']

if platform.system() == 'Linux':
    extensions = ['so', 'SO']
elif platform.system() == 'Darwin':
    extensions = ['dylib', 'DYLIB', 'so', 'SO']
elif platform.system() == 'Windows':
    extensions = ['dll', 'DLL']
else:
    extensions = ['so', 'SO', 'dylib', 'DYLIB', 'dll', 'DLL']


class LibraryNotFound(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return "Linrary not found: %s" % self.msg


def exists_c_library(lib_name, library_path):
    """Returns True if a library with the given name exists in the specified path. """
    library_found = False
    fullpath = None
    for extension in extensions:
        for prefix in prefixes:
            fullpath = library_path + os.path.sep + prefix + lib_name + "." + extension
            if os.path.exists(str(fullpath)):
                library_found = True
                break
        if library_found:
            break
    return library_found, fullpath


def isloadable_c_library(fullpath):
    try:
        L = CDLL(fullpath)
    except:
        return False
    else:
        return True


NOT_FOUND = 0
FOUND = 1
FOUND_NOT_LOADABLE = 2


def find_c_library(library_name, paths=('./')):
    for path in paths:
        (exists, fullpath) = exists_c_library(library_name, path)
        if exists:
            if isloadable_c_library(fullpath):
                return FOUND, fullpath, path
            else:
                return FOUND_NOT_LOADABLE, fullpath, path
    #return NOT_FOUND, None, None


def load_c_library(fullpath):
    """Load the dynamic library with the given name (with path). """
    if not os.path.exists(str(fullpath)):
        raise InstallationError(
            "The library %s could not be found. Please specify the correct location of add location to the system path." % fullpath)
    else:
        try:
            L = CDLL(fullpath)
        except OSError:
            raise InstallationError(
                "The library %s was found but could not be loaded. It is likely due to a linking error, missing libraries. " % fullpath)
        else:
            return L


def export_dl_library_path(path):
    if platform.system() == 'Linux':
        if "LD_LIBRARY_PATH" in os.environ.keys():
            os.environ["LD_LIBRARY_PATH"] += os.pathsep + path
        else:
            os.environ["LD_LIBRARY_PATH"] = path
    elif platform.system() == 'Darwin':
        if "DYLD_LIBRARY_PATH" in os.environ.keys():
            os.environ["DYLD_LIBRARY_PATH"] += os.pathsep + path
        else:
            os.environ["DYLD_LIBRARY_PATH"] = path
    elif platform.system() == 'Windows':
        if "PATH" in os.environ.keys():
            os.environ["PATH"] += os.pathsep + path
        else:
            os.environ["PATH"] = path
    else:
        if "LD_LIBRARY_PATH" in os.environ.keys():
            os.environ["LD_LIBRARY_PATH"] += os.pathsep + path
        else:
            os.environ["LD_LIBRARY_PATH"] = path
        if "PATH" in os.environ.keys():
            os.environ["PATH"] += os.pathsep + path
        else:
            os.environ["PATH"] = path


def call_c_function(c_function, descriptor):
    """Call a C function in a dynamic library. The descriptor is a dictionary 
    that contains that parameters and describes how to use them. """
    # set the return type
    c_function.restype = c_int
    # parse the descriptor, determine the types and instantiate variables if their value is not given 
    argtypes_c = []
    args_c = []
    args = []
    for d in descriptor:
        if d['name'] == 'status':
            DescriptorError("variable name 'status' is reserved. ")
        argtype = d['type']
        arg = d['value']
        if argtype == 'string':
            if arg is None:
                try:
                    if not d.has_key('size'):
                        raise DescriptorError("'string' with 'value'='None' must have 'size' property. ")
                except:
                    if 'size' not in d:
                        raise DescriptorError("'string' with 'value'='None' must have 'size' property. ")
                arg = ' ' * size
            try:
                arg_c = c_char_p(arg)
            except:
                arg_c = c_char_p(arg.encode('utf-8'))
        elif argtype == 'int':
            if arg is None:
                arg = 0
            arg = c_int32(arg)
            arg_c = pointer(arg)
        elif argtype == 'uint':
            if arg is None:
                arg = 0
            arg = c_uint32(arg)
            arg_c = pointer(arg)
        elif argtype == 'long':
            if arg is None:
                arg = 0
            arg = c_longlong(arg)
            arg_c = pointer(arg)
        elif argtype == 'float':
            if arg is None:
                arg = 0.0
            arg = c_float(arg)
            arg_c = pointer(arg)
        elif argtype == 'array':
            if arg is None:
                try:
                    if not d.has_key('size'):
                        raise DescriptorError("'array' with 'value'='None' must have 'size' property. ")
                except:
                    if 'size' not in d:
                        raise DescriptorError("'array' with 'value'='None' must have 'size' property. ")

                try:
                    if not d.has_key('dtype'):
                        raise DescriptorError("'array' with 'value'='None' must have 'dtype' property. ")
                except:
                    if 'dtype' not in d:
                        raise DescriptorError("'array' with 'value'='None' must have 'dtype' property. ")

                try:
                    if d.has_key('order'):
                        order = d['order']
                        if not order in ["C", "A", "F", None]:
                            raise DescriptorError(
                                "'order' property of type 'array' must be 'C' (C array order),'F' (Fortran array order), 'A' (any order, let numpy decide) or None (any order, let numpy decide) ")
                    else:
                        order = None
                except:
                    if 'order' in d:
                        order = d['order']
                        if not order in ["C", "A", "F", None]:
                            raise DescriptorError(
                                "'order' property of type 'array' must be 'C' (C array order),'F' (Fortran array order), 'A' (any order, let numpy decide) or None (any order, let numpy decide) ")

                    else:
                        order = None
                arg = zeros(d['size'], dtype=d['dtype'], order=order)
                # If variable is given (not None) and dtype is specified, change the dtype of the given array if not consistent
            # This also converts lists and tuples to numpy arrays if the given variable is not a numpy array. 
            else:
                try:
                    if d.has_key('dtype'):
                        dtype = d['dtype']
                        arg = dtype(arg)
                    if d.has_key('order'):
                        arg = asarray(arg, order=d['order'])
                except:
                    if 'dtype' in d:
                        dtype = d['dtype']
                        arg = dtype(arg)
                    if 'order' in d:
                        arg = asarray(arg, order=d['order'])
            arg_c = arg.ctypes.data_as(POINTER(c_void_p))
        elif argtype == 'function':
            if arg is None:
                raise DescriptorError("For 'function' type, 'value' must be a function. ")
            try:
                if not d.has_key('arg_types'):
                    raise DescriptorError("For 'function' type, 'arg_types' must be specified. ")
            except:
                if 'arg_types' not in d:
                    raise DescriptorError("For 'function' type, 'arg_types' must be specified. ")
            arg_types = []
            for t in d['arg_types']:
                if t == 'int':
                    arg_types.append(c_int32)
                if t == 'uint':
                    arg_types.append(c_uint32)
                if t == 'long':
                    arg_types.append(c_longlong)
                if t == 'float':
                    arg_types.append(c_float)
            funcCB = CFUNCTYPE(None, *arg_types)
            arg_c = funcCB(arg)
        else:
            raise UnknownType("Type %s is not supported. " % str(argtype))
        argtype_c = type(arg_c)
        argtypes_c.append(argtype_c)
        args_c.append(arg_c)
        args.append(arg)
        # set the arguments types
    c_function.argtypes = argtypes_c
    # call the function 
    status = c_function(*args_c)
    # cast back to Python types 
    for i in range(len(descriptor)):
        argtype = descriptor[i]['type']
        if argtype in ['int', 'uint', 'float', 'long']:
            args[i] = args[i].value
        # swap axes of array if requested
        if descriptor[i]['type'] == 'array':
            try:
                if descriptor[i].has_key('swapaxes'):
                    # 1) reshape
                    shape = args[i].shape
                    shape2 = list(shape)
                    shape = copy.copy(shape2)
                    shape[descriptor[i]['swapaxes'][0]] = shape2[descriptor[i]['swapaxes'][1]]
                    shape[descriptor[i]['swapaxes'][1]] = shape2[descriptor[i]['swapaxes'][0]]
                    args[i] = args[i].reshape(shape)
                    # 2) swap axes
                    args[i] = args[i].swapaxes(descriptor[i]['swapaxes'][0], descriptor[i]['swapaxes'][1])
                    # Assemble wrapper object
            except:
                if 'swapaxes' in descriptor[i]:
                    # 1) reshape
                    shape = args[i].shape
                    shape2 = list(shape)
                    shape = copy.copy(shape2)
                    shape[descriptor[i]['swapaxes'][0]] = shape2[descriptor[i]['swapaxes'][1]]
                    shape[descriptor[i]['swapaxes'][1]] = shape2[descriptor[i]['swapaxes'][0]]
                    args[i] = args[i].reshape(shape)
                    # 2) swap axes
                    args[i] = args[i].swapaxes(descriptor[i]['swapaxes'][0], descriptor[i]['swapaxes'][1])
                    # Assemble wrapper object

    class CallResult():
        pass

    result = CallResult()
    dictionary = {}
    for index in range(len(descriptor)):
        name = descriptor[index]['name']
        arg = args[index]
        setattr(result, name, arg)
        dictionary[name] = arg
    setattr(result, 'status', status)
    setattr(result, 'values', args)
    setattr(result, 'dictionary', dictionary)
    return result


def localpath():
    path = os.path.dirname(os.path.realpath(inspect.getfile(sys._getframe(1))))
    return path


def filepath(fullfilename):
    return os.path.dirname(os.path.realpath(fullfilename))
