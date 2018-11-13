# SimpleWrap - Simple wrapper for C libraries based on ctypes
# Stefano Pedemonte 
# Aalto University, School of Science, Helsinki 
# Oct 2013, Helsinki 

from __future__ import absolute_import, print_function

__all__ = ['call_c_function']
from ctypes import *
from numpy import *
import os, sys, inspect
from .exceptions import *
from .c_python import load_c_library, localpath


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
                if not d.has_key('size'):
                    raise DescriptorError("'string' with 'value'='None' must have 'size' property. ")
                arg = ' ' * size
            arg_c = c_char_p(arg)
        elif argtype == 'int':
            if arg is None:
                arg = 0
            arg = c_int(arg)
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
                if not d.has_key('size'):
                    raise DescriptorError("'array' with 'value'='None' must have 'size' property. ")
                if not d.has_key('dtype'):
                    raise DescriptorError("'array' with 'value'='None' must have 'dtype' property. ")
                arg = zeros(d['size'], dtype=d['dtype'])
            arg_c = arg.ctypes.data_as(POINTER(c_void_p))
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
