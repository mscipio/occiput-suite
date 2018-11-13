# SimpleWrap - Simple wrapper for C libraries based on ctypes
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 

from __future__ import absolute_import, print_function

from .. import c_python
from .. import c_python2

import unittest
import os
import random
import numpy


class TestSimpleWrap(unittest.TestCase):
    """Sequence of tests for SimpleWrap. """

    def setUp(self):
        # Load library
        (found, fullpath, path) = c_python.find_c_library("test_simplewrap_c", [c_python.localpath()])
        self.lib = c_python.load_c_library(fullpath)

    def test_int(self):
        """Wrap a simple function with integer parameters. """
        number = random.randint(1, 1e6)
        descriptor = [{'name': 'input', 'type': 'int', 'value': number},
                      {'name': 'output', 'type': 'int', 'value': None}, ]
        r = c_python.call_c_function(self.lib.echo, descriptor)
        self.assertTrue(r.output == number)

    def test_callback(self):
        """Wrap a simple function that calls a Python callback. """
        global A
        A = 0
        B = random.randint(1, 1e6)

        def callback(value):
            global A
            A = value

        descriptor = [{'name': 'function1', 'type': 'function', 'value': callback, 'arg_types': ['int']},
                      {'name': 'input', 'type': 'int', 'value': B}]
        r = c_python.call_c_function(self.lib.callback_test, descriptor)
        self.assertTrue(A == B)

    def test_numpy_array(self):
        """Wrap a simple function with numpy ndarray parameters. """
        pass

    def test_string(self):
        """Wrap a simple function with string parameter. """
        pass

    def test_cpython2(self):
        lib = c_python2.wrap_c_library("test_matrices_c", c_python2.localpath())
        m = lib.sum_matrices_f(numpy.ones((4, 4, 4)), numpy.ones((4, 4, 4)))
        self.assertTrue(m.sum() == 4 * 4 * 4 * 2)


if __name__ == '__main__':
    unittest.main()
