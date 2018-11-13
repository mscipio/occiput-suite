# ilang - Inference Language
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 

from __future__ import absolute_import, print_function

import unittest
from .. import Samplers
from .. import exceptions
from .. import verbose

verbose.set_verbose_low()


class TestSequenceMetropolisHastings(unittest.TestCase):
    """Sequence of tests for the Metropolis Hastings sampler"""

    def setUp(self):
        pass

    def test_sample(self):
        """.."""
        pass


class TestSequenceGradientDescent(unittest.TestCase):
    """Sequence of tests for the Gradient Descent sampler"""

    def setUp(self):
        pass

    def test_sample(self):
        """.."""
        pass


if __name__ == '__main__':
    unittest.main()
