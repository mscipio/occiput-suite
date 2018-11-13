# petlink - Decode and encode PETlink streams.
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 


from __future__ import absolute_import, print_function
from .. import petlink
import unittest
import os

class TestPetlink(unittest.TestCase):
    """Sequence of tests for petlink. """

    def setUp(self):
        pass

    def test_load_c_libraries(self):
        """Check if the C libraries have been built and linked without problems. """
        self.assertTrue(petlink.test_library_petlink32_c())

    def test_petlink32_info(self):
        """Test function petlink32_info"""
        pass


if __name__ == '__main__':
    unittest.main()
