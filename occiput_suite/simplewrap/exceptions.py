
# SimpleWrap - Simple wrapper for C libraries based on ctypes 
# Stefano Pedemonte 
# Aalto University, School of Science, Helsinki 
# Oct. 2013, Helsinki 
# Harvard University, Martinos Center for Biomedical Imaging 
# Dec. 2013, Boston

__all__ = ['InstallationError', 'UnknownType', 'DescriptorError']

class InstallationError(Exception):
    def __init__(self, value):
        self.value = repr(value)
    def __str__(self):
        return "Installation Error: "+self.value

class UnknownType(Exception):
    def __init__(self, value):
        self.value = repr(value)
    def __str__(self):
        return "Installation Error: "+self.value

class DescriptorError(Exception):
    def __init__(self, value):
        self.value = repr(value)
    def __str__(self):
        return "Error in the descriptor of the C function parameters: "+self.value

