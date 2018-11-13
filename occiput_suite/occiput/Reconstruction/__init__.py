__all__ = ['CT','MR','PET','SPECT']

from . import CT
from . import MR
from . import PET
from . import SPECT

from .PET import *
from .SPECT import *
from .CT import *
from .MR import *

__all__.extend(PET.__all__)
__all__.extend(SPECT.__all__)
__all__.extend(CT.__all__)
__all__.extend(MR.__all__)