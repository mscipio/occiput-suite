# occiput 
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 
# Martinos Center for Biomedical Imaging, Harvard University/MGH, Boston
# Dec. 2013, Boston
# Martinos Center for Biomedical Imaging, Harvard University/MGH, Boston
# Jan. 2015, Boston
# Feb. 2015, Helsinki
# Nov. 2015, Boston 


# Here is the library of PET scanners! Access to listmode data is provided by an external Package. 
# Don't be put off, the mechanism is quite simple. 
from __future__ import absolute_import, print_function

__all__ = ["Generic", "Brain_PET", "Biograph_mMR", "Discovery_RX", "get_scanner_by_name"]

from numpy import linspace, pi, float32
from .PET_meshing import Michelogram

# Import scanner definitions 
try:
    from ....Occiput_Interface_Biograph_mMR import Biograph_mMR
except ImportWarning:
    Biograph_mMR = None

class Generic(object):
    def __init__(self):
        self.model = "Generic PET Scanner"
        self.manufacturer = "Occiput's immagination"
        self.version = "1.0"
        self.supports_listmode = False
        self.uses_meshing = False

        self.N_u = 128
        self.N_v = 128
        self.size_u = 2.0 * 128
        self.size_v = 2.0 * 128
        self.N_azimuthal = 5
        self.N_axial = 120
        self.angles_azimuthal = float32([-0.5, -0.25, 0.0, 0.25, 0.5])
        self.angles_axial = float32(linspace(0, pi - pi / self.N_axial, self.N_axial))

        self.scale_activity = 1.0

        self.activity_N_samples_projection_DEFAULT = 150
        self.activity_N_samples_backprojection_DEFAULT = 150
        self.activity_sample_step_projection_DEFAULT = 2.0
        self.activity_sample_step_backprojection_DEFAULT = 2.0
        self.activity_shape_DEFAULT = [128, 128, 128]
        self.activity_size_DEFAULT = float32([2.0, 2.0, 2.0]) * float32(self.activity_shape_DEFAULT)

        self.attenuation_N_samples_projection_DEFAULT = 150
        self.attenuation_N_samples_backprojection_DEFAULT = 150
        self.attenuation_sample_step_projection_DEFAULT = 2.0
        self.attenuation_sample_step_backprojection_DEFAULT = 2.0
        self.attenuation_shape_DEFAULT = [128, 128, 128]
        self.attenuation_size_DEFAULT = float32([2.0, 2.0, 2.0]) * float32(self.attenuation_shape_DEFAULT)

        self.listmode = None
        self.physiology = None

class Discovery_RX(object):
    def __init__(self):
        self.model = "Discovery_RX"
        self.manufacturer = "GE Healthcare"
        self.version = "n.d."
        self.supports_listmode = False
        self.uses_meshing = True

        self.michelogram = Michelogram(
            n_rings=24, span=3, max_ring_difference=1)

        self.N_u = 367
        self.N_v = self.michelogram.segments_sizes.max()  # 47 ???
        self.size_u = 1.90735 * self.N_u
        self.size_v = 3.34 * 47
        self.N_azimuthal = self.michelogram.n_segments  # 11
        self.N_axial = 315
        self.angles_azimuthal = float32([0.0])
            #[-0.482, -0.373, -0.259, -0.180, -0.105, 0.0, 0.105, 0.180,
        # 0.259, 0.373, 0.482])
        self.angles_axial = float32(
            linspace(
                0,
                pi -
                pi /
                self.N_axial,
                self.N_axial))

        self.scale_activity = 8.58e-05 #TODO: check this

        self.activity_N_samples_projection_DEFAULT = 300
        self.activity_N_samples_backprojection_DEFAULT = 300
        self.activity_sample_step_projection_DEFAULT = 2.0
        self.activity_sample_step_backprojection_DEFAULT = 2.0
        #self.activity_shape_DEFAULT = [367,367,47]
        #self.activity_size_DEFAULT = float32(
        #    [1.90735, 1.90735, 3.34]) * float32(self.activity_shape_DEFAULT)
        self.activity_shape_DEFAULT = [128,128,47]
        self.activity_size_DEFAULT = float32(
            [5.46875, 5.46875, 3.34]) * float32(self.activity_shape_DEFAULT)

        self.attenuation_N_samples_projection_DEFAULT = 300
        self.attenuation_N_samples_backprojection_DEFAULT = 300
        self.attenuation_sample_step_projection_DEFAULT = 2.0
        self.attenuation_sample_step_backprojection_DEFAULT = 2.0
        #self.attenuation_shape_DEFAULT = [344, 344, 127]
        #self.attenuation_size_DEFAULT = float32(
        #        [1.90735,1.90735,3.34]) * float32(
        # self.attenuation_shape_DEFAULT)
        self.attenuation_shape_DEFAULT = [128,128,47]
        self.attenuation_size_DEFAULT = float32(
                [5.46875,5.46875,3.34]) * float32(self.activity_shape_DEFAULT)

class Brain_PET(Generic):
    """
    Not implemented, imported as generic scanner
    """


def get_scanner_by_name(name):
    #print(name)
    if name == "Generic":
        return Generic
    elif name == "Brain_PET" or name == "BrainPET":
        return Brain_PET
    elif name == "Biograph_mMR" or name == "Siemens_Biograph_mMR" or name == "mMR" or name == "Siemens_mMR":
        return Biograph_mMR
    elif name == "Discovery_RX" or name == "GE_Discovery_RX" or name == \
            "GE_RX" or name == "RX" or name == "DRX":
        return Discovery_RX
    else:
        return None
