import sys 
sys.path.insert(0, '/media/DATA/DOCUMENTI/GITHUB/__OCCIPUT_src/occiput-suite-DEV')
from occiput_suite.occiput.Reconstruction.PET.PET_meshing import Michelogram
from numpy import isscalar, linspace, int32, uint32, ones, zeros, pi, sqrt, float32, float64, where, ndarray, nan
from numpy import inf, asarray, concatenate, fromfile, maximum, exp, asfortranarray, fliplr, transpose 

from Biograph_mMR_Listmode import Biograph_mMR_Listmode
from Biograph_mMR_Physiology import Biograph_mMR_Physiology


__all__ = ['Biograph_mMR'] 



class Biograph_mMR(): 
    def __init__(self): 
        self.model        = "Biograph_mMR" 
        self.manufacturer = "Siemens" 
        self.version      = "2008" 
        self.uses_meshing      = True 

        self.michelogram = Michelogram(n_rings=64, span=11, max_ring_difference=60) 

        self.N_u    = 344
        self.N_v    = self.michelogram.segments_sizes.max()                     # 127
        self.size_u = 2.08626*344 
        self.size_v = 2.03125*127
        self.N_azimuthal                        = self.michelogram.n_segments   # 11
        self.N_axial                            = 252 
        self.angles_azimuthal = float32([-0.482, -0.373, -0.259, -0.180, -0.105,  0.0, 0.105,  0.180,  0.259,  0.373,  0.482]) 
        self.angles_axial                       = float32( linspace(0,pi-pi/self.N_axial,self.N_axial) )
        
        self.scale_activity                     = 0.4 * 8.58e-05
        
        self.activity_N_samples_projection_DEFAULT       = 300
        self.activity_N_samples_backprojection_DEFAULT   = 300
        self.activity_sample_step_projection_DEFAULT     = 2.0 
        self.activity_sample_step_backprojection_DEFAULT = 2.0
        self.activity_shape_DEFAULT                      = [344,344,127]
        self.activity_size_DEFAULT              = float32([2.08626, 2.08626, 2.03125])*float32(self.activity_shape_DEFAULT)
        
        self.attenuation_N_samples_projection_DEFAULT       = 300
        self.attenuation_N_samples_backprojection_DEFAULT   = 300
        self.attenuation_sample_step_projection_DEFAULT     = 2.0 
        self.attenuation_sample_step_backprojection_DEFAULT = 2.0
        self.attenuation_shape_DEFAULT                      = [344,344,127]
        self.attenuation_size_DEFAULT           = float32([2.08626, 2.08626, 2.03125])*float32(self.attenuation_shape_DEFAULT)

        self.listmode = Biograph_mMR_Listmode() 
        self.supports_listmode = True 

        self.physiology = Biograph_mMR_Physiology() 
        self.supports_physiology = True 






