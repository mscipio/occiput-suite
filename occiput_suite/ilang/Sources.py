
# ilang - Inference Language 
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 

from __future__ import absolute_import, print_function

__all__ = ['FileSource']

from .Models import *
from .exceptions import *
from .verbose import *
import nibabel


class FileSource(object):
    def __init__(self,filename=None):
        self.nifti_image=None
        if filename != None: 
            self.load(filename)

    def load(self,file): 
        self.nifti_image = nibabel.load(file) 

    def get_data(self): 
        return self.nifti_image.get_data() 
 
    def get_affine(self):
        return self.nifti_image.get_affine()
        
    def get_info(self): 
        return self.nifti_image.get_info() 

    def save(self,file):
        return self.nifti_image.save(file)  
 
    def get_data_dtype(self):
        return self.nifti_image.get_data_type() 
         
    def get_filename(self):
        return self.nifti_image.get_filename() 
         
    def get_header(self):
        return self.nifti_image.get_header()  
        
    def get_qform(self):
        return self.nifti_image.get_qform()  
        
    def get_sform(self):
        return self.nifti_image.get_sform()  
        
    def get_shape(self):
        return self.nifti_image.get_shape()  

    def set_filename(self,filename):
        return self.nifti_image.set_filename(filename) 
        
    def set_qform(self,q):
        return self.nifti_image.set_qform(q)
        
    def set_sform(self,s):
        return self.nifti_image.set_sform(s)
        
    def is_instance_to_filename(self):
        return self.nifti_image.is_instance_to_filename()
        
    # display
    def show(self): 
        from math import floor 
        from numpy import zeros
        from PIL import Image
         #from webgui.QuickDisplay import display_viewport3
        from .webgui.QuickDisplay import display_image as display_viewport3
        imx = self.get_data()[floor(self.get_shape()[0]/2),:,:]
        imy = self.get_data()[:,floor(self.get_shape()[1]/2),:]
        if len(self.get_shape())>=3: #FIXME: handle 4D, 5D, ..
            imz = self.get_data()[:,:,floor(self.get_shape()[2]/2)]
        else: 
            imz = zeros((128,128))
        imx = Image.fromarray(imx).rotate(90)
        imy = Image.fromarray(imy).rotate(90)
        imz = Image.fromarray(imz).rotate(90)
        #return display_viewport3(imx, imy, imz, background=True, new_tab=True)
        display_viewport3(imx, background=True, new_tab=True)
        display_viewport3(imy, background=True, new_tab=True)
        return display_viewport3(imz, background=True, new_tab=True)
        
    def _repr_html_(self): 
        from math import floor 
        from numpy import zeros
        from PIL import Image
        #from webgui.QuickDisplay import display_viewport3_ipython_notebook
        from .webgui.QuickDisplay import display_image as display_viewport3_ipython_notebook
        imx = self.get_data()[floor(self.get_shape()[0]/2),:,:]
        imy = self.get_data()[:,floor(self.get_shape()[1]/2),:]
        if len(self.get_shape())>=3: #FIXME: handle 4D, 5D, ..
            imz = self.get_data()[:,:,floor(self.get_shape()[2]/2)]
        else: 
            imz = zeros((128,128))
        imx = Image.fromarray(imx).rotate(90)
        imy = Image.fromarray(imy).rotate(90)
        imz = Image.fromarray(imz).rotate(90)
        #return display_viewport3_ipython_notebook(imx,imy,imz)
        display_viewport3_ipython_notebook(imx)
        display_viewport3_ipython_notebook(imy)
        return display_viewport3_ipython_notebook(imz)




