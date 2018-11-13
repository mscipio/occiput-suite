# Display Node - Python and Javascript plotting and data visualisation. 
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# 20 Oct 2013, Helsinki 
# Harvard University, Martinos Center for Biomedical Imaging
# Dec 2013, Boston, MA, USA 

from __future__ import absolute_import, print_function, division
from ..DisplayNodeProxy import DisplayNode
from PIL import Image 
from PIL import ImageDraw 
import math

__all__ = ["display_image_example","display_graph_example","display_geometry_example","ExampleObjectThatHasGraphicalRepresentation","polygon_fractal"]

def polygon_fractal(n=4,imgx=256,imgy=256,maxIt=6):
    image = Image.new("RGB", (imgx, imgy))
    m = n % 2
    p = float(n - m - 2.0) / 4.0 + 2.0
    q = p - 1.0
    af = 2.0 * math.pi / n
        
    for ky in range(imgy):
        for kx in range(imgx):
            x = kx * 2.0 / (imgx-1.0) - 1.0
            y = ky * 2.0 / (imgy-1.0) - 1.0
            for i in range(maxIt):      
                a = math.atan2(y, x)
                if a < 0:
                    a = 2.0 * math.pi - math.fabs(a)
                k = int(a / af) % n                
                x = x * p - math.cos(k * af + af / 2.0) * q
                y = y * p - math.sin(k * af + af / 2.0) * q
                if math.hypot(x, y) > 1.0:    
                    break

            r = i % 4 * 64
            g = i % 8 * 32
            b = i % 16 * 16
            image.putpixel((kx, ky), b * 65536 + g * 256 + r)
    return image



def display_graph_example(): 
    """Display a simple (probabilistic) graphical model in the browser. """
    d = DisplayNode() 
    graph = {'nodes':[{'name': 'A', 'type': 0}, {'name': 'B', 'type': 1}, {'name': 'C', 'type': 0}, {'name': 'D', 'type': 2}], 'links': [{'source': 'A', 'target': 'B', 'type': 't1'},
  {'source': 'A', 'target': 'C', 'type': 't1'},{'source': 'C', 'target': 'D', 'type': 't2'} ] }
    d.display_in_browser('graph',graph)



def display_image_example(): 
    """Display an image in the browser using Openseadragon.js """
    N        = 512
    N_tiles  = 6
    max_iter = 6
    d = DisplayNode() 
    print("Wait, generating awesome images .. ")
    image = Image.new("RGB",(N*N_tiles,N))
    for i in range(N_tiles): 
        print("%d / %d "%(i+1,N_tiles))
        tile = polygon_fractal(n=i+3,imgx=N,imgy=N,maxIt=max_iter)
        image.paste(tile,(i*N,0))
    d.display_in_browser('image',image)



def display_geometry_example(): 
    "Under development: webGL display based on three.js " 
    d = DisplayNode() 
    d.display_in_browser("three_cubes",{}) 




class ExampleObjectThatHasGraphicalRepresentation(): 
    """This example shows how to define an object that exposes a hook for Ipython Notebook display. 
    Just create an instance of this class in an Ipython Notebook:  
    from DisplayNode.examples import ExampleObjectThatHasGraphicalRepresentation
    ExampleObjectThatHasGraphicalRepresentation() 
    """
    def make_graph_of_self(self): 
        return {'nodes':[{'name': 'A', 'type': 0}, {'name': 'B', 'type': 1}, {'name': 'C', 'type': 0}, {'name': 'D', 'type': 2}], 'links': [{'source': 'A', 'target': 'B', 'type': 't1'},
  {'source': 'A', 'target': 'C', 'type': 't1'},{'source': 'C', 'target': 'D', 'type': 't2'} ] }
  
    def _repr_html_(self): 
        d = DisplayNode()
        d.display('graph', self.make_graph_of_self() ) 
        return d._repr_html_() 
        



if __name__ == "__main__":
    display_graph_example()
    display_image_example()
    display_geometry_example()
    print("Open examples.py to find out how to integrate DisplayNode with Ipython Notebook (see the definition of ExampleObjectThatHasGraphicalRepresentation). ")
