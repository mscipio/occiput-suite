
# Display Node - Python and Javascript plotting and data visualisation. 
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# 20 Oct 2013, Helsinki 
# Harvard University, Martinos Center for Biomedical Imaging
# Dec 2013, Boston, MA, USA 
# Harvard University, Martinos Center for Biomedical Imaging
# Dec 2014, Boston, MA, USA 

from __future__ import absolute_import

__all__ = ['DisplayNode','examples','set_ports']

from . import DisplayNodeProxy
from .DisplayNodeProxy import DisplayNode
from . import DisplayNodeServer
from . import examples


def set_ports(web,proxy):
    DisplayNodeServer.WEB_PORT = web
    DisplayNodeServer.PROXY_PORT = proxy
    DisplayNodeProxy.WEB_PORT = web
    DisplayNodeProxy.PROXY_PORT = proxy
