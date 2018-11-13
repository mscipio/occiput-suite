# Display Node - Python and Javascript plotting and data visualisation.
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# 20 Oct 2013, Helsinki 

from __future__ import absolute_import, print_function

from .DisplayNodeServer import DisplayNodeServer
from .DisplayNodeServer import PROXY_ADDRESS, PROXY_PORT, WEB_ADDRESS, WEB_PORT

try:
    from xmlrpclib import Server, Binary
except ImportError:
    from xmlrpc.client import Server, Binary
import webbrowser
import sys
import socket

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from PIL import Image

try:
    import Image as IM
except:
    IM = Image

if 0:
    from multiprocessing import Process
    import signal

    USE_MULTIPROCESSING = True
else:
    try:
        import thread
    except ImportError:
        import _thread as thread

    USE_MULTIPROCESSING = False

WIDTH = '900'  # '900' #FIXME: obtain display specific width and height form the server
HEIGHT = '450'

socket.setdefaulttimeout(60)


class ParameterError(Exception):
    def __init__(self, msg):
        self.msg = str(msg)

    def __str__(self):
        return "Unexpected parameter: %s" % (self.msg)


def is_an_image(im):
    is_image = False
    if isinstance(im, Image.Image) or isinstance(im, IM.Image):
        is_image = True
    return is_image


class DisplayNode:
    def __init__(self, proxy_address=(PROXY_ADDRESS, PROXY_PORT), web_address=(WEB_ADDRESS, WEB_PORT)):
        self._proxy = Server('http://%s:%s' % proxy_address, allow_none=True)
        #        print "Proxy: ",proxy_address
        self.start_server(proxy_address, web_address)
        self.data = None
        self.type = None
        self.url = None
        self.width = 0
        self.height = 0
        # FIXME: use introspection to define the methods (for autocompletion) 

    def start_server(self, proxy_address, web_address):
        if not self.is_server_responding():
            self._server = DisplayNodeServer(proxy_address, web_address)
            if USE_MULTIPROCESSING:
                # print "Multiprocessing version! "
                self._server_process = Process(target=self.__run_server_forever, args=())
                self._server_process.start()
            else:
                thread.start_new_thread(self.__run_server_forever, ())

    def __run_server_forever(self):
        if USE_MULTIPROCESSING:
            signal.signal(signal.SIGINT, self.__signal_handler_interrupt)
        self._server.serve_forever()

    def __signal_handler_interrupt(self, signal, frame):
        print('Shutting down DisplayNode server. ')
        sys.exit(0)

    def is_server_responding(self):
        socket.setdefaulttimeout(2)
        try:
            alive = self._proxy.is_alive(1)
        except:
            alive = False
        socket.setdefaulttimeout(60)
        return alive

    def display(self, content_type, data={}, open_browser=False, new_tab=True, autoraise=False):
        # if image: send png content
        if content_type == "image":
            buf = StringIO()
            data.convert("RGB").save(buf, format="png")
            data = Binary(buf.getvalue())
            buf.close()
        # if list of images: send list of png content
        if content_type == "tipix":
            if not type(data) == list:
                raise ParameterError("Parameter for 'tipix' must be a list of images.")
                # 1D array of images:
            if is_an_image(data[0]):
                for i in range(len(data)):
                    if not is_an_image(data[i]):
                        raise ParameterError("Parameter for 'tipix' must be a list of images.")
                    buf = StringIO()
                    data[i].convert("RGB").save(buf, format="png")
                    data[i] = Binary(buf.getvalue())
                    buf.close()
            elif type(data[0]) == list:
                for i in range(len(data)):
                    for j in range(len(data[i])):
                        if not is_an_image(data[i][j]):
                            raise ParameterError("Parameter for 'tipix' must be a list of images.")
                        buf = StringIO()
                        data[i][j].convert("RGB").save(buf, format="png")
                        data[i][j] = Binary(buf.getvalue())
                        buf.close()
        url = self._proxy.display({'type': content_type, 'data': data})
        if open_browser:
            if new_tab:
                webbrowser.open_new_tab(url)
            else:
                webbrowser.open(url, autoraise=autoraise)
        self.data = data
        self.type = content_type
        self.url = url
        self.width = WIDTH  # FIXME: obtain width and height from the server
        self.height = HEIGHT
        return self

    def display_in_browser(self, content_type, data={}, new_tab=False, autoraise=False):
        self.display(content_type, data, open_browser=True, new_tab=new_tab, autoraise=autoraise)
        return None

    def _repr_html_(self):
        # This method is for ipython notebook integration through Rich Display
        return '<iframe src=%s width=%s height=%s frameborder=0></iframe>' % (self.url, self.width, self.height)
