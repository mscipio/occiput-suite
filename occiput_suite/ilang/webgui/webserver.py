# ilang - Inference Language
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 


import sys, os, shutil

try:
    import thread
except:
    import _thread as thread
try:
    import BaseHTTPServer
except:
    import http.server as BaseHTTPServer
try:
    from SimpleHTTPServer import SimpleHTTPRequestHandler
except:
    from http.server import SimpleHTTPRequestHandler
try:
    import urllib2
except:
    import urllib.request as urllib2
from random import random
from time import sleep
from ..verbose import *

address = "127.0.0.1"
port = 8080


def serve(address, port):
    ServerClass = BaseHTTPServer.HTTPServer

    class HandlerClass(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            return

    Protocol = "HTTP/1.0"
    HandlerClass.protocol_version = Protocol
    httpd = ServerClass((address, port), HandlerClass)

    sa = httpd.socket.getsockname()
    # print "Serving HTTP on", sa[0], "port", sa[1], "..."
    httpd.serve_forever()


def run_webserver(background):
    # check if server is running, serving the files in the current working directory: 
    s = str(random())
    cwd = os.getcwd()
    fid = open(cwd + '/a_random_number.txt', 'w')
    fid.write(s)
    fid.close()
    server_ok = False
    try:
        data = str(urllib2.urlopen('http://%s:%s/a_random_number.txt' % (address, str(port))).read())
        if data == s:
            server_ok = True
    except urllib2.URLError:
        server_ok = False
    except:
        raise Exception("Unhandled exception")
    if not server_ok:
        if background:
            thread.start_new_thread(serve, ((address, port)))
        else:
            serve(address, port)
    else:
        print_debug("Display server already running. ")
        if not background:
            while (1):
                sleep(0.5)


if __name__ == "__main__":
    run_webserver(background=False)
