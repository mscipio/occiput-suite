# ilang - Inference Language
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 

from __future__ import absolute_import, print_function

__all__ = ['display_graph']

from .webserver import run_webserver, address, port
from webbrowser import open_new_tab
import json
from ..exceptions import *
import os, shutil


def setup_local_files():
    cwd = os.getcwd() + "/"
    webgui_dir = os.path.dirname(__file__) + "/"
    for f in os.listdir(webgui_dir):
        if f.endswith('.html') or f.endswith('.js') or f.endswith('.css'):
            shutil.copyfile(webgui_dir + f, cwd + f)


def display_image(image, background=False, new_tab=True):
    image.save(os.getcwd() + '/ilang_viewport.png')
    setup_local_files()
    if new_tab:
        open_new_tab('http://%s:%s/ilang_viewport.html' % (address, str(port)))
    run_webserver(background)


def image_ipython_notebook(image):
    display_image(image, background=True, new_tab=False)
    return '<iframe src=http://%s:%s/ilang_viewport.html width=800 height=500 frameborder=0></iframe>' % (
    address, str(port))


def display_graph(graph, background=False, new_tab=True):
    # if not a json string, see if the given object has a .export_json() method. 
    if not isinstance(graph, str):
        if hasattr(graph, 'export_json'):
            graph = graph.export_json()
        else:
            raise UnexpectedParameterType("The given object does not seem to be a graph. ")
    setup_local_files()
    # save json graph file
    s = 'var graph = ' + graph
    cwd = os.getcwd()
    fid = open(cwd + '/ilang_graph_data.js', 'w')
    fid.write(s)
    fid.close()

    if new_tab:
        open_new_tab('http://%s:%s/ilang_graph.html' % (address, str(port)))
    run_webserver(background)


def graph_ipython_notebook(graph):
    display_graph(graph, background=True, new_tab=False)
    return '<iframe src=http://%s:%s/ilang_graph.html width=850 height=450 frameborder=0></iframe>' % (
    address, str(port))


if __name__ == "__main__":
    graph = json.dumps({'nodes': [{"name": "A", "type": 0}, {"name": "B", "type": 1},
                                  {"name": "C", "type": 0}, {"name": "D", "type": 2}],
                        'links': [{"source": "A", "type": "t0", "target": "C"},
                                  {"source": "B", "type": "t1", "target": "C"},
                                  {"source": "D", "type": "t2", "target": "C"}]})
    display_graph(graph)
