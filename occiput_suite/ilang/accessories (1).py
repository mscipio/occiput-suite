from __future__ import absolute_import, print_function

__all__ = ['ProgressBar', 'LIGHT_BLUE', 'LIGHT_RED', 'BLUE', 'RED', 'GRAY', 'LIGHT_GRAY']

import uuid
from IPython.display import HTML, Javascript, display

LIGHT_BLUE = 'rgb(200,228,246)'
BLUE = 'rgb(47,128,246)'
LIGHT_RED = 'rgb(246,228,200)'
RED = 'rgb(246,128,47)'
LIGHT_GRAY = 'rgb(246,246,246)'
GRAY = 'rgb(200,200,200)'


class ProgressBar():
    def __init__(self, height='6px', width='100%%', background_color=LIGHT_BLUE, foreground_color=BLUE):
        self.divid = str(uuid.uuid4())
        self.pb = HTML(
            """
            <div style="border: 1px solid white; width:%s; height:%s; background-color:%s">
                <div id="%s" style="background-color:%s; width:0%%; height:%s"> </div>
            </div> 
            """ % (width, height, background_color, self.divid, foreground_color, height))
        self.visible = False
        self._previous = -1.0

    def show(self):
        display(self.pb)
        self.visible = True

    def set_percentage(self, percentage):
        if not self.visible:
            self.show()
        if percentage < 1:
            percentage = 1
        if percentage > 100:
            percentage = 100
        percentage = int(percentage)
        if percentage != self._previous:
            self._previous = percentage
            display(Javascript("$('div#%s').width('%i%%')" % (self.divid, percentage)))
