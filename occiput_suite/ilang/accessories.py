from __future__ import absolute_import, print_function

__all__ = ['ProgressBar', 'LIGHT_BLUE', 'LIGHT_RED', 'BLUE', 'RED', 'GRAY', 'LIGHT_GRAY']

from IPython.display import display
from ipywidgets import FloatProgress

LIGHT_BLUE = 'rgb(200,228,246)'
BLUE = 'rgb(47,128,246)'
LIGHT_RED = 'rgb(246,228,200)'
RED = 'rgb(246,128,47)'
LIGHT_GRAY = 'rgb(246,246,246)'
GRAY = 'rgb(200,200,200)'


class ProgressBar():
    def __init__(self, height='20', width='500', background_color=LIGHT_BLUE,
                 foreground_color=BLUE, text_color=LIGHT_GRAY, title="Processing:"):
        self._percentage = 0.0
        self.visible = False
        if is_in_ipynb():
            self.set_display_mode("ipynb")
        else:
            self.set_display_mode("text")
        self._pb = FloatProgress(
            value=0.0,
            min=0,
            max=100.0,
            step=0.1,
            description=title,
            bar_style='info',
            orientation='horizontal'
        )

    def show(self):
        if self.mode == "ipynb":
            display(self._pb)
        self.visible = True

    def set_display_mode(self, mode="ipynb"):
        self.mode = mode

    def set_percentage(self, percentage):
        if not self.visible:
            self.show()
        if percentage < 0.0:
            percentage = 0.0
        if percentage > 100.0:
            percentage = 100.0
        percentage = int(percentage)
        self._percentage = percentage
        if self.mode == "ipynb":
            self._pb.value = self._percentage
        else:
            print("%2.1f / 100" % percentage)

    def get_percentage(self):
        return self._percentage


def is_in_ipynb():
    try:
        from IPython import get_ipython
        chk = str(get_ipython()).split(".")[1]
        if chk == 'zmqshell':
            return True
        else:
            return False
    except:
        return False