"""def is_in_ipynb():
    try:
        from IPython import get_ipython
        cfg = get_ipython().config
        if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
        else:
            return False
    except:
        return False"""

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