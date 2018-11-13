# ilang - Inference Language
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 


# Print with 3 levels of verbosity 
verbose = 1


def set_verbose_high():
    """Print everything - DEBUG mode"""
    global verbose
    verbose = 2


def set_verbose_medium():
    """Print runtime information"""
    global verbose
    verbose = 1


def set_verbose_low():
    """Print only important messages"""
    global verbose
    verbose = 0


def set_verbose_no_printing():
    """Do not print messages at all"""
    global verbose
    verbose = -1


def get_verbose_level():
    return verbose


def print_debug(msg):
    """Use this for DEBUG Information"""
    if verbose >= 2:
        print(msg)


def print_runtime(msg):
    """Use this for messages useful at runtime"""
    if verbose >= 1:
        print(msg)


def print_important(msg):
    """Use this for important messages"""
    if verbose >= 0:
        print(msg)
