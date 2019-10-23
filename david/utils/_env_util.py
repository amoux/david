from __future__ import absolute_import, division, print_function


def is_notebook_environment():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # IPython Notebook
        elif shell == 'Shell':
            return True   # Colaboratory Notebook
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other unknown type (?)
    except NameError:
        return False    # Probably standard Python interpreter
