"""Main logging for david.
David uses Python's default logging system.
Each module has it's own logger, so you can control the verbosity of each
module to your needs. To change the overall log level, you can set log levels
at each part of the module hierarchy, including simply at the root, `david`:
```ipython
import logging
logging.getLogger('david').setLevel(logging.INFO)
```
Logger-Modes:
    DEBUG   : information typically of interest only when diagnosing problems.
    INFO    : confirmation that things are working as expected.
    WARN    : indication that something unexpected happened.
    ERROR   : software has not been able to perform some function.
    CRITICAL: program itself may be unable to continue running.
"""

import logging
import warnings
logging.basicConfig(level=logging.WARN)
del logging

# silence tensorflow warnings.
warnings.filterwarnings('ignore', category=DeprecationWarning, module='google')
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="DeprecationWarning")

# seeds random states for sources.
seed = 0

# david version - setup.py imports this value
__version__ = '0.0.2'
