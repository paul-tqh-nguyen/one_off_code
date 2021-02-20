
'''
'''

# TODO fill in doc string

###########
# Imports #
###########

import ctypes
import io
import os
import sys
import tempfile
from contextlib import contextmanager
from typing import Generator, Callable, Optional

from ..misc_utilities import *

# TODO make sure imports are used
# TODO make sure these imports are ordered in some way

#############################
# Globals & Initializations #
#############################

CURRENT_MODULE_PATH = os.path.dirname(os.path.realpath(__file__))

LIBTIBS_SO_LOCATION = os.path.abspath(os.path.join(CURRENT_MODULE_PATH, 'mlir/build/tibs-compiler/libtibs-compiler.so'))

assert os.path.isfile(LIBTIBS_SO_LOCATION), f'The TIBS compiler has not yet been compiled. It is expected to be found at {LIBTIBS_SO_LOCATION}.'

LIBTIBS_SO = ctypes.CDLL(LIBTIBS_SO_LOCATION)

##########################
# Module Generator Class #
##########################

LIBTIBS_SO.newModuleGenerator.argtypes = []
LIBTIBS_SO.newModuleGenerator.restype = ctypes.c_void_p

LIBTIBS_SO.dumpModule.argtypes = [ctypes.c_void_p]
LIBTIBS_SO.dumpModule.restype = None

# TODO get rid of this
LIBTIBS_SO.generateModule.argtypes = [ctypes.c_void_p]
LIBTIBS_SO.generateModule.restype = None

class ModuleGenerator:

    def __init__(self) -> None:
        self.value = LIBTIBS_SO.newModuleGenerator()
        return

    @trace
    def dump_module(self) -> str:
        result_container = []
        print(f"1 {repr(1)}")
        LIBTIBS_SO.generateModule(self.value) # TODO remove this
        print(f"2 {repr(2)}")
        LIBTIBS_SO.dumpModule(self.value) # TODO remove this
        print(f"3 {repr(3)}")
        module_string = only_one(result_container)
        return module_string

# TODO we must declare types for all methods
LIBTIBS_SO.runAllPasses.restype = None
LIBTIBS_SO.runAllPasses.argtypes = []

############
# Compiler #
############

def compile():
    result = 'DUMMY' # result = LIBTIBS_SO.runAllPasses()
    return result

# TODO enable this
# __all__ = [
#     'TODO put something here'
# ]

##########
# Driver #
##########

if __name__ == '__main__':
    print('TODO add something here')
