
'''
'''

# TODO fill in doc string

###########
# Imports #
###########

import os
import ctypes

from ..misc_utilities import *

# TODO make sure imports are used
# TODO make sure these imports are ordered in some way

#############################
# Globals & Initializations #
#############################

CURRENT_MODULE_PATH = os.path.dirname(os.path.realpath(__file__))

LIBTIBS_SO_LOCATION = os.path.abspath(os.path.join(CURRENT_MODULE_PATH, 'mlir/build/tibs-compiler/libtibs-compiler.so'))

assert os.path.isdir(LIBTIBS_SO_LOCATION), 'The Tibs compiler has not yet been compiled.'

LIBTIBS_SO = ctypes.CDLL(LIBTIBS_SO_LOCATION)

#################################
# C++ Method Type Declarations  #
#################################

# TODO we must declare types for all methods
LIBTIBS_SO.runAllPasses.restype = None
LIBTIBS_SO.runAllPasses.argtypes = []

############
# Compiler #
############

def compile():
    return LIBTIBS_SO.runAllPasses()

# TODO enable this
# __all__ = [
#     'TODO put something here'
# ]

##########
# Driver #
##########

if __name__ == '__main__':
    print('TODO add something here')
