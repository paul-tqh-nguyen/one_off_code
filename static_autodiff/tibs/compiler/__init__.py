
'''
'''

# TODO fill in doc string

###########
# Imports #
###########

import ctypes

# TODO make sure imports are used
# TODO make sure these imports are ordered in some way

###########
# Globals #
###########

LIBTIBS_SO = ctypes.CDLL('../build/tibs-main/libtibs.so')

# TODO automatically download the LLVM-project REPO if the directory doesn't exist
# TODO if it does exist, just do a pull

# TOODO set these environment variables
# export PATH=/home/pnguyen/code/one_off_code/mlir/llvm-project/build/bin/:$PATH
# export LD_LIBRARY_PATH=/home/pnguyen/miniconda3/envs/mlir/lib:$LD_LIBRARY_PATH
# export BUILD_DIR=/home/pnguyen/code/one_off_code/mlir/llvm-project/build
# export PREFIX=/home/pnguyen/code/one_off_code/mlir/llvm-project/build

# TODO do a compile of our MLIR compiler to make sure it is up-to-date (redundant compiles will be fast)
# cd /home/pnguyen/code/one_off_code/mlir/tibs/
# mkdir build
# cd build
# cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
# ninja


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
    print("TODO add something here")
