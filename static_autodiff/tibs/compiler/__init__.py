
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

# LIBTIBS_SO_LOCATION = os.path.abspath('../build/tibs-main/libtibs.so') # TODO update this

# LIBTIBS_SO = ctypes.CDLL(LIBTIBS_SO_LOCATION)

CURRENT_MODULE_PATH = os.path.dirname(os.path.realpath(__file__))

LLVM_PROJECT_GIT_URL = 'https://github.com/llvm/llvm-project.git'

LLVM_PROJECT_REPO_LOCATION = os.path.abspath(os.path.join(CURRENT_MODULE_PATH, 'llvm-project'))

@trace
def verify_local_llvm() -> None:
    # TODO use the MLILR conda-forge package https://anaconda.org/conda-forge/mlir
    if not os.path.isdir(LLVM_PROJECT_REPO_LOCATION):
        print(f'Cloning local LLVM repo into {LLVM_PROJECT_REPO_LOCATION}.')
        command = ' ; '.join([
            f'pushd {CURRENT_MODULE_PATH}', 
            f'git clone {LLVM_PROJECT_GIT_URL}', 
            'popd', 
        ])
        clone_stdout, clone_stderr, return_code= shell(command)
        print('LLVM Repo Clone Command:')
        print(command)
        print('LLVM Repo Clone Output:')
        print(clone_stdout)
        print('LLVM Repo Clone Error Messages:')
        print(clone_stderr)
        if return_code != 0:
            raise RuntimeError(f'Could not clone LLVM.')
    else:
        print(f'Updating local LLVM repo at {LLVM_PROJECT_REPO_LOCATION}.')
        command = ' ; '.join([
            f'pushd {LLVM_PROJECT_REPO_LOCATION}', 
            'git pull',
            'popd'
        ])
        pull_stdout, pull_stderr, return_code = shell(command)
        print('LLVM Repo Pull Command:')
        print(command)
        print('LLVM Repo Pull Output:')
        print(pull_stdout)
        print('LLVM Repo Pull Error Messages:')
        print(pull_stderr)
        if return_code != 0:
            raise RuntimeError(f'Could not update local LLVM repo.')
    return

@trace
def set_environment_variables() -> None:
    # Verify BUILD_DIR and PREFIX environment variables
    llvm_repo_build_dir = os.path.join(LLVM_PROJECT_REPO_LOCATION, 'build')
    os.environ['BUILD_DIR'] = llvm_repo_build_dir
    os.environ['PREFIX'] = llvm_repo_build_dir
    # Verify PATH environment variable
    llvm_repo_bin_dir = os.path.join(llvm_repo_build_dir, 'bin')
    assert os.path.isdir(llvm_repo_bin_dir)
    if llvm_repo_bin_dir not in os.environ['PATH'].split(':'):
        os.environ['PATH'] = os.environ['PATH'] + ':' + llvm_repo_bin_dir
    # Verify LD_LIBRARY_PATH environment variable
    if 'CONDA_PREFIX' not in os.environ:
        raise NotImplementedError(f'Compilation not currently supported outside of a conda environment.')
    conda_prefix = os.environ['CONDA_PREFIX']
    conda_lib_dir = os.path.join(conda_prefix, 'lib')
    if conda_lib_dir not in os.environ['LD_LIBRARY_PATH'].split(':'):
        os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ':' + conda_lib_dir
    return

@trace
def compile_local_llvm() -> None:
    print(f'Compiling local LLVM repo at {LLVM_PROJECT_REPO_LOCATION}.')
    llvm_repo_build_dir = os.path.join(LLVM_PROJECT_REPO_LOCATION, "build")
    if not os.path.isdir(llvm_repo_build_dir):
        os.makedirs(llvm_repo_build_dir)
    command = ' ; '.join([
        f'pushd {llvm_repo_build_dir}',
        f'cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON',
        f'cmake --build . --target check-mlir',
        f'popd',
    ])
    llvm_compile_stdout, llvm_compile_stderr, return_code = shell(command)
    print('LLVM Repo Compile Command:')
    print(command)
    print('LLVM Repo Compile Output:')
    print(llvm_compile_stdout)
    print('LLVM Repo Compile Error Messages:')
    print(llvm_compile_stderr)
    if return_code != 0:
        raise RuntimeError(f'Could not compile local LLVM repo.')
    return

@trace
def compile_tibs_compiler() -> None:
    print(f'Compiling the TIBS compiler.')
    build_dir = os.path.join(CURRENT_MODULE_PATH, 'mlir/build')
    if not os.path.isdir(build_dir):
        os.makedirs(build_dir)
    command = ' ; '.join([
        f'pushd {build_dir}',
        'cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit',
        'ninja',
        'popd'
    ])
    compile_stdout, compile_stderr, return_code = shell(command)
    print('Compilation of TIBS Compiler Command:')
    print(command)
    print('Compilation of TIBS Compiler Output:')
    print(compile_stdout)
    print('Compilation of TIBS Compiler Error Messages:')
    print(compile_stderr)
    if return_code != 0:
        raise RuntimeError(f'Could not compile TIBS compiler.')
    return 

verify_local_llvm()
set_environment_variables()
compile_local_llvm()
compile_tibs_compiler()

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
