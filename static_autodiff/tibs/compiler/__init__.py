
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

LLVM_PROJECT_GIT_URL = 'https://github.com/llvm/llvm-project.git'

LLVM_PROJECT_GIT_COMMIT_HASH = '805d59593f50a5c5050c0fc5cb9fbe02cd751511'

LLVM_PROJECT_REPO_LOCATION = os.path.abspath(os.path.join(CURRENT_MODULE_PATH, 'llvm-project'))

@trace
def clone_local_llvm() -> None:
    # TODO use the MLILR conda-forge package https://anaconda.org/conda-forge/mlir
    if not os.path.isdir(LLVM_PROJECT_REPO_LOCATION):
        print(f'Cloning local LLVM repo into {LLVM_PROJECT_REPO_LOCATION}.')
        command = ' && '.join([
            f'pushd {CURRENT_MODULE_PATH}', 
            f'git clone {LLVM_PROJECT_GIT_URL}', 
            'popd', 
        ])
        print('LLVM Repo Clone Command:')
        print(command)
        clone_stdout, clone_stderr, clone_return_code= shell(command)
        print('LLVM Repo Clone Output:')
        print(clone_stdout)
        print('LLVM Repo Clone Error Messages:')
        print(clone_stderr)
        if clone_return_code != 0:
            raise RuntimeError(f'Could not clone LLVM.')
        
        print(f'Resetting local LLVM repo at {LLVM_PROJECT_REPO_LOCATION} to commit {LLVM_PROJECT_GIT_COMMIT_HASH}.')
        command = ' && '.join([
            f'pushd {LLVM_PROJECT_REPO_LOCATION}', 
            f'git reset --hard  {LLVM_PROJECT_GIT_COMMIT_HASH}',
            'popd'
        ])
        print('LLVM Repo Reset Command:')
        print(command)
        reset_stdout, reset_stderr, reset_return_code = shell(command)
        print('LLVM Repo Reset Output:')
        print(reset_stdout)
        print('LLVM Repo Reset Error Messages:')
        print(reset_stderr)
        if reset_return_code != 0:
            raise RuntimeError(f'Could not reset local LLVM repo to correct commit at {LLVM_PROJECT_GIT_COMMIT_HASH}. Try deleting {LLVM_PROJECT_REPO_LOCATION}.')
    return

@trace # TODO move this to be within compile_local_llvm
def set_environment_variables() -> None:
    # Verify PATH environment variable
    llvm_repo_build_dir = os.path.join(LLVM_PROJECT_REPO_LOCATION, "build")
    llvm_repo_bin_dir = os.path.join(llvm_repo_build_dir, 'bin')
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
    llvm_repo_build_dir = os.path.join(LLVM_PROJECT_REPO_LOCATION, 'build')
    llvm_repo_bin_dir = os.path.join(llvm_repo_build_dir, 'bin')
    if not os.path.isdir(llvm_repo_build_dir):
        os.makedirs(llvm_repo_build_dir)
    print(f"os.environ['PATH'] {repr(os.environ['PATH'])}") # TODO Remove this
    print(f"os.environ['LD_LIBRARY_PATH'] {repr(os.environ['LD_LIBRARY_PATH'])}") # TODO Remove this
    command = ' && '.join([
        f'pushd {llvm_repo_build_dir}',
        f'cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON',
        f'cmake --build . --target check-mlir',
        f'popd',
    ])
    print('LLVM Repo Compile Command:')
    print(command)
    llvm_compile_stdout, llvm_compile_stderr, return_code = shell(command)
    print('LLVM Repo Compile Output:')
    print(llvm_compile_stdout)
    print('LLVM Repo Compile Error Messages:')
    print(llvm_compile_stderr)
    if return_code != 0:
        raise RuntimeError(f'Could not compile local LLVM repo.')
    assert os.path.isdir(llvm_repo_bin_dir)
    return

@trace
def compile_tibs_compiler() -> None:
    print(f'Compiling the TIBS compiler.')
    
    # TODO move these environment variables into local pyton variables
    # Verify BUILD_DIR and PREFIX environment variables
    llvm_repo_build_dir = os.path.join(LLVM_PROJECT_REPO_LOCATION, 'build')
    os.environ['BUILD_DIR'] = llvm_repo_build_dir
    os.environ['PREFIX'] = llvm_repo_build_dir
    
    build_dir = os.path.join(CURRENT_MODULE_PATH, 'mlir/build')
    if not os.path.isdir(build_dir):
        os.makedirs(build_dir)
    print(f"os.environ['PATH'] {repr(os.environ['PATH'])}") # TODO Remove this
    print(f"os.environ['LD_LIBRARY_PATH'] {repr(os.environ['LD_LIBRARY_PATH'])}") # TODO Remove this
    print(f"os.environ['BUILD_DIR'] {repr(os.environ['BUILD_DIR'])}") # TODO Remove this
    print(f"os.environ['PREFIX'] {repr(os.environ['PREFIX'])}") # TODO Remove this
    command = ' && '.join([
        'echo PATH $PATH', # TODO Remove this
        'echo LD_LIBRARY_PATH $LD_LIBRARY_PATH', # TODO Remove this
        'echo BUILD_DIR $BUILD_DIR', # TODO Remove this
        'echo PREFIX $PREFIX', # TODO Remove this
        f'pushd {build_dir}',
        'cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit',
        'ninja',
        'popd'
    ])
    print('Compilation of TIBS Compiler Command:')
    print(command)
    compile_stdout, compile_stderr, return_code = shell(command)
    print('Compilation of TIBS Compiler Output:')
    print(compile_stdout)
    print('Compilation of TIBS Compiler Error Messages:')
    print(compile_stderr)
    if return_code != 0:
        raise RuntimeError(f'Could not compile TIBS compiler.')
    return 

clone_local_llvm()
set_environment_variables()
compile_local_llvm()
compile_tibs_compiler()

LIBTIBS_SO_LOCATION = os.path.abspath(os.path.join(CURRENT_MODULE_PATH, 'mlir/build/tibs-compiler/libtibs-compiler.so')) # TODO update this

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
    print("TODO add something here")
