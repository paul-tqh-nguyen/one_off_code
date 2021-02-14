
from setuptools import setup
import distutils.cmd
import distutils.log

import os
import subprocess
import shutil

from typing import Tuple

TIBS_COMPILER_PATH = os.path.abspath('./tibs/compiler/')

LLVM_PROJECT_GIT_URL = 'https://github.com/llvm/llvm-project.git'

LLVM_PROJECT_GIT_COMMIT_HASH = '805d59593f50a5c5050c0fc5cb9fbe02cd751511'

LLVM_PROJECT_REPO_LOCATION = os.path.abspath(os.path.join(TIBS_COMPILER_PATH, 'llvm-project'))

LLVM_REPO_BUILD_DIR = os.path.join(LLVM_PROJECT_REPO_LOCATION, 'build')

LLVM_REPO_BIN_DIR = os.path.join(LLVM_REPO_BUILD_DIR, 'bin')

TIBS_COMPILER_BUILD_DIR = os.path.join(TIBS_COMPILER_PATH, 'mlir/build')

def run_shell_commands(directory: str, *commands: Tuple[str], **environment_variables) -> None:
    command = ' && '.join(
        [f'pushd {directory}'] +
        [f'export {name}={value}' for name, value in environment_variables.items()] + 
        list(commands) +
        ['popd']
    )
    
    process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_string, stderr_string = process.communicate(command.encode("utf-8"))
    stdout_string = stdout_string.decode("utf-8")
    stderr_string = stderr_string.decode("utf-8")
    return_code = process.returncode
    
    if return_code != 0:
        error_string = (
            f'Command Failed wit exit code {return_code}:' + '\n' +
            command + '\n' +
            'STDOUT Messages:' + '\n' +
            stdout_string + '\n' +
            'STDERR Messages:' + '\n' +
            stderr_string
        )
        raise RuntimeError(error_string)
    return

class CompileTibsCompilerCommand(distutils.cmd.Command):
    
    description = 'Compile the TIBS compiler.'
    user_options = [
        ('compile-clean=', None, 'Compile from a clean state.'),
    ]

    def initialize_options(self) -> None:
        self.compile_clean = 'false'
        return

    def finalize_options(self) -> None:
        self.compile_clean = False if self.compile_clean.lower().strip() == 'false' else True
        return

    def run(self) -> None:
        if 'CONDA_PREFIX' not in os.environ:
            raise NotImplementedError(f'Compilation of TIBS compiler not currently supported outside of a conda environment.')
        self.clone_local_llvm()
        self.compile_local_llvm()
        self.compile_tibs_compiler()
        self.announce(f'Finished compiling TIBS compiler.', level=distutils.log.INFO)
        return
    
    def clone_local_llvm(self) -> None:
        if self.compile_clean:
            shutil.rmtree(LLVM_PROJECT_REPO_LOCATION)
        if not os.path.isdir(LLVM_PROJECT_REPO_LOCATION):
            self.announce(f'Cloning local LLVM repo into {LLVM_PROJECT_REPO_LOCATION}.', level=distutils.log.INFO)
            run_shell_commands(TIBS_COMPILER_PATH, f'git clone {LLVM_PROJECT_GIT_URL}')
            
            self.announce(f'Resetting local LLVM repo at {LLVM_PROJECT_REPO_LOCATION} to commit {LLVM_PROJECT_GIT_COMMIT_HASH}.', level=distutils.log.INFO)
            run_shell_commands(LLVM_PROJECT_REPO_LOCATION, f'git reset --hard  {LLVM_PROJECT_GIT_COMMIT_HASH}')
        return
    
    def compile_local_llvm(self) -> None:
        self.announce(f'Compiling local LLVM repo at {LLVM_PROJECT_REPO_LOCATION}.', level=distutils.log.INFO)
        
        if not os.path.isdir(LLVM_REPO_BUILD_DIR):
            os.makedirs(LLVM_REPO_BUILD_DIR)
        
        if self.compile_clean or not os.path.isdir(LLVM_REPO_BIN_DIR):
            PATH = os.environ['PATH']
            LD_LIBRARY_PATH = os.environ['LD_LIBRARY_PATH']

            if LLVM_REPO_BIN_DIR not in PATH.split(':'):
                PATH = PATH + ':' + LLVM_REPO_BIN_DIR
            
            conda_prefix = os.environ['CONDA_PREFIX']
            conda_lib_dir = os.path.join(conda_prefix, 'lib')
            if conda_lib_dir not in LD_LIBRARY_PATH.split(':'):
                LD_LIBRARY_PATH = LD_LIBRARY_PATH + ':' + conda_lib_dir
                
            run_shell_commands(
                LLVM_REPO_BUILD_DIR,
                f'cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON',
                f'cmake --build . --target check-mlir',
                PATH=PATH,
                LD_LIBRARY_PATH=LD_LIBRARY_PATH,
            )
        
        assert os.path.isdir(LLVM_REPO_BIN_DIR)
        
        return

    def compile_tibs_compiler(self) -> None:
        self.announce(f'Compiling the TIBS compiler.', level=distutils.log.INFO)
        
        if not os.path.isdir(TIBS_COMPILER_BUILD_DIR):
            os.makedirs(TIBS_COMPILER_BUILD_DIR)
        
        run_shell_commands(
            TIBS_COMPILER_BUILD_DIR,
            'cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit',
            'ninja',
            BUILD_DIR=LLVM_REPO_BUILD_DIR,
            PREFIX=LLVM_REPO_BUILD_DIR
        )

        return
        

setup( # TODO add more here
    name="TIBS",
    description="TIBS (Tensor Instruction Based Semantics) Programming Language",
    author="Paul Nguyen",
    packages=["tibs"],
    python_requires=">=3.7",
    cmdclass={
        'compile_tibs_compiler': CompileTibsCompilerCommand,
    },

)
