
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
from typing import Any, Tuple, Union, Generator, Callable, Optional

from ..misc_utilities import *

# TODO make sure imports are used
# TODO make sure these imports are ordered in some way

#############################
# Globals & Initializations #
#############################

CURRENT_MODULE_PATH = os.path.dirname(os.path.realpath(__file__))

LIBTIBS_SO_LOCATION = os.path.abspath(os.path.join(CURRENT_MODULE_PATH, 'mlir/build/tibs-compiler/libtibs-compiler.so'))

assert os.path.isfile(LIBTIBS_SO_LOCATION), f'The TIBS compiler has not yet been compiled. It is expected to be found at {LIBTIBS_SO_LOCATION}.'

####################
# ctypes Utilities #
####################

class SafeSharedObjectCallable:

    @staticmethod
    def ctypes_compatible(instance: Any, data_type: '_ctypes._CData') -> bool:
        expected_class = {
            ctypes.c_int: int,
            ctypes.c_void_p: int,
            ctypes.c_bool: bool,
        }.get(data_type, data_type)
        result = isinstance(instance, expected_class)
        return result

    def __init__(self, so_callable: '_ctypes.PyCFuncPtr') -> None:
        self.so_callable = so_callable
        self.input_types = BOGUS_TOKEN
        self.return_type = BOGUS_TOKEN
        return
    
    def __setitem__(self, item: Union[Tuple['_ctypes._CData', ...], '_ctypes._CData'], return_type: '_ctypes._CData') -> None:
        ASSERT.RuntimeError(self.input_types is BOGUS_TOKEN and self.so_callable.argtypes is None, f'Input types of {self.input_types} already declared for {self.so_callable.__name__}')
        ASSERT.RuntimeError(self.return_type is BOGUS_TOKEN and self.so_callable.restype is ctypes.c_int, f'Return type of {self.return_type} already declared for {self.so_callable.__name__}')
        self.input_types = item if isinstance(item, tuple) else (item,)
        self.return_type = return_type
        self.so_callable.argtypes = self.input_types
        self.so_callable.restype = self.return_type
        return
    
    def __call__(self, *args) -> '_ctypes._CData':
        ASSERT.RuntimeError(self.input_types is not BOGUS_TOKEN, f'Input types not declared for {self.so_callable.__name__}')
        ASSERT.RuntimeError(self.return_type is not BOGUS_TOKEN, f'Return type not declared for {self.so_callable.__name__}')
        ASSERT.ValueError(len(args) == len(self.input_types), f'{self.so_callable.__name__} expects {len(self.input_types)} inputs but got {len(args)}.')
        for arg, input_type in zip(args, self.input_types):
            ASSERT.TypeError(self.ctypes_compatible(arg, input_type), f'{arg} is not compatible with {input_type}.')
        result = self.so_callable(*args)
        expected_result_type = NoneType if self.return_type is None else self.return_type
        ASSERT.TypeError(self.ctypes_compatible(result, expected_result_type), f'{self.so_callable.__name__} returned {result}, an instance of {type(result)}, which is not compatible with {self.return_type}.')
        return result

class SafeDLL():

    def __init__(self, so_location: str) -> None:
        self.so = ctypes.CDLL(so_location)
        self.func_name_to_func: Dict[str, SafeSharedObjectCallable] = dict()
        return
    
    def __getattr__(self, func_name: str) -> SafeSharedObjectCallable:
        if func_name not in self.func_name_to_func:
            func = getattr(self.so, func_name)
            self.func_name_to_func[func_name] = SafeSharedObjectCallable(func)
        return self.func_name_to_func[func_name]

NO_INPUTS = ()
''' # TODO enable this
###################################
# libtibs Shared Object Utilities #
###################################

c = SafeDLL(LIBTIBS_SO_LOCATION)

# Module Generator Functions

c.generateModule[ctypes.c_void_p, ctypes.c_void_p] = None # TODO remove this

c.newModuleGenerator[NO_INPUTS] = ctypes.c_void_p
c.deleteModuleGenerator[ctypes.c_void_p] = None
c.dumpModule[ctypes.c_void_p] = None
c.runPassManager[ctypes.c_void_p] = ctypes.c_bool
c.compileAndExecuteModule[ctypes.c_void_p] = ctypes.c_bool
c.newLocation[ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int] = ctypes.c_void_p
c.deleteLocation[ctypes.c_void_p] = None

##########################
# Module Generator Class #
##########################

class ModuleGenerator:
    
    def __init__(self) -> None:
        self.module_generator_pointer: int = c.newModuleGenerator()
        return
    
    def __del__(self) -> None:
        c.deleteModuleGenerator(self.module_generator_pointer)
        return
    
    def dump_module(self) -> str:
        results = []
        with redirected_standard_streams(lambda *args: results.append(args)):
            c.dumpModule(self.module_generator_pointer)
        stdout_string, stderr_string = only_one(results)
        assert stdout_string == '', f'Expected stdout stream to be empty but got {repr(stdout_string)}.'
        module_string = stderr_string
        return module_string
    
    def run_pass_manager(self) -> bool:
        success_status = c.runPassManager(self.module_generator_pointer)
        return success_status

    def compile_and_execute_module(self) -> bool:
        success_status = c.compileAndExecuteModule(self.module_generator_pointer)
        return success_status

    def new_location(self, file_name: bytes, line_number: int, column_number: int) -> int:
        # TODO use this somewhere
        # TODO test this
        file_name = ctypes.c_char_p(file_name)
        location_pointer = c.newLocation(self.module_generator_pointer, file_name, line_number, column_number)
        return location_pointer

    def delete_location(self, location_pointer: int) -> None:
        # TODO use this somewhere
        # TODO test this
        c.deleteLocation(location_pointer)
        return
'''    
############
# Compiler #
############

def compile():
    # TODO update this
    result = 'DUMMY'
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
