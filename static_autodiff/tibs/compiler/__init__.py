
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
from functools import lru_cache
from contextlib import contextmanager
from typing import Tuple, Union, Generator, Callable, Optional

from ..misc_utilities import *

# TODO make sure imports are used
# TODO make sure these imports are ordered in some way

#############################
# Globals & Initializations #
#############################

CURRENT_MODULE_PATH = os.path.dirname(os.path.realpath(__file__))

LIBTIBS_SO_LOCATION = os.path.abspath(os.path.join(CURRENT_MODULE_PATH, 'mlir/build/tibs-compiler/libtibs-compiler.so'))

assert os.path.isfile(LIBTIBS_SO_LOCATION), f'The TIBS compiler has not yet been compiled. It is expected to be found at {LIBTIBS_SO_LOCATION}.'

###########################
# Type Checking Utilities #
###########################

BOGUS_TOKEN = object()

class ASSERTMetaClass(type):

    
    @staticmethod
    @lru_cache(maxsize=1)
    def exceptions_by_name() -> None:
        return {exception.__name__: exception for exception in child_classes(Exception)}
    
    @lru_cache(maxsize=16)
    def __getitem__(self, exception_type: Union[type, str]) -> Callable[[bool], None]:
        if isinstance(exception_type, str):
            if exception_type not in self.exceptions_by_name():
                self.exceptions_by_name.cache_clear()
            exception_type = self.exceptions_by_name().get(exception_type, exception_type)
        if not isinstance(exception_type, type):
            raise TypeError(f'{exception_type} does not describe a subclass of {BaseException.__qualname__}.')
        if not issubclass(exception_type, BaseException):
            raise TypeError(f'{exception_type} is not a subclass of {BaseException.__qualname__}.')
        def assert_func(invariant: bool, error_message: str) -> None:
            if not invariant:
                raise exception_type(error_message)
            return
        return assert_func
    
    def __getattr__(self, exception_type: type) -> Callable[[bool], None]:
        return self[exception_type]

class ASSERT(metaclass=ASSERTMetaClass):
    pass

####################
# ctypes Utilities #
####################

class SafeSharedObjectCallable:

    def __init__(self, so_callable: '_ctypes.PyCFuncPtr') -> None:
        self.so_callable = so_callable
        self.input_types = BOGUS_TOKEN
        self.return_type = BOGUS_TOKEN
        return
    
    def __setitem__(self, item: Union[Tuple['_ctypes._CData', ...], '_ctypes._CData'], return_type: '_ctypes._CData') -> None:
        ASSERT.RuntimeError(self.input_types is BOGUS_TOKEN, f'Input types of {self.input_types} already declared for {self.so_callable.__name__}')
        ASSERT.RuntimeError(self.return_type is BOGUS_TOKEN, f'Return type of {self.return_type} already declared for {self.so_callable.__name__}')
        self.input_types = item if isinstance(item, tuple) else (item,)
        self.return_type = return_type
        self.so_callable.argtypes = self.input_types
        self.so_callable.restype = self.return_type
        # TODO make assertions about the types and values of self.so_callable.argtypes and self.so_callable.restype
        # TODO get rid of self.input_types and self.return_type
        return
    
    def __call__(self, *args) -> '_ctypes._CData':
        ASSERT.RuntimeError(self.input_types is not BOGUS_TOKEN, f'Input types not declared for {self.so_callable.__name__}')
        ASSERT.RuntimeError(self.return_type is not BOGUS_TOKEN, f'Return type not declared for {self.so_callable.__name__}')
        ASSERT.ValueError(len(args) == len(self.input_types), f'{self.so_callable.__name__} expects {len(self.input_types)} inputs but got {len(args)}.')
        for arg, input_type in zip(args, self.input_types):
            ASSERT.TypeError(isinstance(arg, input_type), f'{arg} is not an instance of {input_type}.')
        result = self.so_callable(*args)
        ASSERT.TypeError(self.return_type is None or isinstance(result, self.return_type), f'{self.so_callable.__name__} returned {result} (an instance of {type(result)}) when {self.return_type} was expected.')
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

###################################
# libtibs Shared Object Utilities #
###################################

LIBTIBS_SO = SafeDLL(LIBTIBS_SO_LOCATION)

LIBTIBS_SO.newModuleGenerator[NO_INPUTS] = ctypes.c_void_p # TODO remove this
LIBTIBS_SO.generateModule[NO_INPUTS] = None
LIBTIBS_SO.runAllPasses[NO_INPUTS] = None # TODO remove this
LIBTIBS_SO.dumpModule[ctypes.c_void_p] = None

##########################
# Module Generator Class #
##########################

class ModuleGenerator:

    c = LIBTIBS_SO

    def __init__(self) -> None:
        self.module_generator = self.c.newModuleGenerator() # TODO add type declaration
        return

    @trace
    def dump_module(self) -> str:
        self.c.generateModule(self.value) # TODO remove this
        self.c.dumpModule(self.value)
        module_string = 'DUMMY' # TODO unstub this
        return module_string

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
