
# Start Up Prints

def print_header() -> None:
    import time
    import os
    print(f'')
    print(f'Start time: {time.strftime("%Y/%m/%d %H:%M:%S")}')
    print(f'Current Working Directory: {os.getcwd()}')
    print(f'')
    return

print_header()

# Imports purely for accessibility, not use in helper utilities

import os
import sys
import random
import time
import re
import time
import math
import json
import subprocess
import multiprocessing
import functools
import itertools
import inspect
import signal
from importlib import reload
from inspect import getfile, getsource, getsourcefile
from inspect import getmodule
from inspect import getdoc
from inspect import signature
from statistics import mean
from functools import reduce

import pandas as pd
import numpy as np

# Debugging Utilities

from typing import Generator
from contextlib import contextmanager
@contextmanager
def safe_cuda_memory() -> Generator:
    try:
        yield
    except RuntimeError as err:
        if not any(cuda_err_substring in str(err) for cuda_err_substring in {'CUDA error: out of memory', 'CUDA out of memory'}):
            raise
        else:
            print("CUDA ran out of memory.")

from typing import Union, Callable
from functools import lru_cache
class ASSERTMetaClass(type):
    
    @staticmethod
    @lru_cache(maxsize=1)
    def exceptions_by_name() -> None:
        return {exception.__name__: exception for exception in child_classes(BaseException)}
    
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
        def assert_func(invariant: bool, *args) -> None:
            if not invariant:
                raise exception_type(*args)
            return
        return assert_func
    
    def __getattr__(self, exception_type: type) -> Callable[[bool], None]:
        return self[exception_type]

class ASSERT(metaclass=ASSERTMetaClass):
    pass

from typing import Generator
from contextlib import contextmanager
@contextmanager
def warnings_suppressed() -> Generator:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
    return

from contextlib import contextmanager
@contextmanager
def exceptions_suppressed() -> Generator:
    try:
        yield
    except Exception:
        pass
    return

import io
from typing import Generator
from contextlib import contextmanager
@contextmanager
def std_out(stream: io.TextIOWrapper) -> Generator:
    import sys
    original_std_out = sys.stdout
    sys.stdout = stream
    yield
    sys.stdout = original_std_out
    return

from typing import Generator
from contextlib import contextmanager
@contextmanager
def suppressed_output() -> Generator:
    import os
    import sys
    with open(os.devnull, 'w') as dev_null:
        with std_out(dev_null):
            yield
    return

try:
    from contextlib import contextmanager
    @contextmanager
    class redirected_standard_streams:
        
        import ctypes
        from typing import Callable, Union, Generator
        
        libc = ctypes.CDLL(None)
        c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')
        c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')
    
        def __new__(cls, exitCallback: Union[None, Callable[[str], None]] = None) -> Generator:
            
            import io
            import os
            import sys
            import tempfile
            
            stdout_stream = io.BytesIO()
            stderr_stream = io.BytesIO()
            
            original_stdout_file_descriptor = sys.stdout.fileno()
            original_stderr_file_descriptor = sys.stderr.fileno()
    
            def _redirect(destination_stdout_file_descriptor: int, destination_stderr_file_descriptor: int) -> None:
                cls.libc.fflush(cls.c_stdout)
                cls.libc.fflush(cls.c_stderr)
                sys.stdout.close()
                sys.stderr.close()
                os.dup2(destination_stdout_file_descriptor, original_stdout_file_descriptor)
                os.dup2(destination_stderr_file_descriptor, original_stderr_file_descriptor)
                sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_file_descriptor, 'wb'))
                sys.stderr = io.TextIOWrapper(os.fdopen(original_stderr_file_descriptor, 'wb'))
                return
    
            with tempfile.TemporaryFile(mode='w+b') as stdout_tmp_file:
                with tempfile.TemporaryFile(mode='w+b') as stderr_tmp_file:
                    try:
                        saved_stdout_file_descriptor = os.dup(original_stdout_file_descriptor)
                        saved_stderr_file_descriptor = os.dup(original_stderr_file_descriptor)
                        _redirect(stdout_tmp_file.fileno(), stderr_tmp_file.fileno())
                        yield
                        _redirect(saved_stdout_file_descriptor, saved_stderr_file_descriptor)
                        stdout_tmp_file.flush()
                        stderr_tmp_file.flush()
                        stdout_tmp_file.seek(0, io.SEEK_SET)
                        stderr_tmp_file.seek(0, io.SEEK_SET)
                        stdout_stream.write(stdout_tmp_file.read())
                        stderr_stream.write(stderr_tmp_file.read())
                    finally:
                        os.close(saved_stdout_file_descriptor)
                        os.close(saved_stderr_file_descriptor)
                        
            if exitCallback is not None:
                exitCallback(stdout_stream.getvalue().decode('utf-8'), stderr_stream.getvalue().decode('utf-8'))
            
            return
except Exception:
    pass
        
from typing import Callable, Union, Generator
from contextlib import contextmanager
@contextmanager
def redirected_output(exitCallback: Union[None, Callable[[str], None]] = None) -> Generator:
    import sys
    from io import StringIO
    temporary_std_out = StringIO()
    with std_out(temporary_std_out):
        yield
    printed_output: str = temporary_std_out.getvalue()
    if exitCallback is not None:
        exitCallback(printed_output)
    return

from typing import Tuple
def shell(input_command: str) -> Tuple[str, str]:
    '''Handles multi-line input_command'''
    import subprocess
    command = input_command.encode()
    process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_string, stderr_string = map(bytes.decode, process.communicate(command))
    return stdout_string, stderr_string

def pid() -> int:
    import os
    return os.getpid()

from typing import Optional
def file(obj) -> Optional[str]:
    import inspect
    file_location: Optional[str] = None
    try:
        file_location = inspect.getfile(obj)
        source_file_location = inspect.getsourcefile(obj)
    except TypeError as err:
        module = inspect.getmodule(obj)
        if module:
            file_location = inspect.getfile(module)
            source_file_location = inspect.getsourcefile(module)
        else:
            file_location = source_file_location = None
    if None in {file_location, source_file_location} or file_location != source_file_location:
        print("Consider using inspect.getsourcefile as well.")
    return file_location

def source(obj) -> None:
    import inspect
    try:
        source_code = inspect.getsource(obj)
    except TypeError as err:
        obj_type = type(obj)
        source_code = inspect.getsource(obj_type)
    print(source_code)
    return

def module(obj):
    return getmodule(obj)

def doc(obj) -> None:
    import inspect
    print(inspect.getdoc(obj))
    return

from typing import Tuple
def parent_classes(obj) -> Tuple[type, ...]:
    import inspect
    cls = obj if inspect.isclass(obj) else type(obj)
    return inspect.getmro(cls)

from typing import Tuple
def child_classes(obj) -> Tuple[type, ...]:
    import inspect
    cls = obj if inspect.isclass(obj) else type(obj)
    def _child_classes(cls):
        for subclass in cls.__subclasses__():
            yield from _child_classes(subclass)
            yield subclass
    return tuple(_child_classes(cls))

from typing import Iterable
def p1(iterable: Iterable) -> None:
    for e in iterable:
        print(e)
    return

def pdir(arbitrary_object: object) -> None:
    for e in dir(arbitrary_object):
        print(e)
    return

from typing import List
def current_tensors() -> List:
    import torch
    import gc
    return [e for e in gc.get_objects() if isinstance(e, torch.Tensor)]

def _dummy_tqdm_message_func(index: int):
    return ''
def tqdm_with_message(iterable,
                      pre_yield_message_func: Callable[[int], str] = _dummy_tqdm_message_func,
                      post_yield_message_func: Callable[[int], str] = _dummy_tqdm_message_func,
                      *args, **kwargs):
    import tqdm
    if 'bar_format' not in kwargs:
        kwargs['bar_format']='{l_bar}{bar:50}{r_bar}'
    progress_bar_iterator = tqdm.tqdm(iterable, *args, **kwargs)
    for index, element in enumerate(progress_bar_iterator):
        if pre_yield_message_func != _dummy_tqdm_message_func:
            pre_yield_message = pre_yield_message_func(index)
            progress_bar_iterator.set_description(pre_yield_message)
            progress_bar_iterator.refresh()
        yield element
        if post_yield_message_func != _dummy_tqdm_message_func:
            post_yield_message = post_yield_message_func(index)
            progress_bar_iterator.set_description(post_yield_message)
            progress_bar_iterator.refresh()

from typing import Callable
def debug_on_error(func: Callable) -> Callable:
    import pdb
    import traceback
    import sys
    def decorating_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            print(f'Exception Class: {type(err)}')
            print(f'Exception Args: {err.args}')
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
    return decorating_function

from typing import Callable
def raise_on_warn(func: Callable) -> Callable:
    import warnings
    def decorating_function(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = func(*args, **kwargs)
        return result
    return decorating_function

TRACE_INDENT_LEVEL = 0
TRACE_INDENTATION = '    '
TRACE_VALUE_SIZE_LIMIT = 200
from typing import Callable
def trace(func: Callable) -> Callable:
    from inspect import signature
    import sys
    import random
    def human_readable_value(value) -> str:
        readable_value = repr(value)
        if len(readable_value) > TRACE_VALUE_SIZE_LIMIT:
            readable_value = readable_value[:TRACE_VALUE_SIZE_LIMIT]+'...'
        return readable_value
    def decorating_function(*args, **kwargs):
        arg_values_string = ', '.join((f'{param_name}={human_readable_value(value)}' for param_name, value in signature(func).bind(*args, **kwargs).arguments.items()))
        probably_unique_id = random.randint(10,99)
        global TRACE_INDENT_LEVEL, TRACE_INDENTATION
        entry_line = f' {TRACE_INDENTATION * TRACE_INDENT_LEVEL}[{TRACE_INDENT_LEVEL}:{probably_unique_id}] {func.__qualname__}({arg_values_string})'
        with std_out(sys.__stdout__):
            print()
            print(entry_line)
            print()
        TRACE_INDENT_LEVEL += 1
        result = func(*args, **kwargs)
        TRACE_INDENT_LEVEL -= 1
        with std_out(sys.__stdout__):
            print()
            print(f' {TRACE_INDENTATION * TRACE_INDENT_LEVEL}[{TRACE_INDENT_LEVEL}:{probably_unique_id}] returned {human_readable_value(result)}')
            print()
        return result
    return decorating_function

BOGUS_TOKEN = object()
import types
def dpn(expression_string: str, given_frame=Optional[types.FrameType]):
    """dpn == debug print name"""
    import os
    import sys
    import inspect
    try:
        frame = inspect.currentframe() if given_frame is None else given_frame
        prev_frame = frame.f_back
        macro_caller_locals = prev_frame.f_locals
        macro_caller_globals = prev_frame.f_globals
        new_var_name = f'paul_dpf_hack_{id(expression_string)}'
        new_globals = dict(macro_caller_globals)
        new_globals.update({new_var_name: BOGUS_TOKEN})
        sys.stdout = open(os.devnull, 'w')
        exec(f'{new_var_name} = {expression_string}', macro_caller_locals, new_globals)
        sys.stdout = sys.__stdout__
        var_value = new_globals[new_var_name]
        if id(var_value) == id(BOGUS_TOKEN):
            raise NameError(f"Cannot determine value of {expression_string}")
        print(f'{expression_string}: {repr(var_value)}')
    finally:
        del frame
    return var_value

class __dpf_hack_by_paul__():
    def __init__(self):
        pass
    
    def __getattr__(self, var_name):
        import inspect
        frame = inspect.currentframe()
        return dpn(var_name, frame)

dpf = __dpf_hack_by_paul__() # usage is like a='a' ; dpf.a

# @todo make apropos methods

# Timing Utilities

from typing import Callable, Generator
from contextlib import contextmanager
@contextmanager
def timeout(time: int, functionToExecuteOnTimeout: Callable[[], None] = None) -> Generator:
    """NB: This cannot be nested."""
    import signal
    def _raise_timeout(*args, **kwargs):
        raise TimeoutError
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(time)
    try:
        yield
    except TimeoutError:
        if functionToExecuteOnTimeout is not None:
            functionToExecuteOnTimeout()
    finally:
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
    return

from typing import Callable, Generator
from contextlib import contextmanager
@contextmanager
def timer(section_name: str = None, exitCallback: Callable[[float], None] = None) -> Generator:
    import time
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    if exitCallback != None:
        exitCallback(elapsed_time)
    elif section_name:
        print(f'{section_name} took {elapsed_time} seconds.')
    else:
        print(f'Execution took {elapsed_time} seconds.')
    return

# General Utilities

NoneType = type(None)

from collections import defaultdict
def recursive_defaultdict() -> defaultdict:
    return defaultdict(recursive_defaultdict)

def is_ascii(input_string: str) -> bool:
    return all(ord(character) < 128 for character in input_string)

from typing import Generator
from contextlib import contextmanager
@contextmanager
def temp_plt_figure(*args, **kwargs) -> Generator:
    import matplotlib.pyplot as plt
    figure = plt.figure(*args, **kwargs)
    yield figure
    plt.close(figure)
    return

from typing import Iterable
def only_one(items: Iterable):
    assert isinstance(items, Iterable)
    items = [e for e in items]
    assert len(items) == 1
    return items[0]

from typing import List
def at_most_one(items: List):
    return only_one(items) if items else None

from typing import List
def parallel_map(*args, **kwargs) -> List:
    import multiprocessing
    p = multiprocessing.Pool()
    result = p.map(*args, **kwargs)
    p.close()
    p.join()
    return result

from typing import List
def parallel_mapcar(func, *args) -> List:
    import multiprocessing
    p = multiprocessing.Pool()
    result = p.starmap(func, zip(*args))
    p.close()
    p.join()
    return result

from typing import Iterable, Callable,  List
def eager_map(func: Callable, iterable: Iterable) -> List:
    return list(map(func, iterable))

from typing import Iterable, Callable,  List
def eager_map_reduce(func: Callable, iterable: Iterable) -> List:
    return list(map(func, iterable))

from typing import Iterable, Callable, List
def eager_filter(func: Callable, iterable: Iterable) -> List:
    return list(filter(func, iterable))

from typing import List
def eager_zip(*args) -> List:
    args = list(map(tuple, args))
    assert len(set(map(len, args))) == 1
    return list(zip(*args))

def identity(input):
    return input

def xor(disjunct_a: bool, disjunct_b: bool) -> bool:
    return bool(disjunct_a) ^ bool(disjunct_b)

def implies(antecedent: bool, consequent: bool) -> bool:
    return not antecedent or consequent

def iff(antecedent: bool, consequent: bool) -> bool:
    return bool(antecedent) == bool(consequent)

from typing import Generator
def uniq(iterator: Iterable) -> Generator:
    previous = BOGUS_TOKEN
    for value in iterator:
        if previous != value:
            yield value
            previous = value

from itertools import cycle, islice
from typing import Iterable, Generator
def roundrobin(*iterables: Iterable) -> Generator:
    number_of_active_iterables = len(iterables)
    nexts = cycle(iter(iterable).__next__ for iterable in iterables)
    while number_of_active_iterables > 0:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            number_of_active_iterables -= 1
            nexts = cycle(islice(nexts, number_of_active_iterables))

from typing import Iterable 
def powerset(iterable: Iterable) -> Iterable:
    from itertools import chain, combinations
    items = list(iterable)
    number_of_items = len(items)
    subset_iterable = chain.from_iterable(combinations(items, length) for length in range(1, number_of_items+1))
    return subset_iterable

def n_choose_k(n: int, k: int):
    from functools import reduce
    k = min(k, n-k)
    numerator = reduce(int.__mul__, range(n, n-k, -1), 1)
    denominator = reduce(int.__mul__, range(1, k+1), 1)
    return numerator // denominator

def lerp(start: float, end: float, floatValue: float) -> float:
    return start + floatValue * (end - start);

def true(*args, **kwargs) -> bool:
    return True

def false(*args, **kwargs) -> bool:
    return False

def current_timestamp_string() -> str:
    import time
    return time.strftime("%Y_%m_%d_%H_%M_%S")

from typing import Iterable 
def unzip(zipped_item: Iterable) -> Iterable:
    return zip(*zipped_item)

from collections import Counter
def histogram(iterator: Iterable) -> Counter:
    counter: Counter = Counter()
    for element in iterator:
        counter[element]+=1
    return counter

from typing import Callable
def _compose(f: Callable, g: Callable):
    return lambda *args, **kwargs: f(g(*args, **kwargs))
def compose(*functions: Callable):
    return reduce(_compose, functions)

from typing import Iterable, List
def quadratic_unique(iterator: Iterable) -> List:
    answer = []
    for element in iterator:
        if element not in answer:
            answer.append(element)
    return answer    
