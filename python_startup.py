import pdb
import traceback
import sys
import os
import time
import inspect
import signal
from inspect import getfile, getsource, getsourcefile
from inspect import getmodule
from inspect import getdoc
from inspect import signature
from functools import reduce
from contextlib import contextmanager
from itertools import chain, combinations
from typing import Iterable, Callable, Generator, List

# Imports purely for accessibility, not necessarily use in these functions

import random

# Start Up Prints

def print_header() -> None:
    print(f'')
    print(f'Start time: {time.strftime("%Y/%m/%d %H:%M:%S")}')
    print(f'Current Working Directory: {os.getcwd()}')
    print(f'')
    print(f'''Useful Forms:

import metagraph_cuda.tests.algorithms.test_pagerank ; from importlib import reload ; reload(metagraph_cuda.tests.algorithms.test_pagerank) ; from metagraph_cuda.tests.algorithms.test_pagerank import *

os.chdir(os.path.expanduser('~/code/one_off_code/cugraph_experiments/')); import test_community, importlib; importlib.reload(test_community); from test_community import * 

@debug_on_error
def test():
    return test_katz_centrality_undirected()
''')
    return

print_header()

# Debugging Utilities

def pid() -> int:
    return os.getpid()

def file(obj) -> str:
    try:
        file_location = inspect.getfile(obj)
        source_file_location = inspect.getsourcefile(obj)
    except TypeError as err:
        module = inspect.getmodule(obj)
        file_location = inspect.getfile(module)
        source_file_location = inspect.getsourcefile(module)
    if file_location != source_file_location:
        print("Consider using inspect.getsourcefile as well.")
    return file_location

def source(obj) -> None:
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
    print(inspect.getdoc(batch))
    return

def debug_on_error(func: Callable) -> Callable:
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

TRACE_INDENT_LEVEL = 0
TRACE_INDENTATION = '    '

def trace(func: Callable) -> Callable:
    def decorating_function(*args, **kwargs):
        arg_values_string = ', '.join((f'{param_name}={value}' for param_name, value in signature(func).bind(*args, **kwargs).arguments.items()))
        global TRACE_INDENT_LEVEL, TRACE_INDENTATION
        entry_line = f' {TRACE_INDENTATION * TRACE_INDENT_LEVEL}[{TRACE_INDENT_LEVEL}] {func.__name__}({arg_values_string})'
        print(entry_line)
        TRACE_INDENT_LEVEL += 1
        result = func(*args, **kwargs)
        TRACE_INDENT_LEVEL -= 1
        print(f' {TRACE_INDENTATION * TRACE_INDENT_LEVEL}[{TRACE_INDENT_LEVEL}] returned {result}')
        return result
    return decorating_function

BOGUS_TOKEN = lambda x:x

def dpn(expression_string: str, given_frame=None):
    """dpn == debug print name"""
    try:
        frame = inspect.currentframe() if given_frame is None else given_frame
        prev_frame = frame.f_back
        macro_caller_locals = prev_frame.f_locals
        macro_caller_globals = prev_frame.f_globals
        new_var_name = f'paul_dpf_hack_{id(expression_string)}'
        new_globals = dict(macro_caller_globals)
        new_globals.update({new_var_name: BOGUS_TOKEN})
        exec(f'{new_var_name} = {expression_string}', macro_caller_locals, new_globals)
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
        frame = inspect.currentframe()
        return dpn(var_name, frame)

dpf = __dpf_hack_by_paul__() # usage is like a='a' ; dpf.a

def p1(iterable: Iterable) -> None:
    for e in iterable:
        print(e)
    return

# @todo make apropos methods

# Timing Utilities

@contextmanager
def timeout(time: float, functionToExecuteOnTimeout: Callable[[], None] = None):
    """NB: This cannot be nested."""
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

@contextmanager
def timer(section_name: str = None, exitCallback: Callable[[], None] = None):
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    if exitCallback != None:
        exitCallback(elapsed_time)
    elif section_name:
        print('Execution of "{section_name}" took {elapsed_time} seconds.'.format(section_name=section_name, elapsed_time=elapsed_time))
    else:
        print('Execution took {elapsed_time} seconds.'.format(elapsed_time=elapsed_time))

# General Utilities

def only_one(items: List):
    assert isinstance(items, list)
    assert len(items) == 1
    return items[0]

def at_most_one(items: List):
    return only_one(items) if items else None

def eager_map(func: Callable, iterable: Iterable) -> List:
    return list(map(func, iterable))

def eager_filter(func: Callable, iterable: Iterable) -> List:
    return list(filter(func, iterable))

def identity(input):
    return input

def implies(antecedent: bool, consequent: bool) -> bool:
    return not antecedent or consequent

UNIQUE_BOGUS_RESULT_IDENTIFIER = (lambda x: x)

def uniq(iterator: Iterable) -> Generator:
    previous = UNIQUE_BOGUS_RESULT_IDENTIFIER
    for value in iterator:
        if previous != value:
            yield value
            previous = value

def powerset(iterable: Iterable) -> Iterable:
    items = list(iterable)
    number_of_items = len(items)
    subset_iterable = chain.from_iterable(combinations(items, length) for length in range(1, number_of_items+1))
    return subset_iterable

def n_choose_k(n, k):
    k = min(k, n-k)
    numerator = reduce(int.__mul__, range(n, n-k, -1), 1)
    denominator = reduce(int.__mul__, range(1, k+1), 1)
    return numerator // denominator

def false(*args, **kwargs) -> bool:
    return False

def current_timestamp_string() -> str:
    return time.strftime("%Y_%m_%d_%H_%M_%S")

def unzip(zipped_item: Iterable) -> Iterable:
    return zip(*zipped_item)
