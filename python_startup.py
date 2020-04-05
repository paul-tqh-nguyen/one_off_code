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

# Printing Utilities

def p1(iterable: Iterable) -> None:
    for e in iterable:
        print(e)
    return

# Debugging Utilities

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
        module = inspect.getmodule(obj)
        source_code = inspect.getsource(module)
    print(source_code)
    return

def doc(obj) -> None:
    print(inspect.getdoc(batch))
    return

def debug_on_error(func: Callable) -> Callable:
    def func_wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            print(f'Exception Class: {type(err)}')
            print(f'Exception Args: {err.args}')
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
    return func_wrapped

def trace(func: Callable) -> Callable:
    def func_wrapped(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except:
            for frame in inspect.trace():
                print(frame.code_context, frame.lineno)
                print(dir(frame))
    return func_wrapped

def dpn(var_name: str, given_frame=None):
    """dpn == debug print name"""
    try:
        frame = inspect.currentframe() if given_frame is None else given_frame
        prev_frame = frame.f_back
        macro_caller_locals = prev_frame.f_locals
        macro_caller_globals = prev_frame.f_globals
        bogus_token = lambda x:x
        var_value = macro_caller_locals[var_name] if var_name in macro_caller_locals else macro_caller_globals[var_name] if var_name in macro_caller_globals else bogus_token
        if var_value == bogus_token:
            raise NameError(f"Cannot determine value of {var_name}")
        print(f'{var_name}: {var_value}')
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
