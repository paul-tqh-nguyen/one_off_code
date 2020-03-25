import pdb
import traceback
import sys
import time
import signal
from contextlib import contextmanager
from itertools import chain, combinations

# General Utilities

def identity(args):
    return args

def implies(antecedent, consequent) -> bool:
    return not antecedent or consequent

UNIQUE_BOGUS_RESULT_IDENTIFIER = (lambda x: x)

def uniq(iterator):
    previous = UNIQUE_BOGUS_RESULT_IDENTIFIER
    for value in iterator:
        if previous != value:
            yield value
            previous = value

def powerset(iterable):
    items = list(iterable)
    number_of_items = len(items)
    subset_iterable = chain.from_iterable(combinations(items, length) for length in range(1, number_of_items+1))
    return subset_iterable
        
def false(*args, **kwargs):
    return False

def current_timestamp_string() -> str:
    return time.strftime("%Y_%m_%d_%H_%M_%S")

# Debugging Utilities

def debug_on_error(func):
    def func_wrapped(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as err:
            print(f'Exception Class: {type(err)}')
            print(f'Exception Args: {err.args}')
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
    return func_wrapped

# Timing Utilities

@contextmanager
def timeout(time, functionToExecuteOnTimeout=None):
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
def timer(section_name=None, exitCallback=None):
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
