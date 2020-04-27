#!/usr/bin/python3

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
        entry_line = f' {TRACE_INDENTATION * TRACE_INDENT_LEVEL}[{TRACE_INDENT_LEVEL}:{probably_unique_id}] {func.__name__}({arg_values_string})'
        with std_out(sys.__stdout__):
            print()
            print(entry_line)
            print()
        TRACE_INDENT_LEVEL += 1
        result = func(*args, **kwargs)
        TRACE_INDENT_LEVEL -= 1
        with std_out(sys.__stdout__):
            print()
            print(f' {TRACE_INDENTATION * TRACE_INDENT_LEVEL}[{TRACE_INDENT_LEVEL}:{probably_unique_id}] returned {result}')
            print()
        return result
    return decorating_function

from typing import List
def parallel_map(*args, **kwargs) -> List:
    import multiprocessing
    p = multiprocessing.Pool()
    result = p.map(*args, **kwargs)
    p.close()
    p.join()
    return result

from typing import Callable, Union
from contextlib import contextmanager
@contextmanager
def redirected_output(exitCallback: Union[None, Callable[[str], None]] = None) -> None:
    import sys
    from io import StringIO
    original_stdout = sys.stdout
    temporary_std_out = StringIO()
    sys.stdout = temporary_std_out
    yield
    sys.stdout = original_stdout
    printed_output: str = temporary_std_out.getvalue()
    if exitCallback is not None:
        exitCallback(printed_output)
    return

from contextlib import contextmanager
@contextmanager
def temp_plt_figure(*args, **kwargs):
    import matplotlib.pyplot as plt
    figure = plt.figure(*args, **kwargs)
    yield figure
    plt.close(figure)

from typing import Callable
from contextlib import contextmanager
@contextmanager
def timeout(time: float, functionToExecuteOnTimeout: Callable[[], None] = None):
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

from typing import Callable
from contextlib import contextmanager
@contextmanager
def timer(section_name: str = None, exitCallback: Callable[[], None] = None):
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

from typing import Iterable
from collections import Counter
def histogram(iterator: Iterable) -> Counter:
    from collections import Counter
    counter = Counter()
    for element in iterator:
        counter[element]+=1
    return counter

import io
from contextlib import contextmanager
@contextmanager
def std_out(stream: io.TextIOWrapper) -> None:
    import sys
    original_std_out = sys.stdout
    sys.stdout = stream
    yield
    sys.stdout = original_std_out
    return

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

from typing import Callable, Union
from tqdm import tqdm
def tqdm_with_message(iterable,
                      pre_yield_message_func: Union[None, Callable[[int], str]] = None,
                      post_yield_message_func: Union[None, Callable[[int], str]] = None,
                      *args, **kwargs):
    progress_bar_iterator = tqdm(iterable, *args, **kwargs)
    for index, element in enumerate(progress_bar_iterator):
        if pre_yield_message_func is not None:
            pre_yield_message = pre_yield_message_func(index)
            progress_bar_iterator.set_description(pre_yield_message)
            progress_bar_iterator.refresh()
        yield element
        if post_yield_message_func is not None:
            post_yield_message = post_yield_message_func(index)
            progress_bar_iterator.set_description(post_yield_message)
            progress_bar_iterator.refresh()
