#!/usr/bin/python3

from typing import Callable, Union
from contextlib import contextmanager
@contextmanager
def redirected_output(exitCallback: Union[None, Callable[[str], None]] = None) -> None:
    import sys
    print('redirected_output 1')
    from io import StringIO
    print('redirected_output 2')
    original_stdout = sys.stdout
    print('redirected_output 3')
    temporary_std_out = StringIO()
    print('redirected_output 4')
    sys.stdout = temporary_std_out
    print('redirected_output 5')
    yield
    print('redirected_output 6')
    sys.stdout = original_stdout
    print('redirected_output 7')
    printed_output: str = temporary_std_out.getvalue()
    print('redirected_output 8')
    print(f"printed_output {printed_output}")
    if exitCallback is not None:
        print('redirected_output 8.5')
        exitCallback(printed_output)
    print('redirected_output 9')
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

TRACE_INDENT_LEVEL = 0
TRACE_INDENTATION = '    '
from typing import Callable
def trace(func: Callable) -> Callable:
    from inspect import signature
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
