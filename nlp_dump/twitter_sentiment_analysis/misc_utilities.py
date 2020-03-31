#!/usr/bin/python3 -O

"""

This is a module of miscellaneous utilities.

Owner : paul-tqh-nguyen

Created : 12/13/2019

File Name : misc_utilities.py

File Organization:
* Imports
* Misc. Utilities
"""

###########
# Imports #
###########

import signal
import time
import os
import socket
from contextlib import contextmanager
from itertools import chain, combinations

###################
# Misc. Utilities #
###################

LOGGING_FILE = os.path.expanduser("~/Desktop/log_sentiment_analysis.txt")

def logging_print(input='') -> None:
    input_string = str(input)
    with open(LOGGING_FILE, 'a') as f:
        lines = input_string.split('\n')
        lines_with_machine_name_appended = map(lambda line: socket.gethostname()+': '+line, lines)
        f.write('\n'.join(lines_with_machine_name_appended)+'\n')
    print(input_string)
    return None

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

@contextmanager
def timeout(time, functionToExecuteOnTimeout=None):
    """NB: This cannot be nested."""
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
        logging_print('Execution of "{section_name}" took {elapsed_time} seconds.'.format(section_name=section_name, elapsed_time=elapsed_time))
    else:
        logging_print('Execution took {elapsed_time} seconds.'.format(elapsed_time=elapsed_time))

def current_timestamp_string() -> str:
    return time.strftime("%Y_%m_%d_%H_%M_%S")
        
def false(*args, **kwargs):
    return False

def main():
    print("This is a module of miscellaneous utilities.")

if __name__ == '__main__':
    main()
