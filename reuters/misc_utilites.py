#!/usr/bin/python3 -OO

"""
"""
# @todo figure out this doc string

###########
# Imports #
###########

import os
import sys
import traceback
import pdb
import time
from tqdm import tqdm
from contextlib import contextmanager
from typing import Iterable, List, Callable

###################
# Misc. Utilities #
###################

def only_one(items: List):
    assert isinstance(items, list)
    assert len(items) == 1
    return items[0]

def at_most_one(items: List):
    return only_one(items) if items else None

def eager_map(func: Callable, iterable: Iterable) -> List:
    return list(map(func, iterable))

def implies(antecedent: bool, consequent: bool) -> bool:
    return not antecedent or consequent

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

def _dummy_tqdm_message_func(index: int):
    return ''

def tqdm_with_message(iterable,
                      pre_yield_message_func: Callable[[int], str] = _dummy_tqdm_message_func,
                      post_yield_message_func: Callable[[int], str] = _dummy_tqdm_message_func,
                      *args, **kwargs):
    progress_bar_iterator = tqdm(iterable, *args, **kwargs)
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

if __name__ == '__main__':
    print("This file contains miscellaneous utilities.")
