#!/usr/bin/python3

import time
from typing import Callable, Iterable
from contextlib import contextmanager
from collections import Counter

@contextmanager
def timer(section_name: str = None, exitCallback: Callable[[], None] = None):
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

def histogram(iterator: Iterable) -> Counter:
    counter = Counter()
    for element in iterator:
        counter[element]+=1
    return counter
