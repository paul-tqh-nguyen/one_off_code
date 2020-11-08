import pytest
import numpy as np
from uuid import uuid4
from contextlib import contextmanager
from typing import Generator, Callable

import sys ; sys.path.append("..")
import autograd
from autograd import Variable

################
# Test Helpers #
################

@contextmanager
def temp_numpy_func(temp_func: Callable) -> Generator:
    assert not hasattr(np, temp_func.__qualname__)
    yield
    delattr(np, temp_func.__qualname__)
    assert not hasattr(np, temp_func.__qualname__)
    return

#########
# Tests #
#########

def test_differentiable_method():
    def dummy_func(array: np.ndarray) -> np.ndarray:
        return array*10
    with temp_numpy_func(dummy_func):
        
