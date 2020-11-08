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
def dummy_numpy_func(dummy_func: Callable) -> Generator:
    assert not hasattr(np, dummy_func.__qualname__)
    yield
    delattr(np, dummy_func.__qualname__)
    assert not hasattr(np, dummy_func.__qualname__)
    return

#########
# Tests #
#########

def test_differentiable_method():
    with dummy_numpy_func_name('dummy_func'):
        
        def dummy_func(array: np.ndarray) -> np.ndarray:
            return array*10
        assert dummy_func.__qualname__ == dummy_func_name
