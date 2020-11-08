import pytest
import numpy as np
from uuid import uuid4
from contextlib import contextmanager

import sys ; sys.path.append("..")
import autograd
from autograd import Variable

@contextmanager
def dummy_numpy_func(dummy_func_name: str) -> Generator:
    assert not hasattr(np, dummy_func)
    yield
    return

def test_differentiable_method():
    with dummy_numpy_func
    def dummy_func(array: np.ndarray) -> np.ndarray:
        return array*10    
    assert dummy_func.__qualname__ == dummy_func
    
    setattr(np, dummy_func, dummy_func)
    assert np.dummy_func(np.
    
    delattr(np, dummy_func)
    assert not hasattr(np, dummy_func)
    
