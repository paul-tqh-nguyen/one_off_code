import pytest
import numpy as np
from uuid import uuid4
from contextlib import contextmanager

import sys ; sys.path.append("..")
import autograd
from autograd import Variable

@contextmanager
def dummy_numpy_operation(dummy_func_name: str) -> Generator:
    yield
    return

def test_differentiable_method():
    unique_operation_name = f'dummy_func'
    assert not hasattr(np, unique_operation_name)

    def dummy_func(array: np.ndarray) -> np.ndarray:
        return array*10    
    assert dummy_func.__qualname__ == unique_operation_name
    
    setattr(np, unique_operation_name, dummy_func)
    assert np.dummy_func(np.
    
    delattr(np, unique_operation_name)
    assert not hasattr(np, unique_operation_name)
    
