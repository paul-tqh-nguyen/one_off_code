import pytest
import numpy as np
from uuid import uuid4

import sys ; sys.path.append("..")
import autograd
from autograd import Variable

def test_():
    def dummy_func(array: np.ndarray) -> np.ndarray:
        return array*10
    
    assert not hasattr(np, dummy_func.__qualname__)
    setattr(np, unique_operation_name, dummy_func)
    assert np.dummy_func(np.
    
    delattr(np, unique_operation_name)
    assert not hasattr(np, dummy_func.__qualname__)
