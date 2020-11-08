import pytest
import numpy as np
from uuid import uuid4
from contextlib import contextmanager

import sys ; sys.path.append("..")
import autograd
from autograd import Variable

################
# Test Helpers #
################

@contextmanager
def dummy_numpy_func_name(dummy_func_name: str) -> Generator:
    '''Only does checking and removal ; does not do any adding.'''
    assert not hasattr(np, dummy_func_name)
    yield
    if hasattr(np, dummy_func_name):
        delattr(np, dummy_func_name)
    assert not hasattr(np, dummy_func_name)
    return

#########
# Tests #
#########

def test_differentiable_method():
    with dummy_numpy_func_name('dummy_func'):
        
        def dummy_func(array: np.ndarray) -> np.ndarray:
            return array*10
        assert dummy_func.__qualname__ == dummy_func_name
