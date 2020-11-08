import pytest
import numpy as np
from uuid import uuid4
from contextlib import contextmanager
from typing import Union, Generator, Callable

import sys ; sys.path.append("..")
import autograd
from autograd import Variable, VariableOperand

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

def test_basic_numpy_replacement():
    def mult_ten(array: np.ndarray) -> np.ndarray:
        return array*10
    with temp_numpy_func(mult_ten):
        assert np.all(np.mult_ten(np.ones(4)) == np.full([4], 10))
        @Variable.numpy_replacement(np_mult_ten='np.mult_ten') # @todo test these numpy methods
        def new_mult_ten(array: VariableOperand) -> np.ndarray:
            
            return array*10
