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
    assert not hasattr(np, temp_func.__name__)
    setattr(np, temp_func.__name__, temp_func)
    assert hasattr(np, temp_func.__name__)
    yield
    delattr(np, temp_func.__name__)
    assert not hasattr(np, temp_func.__name__)
    return

#########
# Tests #
#########

def test_basic_numpy_replacement():
    def mult_ten(operand: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return operand*10
    with temp_numpy_func(mult_ten):
        
        assert np.all(np.mult_ten(np.ones(4)) == np.full([4], 10))
        
        @Variable.numpy_replacement(np_mult_ten='np.mult_ten') # @todo test these numpy methods
        def mult_ten(operand: VariableOperand, np_mult_ten: Callable) -> np.ndarray:
            if isinstance(operand, Variable):
                return Variable(np_mult_ten(operand.data))
            else:
                return np_mult_ten(operand)

        # Verify numpy still works
        assert np.all(np.mult_ten(np.ones(4)) == np.full([4], 10))

        # Verify 1-D arrays
        var = Variable(np.arange(3))
        assert np.all(np.mult_ten(var).data == np.array([00, 10, 20]))

        # Verify 2-D arrays
        var = Variable(np.arange(4).reshape([2,2]))
        assert np.all(np.mult_ten(var).data == np.array([[00, 10], [20, 30]]))

        # Verify 3-D arrays
        var = Variable(np.arange(8).reshape([2,2]))
        assert np.all(np.mult_ten(var).data == np.array([[00, 10], [20, 30]]))
