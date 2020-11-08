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

def test_basic_numpy_replacement():
    def mult_ten(array: np.ndarray) -> np.ndarray:
        return array*10
    with temp_numpy_func(mult_ten):
        assert np.all(np.mult_ten(np.ones(4)) == np.full([4], 10))
        @Variable.differentiable_method() # @todo test this method
        @Variable.numpy_replacement(np_dot='np.dot') # @todo test these numpy methods
        def dot(a: VariableOperand, b: VariableOperand, np_dot: Callable, **kwargs) -> VariableOperand:
            a_is_variable = isinstance(a, Variable)
            b_is_variable = isinstance(b, Variable)
            a_data = a.data if a_is_variable else a
            b_data = b.data if b_is_variable else b
            dot_product = np_dot(a_data, b_data, **kwargs)
            if not a_is_variable and not b_is_variable:
                return dot_product
            if len(kwargs) > 0:
                raise ValueError(f'The parameters {[repr(kwarg_name) for kwarg_name in kwargs.keys()]} are not supported for {Variable.__qualname__}')
            variable_depended_on_by_dot_product_to_backward_propagation_function = {}
            if a_is_variable:
                variable_depended_on_by_dot_product_to_backward_propagation_function[a] = lambda d_minimization_target_over_d_dot_product: d_minimization_target_over_d_dot_product * b_data
            if b_is_variable:
                variable_depended_on_by_dot_product_to_backward_propagation_function[b] = lambda d_minimization_target_over_d_dot_product: d_minimization_target_over_d_dot_product * a_data
            dot_product_variable = Variable(dot_product, variable_depended_on_by_dot_product_to_backward_propagation_function)
            return dot_product_variable
