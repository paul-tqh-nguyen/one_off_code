import pytest
import numpy as np
from uuid import uuid4
from contextlib import contextmanager
from typing import List, Union, Generator, Callable

import sys ; sys.path.append('..')
import autograd
from autograd import Variable, VariableOperand

################
# Test Helpers #
################

@contextmanager
def temp_numpy_funcs(*temp_funcs: List[Callable]) -> Generator:
    for temp_func in temp_funcs:
        assert not hasattr(np, temp_func.__name__)
        setattr(np, temp_func.__name__, temp_func)
        assert hasattr(np, temp_func.__name__)
    yield
    for temp_func in temp_funcs:
        delattr(np, temp_func.__name__)
        assert not hasattr(np, temp_func.__name__)
    return

###############################
# Tests for numpy_replacement #
###############################

def test_basic_numpy_replacement():
    def mult_ten(operand: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return operand*10
    
    with temp_numpy_funcs(mult_ten):
        
        assert np.all(np.mult_ten(np.ones(4)) == np.full([4], 10))
        
        @Variable.numpy_replacement(np_mult_ten='np.mult_ten')
        def mult_ten(operand: VariableOperand, np_mult_ten: Callable) -> np.ndarray:
            if isinstance(operand, Variable):
                return Variable(np_mult_ten(operand.data))
            else:
                return np_mult_ten(operand)

        assert np.mult_ten.__name__ == 'mult_ten'

        # Verify numpy still works
        assert np.all(np.mult_ten(np.ones(4)) == np.full([4], 10))

        # Verify 1-D arrays
        var = Variable(np.arange(3))
        assert np.all(np.mult_ten(var).data == np.array([00, 10, 20]))

        # Verify 2-D arrays
        var = Variable(np.arange(4).reshape([2,2]))
        assert np.all(np.mult_ten(var).data == np.array([[00, 10], [20, 30]]))

        # Verify 3-D arrays
        var = Variable(np.arange(8).reshape([2,2,2]))
        assert np.all(np.mult_ten(var).data == np.array([[[00, 10], [20, 30]], [[40, 50], [60, 70]]]))

def test_numpy_replacement_fails_on_multiple_inputs():
    def mult_ten(operand: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return operand*10
    
    def mult_five(operand: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return operand*5
    
    with temp_numpy_funcs(mult_ten, mult_five):
        
        assert np.all(np.mult_ten(np.ones(4)) == np.full([4], 10))
        assert np.all(np.mult_five(np.ones(8)) == np.full([8], 5))

        with pytest.raises(ValueError, match='Only one numpy callable can be replaced.'):
            @Variable.numpy_replacement(np_mult_ten='np.mult_ten', np_mult_five='np.mult_five')
            def mult_ten(operand: VariableOperand, np_mult_ten: Callable) -> np.ndarray:
                if isinstance(operand, Variable):
                    return Variable(np_mult_ten(operand.data))
                else:
                    return np_mult_ten(operand)

def test_numpy_replacement_fails_on_bogus_internally_used_name():
    def mult_ten(operand: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return operand*10
        
    with temp_numpy_funcs(mult_ten):
        
        assert np.all(np.mult_ten(np.ones(4)) == np.full([4], 10))

        with pytest.raises(ValueError, match='is not a vaild identifier name.'):
            @Variable.numpy_replacement(**{'np\mult/ten': 'np.mult_ten'})
            def mult_ten(operand: VariableOperand, np_mult_ten: Callable) -> np.ndarray:
                if isinstance(operand, Variable):
                    return Variable(np_mult_ten(operand.data))
                else:
                    return np_mult_ten(operand)
 
def test_numpy_replacement_fails_on_bogus_numpy_names():
    def mult_ten(operand: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return operand*10
        
    with temp_numpy_funcs(mult_ten):
        with pytest.raises(ValueError, match='does not specify a numpy function.'):
            @Variable.numpy_replacement(np_mult_ten='np.non.existent.path.mult_ten')
            def mult_ten(*args, **kwargs) -> None:
                pass

    with temp_numpy_funcs(mult_ten):
        with pytest.raises(ValueError, match='does not specify a numpy function.'):
            @Variable.numpy_replacement(np_mult_ten='np.non_existent_name')
            def mult_ten(*args, **kwargs) -> None:
                pass
    
    with temp_numpy_funcs(mult_ten):
        with pytest.raises(ValueError, match='does not specify a numpy function.'):
            @Variable.numpy_replacement(np_mult_ten='')
            def mult_ten(*args, **kwargs) -> None:
                pass
    
    with temp_numpy_funcs(mult_ten):
        with pytest.raises(ValueError, match='does not specify a numpy function.'):
            @Variable.numpy_replacement(np_mult_ten='np.bogus\name\with\slashes')
            def mult_ten(*args, **kwargs) -> None:
                pass 

###################################
# Tests for differentiable_method #
###################################

def test_differentiable_method_unary_no_name():
    def mult_ten(operand: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return operand*10
    
    with temp_numpy_funcs(mult_ten):
        
        assert np.all(np.mult_ten(np.ones(4)) == np.full([4], 10))
        
        @Variable.differentiable_method()
        def mult_ten(operand: VariableOperand) -> np.ndarray:
            if isinstance(operand, Variable):
                return Variable(np.mult_ten(operand.data))
            else:
                return np.mult_ten(operand)
        
        # Verify 1-D arrays
        var = Variable(np.arange(3))
        assert np.all(var.mult_ten().data == np.array([00, 10, 20]))

        # Verify 2-D arrays
        var = Variable(np.arange(4).reshape([2,2]))
        assert np.all(var.mult_ten().data == np.array([[00, 10], [20, 30]]))

        # Verify 3-D arrays
        var = Variable(np.arange(8).reshape([2,2,2]))
        assert np.all(var.mult_ten().data == np.array([[[00, 10], [20, 30]], [[40, 50], [60, 70]]]))

def test_differentiable_method_unary_one_name():
    def mult_ten(operand: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return operand*10
    
    with temp_numpy_funcs(mult_ten):
        
        assert np.all(np.mult_ten(np.ones(4)) == np.full([4], 10))
        
        @Variable.differentiable_method('mult_ten_special_name')
        def mult_ten(operand: VariableOperand) -> np.ndarray:
            if isinstance(operand, Variable):
                return Variable(np.mult_ten(operand.data))
            else:
                return np.mult_ten(operand)
        
        # Verify 1-D arrays
        var = Variable(np.arange(3))
        assert np.all(var.mult_ten_special_name().data == np.array([00, 10, 20]))

        # Verify 2-D arrays
        var = Variable(np.arange(4).reshape([2,2]))
        assert np.all(var.mult_ten_special_name().data == np.array([[00, 10], [20, 30]]))

        # Verify 3-D arrays
        var = Variable(np.arange(8).reshape([2,2,2]))
        assert np.all(var.mult_ten_special_name().data == np.array([[[00, 10], [20, 30]], [[40, 50], [60, 70]]]))

def test_differentiable_method_unary_two_names():
    def mult_ten(operand: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return operand*10
    
    with temp_numpy_funcs(mult_ten):
        
        assert np.all(np.mult_ten(np.ones(4)) == np.full([4], 10))
        
        @Variable.differentiable_method('mult_ten_first', 'mult_ten_second')
        def mult_ten(operand: VariableOperand) -> np.ndarray:
            if isinstance(operand, Variable):
                return Variable(np.mult_ten(operand.data))
            else:
                return np.mult_ten(operand)
        
        # Verify 1-D arrays
        var = Variable(np.arange(3))
        assert np.all(var.mult_ten_first().data == np.array([00, 10, 20]))
        assert np.all(var.mult_ten_second().data == np.array([00, 10, 20]))

        # Verify 2-D arrays
        var = Variable(np.arange(4).reshape([2,2]))
        assert np.all(var.mult_ten_first().data == np.array([[00, 10], [20, 30]]))
        assert np.all(var.mult_ten_second().data == np.array([[00, 10], [20, 30]]))

        # Verify 3-D arrays
        var = Variable(np.arange(8).reshape([2,2,2]))
        assert np.all(var.mult_ten_first().data == np.array([[[00, 10], [20, 30]], [[40, 50], [60, 70]]]))
        assert np.all(var.mult_ten_second().data == np.array([[[00, 10], [20, 30]], [[40, 50], [60, 70]]]))

def test_differentiable_method_binary_no_name():
    def multiply_then_halve(a: Union[int, float, np.number, np.ndarray], b: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return (a*b)/2
    
    with temp_numpy_funcs(multiply_then_halve):
        
        assert np.all(np.multiply_then_halve(np.ones(4), np.arange(4)) == np.array([0, 0.5, 2, 1.5]))
        
        @Variable.differentiable_method()
        def multiply_then_halve(a: Union[int, float, np.number, np.ndarray], b: Union[int, float, np.number, np.ndarray]) -> np.ndarray:
            a_is_var = isinstance(a, Variable)
            b_is_var = isinstance(b, Variable)
            if a_is_var and b_is_var:
                return Variable(np.multiply_then_halve(a.data, b.data))
            elif a_is_var:
                return Variable(np.multiply_then_halve(a.data, b))
            elif b_is_var:
                return Variable(np.multiply_then_halve(a, b.data))
            else:
                return np.multiply_then_halve(a, b)
        
        var_a = Variable(10)
        var_b = Variable(5)
        assert np.all(var_a.multiply_then_halve(var_b).data == 25)

# def test_differentiable_method_binary_one_name():
#     def multiply_then_halve(a: Union[int, float, np.number, np.ndarray], b: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
#         return (a*b)/2
    
#     with temp_numpy_funcs(multiply_then_halve):
        
        
# def test_differentiable_method_binary_two_names():
#     def multiply_then_halve(a: Union[int, float, np.number, np.ndarray], b: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
#         return (a*b)/2
    
#     with temp_numpy_funcs(multiply_then_halve):
        
