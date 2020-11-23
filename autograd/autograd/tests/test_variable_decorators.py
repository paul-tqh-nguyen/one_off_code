import pytest
import numpy as np
from contextlib import contextmanager
from typing import List, Union, Generator, Callable

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

@contextmanager
def temp_variable_method_names(*method_names: List[str]) -> Generator:
    for method_name in method_names:
        assert not hasattr(Variable, method_name)
    yield
    for method_name in method_names:
        assert hasattr(Variable, method_name)
        delattr(Variable, method_name)
        assert not hasattr(Variable, method_name)
    return

########################################
# Tests for Variable.numpy_replacement #
########################################

def test_basic_numpy_replacement():
    def mult_ten(operand: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return operand*10
    def multiply_by_ten(operand: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return operand*2*5
    
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
    
    with temp_numpy_funcs(mult_ten):
        
        assert np.all(np.mult_ten(np.ones(4)) == np.full([4], 10))
        
        @Variable.numpy_replacement(np_mult_ten=['np.mult_ten'])
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
    
    with temp_numpy_funcs(mult_ten, multiply_by_ten):
        
        assert np.all(np.mult_ten(np.ones(4)) == np.full([4], 10))
        
        @Variable.numpy_replacement(np_mult_ten=['np.mult_ten', 'np.multiply_by_ten'])
        def mult_ten(operand: VariableOperand, np_mult_ten: Callable) -> np.ndarray:
            if isinstance(operand, Variable):
                return Variable(np_mult_ten(operand.data))
            else:
                return np_mult_ten(operand)

        assert np.mult_ten.__name__ == 'mult_ten'

        # Verify numpy still works
        assert np.all(np.mult_ten(np.ones(4)) == np.full([4], 10))
        assert np.all(np.multiply_by_ten(np.ones(4)) == np.full([4], 10))

        # Verify 1-D arrays
        var = Variable(np.arange(3))
        assert np.all(np.mult_ten(var).data == np.array([00, 10, 20]))
        assert np.all(np.multiply_by_ten(var).data == np.array([00, 10, 20]))

        # Verify 2-D arrays
        var = Variable(np.arange(4).reshape([2,2]))
        assert np.all(np.mult_ten(var).data == np.array([[00, 10], [20, 30]]))
        assert np.all(np.multiply_by_ten(var).data == np.array([[00, 10], [20, 30]]))

        # Verify 3-D arrays
        var = Variable(np.arange(8).reshape([2,2,2]))
        assert np.all(np.mult_ten(var).data == np.array([[[00, 10], [20, 30]], [[40, 50], [60, 70]]]))
        assert np.all(np.multiply_by_ten(var).data == np.array([[[00, 10], [20, 30]], [[40, 50], [60, 70]]]))

def test_numpy_replacement_fails_on_multiple_internally_used_names():
    def mult_ten(operand: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return operand*10
    
    def mult_five(operand: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return operand*5
    
    with temp_numpy_funcs(mult_ten, mult_five):
        
        assert np.all(np.mult_ten(np.ones(4)) == np.full([4], 10))
        assert np.all(np.mult_five(np.ones(8)) == np.full([8], 5))

        with pytest.raises(ValueError, match='Only one internally used name can be specified. 2 were given.'):
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
        with pytest.raises(ValueError, match='"np.non" does not exist.'):
            @Variable.numpy_replacement(np_mult_ten='np.non.existent.path.mult_ten')
            def mult_ten(*args, **kwargs) -> None:
                pass

    with temp_numpy_funcs(mult_ten):
        with pytest.raises(ValueError, match='"np.non_existent_name" does not specify a numpy callable.'):
            @Variable.numpy_replacement(np_mult_ten='np.non_existent_name')
            def mult_ten(*args, **kwargs) -> None:
                pass
    
    with temp_numpy_funcs(mult_ten):
        with pytest.raises(ValueError, match='"" does not specify a numpy callable.'):
            @Variable.numpy_replacement(np_mult_ten='')
            def mult_ten(*args, **kwargs) -> None:
                pass
    
    with temp_numpy_funcs(mult_ten):
        with pytest.raises(ValueError, match='slashes" does not specify a numpy callable.'):
            @Variable.numpy_replacement(np_mult_ten='np.bogus\name\with\slashes')
            def mult_ten(*args, **kwargs) -> None:
                pass
            
    with temp_numpy_funcs(mult_ten):
        with pytest.raises(ValueError, match='slashes" does not specify a numpy callable.'):
            @Variable.numpy_replacement(np_mult_ten=['np.mult_ten', 'np.bogus\name\with\slashes'])
            def mult_ten(*args, **kwargs) -> None:
                pass
    
    with temp_numpy_funcs(mult_ten):
        with pytest.raises(ValueError, match='No numpy callable specified to be replaced.'):
            @Variable.numpy_replacement(np_mult_ten=[])
            def mult_ten(*args, **kwargs) -> None:
                pass
    
    with temp_numpy_funcs(mult_ten):
        with pytest.raises(ValueError, match='1 does not specify a numpy callable.'):
            @Variable.numpy_replacement(np_mult_ten=1)
            def mult_ten(*args, **kwargs) -> None:
                pass
    
    with temp_numpy_funcs(mult_ten):
        with pytest.raises(ValueError, match='\[1\] does not specify a numpy callable.'):
            @Variable.numpy_replacement(np_mult_ten=[1])
            def mult_ten(*args, **kwargs) -> None:
                pass
    
    with temp_numpy_funcs(mult_ten):
        with pytest.raises(ValueError, match='\[1, \'np.mult_ten\'\] does not specify a numpy callable.'):
            @Variable.numpy_replacement(np_mult_ten=[1, 'np.mult_ten'])
            def mult_ten(*args, **kwargs) -> None:
                pass 

#################################
# Tests for Variable.new_method #
#################################

def test_new_method_unary_no_name():
    def _mult_ten(operand: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return operand*10
        
    assert np.all(_mult_ten(np.ones(4)) == np.full([4], 10))
    
    with temp_variable_method_names('mult_ten'):
        
        @Variable.new_method()
        def mult_ten(operand: VariableOperand) -> np.ndarray:
            if isinstance(operand, Variable):
                return Variable(_mult_ten(operand.data))
            else:
                return _mult_ten(operand)
        
        # Verify 1-D arrays
        var = Variable(np.arange(3))
        assert np.all(var.mult_ten().data == np.array([00, 10, 20]))

        # Verify 2-D arrays
        var = Variable(np.arange(4).reshape([2,2]))
        assert np.all(var.mult_ten().data == np.array([[00, 10], [20, 30]]))

        # Verify 3-D arrays
        var = Variable(np.arange(8).reshape([2,2,2]))
        assert np.all(var.mult_ten().data == np.array([[[00, 10], [20, 30]], [[40, 50], [60, 70]]]))

def test_new_method_unary_one_name():
    def _mult_ten(operand: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return operand*10
        
    assert np.all(_mult_ten(np.ones(4)) == np.full([4], 10))
    
    with temp_variable_method_names('mult_ten_special_name'):
        
        @Variable.new_method('mult_ten_special_name')
        def mult_ten(operand: VariableOperand) -> np.ndarray:
            if isinstance(operand, Variable):
                return Variable(_mult_ten(operand.data))
            else:
                return _mult_ten(operand)
        
        # Verify 1-D arrays
        var = Variable(np.arange(3))
        assert np.all(var.mult_ten_special_name().data == np.array([00, 10, 20]))

        # Verify 2-D arrays
        var = Variable(np.arange(4).reshape([2,2]))
        assert np.all(var.mult_ten_special_name().data == np.array([[00, 10], [20, 30]]))

        # Verify 3-D arrays
        var = Variable(np.arange(8).reshape([2,2,2]))
        assert np.all(var.mult_ten_special_name().data == np.array([[[00, 10], [20, 30]], [[40, 50], [60, 70]]]))

def test_new_method_unary_two_names():
    def _mult_ten(operand: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return operand*10
        
    assert np.all(_mult_ten(np.ones(4)) == np.full([4], 10))
    
    with temp_variable_method_names('mult_ten_first', 'mult_ten_second'):
        
        @Variable.new_method('mult_ten_first', 'mult_ten_second')
        def mult_ten(operand: VariableOperand) -> np.ndarray:
            if isinstance(operand, Variable):
                return Variable(_mult_ten(operand.data))
            else:
                return _mult_ten(operand)
        
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

def test_new_method_binary_no_name():
    def _multiply_then_halve(a: Union[int, float, np.number, np.ndarray], b: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return (a*b)/2
        
    assert np.all(_multiply_then_halve(np.ones(4), np.arange(4)) == np.array([0, 0.5, 1, 1.5]))
    
    with temp_variable_method_names('multiply_then_halve'):
        
        @Variable.new_method()
        def multiply_then_halve(a: Union[int, float, np.number, np.ndarray], b: Union[int, float, np.number, np.ndarray]) -> np.ndarray:
            a_is_var = isinstance(a, Variable)
            b_is_var = isinstance(b, Variable)
            if a_is_var and b_is_var:
                return Variable(_multiply_then_halve(a.data, b.data))
            elif a_is_var:
                return Variable(_multiply_then_halve(a.data, b))
            elif b_is_var:
                return Variable(_multiply_then_halve(a, b.data))
            else:
                return _multiply_then_halve(a, b)
        
        var_a = Variable(10)
        var_b = Variable(5)
        assert np.all(var_a.multiply_then_halve(var_b).data == 25)

def test_new_method_binary_one_name():
    def _multiply_then_halve(a: Union[int, float, np.number, np.ndarray], b: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return (a*b)/2
        
    assert np.all(_multiply_then_halve(np.ones(4), np.arange(4)) == np.array([0, 0.5, 1, 1.5]))
    
    with temp_variable_method_names('mth'):
        
        @Variable.new_method('mth')
        def multiply_then_halve(a: Union[int, float, np.number, np.ndarray], b: Union[int, float, np.number, np.ndarray]) -> np.ndarray:
            a_is_var = isinstance(a, Variable)
            b_is_var = isinstance(b, Variable)
            if a_is_var and b_is_var:
                return Variable(_multiply_then_halve(a.data, b.data))
            elif a_is_var:
                return Variable(_multiply_then_halve(a.data, b))
            elif b_is_var:
                return Variable(_multiply_then_halve(a, b.data))
            else:
                return _multiply_then_halve(a, b)
        
        var_a = Variable(10)
        var_b = Variable(5)
        assert np.all(var_a.mth(var_b).data == 25)
        assert 'multiply_then_halve' not in dir(var_a)
        
def test_new_method_binary_two_names():
    def _multiply_then_halve(a: Union[int, float, np.number, np.ndarray], b: Union[int, float, np.number, np.ndarray]) -> Union[int, float, np.number, np.ndarray]:
        return (a*b)/2
        
    assert np.all(_multiply_then_halve(np.ones(4), np.arange(4)) == np.array([0, 0.5, 1, 1.5]))
    
    with temp_variable_method_names('mth', 'multiple_and_then_take_half_after'):
        
        @Variable.new_method('mth', 'multiple_and_then_take_half_after')
        def multiply_then_halve(a: Union[int, float, np.number, np.ndarray], b: Union[int, float, np.number, np.ndarray]) -> np.ndarray:
            a_is_var = isinstance(a, Variable)
            b_is_var = isinstance(b, Variable)
            if a_is_var and b_is_var:
                return Variable(_multiply_then_halve(a.data, b.data))
            elif a_is_var:
                return Variable(_multiply_then_halve(a.data, b))
            elif b_is_var:
                return Variable(_multiply_then_halve(a, b.data))
            else:
                return _multiply_then_halve(a, b)
        
        var_a = Variable(10)
        var_b = Variable(5)
        assert np.all(var_a.mth(var_b).data == 25)
        assert np.all(var_a.multiple_and_then_take_half_after(var_b).data == 25)
        assert 'multiply_then_halve' not in dir(var_a)

def test_numpy_replacement_fails_on_bogus_name():

    with pytest.raises(ValueError, match='is not a valid method name.'):
        @Variable.new_method('bad!name')
        def multiply_then_halve(*args, **kwargs) -> None:
            pass

    with pytest.raises(ValueError, match='is not a valid method name.'):
        @Variable.new_method('good_name', 'bad!name')
        def multiply_then_halve(*args, **kwargs) -> None:
            pass
