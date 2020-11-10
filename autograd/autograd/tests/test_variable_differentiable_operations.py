import pytest
import numpy as np

import sys ; sys.path.append('..')
import autograd
from autograd import Variable

def test_variable_dot():
    a_array = np.arange(5)
    b_array = np.array([3, 8, 5, 6, 8])
    a = Variable(np.arange(5, dtype=float))
    b = Variable(np.array([3, 8, 5, 6, 8], dtype=float))
    expected_result = 68

    assert np.all(a_array == a.data)
    assert np.all(b_array == b.data)
    assert id(a_array) != id(a.data)
    assert id(b_array) != id(b.data)

    def validate_result(result, expected_type: type) -> None:
        assert result == expected_result
        assert isinstance(result, expected_type)
        return
    
    # Variable + Variable
    validate_result(a.dot(b), Variable)
    validate_result(np.dot(a, b), Variable)
    
    # nupmy + numpy
    validate_result(np.dot(a_array, b_array), np.int64)
    validate_result(np.ndarray.dot(a_array, b_array), np.int64)
    validate_result(a_array.dot(b_array), np.int64)
    
    # Variable + numpy
    validate_result(a.dot(b_array), Variable)
    validate_result(np.dot(a, b_array), Variable)
    
    # numpy + Variable
    validate_result(np.dot(a_array, b), Variable)
    validate_result(np.ndarray.dot(a_array, b), Variable)
    validate_result(a_array.dot(b), Variable)

    # Verify Derivative
    sgd = autograd.optimizer.SGD(learning_rate=1e-3)
    dot_product = a.dot(b)
    variable_to_gradient = sgd.take_training_step(dot_product)
    assert np.all(variable_to_gradient[a] == b_array)
    assert np.all(variable_to_gradient[b] == a_array)

def test_variable_multiply():
    a_array = np.arange(5)
    b_array = np.array([3, 8, 5, 6, 8])
    a = Variable(np.arange(5))
    b = Variable(np.array([3, 8, 5, 6, 8]))
    expected_result_variable = Variable(np.array([0, 8, 10, 18, 32]))
    expected_result_array = np.array([0, 8, 10, 18, 32])
    
    assert np.all(a_array == a.data)
    assert np.all(b_array == b.data)
    assert np.all(expected_result_variable == expected_result_array)
    
    assert id(a_array) != id(a.data)
    assert id(b_array) != id(b.data)
    assert id(expected_result_variable) != id(expected_result_array)

    def validate_variable_result(result) -> None:
        assert result.eq(expected_result_variable).all()
        assert isinstance(result, Variable)
        return

    def validate_array_result(result) -> None:
        assert np.all(result == expected_result_array)
        assert isinstance(result, np.ndarray)
        return
        
    # Variable + Variable
    validate_variable_result(a.multiply(b))
    validate_variable_result(a * b)
    validate_variable_result(np.multiply(a, b))
    
    # nupmy + numpy
    validate_array_result(np.multiply(a_array, b_array))
    validate_array_result(a_array * b_array)
    
    # Variable + numpy
    validate_variable_result(a.multiply(b_array))
    validate_variable_result(a * b_array)
    validate_variable_result(np.multiply(a, b_array))
    
    # numpy + Variable
    validate_variable_result(np.multiply(a_array, b))
    # validate_variable_result(a_array * b) # @todo make this work

    # Verify Derivative
    sgd = autograd.optimizer.SGD(learning_rate=1e-3)
    product = a*b
    variable_to_gradient = sgd.take_training_step(product)
    assert np.all(variable_to_gradient[a] == b_array)
    assert np.all(variable_to_gradient[b] == a_array)

def test_variable_subtract():
    a_array = np.arange(5)
    b_array = np.array([3, 8, 5, 6, 8])
    a = Variable(np.arange(5))
    b = Variable(np.array([3, 8, 5, 6, 8]))
    expected_result_variable = Variable(np.array([-3, -7, -3, -3, -4]))
    expected_result_array = np.array([-3, -7, -3, -3, -4])
    
    assert np.all(a_array == a.data)
    assert np.all(b_array == b.data)
    assert np.all(expected_result_variable == expected_result_array)
    
    assert id(a_array) != id(a.data)
    assert id(b_array) != id(b.data)
    assert id(expected_result_variable) != id(expected_result_array)

    def validate_variable_result(result) -> None:
        assert result.eq(expected_result_variable).all()
        assert isinstance(result, Variable)
        return

    def validate_array_result(result) -> None:
        assert np.all(result == expected_result_array)
        assert isinstance(result, np.ndarray)
        return
        
    # Variable + Variable
    validate_variable_result(a.subtract(b))
    validate_variable_result(a - b)
    validate_variable_result(np.subtract(a, b))
    
    # nupmy + numpy
    validate_array_result(np.subtract(a_array, b_array))
    validate_array_result(a_array - b_array)
    
    # Variable + numpy
    validate_variable_result(a.subtract(b_array))
    validate_variable_result(a - b_array)
    validate_variable_result(np.subtract(a, b_array))
    
    # numpy + Variable
    validate_variable_result(np.subtract(a_array, b))
    # validate_variable_result(a_array - b) # @todo make this work

def test_variable_pow():
    a_array = np.arange(5)
    b_array = np.array([0, 2, 2, 3, 3])
    a = Variable(np.arange(5))
    b = Variable(np.array(([0, 2, 2, 3, 3])))
    expected_result_variable = Variable(np.array([1, 1, 4, 27, 64]))
    expected_result_array = np.array([1, 1, 4, 27, 64])
    
    assert np.all(a_array == a.data)
    assert np.all(b_array == b.data)
    assert np.all(expected_result_variable == expected_result_array)
    
    assert id(a_array) != id(a.data)
    assert id(b_array) != id(b.data)
    assert id(expected_result_variable) != id(expected_result_array)

    def validate_variable_result(result) -> None:
        assert result.eq(expected_result_variable).all()
        assert isinstance(result, Variable)
        return

    def validate_array_result(result) -> None:
        assert np.all(result == expected_result_array)
        assert isinstance(result, np.ndarray)
        return
        
    # Variable + Variable
    validate_variable_result(a.power(b))
    validate_variable_result(a.pow(b))
    validate_variable_result(a ** b)
    validate_variable_result(np.float_power(a, b))
    
    # nupmy + numpy
    validate_array_result(np.float_power(a_array, b_array))
    validate_array_result(a_array ** b_array)
    
    # Variable + numpy
    validate_variable_result(a.power(b_array))
    validate_variable_result(a ** b_array)
    validate_variable_result(np.float_power(a, b_array))
    
    # numpy + Variable
    validate_variable_result(np.float_power(a_array, b))
    # validate_variable_result(a_array ** b) # @todo make this work
