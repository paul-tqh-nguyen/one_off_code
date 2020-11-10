import pytest
import numpy as np

import sys ; sys.path.append('..')
import autograd
from autograd import Variable

def test_variable_dot():
    a_array = np.arange(5)
    b_array = np.array([3, 8, 5, 6, 8])
    a = Variable(np.arange(5))
    b = Variable(np.array([3, 8, 5, 6, 8]))
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
    validate_result(np.dot(a, b), Variable)
    validate_result(np.ndarray.dot(a, b), Variable)
    validate_result(a_array.dot(b), Variable)    

def test_variable_subtract():
    a_array = np.arange(5)
    b_array = np.array([3, 8, 5, 6, 8])
    a = Variable(np.arange(5))
    b = Variable(np.array([3, 8, 5, 6, 8]))
    expected_result_variable = np.array([-3. 7, 3, 2, 3])
    expected_result_array = np.array([-3. 7, 3, 2, 3])
    
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
    validate_result(np.dot(a, b), Variable)
    validate_result(np.ndarray.dot(a, b), Variable)
    validate_result(a_array.dot(b), Variable)
    
