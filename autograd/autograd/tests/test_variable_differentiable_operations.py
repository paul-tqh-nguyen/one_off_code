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

    def validate_result(result, expected_type) -> None:
        assert result == expected_result
        assert isinstance(result, expected_type)
        return
    
    # Variable + Variable
    assert validate_result(a.dot(b) == expected_result
    assert validate_result(np.dot(a, b) == expected_result
    
    # nupmy + numpy
    assert validate_result(np.dot(a_array, b_array) == expected_result
    assert validate_result(np.ndarray.dot(a_array, b_array) == expected_result
    assert validate_result(a_array.dot(b_array) == expected_result
    
    # Variable + numpy
    assert validate_result(a.dot(b_array) == expected_result
    assert validate_result(np.dot(a, b_array) == expected_result
    
    # numpy + Variable
    assert validate_result(np.dot(a, b) == expected_result
    assert validate_result(np.ndarray.dot(a, b) == expected_result
    assert validate_result(a_array.dot(b) == expected_result
    
