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

    # Verify Trainability
    x = Variable(np.random.rand(2))
    sgd = autograd.optimizer.SGD(learning_rate=1e-4)
    for _ in range(50):
        y = x.dot(np.array([-10, 50]))
        y_hat = 0
        diff = np.subtract(y, y_hat)
        loss = diff ** 2
        sgd.take_training_step(loss)
        if 0 < loss.data < 1e-3:
            break
    assert 0 < loss.data < 1e-3

def test_variable_multiply():
    a_array = np.arange(5)
    b_array = np.array([3, 8, 5, 6, 8])
    a = Variable(np.arange(5, dtype=float))
    b = Variable(np.array([3, 8, 5, 6, 8], dtype=float))
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

    # Verify Trainability
    x = Variable(np.random.rand(2))
    sgd = autograd.optimizer.SGD(learning_rate=1e-4)
    for _ in range(1_000):
        y = x.multiply(np.array([-10, 50]))
        y_hat = 0
        diff = np.subtract(y, y_hat)
        loss = diff ** 2
        sgd.take_training_step(loss)
        if np.all(loss.data < 1e-3):
            break
    assert np.abs(x.data).sum() < 5e-3
    assert np.all(loss.data < 1e-3)

def test_variable_subtract():
    a_array = np.arange(5)
    b_array = np.array([3, 8, 5, 6, 8])
    a = Variable(np.arange(5, dtype=float))
    b = Variable(np.array([3, 8, 5, 6, 8], dtype=float))
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
    
    # Verify Derivative
    sgd = autograd.optimizer.SGD(learning_rate=1e-3)
    difference = a-b
    variable_to_gradient = sgd.take_training_step(difference)
    assert np.all(variable_to_gradient[a] == np.ones(a.shape))
    assert np.all(variable_to_gradient[b] == np.full(b.shape, -1))

    # Verify Trainability
    x = Variable(np.random.rand(2))
    sgd = autograd.optimizer.SGD(learning_rate=1e-1)
    for _ in range(1_000):
        y = x.subtract(np.array([-10, 50]))
        y_hat = np.array([10, 10])
        diff = np.subtract(y, y_hat)
        loss = diff ** 2
        sgd.take_training_step(loss)
        if loss.data.sum() < 1e-10:
            break
    assert np.abs(x.data - np.array([0, 60])).sum() < 1
    assert loss.data.sum() < 1e-10

def test_variable_pow():
    a_array = np.arange(5, dtype=float)+1
    b_array = np.array([0, 2, 2, 3, 3], dtype=float)
    a = Variable(np.arange(5, dtype=float)+1)
    b = Variable(np.array([0, 2, 2, 3, 3], dtype=float))
    expected_result_variable = Variable(np.array([1, 4, 9, 64, 125]))
    expected_result_array = np.array([1, 4, 9, 64, 125])
    
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
    
    # Verify Derivative
    sgd = autograd.optimizer.SGD(learning_rate=1e-3)
    result = a**b
    variable_to_gradient = sgd.take_training_step(result)
    assert np.all(variable_to_gradient[a] == b_array*(a_array**(b_array-1)))
    assert np.all(variable_to_gradient[b] == np.log(a_array)*(a_array**b_array))

    # Verify Trainability (Base)
    x = Variable(np.random.rand(2))
    sgd = autograd.optimizer.SGD(learning_rate=1e-4)
    for _ in range(10_000):
        y = x.pow(np.array([2, 3]))
        y_hat = np.array([100, 8])
        diff = np.subtract(y, y_hat)
        loss = diff ** 2
        variable_to_gradient = sgd.take_training_step(loss)
        if loss.data.sum() < 1e-4:
            break
    assert np.abs(x.data - np.array([10, 2])).sum() < 2e-3
    assert loss.data.sum() < 1e-4

    # Verify Trainability (Exponent)
    x = Variable(np.array([1.9, 2.9], dtype=float))
    sgd = autograd.optimizer.SGD(learning_rate=1e-3)
    for _ in range(1_000):
        y = np.float_power(np.array([3, 2], dtype=float), x)
        y_hat = np.array([9, 8])
        diff = np.subtract(y, y_hat)
        loss = diff ** 2
        sgd.take_training_step(loss)
        if loss.data.sum() < 1e-4:
            break
    assert np.abs(x.data - np.array([2, 3])).sum() < 9e-3
    assert loss.data.sum() < 1e-4

def test_variable_add():
    a_array = np.arange(5)
    b_array = np.array([3, 8, 5, 6, 8])
    a = Variable(np.arange(5, dtype=float))
    b = Variable(np.array([3, 8, 5, 6, 8], dtype=float))
    expected_result_variable = Variable(np.array([3, 9,7, 9, 12]))
    expected_result_array = np.array([3, 9,7, 9, 12])
    
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
    validate_variable_result(a.add(b))
    validate_variable_result(a + b)
    validate_variable_result(np.add(a, b))
    
    # nupmy + numpy
    validate_array_result(np.add(a_array, b_array))
    validate_array_result(a_array + b_array)
    
    # Variable + numpy
    validate_variable_result(a.add(b_array))
    validate_variable_result(a + b_array)
    validate_variable_result(np.add(a, b_array))
    
    # numpy + Variable
    validate_variable_result(np.add(a_array, b))
    # validate_variable_result(a_array + b) # @todo make this work
    
    # Verify Derivative
    sgd = autograd.optimizer.SGD(learning_rate=1e-3)
    difference = a+b
    variable_to_gradient = sgd.take_training_step(difference)
    assert np.all(variable_to_gradient[a] == np.ones(a.shape))
    assert np.all(variable_to_gradient[b] == np.ones(b.shape))

    # Verify Trainability
    x = Variable(np.random.rand(2))
    sgd = autograd.optimizer.SGD(learning_rate=1e-1)
    for _ in range(1_000):
        y = x.add(np.array([-10, 50]))
        y_hat = np.array([10, 10])
        diff = np.subtract(y, y_hat)
        loss = diff ** 2
        sgd.take_training_step(loss)
        if loss.data.sum() < 1e-10:
            break
    assert np.abs(x.data - np.array([20, -40])).sum() < 1
    assert loss.data.sum() < 1e-10

def test_variable_sum():
    a_array = np.arange(5)
    a = Variable(np.arange(5, dtype=float))
    expected_result_variable = Variable(10)
    expected_result_number = 10
    
    assert np.all(a_array == a.data)
    assert np.all(expected_result_variable == expected_result_number)
    
    assert id(a_array) != id(a.data)
    assert id(expected_result_variable) != id(expected_result_number)

    def validate_variable_result(result) -> None:
        assert result.eq(expected_result_variable).all()
        assert isinstance(result, Variable)
        return

    def validate_array_result(result) -> None:
        assert np.all(result == expected_result_number)
        assert isinstance(result, np.ndarray)
        return
    
    # Variable
    validate_variable_result(a.sum())
    validate_variable_result(np.sum(a))
    
    # nupmy + numpy
    validate_array_result(np.sum(a_array))
    validate_array_result(a_array.sum())
        
    # Verify Derivative
    sgd = autograd.optimizer.SGD(learning_rate=1e-3)
    difference = a+b
    variable_to_gradient = sgd.take_training_step(difference)
    assert np.all(variable_to_gradient[a] == np.ones(a.shape))
    assert np.all(variable_to_gradient[b] == np.ones(b.shape))

    # Verify Trainability
    x = Variable(np.random.rand(2))
    sgd = autograd.optimizer.SGD(learning_rate=1e-1)
    for _ in range(1_000):
        y = x.sum(np.array([-10, 50]))
        y_hat = np.array([10, 10])
        diff = np.subtract(y, y_hat)
        loss = diff ** 2
        sgd.take_training_step(loss)
        if loss.data.sum() < 1e-10:
            break
    assert np.abs(x.data - np.array([20, -40])).sum() < 1
    assert loss.data.sum() < 1e-10
