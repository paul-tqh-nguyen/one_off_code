import pytest
import numpy as np

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
    
    # numpy + numpy
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
    assert all(isinstance(var, Variable) and isinstance(grad, np.ndarray) for var, grad in variable_to_gradient.items())
    assert np.all(variable_to_gradient[a] == b_array)
    assert np.all(variable_to_gradient[b] == a_array)

    # Verify Trainability
    x = Variable(np.random.rand(2))
    y = 0
    sgd = autograd.optimizer.SGD(learning_rate=1e-4)
    for training_step_index in range(50):
        y_hat = x.dot(np.array([-10, 50]))
        diff = np.subtract(y, y_hat)
        loss = diff ** 2
        if 0 < loss < 1e-3:
            break
        sgd.take_training_step(loss)
    assert 0 < loss < 1e-3

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
    
    # numpy + numpy
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
    assert all(isinstance(var, Variable) and isinstance(grad, np.ndarray) for var, grad in variable_to_gradient.items())
    assert np.all(variable_to_gradient[a] == b_array)
    assert np.all(variable_to_gradient[b] == a_array)

    # Verify Trainability
    x = Variable(np.random.rand(2))
    y = 0
    sgd = autograd.optimizer.SGD(learning_rate=1e-4)
    for training_step_index in range(1_000):
        y_hat = x.multiply(np.array([-10, 50]))
        diff = np.subtract(y, y_hat)
        loss = diff ** 2
        if np.all(loss < 1e-3):
            break
        sgd.take_training_step(loss)
    assert np.abs(x).sum() < 5e-3
    assert np.all(loss < 1e-3)

def test_variable_divide():
    a_array = np.arange(5)
    b_array = np.array([3, 8, 5, 6, 8])
    a = Variable(np.arange(5, dtype=float))
    b = Variable(np.array([3, 8, 5, 6, 8], dtype=float))
    expected_result_variable = Variable(np.array([0, 0.125, 0.4, 0.5, 0.5]))
    expected_result_array = np.array([0, 0.125, 0.4, 0.5, 0.5])
    
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
    validate_variable_result(a.divide(b))
    validate_variable_result(a / b)
    validate_variable_result(np.divide(a, b))
    
    # numpy + numpy
    validate_array_result(np.divide(a_array, b_array))
    validate_array_result(a_array / b_array)
    
    # Variable + numpy
    validate_variable_result(a.divide(b_array))
    validate_variable_result(a / b_array)
    validate_variable_result(np.divide(a, b_array))
    
    # numpy + Variable
    validate_variable_result(np.divide(a_array, b))
    # validate_variable_result(a_array / b) # @todo make this work

    # Verify Derivative
    sgd = autograd.optimizer.SGD(learning_rate=1e-3)
    product = a/b
    variable_to_gradient = sgd.take_training_step(product)
    assert all(isinstance(var, Variable) and isinstance(grad, np.ndarray) for var, grad in variable_to_gradient.items())
    assert np.all(variable_to_gradient[a] == 1/b_array)
    assert np.all(variable_to_gradient[b] == -a_array * (b_array ** -2.0))

    # Verify Trainability
    x = Variable(np.random.rand(2))
    y = np.array([3,7])
    sgd = autograd.optimizer.SGD(learning_rate=1)
    for training_step_index in range(1_000):
        y_hat = x.divide(np.array([10, 20]))
        diff = np.subtract(y, y_hat)
        loss = diff ** 2
        if np.all(loss < 1e-3):
            break
        sgd.take_training_step(loss)
    assert np.abs(x - np.array([30, 140])).sum() < 1
    assert np.all(loss < 1e-3)

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
    
    # numpy + numpy
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
    assert all(isinstance(var, Variable) and isinstance(grad, np.ndarray) for var, grad in variable_to_gradient.items())
    assert np.all(variable_to_gradient[a] == np.ones(a.shape))
    assert np.all(variable_to_gradient[b] == np.full(b.shape, -1))

    # Verify Trainability
    x = Variable(np.random.rand(2))
    y = np.array([10, 10])
    sgd = autograd.optimizer.SGD(learning_rate=1e-1)
    for training_step_index in range(1_000):
        y_hat = x.subtract(np.array([-10, 50]))
        diff = np.subtract(y, y_hat)
        loss = diff ** 2
        if training_step_index > 10 and loss.sum() < 1e-10:
            break
        sgd.take_training_step(loss)
    assert np.abs(x - np.array([0, 60])).sum() < 1
    assert loss.sum() < 1e-10

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
    
    # numpy + numpy
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
    assert all(isinstance(var, Variable) and isinstance(grad, np.ndarray) for var, grad in variable_to_gradient.items())
    assert np.all(variable_to_gradient[a] == b_array*(a_array**(b_array-1)))
    assert np.all(variable_to_gradient[b] == np.log(a_array)*(a_array**b_array))

    # Verify Trainability (Base)
    x = Variable(np.random.rand(2))
    y = np.array([100, 8])
    sgd = autograd.optimizer.SGD(learning_rate=1e-4)
    for training_step_index in range(10_000):
        y_hat = x.pow(np.array([2, 3]))
        diff = np.subtract(y, y_hat)
        loss = diff ** 2
        if training_step_index > 10 and loss.sum() < 1e-4:
            break
        sgd.take_training_step(loss)
    assert np.abs(x - np.array([10, 2])).sum() < 3e-3
    assert loss.sum() < 1e-4

    # Verify Trainability (Exponent)
    x = Variable(np.array([1.9, 2.9], dtype=float))
    y = np.array([9, 8])
    sgd = autograd.optimizer.SGD(learning_rate=1e-3)
    for training_step_index in range(1_000):
        y_hat = np.float_power(np.array([3, 2], dtype=float), x)
        diff = np.subtract(y, y_hat)
        loss = diff ** 2
        if training_step_index > 10 and loss.sum() < 1e-4:
            break
        sgd.take_training_step(loss)
    assert np.abs(x - np.array([2, 3])).sum() < 9e-3
    assert loss.sum() < 1e-4

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
    
    # numpy + numpy
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
    summation = a+b
    variable_to_gradient = sgd.take_training_step(summation)
    assert all(isinstance(var, Variable) and isinstance(grad, np.ndarray) for var, grad in variable_to_gradient.items())
    assert np.all(variable_to_gradient[a] == np.ones(a.shape))
    assert np.all(variable_to_gradient[b] == np.ones(b.shape))

    # Verify Trainability
    x = Variable(np.random.rand(2))
    y = np.array([10, 10])
    sgd = autograd.optimizer.SGD(learning_rate=1e-1)
    for training_step_index in range(1_000):
        y_hat = x.add(np.array([-10, 50]))
        diff = np.subtract(y, y_hat)
        loss = diff ** 2
        if training_step_index > 10 and loss.sum() < 1e-10:
            break
        sgd.take_training_step(loss)
    assert np.abs(x - np.array([20, -40])).sum() < 1
    assert loss.sum() < 1e-10

def test_variable_sum():
    a_array = np.arange(5) # @todo test single number case as well
    a = Variable(np.arange(5, dtype=float)) # @todo test single number case as well
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

    def validate_number_result(result) -> None:
        assert np.all(result == expected_result_number)
        float(result) # error means it can't be converted to a float
        assert isinstance(result, np.number)
        return
    
    # Variable
    validate_variable_result(a.sum())
    validate_variable_result(np.sum(a))
    
    # numpy
    validate_number_result(a_array.sum())
    validate_number_result(np.sum(a_array))
        
    # Verify Derivative
    sgd = autograd.optimizer.SGD(learning_rate=1e-3)
    summation = a.sum()
    variable_to_gradient = sgd.take_training_step(summation)
    assert all(isinstance(var, Variable) and isinstance(grad, np.ndarray) for var, grad in variable_to_gradient.items())
    assert np.all(variable_to_gradient[a] == np.ones(a.shape))

    # Verify Trainability
    x = Variable(np.random.rand(2))
    y = 10
    sgd = autograd.optimizer.SGD(learning_rate=1e-1)
    for training_step_index in range(1_000):
        y_hat = x.sum()
        diff = np.subtract(y, y_hat)
        loss = diff ** 2
        if training_step_index > 10 and loss.sum() < 1e-10:
            break
        sgd.take_training_step(loss)
    assert np.abs(x.sum() - 10) < 1e-4
    assert loss.sum() < 1e-10

def test_variable_abs():
    a_array = np.array([0, -1, -2, 3]) # @todo test single number case as well
    a = Variable(np.array([0, -1, -2, 3], dtype=float)) # @todo test single number case as well
    expected_result_variable = Variable(np.arange(4))
    expected_result_array = np.arange(4)
    
    assert np.all(a_array == a.data)
    assert np.all(expected_result_variable == expected_result_array)
    
    assert id(a_array) != id(a.data)
    assert id(expected_result_variable) != id(expected_result_array)

    def validate_variable_result(result) -> None:
        assert expected_result_variable.eq(result).all()
        assert isinstance(result, Variable)
        return

    def validate_array_result(result) -> None:
        assert np.all(result == expected_result_array)
        assert isinstance(result, np.ndarray)
        return
    
    # Variable
    validate_variable_result(abs(a))
    validate_variable_result(a.abs())
    validate_variable_result(np.abs(a))
    
    # numpy
    validate_array_result(abs(a_array))
    validate_array_result(np.abs(a_array))
        
    # Verify Derivative
    sgd = autograd.optimizer.SGD(learning_rate=1e-3)
    absolute_value = a.abs()
    variable_to_gradient = sgd.take_training_step(absolute_value)
    assert all(isinstance(var, Variable) and isinstance(grad, np.ndarray) for var, grad in variable_to_gradient.items())
    assert np.all(variable_to_gradient[a] == np.array([0, -1, -1, 1]))

    # Verify Trainability
    x = Variable(np.array([-2, -1, 0, 1, 2], dtype=float))
    sgd = autograd.optimizer.SGD(learning_rate=1e-1)
    for training_step_index in range(1_000):
        absolute_value = x.abs()
        if np.all(np.abs(absolute_value) < 1e-10):
            break
        sgd.take_training_step(absolute_value)
    assert np.all(np.abs(absolute_value) < 1e-10)
    assert np.all(np.abs(x) < 1e-10)

def test_variable_matmul():
    a_matrix = np.arange(10, dtype=float).reshape(2,5)
    b_matrix = np.arange(10, dtype=float).reshape(2,5).T
    a = Variable(np.array([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
    ], dtype=float))
    b = Variable(np.array([
        [0, 5],
        [1, 6],
        [2, 7],
        [3, 8],
        [4, 9],
    ], dtype=float))
    expected_result_variable = Variable(np.array([
        [30, 80],
        [80, 255],
    ], dtype=float))
    expected_result_matrix = np.array([
        [30, 80],
        [80, 255],
    ], dtype=float)
    
    assert np.all(a_matrix == a.data)
    assert np.all(b_matrix == b.data)
    assert np.all(expected_result_variable == expected_result_matrix)
    
    assert id(a_matrix) != id(a.data)
    assert id(b_matrix) != id(b.data)
    assert id(expected_result_variable) != id(expected_result_matrix)

    def validate_variable_result(result) -> None:
        assert tuple(result.shape) == (2, 2)
        assert result.eq(expected_result_variable).all()
        assert isinstance(result, Variable)
        return

    def validate_matrix_result(result) -> None:
        assert tuple(result.shape) == (2, 2)
        assert np.all(result == expected_result_matrix)
        assert isinstance(result, np.ndarray)
        return
    
    # Variable + Variable
    validate_variable_result(a.matmul(b))
    validate_variable_result(a @ b)
    validate_variable_result(np.matmul(a, b))
    
    # numpy + numpy
    validate_matrix_result(np.matmul(a_matrix, b_matrix))
    validate_matrix_result(a_matrix @ b_matrix)
    
    # Variable + numpy
    validate_variable_result(a.matmul(b_matrix))
    validate_variable_result(a @ b_matrix)
    validate_variable_result(np.matmul(a, b_matrix))
    
    # numpy + Variable
    validate_variable_result(np.matmul(a_matrix, b))
    # validate_variable_result(a_matrix @ b) # @todo make this work
    
    # Verify Derivative
    sgd = autograd.optimizer.SGD(learning_rate=1e-3)
    matrix_product = a @ b
    variable_to_gradient = sgd.take_training_step(matrix_product)
    assert all(isinstance(var, Variable) and isinstance(grad, np.ndarray) for var, grad in variable_to_gradient.items())
    assert np.all(variable_to_gradient[a].shape == a.shape)
    assert np.all(variable_to_gradient[b].shape == b.shape)
    assert np.all(variable_to_gradient[a] == b_matrix.T)
    assert np.all(variable_to_gradient[b] == a_matrix.T)
    
    # Verify Trainability
    x = Variable(np.array([[1.1, 1.9], [2.9, 4.1]]))
    y = np.array([[7, 10], [15, 22]])
    sgd = autograd.optimizer.SGD(learning_rate=1e-2)
    for training_step_index in range(1_000):
        y_hat = x.matmul(np.array([[1, 2], [3, 4]]))
        diff = np.subtract(y, y_hat)
        loss = diff ** 2
        if training_step_index > 10 and loss.sum() < 1e-10:
            break
        sgd.take_training_step(loss)
    assert np.abs(x.sum() - np.array([[1, 2], [3, 4]]).sum()) < 0.04
    assert loss.sum() < 1e-10

def test_variable_expand_dims():
    a_array = np.arange(5)
    a = Variable(np.arange(5, dtype=float))
    expected_result_variable = Variable(np.array([[0, 1, 2, 3, 4]]))
    expected_result_number = np.array([[0, 1, 2, 3, 4]])
    
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
    validate_variable_result(a.expand_dims(0))
    validate_variable_result(a.expand_dims((0,)))
    validate_variable_result(np.expand_dims(a, 0))
    validate_variable_result(np.expand_dims(a, (0,)))
    
    # numpy
    validate_array_result(np.expand_dims(a_array, 0))
    validate_array_result(np.expand_dims(a_array, (0,)))
        
    # Verify Derivative
    sgd = autograd.optimizer.SGD(learning_rate=1e-3)
    a_expanded = a.expand_dims(0)
    diff = a_expanded - np.zeros(5)
    loss = np.sum(diff ** 2)
    variable_to_gradient = sgd.take_training_step(loss)
    assert all(isinstance(var, Variable) and isinstance(grad, np.ndarray) for var, grad in variable_to_gradient.items())
    assert a_expanded.data.base is not a.data
    assert np.all(variable_to_gradient[a] == np.arange(5)*2)
    assert tuple(variable_to_gradient[a].shape) == (5,)
    assert np.all(variable_to_gradient[a_expanded] == np.arange(5)*2)
    assert tuple(variable_to_gradient[a_expanded].shape) == (1,5)
    assert np.all(variable_to_gradient[a_expanded].squeeze(0) == variable_to_gradient[a_expanded])
    assert np.all(variable_to_gradient[a_expanded].squeeze() == variable_to_gradient[a_expanded])
    assert np.all(a.data == a_expanded.data)

    # Verify Trainability
    x = Variable(np.random.rand(2))
    y = np.array([[10], [30]])
    sgd = autograd.optimizer.SGD(learning_rate=1e-1)
    for training_step_index in range(1_000):
        y_hat = x.expand_dims(1)
        assert y_hat.data.base is not x.data
        diff = np.subtract(y, y_hat)
        loss = np.sum(diff ** 2)
        if training_step_index > 10 and loss.sum() < 1e-10:
            break
        sgd.take_training_step(loss)
    assert np.abs(np.sum(x - np.array([[10], [30]]))) < 1e-4
    assert loss.sum() < 1e-10

def test_variable_log():
    a_array = np.array([1, 2]) # @todo test single number case as well
    a = Variable(np.array([1, 2], dtype=float)) # @todo test single number case as well
    expected_result_variable = Variable(np.array([0, 0.69314718]))
    expected_result_array = np.array([0, 0.69314718])
    
    assert np.all(a_array == a.data)
    assert np.all(expected_result_variable == expected_result_array)
    
    assert id(a_array) != id(a.data)
    assert id(expected_result_variable) != id(expected_result_array)
    
    def validate_variable_result(result) -> None:
        assert expected_result_variable.isclose(result).all()
        assert isinstance(result, Variable)
        return
    
    def validate_array_result(result) -> None:
        assert np.isclose(result, expected_result_array).all()
        assert isinstance(result, np.ndarray)
        return
    
    # Variable
    validate_variable_result(np.log(a))
    validate_variable_result(a.log())
    validate_variable_result(a.natural_log())
    
    # numpy
    validate_array_result(np.log(a_array))
    
    # Verify Derivative
    sgd = autograd.optimizer.SGD(learning_rate=1e-3)
    log_result = a.log()
    variable_to_gradient = sgd.take_training_step(log_result)
    assert all(isinstance(var, Variable) and isinstance(grad, np.ndarray) for var, grad in variable_to_gradient.items())
    assert np.all(variable_to_gradient[a] == np.array([1, 0.5]))
    
    # Verify Trainability
    x = Variable(np.array([0.1, 0.2]))
    y = 1
    sgd = autograd.optimizer.SGD(learning_rate=1)
    for training_step_index in range(1_000):
        y_hat = x.log()
        diff = np.subtract(y, y_hat)
        loss = np.sum(diff ** 2)
        if training_step_index > 10 and loss.sum() < 1e-10:
            break
        sgd.take_training_step(loss)
    assert np.all(loss < 1e-10)
    assert np.all(np.abs(x - np.e) < 1e-4)

def test_variable_exp():
    a_array = np.array([1, 2]) # @todo test single number case as well
    a = Variable(np.array([1, 2], dtype=float)) # @todo test single number case as well
    expected_result_variable = Variable(np.array([2.718281828459045, 7.3890560989306495]))
    expected_result_array = np.array([2.718281828459045, 7.3890560989306495])
    
    assert np.all(a_array == a.data)
    assert np.all(expected_result_variable == expected_result_array)
    
    assert id(a_array) != id(a.data)
    assert id(expected_result_variable) != id(expected_result_array)
    
    def validate_variable_result(result) -> None:
        assert expected_result_variable.isclose(result).all()
        assert isinstance(result, Variable)
        return
    
    def validate_array_result(result) -> None:
        assert np.isclose(result, expected_result_array).all()
        assert isinstance(result, np.ndarray)
        return
    
    # Variable
    validate_variable_result(np.exp(a))
    validate_variable_result(a.exp())
    
    # numpy
    validate_array_result(np.exp(a_array))
    
    # Verify Derivative
    sgd = autograd.optimizer.SGD(learning_rate=1e-3)
    exp_result = a.exp()
    variable_to_gradient = sgd.take_training_step(exp_result)
    assert all(isinstance(var, Variable) and isinstance(grad, np.ndarray) for var, grad in variable_to_gradient.items())
    assert np.all(np.isclose(variable_to_gradient[a], exp_result, rtol=1e-3, atol=1e-4))
    
    # Verify Trainability
    x = Variable(np.random.rand(2))
    y = 1
    sgd = autograd.optimizer.SGD(learning_rate=1)
    for training_step_index in range(1_000):
        y_hat = x.exp()
        diff = np.subtract(y, y_hat)
        loss = np.sum(diff ** 2)
        if training_step_index > 10 and loss.sum() < 1e-15:
            break
        sgd.take_training_step(loss)
    assert np.all(loss < 1e-15)
    assert np.all(np.abs(x) < 1e-2)

def test_variable_negative():
    a_array = np.array([1, 2]) # @todo test single number case as well
    a = Variable(np.array([1, 2], dtype=float)) # @todo test single number case as well
    expected_result_variable = Variable(np.array([-1.0, -2.0]))
    expected_result_array = np.array([-1, -2])
    
    assert np.all(a_array == a.data)
    assert np.all(expected_result_variable == expected_result_array)
    
    assert id(a_array) != id(a.data)
    assert id(expected_result_variable) != id(expected_result_array)
    
    def validate_variable_result(result) -> None:
        assert expected_result_variable.equal(result).all()
        assert isinstance(result, Variable)
        return
    
    def validate_array_result(result) -> None:
        assert np.equal(result, expected_result_array).all()
        assert isinstance(result, np.ndarray)
        return
    
    # Variable
    validate_variable_result(np.negative(a))
    validate_variable_result(a.negative())
    
    # numpy
    validate_array_result(np.negative(a_array))
    
    # Verify Derivative
    sgd = autograd.optimizer.SGD(learning_rate=1e-3)
    negative_result = a.negative()
    variable_to_gradient = sgd.take_training_step(negative_result)
    assert all(isinstance(var, Variable) and isinstance(grad, np.ndarray) for var, grad in variable_to_gradient.items())
    assert np.all(np.equal(variable_to_gradient[a], np.full(a.shape, -1.0)))
    
    # Verify Trainability
    x = Variable(np.random.rand(2))
    y = 1
    sgd = autograd.optimizer.SGD(learning_rate=1e-1)
    for training_step_index in range(1_000):
        y_hat = x.negative()
        diff = np.subtract(y, y_hat)
        loss = np.sum(diff ** 2)
        if training_step_index > 10 and loss.sum() < 1e-15:
            break
        var2grad = sgd.take_training_step(loss)
    assert np.all(loss < 1e-15)
    assert np.all(x+1 < 1e-2)
