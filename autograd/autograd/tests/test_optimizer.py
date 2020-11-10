import pytest
import numpy as np

import sys ; sys.path.append("..")
import autograd
from autograd import Variable

def test_basic_sgd_1():
    # Variables
    x = Variable(np.random.rand(2))

    # Optimizer
    learning_rate = 0.1
    sgd = autograd.optimizer.SGD(learning_rate)
    
    # Training
    for _ in range(100):
        y = x - np.array([100, 500])
        y_hat = np.array([25, 65])
        diff = y - y_hat
        loss = diff ** 2
        sgd.take_training_step(loss)
        if np.all(np.abs(loss.data) < 1e-1):
            break
    
    # Verify Results
    assert np.all(np.abs(loss.data) < 1e-1)

def test_basic_sgd_2():
    # Variables
    x = Variable(np.random.rand(2))

    # Optimizer
    learning_rate = 1e-4
    sgd = autograd.optimizer.SGD(learning_rate)
    
    # Training
    for _ in range(50):
        y = x.dot(np.array([-10, 50]))
        y_hat = 0
        diff = np.subtract(y, y_hat)
        loss = diff ** 2
        sgd.take_training_step(loss)
        if np.abs(loss.data) < 1e-3:
            break
    
    # Verify Results
    assert np.abs(loss.data) < 1e-3
    

def test_squaring_equal_to_self_multiplication():
    # Variables
    x_1 = Variable(np.arange(4, dtype=float))
    x_2 = Variable(np.arange(4, dtype=float))
    x_3 = Variable(np.arange(4, dtype=float))

    # Optimizer
    learning_rate = 1e-3
    sgd_1 = autograd.optimizer.SGD(learning_rate)
    sgd_2 = autograd.optimizer.SGD(learning_rate)
    sgd_3 = autograd.optimizer.SGD(learning_rate)
    
    # Verify Results
    for _ in range(500):
        
        diff_1 = np.subtract(40, Variable(np.full(4, 5, dtype=float)).dot(x_1))
        diff_2 = np.subtract(40, Variable(np.full(4, 5, dtype=float)).dot(x_2))
        diff_3_a = np.subtract(40, Variable(np.full(4, 5, dtype=float)).dot(x_3))
        diff_3_b = np.subtract(40, Variable(np.full(4, 5, dtype=float)).dot(x_3))

        assert id(diff_3_a) != id(diff_3_b)
        
        loss_1 = diff_1 ** 2
        loss_2 = diff_2 * diff_2
        loss_3 = diff_3_a * diff_3_b

        assert np.all(loss_1.data == loss_2.data)
        assert np.all(loss_2.data == loss_3.data)
        assert np.all(loss_3.data == loss_1.data)
        
        variable_to_gradient_1 = sgd_1.take_training_step(loss_1)
        variable_to_gradient_2 = sgd_2.take_training_step(loss_2)
        variable_to_gradient_3 = sgd_3.take_training_step(loss_3)

        assert np.all(variable_to_gradient_1[x_1] == variable_to_gradient_2[x_2])
        assert np.all(variable_to_gradient_2[x_2] == variable_to_gradient_3[x_3])
        assert np.all(variable_to_gradient_3[x_3] == variable_to_gradient_1[x_1])
        
        if loss_1 < 1e-6:
            break
        
    assert loss_1 < 1e-6
    assert loss_2 < 1e-6
    assert loss_3 < 1e-6
