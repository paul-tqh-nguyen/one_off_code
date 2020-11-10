import pytest
import numpy as np

import sys ; sys.path.append("..")
import autograd
from autograd import Variable

def test_sgd_1():
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

def test_sgd_dot():
    # Variables
    x = Variable(np.random.rand(2))

    # Optimizer
    learning_rate = 1e-4
    sgd = autograd.optimizer.SGD(learning_rate)
    
    # Training
    for _ in range(50):
        y = x.dot(np.array([-10, 50]))
        y_hat = 0
        print(f"y {repr(y)}")
        print(f"y.data.shape {repr(y.data.shape)}")
        diff = y - y_hat
        loss = diff ** 2
        sgd.take_training_step(loss)
        if np.abs(loss.data) < 1e-3:
            break
    
    # Verify Results
    assert np.abs(loss.data) < 1e-3
    
