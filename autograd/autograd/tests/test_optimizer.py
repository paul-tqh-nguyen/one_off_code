import pytest

import numpy as np

import ..autograd
from ..autograd import Variable

def test_sgd():
    # Variables
    x = Variable(np.random.rand(2))

    # Optimizer
    learning_rate = 1e-4
    sgd = autograd.optimizer.SGD(learning_rate)
    
    # Training
    for training_step_index in range(100):
        y = x.dot(np.array([-10, 50]))
        y_hat = 0
        diff = (y - y_hat)
        loss = diff ** 2
        sgd.take_training_step(loss)
    
    # Verify Results
    assert np.abs(loss.data) < 1e-3
    
