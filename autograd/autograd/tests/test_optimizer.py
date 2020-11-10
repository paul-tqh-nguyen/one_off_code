import pytest
import numpy as np

import sys ; sys.path.append("..")
import autograd
from autograd import Variable

def test_sgd():
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
        print(f"y_hat {repr(y_hat)}")
        diff = (y - y_hat)
        loss = diff ** 2
        sgd.take_training_step(loss)
        if np.abs(loss.data) < 1e-3:
            break
    
    # Verify Results
    assert np.abs(loss.data) < 1e-3
    
