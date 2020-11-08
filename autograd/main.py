
'''

This implementation is intentionally non-optimal/slow for readability and understandability purposes.

'''

# @todo fill in docstring

###########
# Imports #
###########

import numpy as np

import autograd
from autograd import Variable
from autograd.misc_utilities import *

# @todo verify these imports are used

##########
# Driver #
##########

@debug_on_error
def main() -> None:
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
        print(f"training_step_index {repr(training_step_index)}")
        print(f"x.data {repr(x.data)}")
        print(f"y.data {repr(y.data)}")
        print(f"diff.data {repr(diff.data)}")
        print(f"loss.data {repr(loss.data)}")
        print()
        sgd.take_training_step(loss)
    
    # Verify Results
    assert np.abs(loss.data) < 1e-3
    
    return

if __name__ == '__main__':
    main()
