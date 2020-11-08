
'''

'''

# @todo fill in docstring

###########
# Imports #
###########

from collections import defaultdict
import numpy as np
from typing import List, Set, Tuple, Dict, Callable, Union, Generator, Optional

from .variable import Variable
from .misc_utilities import *

# @todo verify these imports are used

#############################
# Backpropagation Execution #
#############################

def execute_backpropagation(dependent_variable: Variable): # @todo make this a method of the optimizer abstract class
    '''
    Makes it so that all the variables that dependent_variable depends on has an entry 
    (dependent_variable, d_dependent_variable_over_d_depended_on_variable)
    in their variable_depended_upon_by_self_to_backward_propagation_function attributes.
    '''
    d_dependent_variable_over_d_dependent_variable = np.ones(dependent_variable.data.shape)
    dependent_variable.minimization_target_variable_to_d_minimization_target_variable_over_d_self[dependent_variable]= d_dependent_variable_over_d_dependent_variable
    for depended_upon_variable in dependent_variable.depended_upon_variables_iterator():
        depended_upon_variable.backward_propagate_gradient(dependent_variable)
    return

##############
# Optimizers #
##############

class SGD: # @todo test this with a basic test
    # @todo make a parent abstract class where take_training_step is not implemented but required
    
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        return

    def take_training_step(self, minimization_variable: Variable) -> None:
        minimization_variable.zero_out_gradients() # @todo consider making this a context manager
        execute_backpropagation(minimization_variable)
        for depended_upon_variable in minimization_variable.depended_upon_variables_iterator():
            d_minimization_variable_over_d_depended_upon_variable = depended_upon_variable.minimization_target_variable_to_d_minimization_target_variable_over_d_self[minimization_variable]
            depended_upon_variable.data -= self.learning_rate * d_minimization_variable_over_d_depended_upon_variable
        return

##########
# Driver #
##########

if __name__ == '__main__':
    print('@todo fill this in')
