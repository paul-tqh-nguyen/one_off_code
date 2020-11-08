
'''

'''

# @todo fill in docstring

###########
# Imports #
###########

from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
from typing import List, Set, Tuple, DefaultDict, Dict, Callable, Union, Generator, Optional

from .variable import Variable
from .misc_utilities import *

# @todo verify these imports are used

#############################
# Backpropagation Execution #
#############################

##############
# Optimizers #
##############

class Optimizer(ABC):
    
    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def execute_backpropagation(dependent_variable: Variable) -> DefaultDict[Variable, Union[int, float, np.number, np.ndarray]]:
        '''Returns a mapping from Variable instances to their gradients, i.e. d_dependent_variable_over_d_variable .'''
        variable_to_gradient = defaultdict(lambda: 0)
        d_dependent_variable_over_d_dependent_variable = np.ones(dependent_variable.data.shape)
        variable_to_gradient[dependent_variable] = d_dependent_variable_over_d_dependent_variable
        for depended_upon_variable in dependent_variable.depended_upon_variables_iterator():
            assert depended_upon_variable in variable_to_gradient
            depended_upon_variable_gradient = variable_to_gradient[depended_upon_variable]
            # @todo rename these variables
            variable_depended_upon_by_depended_upon_variable_to_gradient = depended_upon_variable.backward_propagate_gradient(depended_upon_variable_gradient)
            for variable_depended_upon_by_depended_upon_variable, gradient in variable_depended_upon_by_depended_upon_variable_to_gradient.items():
                variable_to_gradient[variable_depended_upon_by_depended_upon_variable] += gradient
        return variable_to_gradient

    @abstractmethod
    def take_training_step(self, minimization_variable: Variable) -> None:
        raise NotImplementedError

class SGD(Optimizer): # @todo test this with a basic test
    
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        return

    def take_training_step(self, minimization_variable: Variable) -> None:
        minimization_variable.zero_out_gradients() # @todo consider making this a context manager
        variable_to_gradient = self.__class__.execute_backpropagation(minimization_variable)
        for variable, d_minimization_variable_over_d_variable in variable_to_gradient.items(): # @todo consider renaming d_minimization_variable_over_d_variable to gradient or adding a comment somewhere
            variable.data -= self.learning_rate * d_minimization_variable_over_d_variable
        return

##########
# Driver #
##########

if __name__ == '__main__':
    print('@todo fill this in')
