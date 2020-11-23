
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

##############
# Optimizers #
##############

# @todo add gradient clipping as a parameter to take_training_step or as an initialization parameter

class Optimizer(ABC):
    
    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def execute_backpropagation(dependent_variable: Variable) -> Dict[Variable, Union[int, float, np.number, np.ndarray]]:
        '''
        Executes the backpropagation algorithm.
        dependent_variable is a variable that relies on (i.e. is a function of) other variables (directly or indirectly).
        This method returns a mapping (dictionary) from each relied upon variable (let's call it var) to d_dependent_variable_over_d_var (conventionally referred to as the gradient).
        In other words, this function accumulates the gradients of all variables that dependent_variable directly or indirectly is a function of.
        '''
        variable_to_gradient = defaultdict(lambda: 0)
        if len(dependent_variable.data.shape) > 2:
            raise NotImplementedError(f'Backpropagation on arrays with dimensionality > 2 not yet supported.')
        d_dependent_variable_over_d_dependent_variable = np.ones(dependent_variable.data.shape) if len(dependent_variable.data.shape) < 2 else np.eye(dependent_variable.data.shape[1])
        variable_to_gradient[dependent_variable] = d_dependent_variable_over_d_dependent_variable
        # iterate over variables depended on by dependent_variable (directly or indirectly) in topologically sorted order (i.e. DFS ordedr with no repeats)
        for var in dependent_variable.depended_on_variables():
            assert var in variable_to_gradient
            d_dependent_variable_over_d_var = variable_to_gradient[var]
            # backpropagate one step gradient from var to variables it directly relies on
            variable_depended_on_by_var_to_gradient = var.directly_backward_propagate_gradient(d_dependent_variable_over_d_var)
            for variable_depended_on_by_var, gradient in variable_depended_on_by_var_to_gradient.items():
                variable_to_gradient[variable_depended_on_by_var] += gradient
        variable_to_gradient = dict(variable_to_gradient)
        # # @todo remove this
        # for v, g in variable_to_gradient.items():
        #     variable_to_gradient[v] = g/np.linalg.norm(g)
        return variable_to_gradient

    @abstractmethod
    def take_training_step(self, minimization_variable: Variable) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        return

    def take_training_step(self, minimization_variable: Variable) -> Dict[Variable, Union[int, float, np.number, np.ndarray]]:
        variable_to_gradient = self.execute_backpropagation(minimization_variable)
        # @todo make this class track the variables we only care about changing. Don't bother taking steps for variables that aren't important.
        for variable, d_minimization_variable_over_d_variable in variable_to_gradient.items():
            variable.data -= self.learning_rate * d_minimization_variable_over_d_variable
        return variable_to_gradient

# @todo add Adam optimizer
