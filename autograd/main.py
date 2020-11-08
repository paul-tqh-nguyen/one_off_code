
'''

This implementation is intentionally non-optimal/slow for readability and understandability purposes.

'''

# @todo fill in docstring

###########
# Imports #
###########

from collections import defaultdict
import numpy as np
from typing import List, Set, Tuple, Dict, Callable, Union, Generator, Optional

from misc_utilities import *

# @todo verify these imports are used

####################
# Variable Classes #
####################

class Variable:

    ##############
    # Decorators #
    ##############

    # numpy_replacement Decorator
    
    @staticmethod
    def _numpy_replacement_extract_inputs(internally_used_name_to_np_path: Dict[str, str]) -> Tuple[str, Callable]:
        if len(internally_used_name_to_np_path) != 1:
            raise ValueError(f'Only one numpy callable can be replaced. {len(internally_used_name_to_np_path)} were specified.') # @todo test this
        
        internally_used_name, np_path = only_one(internally_used_name_to_np_path.items())
        
        if not internally_used_name.isidentifier(): # @todo test this
            raise ValueError(f'"{internally_used_name}" is not a vaild identifier name.')
        
        replaced_callable_parent_attribute = np
        np_path_sub_attributes = np_path.split('.')

        if globals()[np_path_sub_attributes[0]] != np:
            raise ValueError(f'"{np_path}" does not specify a numpy function')
        
        # @todo test this with simple names, not present names, empty string names, and "invalid variable name" names passed into np_path; make sure we get errors in the right cases
        for np_path_sub_attribute_index, np_path_sub_attribute in enumerate(np_path_sub_attributes[1:-1], start=1):
            if not hasattr(replaced_callable_parent_attribute, np_path_sub_attribute):
                raise ValueError(f'"{".".join(np_path_sub_attributes[:np_path_sub_attribute_index])}" does not specify a numpy function')
            replaced_callable_parent_attribute = getattr(replaced_callable_parent_attribute, np_path_sub_attribute)

        if not hasattr(replaced_callable_parent_attribute, np_path_sub_attributes[-1]):
            raise ValueError(f'"{np_path}" does not specify a numpy function')

        replaced_callable = getattr(replaced_callable_parent_attribute, np_path_sub_attributes[-1])

        return (internally_used_name, replaced_callable)
    
    @classmethod
    def numpy_replacement(cls, **internally_used_name_to_np_path: Dict[str, str]) -> Callable:
        '''Replaces numpy methods via monkey patching.'''
        internally_used_name, replaced_callable = cls._numpy_replacement_extract_inputs(internally_used_name_to_np_path)
        # @todo inspect the signature of replaced_callable to make sure internally_used_name is not a parameter name
        def decorator(func: Callable):
            def decorated_function(*args, **kwargs):
                assert internally_used_name not in kwargs.keys()
                kwargs[internally_used_name] = replaced_callable
                return func(*args, **kwargs)
            decorated_function.__qualname__ = func.__qualname__ # @todo test invariant holds
            decorated_function.__name__ = func.__name__ # @todo test invariant holds
            return decorated_function
        return decorator

    # differentiable_method Decorator
    
    class _DifferentiableMethodDecorator:
        def __init__(self, method_names: List[str]):
            for method_name in method_names:
                if not method_name.isidentifier():
                    raise ValueError(f'"{method_name}" is not a valid method name.') # @todo test this
            self.method_names = method_names
            return
        
        def __call__(self, func: Callable) -> Callable:
            if len(self.method_names) == 0: # @todo test single, zero, and multiple method name cases
                self.method_names = [func.__qualname__]
            for method_name in self.method_names:
                setattr(Variable, method_name, func)
            return func
    
    @classmethod
    def differentiable_method(cls, *method_names: List[str]) -> Callable:
        return cls._DifferentiableMethodDecorator(method_names)

    ###########
    # Methods #
    ###########

    def __init__(self, data: np.ndarray, variable_depended_upon_by_self_to_backward_propagation_function: Dict['Variable', Callable] = dict()):
        self.data = data
        # @todo make this be handled by a context manager that clears out gradients inside the optimizer
        self.minimization_target_variable_to_d_minimization_target_variable_over_d_self: Dict[Variable, np.ndarray] = defaultdict(lambda: 0) # values are conventionally referred to as "the accumulated gradient"
        self.variable_depended_upon_by_self_to_backward_propagation_function = variable_depended_upon_by_self_to_backward_propagation_function
        return
    
    def depended_upon_variables_iterator(self) -> Generator: # @todo test just this iterator (both cases)
        '''Yields all variables that sef directly or indirectly relies on.'''
        yield from self._depended_upon_variables_iterator(set())
        
    def _depended_upon_variables_iterator(self, visited_variables: Set['Variable']) -> Generator:
        '''Yields variables in topological order.'''
        yield self
        visited_variables.add(self)
        for variable_directly_depended_upon_by_self in self.variable_depended_upon_by_self_to_backward_propagation_function.keys():
            if variable_directly_depended_upon_by_self not in visited_variables:
                yield variable_directly_depended_upon_by_self
                visited_variables.add(variable_directly_depended_upon_by_self)
                for variable_indirectly_depended_upon_by_self in variable_directly_depended_upon_by_self._depended_upon_variables_iterator(visited_variables):
                    if variable_indirectly_depended_upon_by_self not in visited_variables:
                        yield variable_indirectly_depended_upon_by_self
                        visited_variables.add(variable_indirectly_depended_upon_by_self)
    
    def zero_out_gradients(self, *minimization_target_variables: List['Variable']): # @todo test this with single, zero, and many minimization_target_variables
        '''Clears gradients of minimization_target_variables w.r.t self and w.r.t all the variables that self depends on (directly or indirectly).'''
        for minimization_target_variable in minimization_target_variables:
            for depended_upon_variable in self.depended_upon_variables_iterator():
                del depended_upon_variable.variable_depended_upon_by_self_to_backward_propagation_function[minimization_target_variable]
    
    def backward_propagate_gradient(self, minimization_target_variable: 'Variable') -> None:
        '''
        By the chain rule (see https://bit.ly/36hTKM2),
        
            dLOSS/dCUR = sum(
                dLOSS   dVAR
                ----- * ----
                dVAR    dCUR
            for all variables VAR that LOSS is a direct function of
            )
        
        We want to make it so that each CUR eventually has a correct dLOSS/dCUR value stored. 
            - dLOSS/dCUR is stored in CUR.minimization_target_variable_to_d_minimization_target_variable_over_d_self[LOSS]
        
        dLOSS/dCUR is a sum.
        
        dLOSS/dCUR is made correct by adding in each summand over time.
        
        This function adds the following summand to dLOSS/dCUR :
            dLOSS   dVAR
            ----- * ----
            dVAR    dCUR
        
        In the example immediately above, 
            - VAR is self
            - CUR is each of the variables that VAR, i.e. self, depends on.
                - The number of summand additions done by this function is equal to the number of variables self depends on.
                
        For any CUR, when this function is called for all variables that depend on CUR (not necessarily just VAR, i.e. self), dLOSS/dCUR will be correct.
            - This is handled by execute_backpropagation.
        
        This function assumes that there's already an entry (minimization_target_variable, d_minimization_target_variable_over_d_self) in 
        self.minimization_target_variable_to_d_minimization_target_variable_over_d_self .
        '''
        d_minimization_target_variable_over_d_self = self.minimization_target_variable_to_d_minimization_target_variable_over_d_self[minimization_target_variable]
        for variable_depended_upon_by_self, backward_propagation_function in self.variable_depended_upon_by_self_to_backward_propagation_function.items():
            '''
            backward_propagation_function updates d_minimization_target_variable_over_d_variable_depended_upon_by_self
            self is a function of variable_depended_upon_by_self
            self passes d_minimization_target_variable_over_d_self to variable_depended_upon_by_self
            variable_depended_upon_by_self calculates d_self_over_variable_depended_upon_by_self
            variable_depended_upon_by_self can then calculate d_minimization_target_variable_over_d_variable_depended_upon_by_self = d_minimization_target_variable_over_d_self * d_self_over_variable_depended_upon_by_self
            This is what backward_propagation_function does.
            '''
            backward_propagation_function(minimization_target_variable, d_minimization_target_variable_over_d_self)
            # @todo add shape assertions here
        return

#######################
# Variable Operations #
#######################

VariableOperand = Union[int, float, np.number, np.ndarray, Variable]

# @todo test this with all combinations of types
@Variable.differentiable_method() # @todo test this method
@Variable.numpy_replacement(np_dot='np.dot') # @todo test these numpy methods
def dot(a: VariableOperand, b: VariableOperand, np_dot: Callable, out: Optional[np.ndarray]=None) -> VariableOperand:
    a_is_variable = isinstance(a, Variable)
    b_is_variable = isinstance(b, Variable)
    a_data = a.data if a_is_variable else a
    b_data = b.data if b_is_variable else b
    dot_product = np_dot(a_data, b_data, out)
    if not a_is_variable and not b_is_variable:
        return dot_product
    if out is not None:
        raise ValueError(f'"out" parameter not supported for {Variable.__qualname__}')
    variable_depended_upon_by_dot_product_to_backward_propagation_function = {}
    if a_is_variable:
        def propagate_gadient_to_a(minimization_target: Variable, d_minimization_target_over_d_dot_product: np.ndarray) -> None:
            a.minimization_target_variable_to_d_minimization_target_variable_over_d_self[minimization_target] += d_minimization_target_over_d_dot_product * b_data
            return
        variable_depended_upon_by_dot_product_to_backward_propagation_function[a] = propagate_gadient_to_a
    if b_is_variable:
        def propagate_gadient_to_b(minimization_target: Variable, d_minimization_target_over_d_dot_product: np.ndarray) -> None:
            b.minimization_target_variable_to_d_minimization_target_variable_over_d_self[minimization_target] += d_minimization_target_over_d_dot_product * a_data
            return
        variable_depended_upon_by_dot_product_to_backward_propagation_function[b] = propagate_gadient_to_b
    dot_product_variable = Variable(dot_product, variable_depended_upon_by_dot_product_to_backward_propagation_function)
    return dot_product_variable

# @todo test this with all combinations of types
@Variable.differentiable_method('subtract', '__sub__') # @todo test these methods
@Variable.numpy_replacement(np_subtract='np.subtract') # @todo test these numpy methods
def subtract(a: VariableOperand, b: VariableOperand, np_subtract: Callable, **kwargs) -> VariableOperand:
    a_is_variable = isinstance(a, Variable)
    b_is_variable = isinstance(b, Variable)
    a_data = a.data if a_is_variable else a
    b_data = b.data if b_is_variable else b
    difference = np_subtract(a_data, b_data, **kwargs)
    if not a_is_variable and not b_is_variable:
        return difference
    if len(kwargs) > 0:
        raise ValueError(f'The parameters {[repr(kwarg_name) for kwarg_name in kwargs.keys()]} are not supported for {Variable.__qualname__}')
    variable_depended_upon_by_difference_to_backward_propagation_function = {}
    if a_is_variable:
        def propagate_gadient_to_a(minimization_target: Variable, d_minimization_target_over_d_difference: np.ndarray) -> None:
            a.minimization_target_variable_to_d_minimization_target_variable_over_d_self[minimization_target] += d_minimization_target_over_d_difference
            return
        variable_depended_upon_by_difference_to_backward_propagation_function[a] = propagate_gadient_to_a
    if b_is_variable:
        def propagate_gadient_to_b(minimization_target: Variable, d_minimization_target_over_d_difference: np.ndarray) -> None:
            b.minimization_target_variable_to_d_minimization_target_variable_over_d_self[minimization_target] += d_minimization_target_over_d_difference
            return
        variable_depended_upon_by_difference_to_backward_propagation_function[b] = propagate_gadient_to_b
    difference_variable = Variable(difference, variable_depended_upon_by_difference_to_backward_propagation_function)
    return difference_variable

# @todo test this with all combinations of types
@Variable.differentiable_method('power', 'pow', '__pow__') # @todo test these methods
@Variable.numpy_replacement(np_float_power='np.float_power') # @todo test these numpy methods
def float_power(base: VariableOperand, exponent: VariableOperand, np_float_power: Callable, **kwargs) -> VariableOperand:
    base_is_variable = isinstance(base, Variable)
    exponent_is_variable = isinstance(exponent, Variable)
    base_data = base.data if base_is_variable else base
    exponent_data = exponent.data if exponent_is_variable else exponent
    power = np_float_power(base_data, exponent_data, **kwargs)
    if not base_is_variable and not exponent_is_variable:
        return power
    if len(kwargs) > 0:
        raise ValueError(f'The parameters {[repr(kwarg_name) for kwarg_name in kwargs.keys()]} are not supported for {Variable.__qualname__}')
    variable_depended_upon_by_power_to_backward_propagation_function = {}
    if base_is_variable:
        def propagate_gadient_to_base(minimization_target: Variable, d_minimization_target_over_d_power: np.ndarray) -> None:
            base.minimization_target_variable_to_d_minimization_target_variable_over_d_self[minimization_target] += d_minimization_target_over_d_power * exponent_data * np_float_power(base_data, exponent_data-1)
            return
        variable_depended_upon_by_power_to_backward_propagation_function[base] = propagate_gadient_to_base
    if exponent_is_variable:
        def propagate_gadient_to_b(minimization_target: Variable, d_minimization_target_over_d_power: np.ndarray) -> None:
            exponent.minimization_target_variable_to_d_minimization_target_variable_over_d_self[minimization_target] += d_minimization_target_over_d_power * power.data*np.log(base_data)
            return
        variable_depended_upon_by_power_to_backward_propagation_function[b] = propagate_gadient_to_exponent
    power_variable = Variable(power, variable_depended_upon_by_power_to_backward_propagation_function)
    return power_variable

#############################
# Backpropagation Execution #
#############################

def execute_backpropagation(dependent_variable: Variable):
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

class SGD:
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

@debug_on_error
def main() -> None:
    # Variables
    x = Variable(np.random.rand(3))

    # Optimizer
    learning_rate = 1e-3
    sgd = SGD(learning_rate)
    
    # Training
    for training_step_index in range(100):
        y = x.dot(np.array([10, 20, 30]))
        y_hat = np.array([100, 200, 300])
        loss = (y - y_hat) ** 2
        sgd.take_training_step(loss)
        print(f"training_step_index {repr(training_step_index)}")
        print(f"loss.data {repr(loss.data)}")
    
    # Verify Results
    assert np.allclose(x.data, np.array([100, 200, 300]) < 1e-3)
    
    return

if __name__ == '__main__':
    main()
