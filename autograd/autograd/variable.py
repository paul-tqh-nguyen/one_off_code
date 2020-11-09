
'''

'''

# @todo fill in docstring

###########
# Imports #
###########

from itertools import chain
from collections import defaultdict
import numpy as np
from typing import List, Set, Tuple, DefaultDict, Dict, Callable, Union, Generator, Optional

from .misc_utilities import *

# @todo verify these imports are used

####################
# Variable Classes #
####################

VariableOperand = Union[int, float, np.number, np.ndarray, 'Variable']

class Variable:

    ##############
    # Decorators #
    ##############

    # numpy_replacement Decorator
    
    @staticmethod
    def _numpy_replacement_extract_inputs(internally_used_name_to_np_path: Dict[str, str]) -> Tuple[str, str, Callable]:
        if len(internally_used_name_to_np_path) != 1:
            raise ValueError(f'Only one numpy callable can be replaced. {len(internally_used_name_to_np_path)} were specified.')
        
        internally_used_name, np_path = only_one(internally_used_name_to_np_path.items())
        
        if not internally_used_name.isidentifier():
            raise ValueError(f'"{internally_used_name}" is not a vaild identifier name.')
        
        replaced_callable_parent_attribute = np
        np_path_sub_attributes = np_path.split('.')

        if globals().get(np_path_sub_attributes[0]) != np:
            raise ValueError(f'"{np_path}" does not specify a numpy function.')
        
        for np_path_sub_attribute_index, np_path_sub_attribute in enumerate(np_path_sub_attributes[1:-1], start=1):
            if not hasattr(replaced_callable_parent_attribute, np_path_sub_attribute):
                raise ValueError(f'"{".".join(np_path_sub_attributes[:np_path_sub_attribute_index])}" does not specify a numpy function.')
            replaced_callable_parent_attribute = getattr(replaced_callable_parent_attribute, np_path_sub_attribute)

        if not hasattr(replaced_callable_parent_attribute, np_path_sub_attributes[-1]):
            raise ValueError(f'"{np_path}" does not specify a numpy function.')

        replaced_callable = getattr(replaced_callable_parent_attribute, np_path_sub_attributes[-1])

        return internally_used_name, np_path, replaced_callable
    
    @staticmethod
    def _replace_numpy_method(np_path: str, replacement_function: Callable) -> None:
        np_path_sub_attributes = np_path.split('.')
        module = np
        for np_path_sub_attribute in np_path_sub_attributes[1:-1]:
            module = getattr(module, np_path_sub_attribute)
        setattr(module, np_path_sub_attributes[-1], replacement_function)
        return
    
    @classmethod
    def numpy_replacement(cls, **internally_used_name_to_np_path: Dict[str, str]) -> Callable:
        '''Replaces numpy methods via monkey patching.'''
        internally_used_name, np_path, replaced_callable = cls._numpy_replacement_extract_inputs(internally_used_name_to_np_path)
        def decorator(func: Callable):
            def decorated_function(*args, **kwargs):
                assert internally_used_name not in kwargs.keys()
                kwargs[internally_used_name] = replaced_callable
                return func(*args, **kwargs)
            decorated_function.__name__ = func.__name__
            cls._replace_numpy_method(np_path, decorated_function)
            return decorated_function
        return decorator
    
    # differentiable_method Decorator
    
    class _DifferentiableMethodDecorator:
        def __init__(self, method_names: List[str]):
            for method_name in method_names:
                if not method_name.isidentifier():
                    raise ValueError(f'"{method_name}" is not a valid method name.')
            self.method_names = method_names
            return
        
        def __call__(self, func: Callable) -> Callable:
            if len(self.method_names) == 0:
                self.method_names = [func.__name__]
            for method_name in self.method_names:
                setattr(Variable, method_name, func)
            return func
    
    @classmethod
    def differentiable_method(cls, *method_names: List[str]) -> Callable:
        return cls._DifferentiableMethodDecorator(method_names)

    ###########
    # Methods #
    ###########
    
    def __init__(self, data: np.ndarray, directly_depended_on_variable_to_backward_propagation_function: Dict['Variable', Callable] = dict()):
        '''
        For self.directly_depended_on_variable_to_backward_propagation_function, 
            - Each key is a variable (let's call it var)
            - Each value is a function that takes in d_minimization_target_variable_over_d_self (i.e. the gradient for self) and returns d_minimization_target_variable_over_d_var (i.e. the gradient for var).
                - This function performs one step of backward propagation along one edge in the computation graph.
        '''
        self.data = data
        self.directly_depended_on_variable_to_backward_propagation_function = directly_depended_on_variable_to_backward_propagation_function
        return

    @property
    def directly_depended_on_variables(self):
        return self.directly_depended_on_variable_to_backward_propagation_function.keys()
    
    def depended_on_variables(self) -> Generator:
        return reversed(list(self._depended_on_variables()))
    
    def _depended_on_variables(self) -> Generator:
        '''Yields all variables that self directly or indirectly relies on in reverse topologically sorted order.'''
        visited_variables: Set[Variable] = set()
        def _traverse(var: Variable) -> Generator:
            visited_variables.add(var)
            yield from chain(*map(_traverse, filter(lambda next_var: next_var not in visited_variables, var.directly_depended_on_variables)))
            yield var
        yield from _traverse(self)
    
    def calculate_gradient(self, d_minimization_target_variable_over_d_self: Union[int, float, np.number, np.ndarray]) -> Dict['Variable', Union[int, float, np.number, np.ndarray]]:
        '''
        Backward propagates the gradient (i.e. d_minimization_target_variable_over_d_self) to variables that self directly 
        relies on, i.e. is a direct function of (does not include variables it is transitively or indirectly dependent on).
        
        Returns a dictionary mapping where:
            Each entry's key is a variable (let's call it var) that self directly relies on.
            Each entry's value is gradient for var (i.e. d_minimization_target_variable_over_d_var).
        '''
        directly_depended_on_variable_to_gradient = {}
        for depended_on_variable, calculate_depended_on_variable_gradient in self.directly_depended_on_variable_to_backward_propagation_function.items():
            gradient = calculate_depended_on_variable_gradient(d_minimization_target_variable_over_d_self)
            directly_depended_on_variable_to_gradient[depended_on_variable] = gradient
        return directly_depended_on_variable_to_gradient

    def all(self) -> Union[bool, np.ndarray]: # @toddo test this
        if isinstance(self.data, np.ndarray):
            return self.data.all()
    
    def __eq__(self, other: VariableOperand) -> Union[bool, np.ndarray]: # @toddo test this
        other_data = other.data if isinstance(other, Variable) else other
        return self.data == other_data
    
     def eq(self, other: VariableOperand) -> Union[bool, np.ndarray]: # @toddo test this
         return self == other

     # @todo add gt, gte, le, lte, neq

#######################
# Variable Operations #
#######################

# @todo lots of boiler plate here; can we abstract it out?

# @todo test this with all combinations of types
@Variable.differentiable_method() # @todo test this method
@Variable.numpy_replacement(np_dot='np.dot') # @todo test these numpy methods
def dot(a: VariableOperand, b: VariableOperand, np_dot: Callable, **kwargs) -> VariableOperand:
    a_is_variable = isinstance(a, Variable)
    b_is_variable = isinstance(b, Variable)
    a_data = a.data if a_is_variable else a
    b_data = b.data if b_is_variable else b
    dot_product = np_dot(a_data, b_data, **kwargs)
    if not a_is_variable and not b_is_variable:
        return dot_product
    if len(kwargs) > 0:
        raise ValueError(f'The parameters {[repr(kwarg_name) for kwarg_name in kwargs.keys()]} are not supported for {Variable.__qualname__}.')
    variable_depended_on_by_dot_product_to_backward_propagation_function = {}
    if a_is_variable:
        variable_depended_on_by_dot_product_to_backward_propagation_function[a] = lambda d_minimization_target_over_d_dot_product: d_minimization_target_over_d_dot_product * b_data
    if b_is_variable:
        variable_depended_on_by_dot_product_to_backward_propagation_function[b] = lambda d_minimization_target_over_d_dot_product: d_minimization_target_over_d_dot_product * a_data
    dot_product_variable = Variable(dot_product, variable_depended_on_by_dot_product_to_backward_propagation_function)
    return dot_product_variable

# @todo test this with all combinations of types
@Variable.differentiable_method('subtract', '__sub__') # @todo test these methods
@Variable.numpy_replacement(np_subtract='np.subtract') # @todo test these numpy methods
def subtract(minuend: VariableOperand, subtrahend: VariableOperand, np_subtract: Callable, **kwargs) -> VariableOperand:
    minuend_is_variable = isinstance(minuend, Variable)
    subtrahend_is_variable = isinstance(subtrahend, Variable)
    minuend_data = minuend.data if minuend_is_variable else minuend
    subtrahend_data = subtrahend.data if subtrahend_is_variable else subtrahend
    difference = np_subtract(minuend_data, subtrahend_data, **kwargs)
    if not minuend_is_variable and not subtrahend_is_variable:
        return difference
    if len(kwargs) > 0:
        raise ValueError(f'The parameters {[repr(kwarg_name) for kwarg_name in kwargs.keys()]} are not supported for {Variable.__qualname__}.')
    variable_depended_on_by_difference_to_backward_propagation_function = {}
    if minuend_is_variable:
        variable_depended_on_by_difference_to_backward_propagation_function[minuend] = lambda d_minimization_target_over_d_difference: d_minimization_target_over_d_difference
    if subtrahend_is_variable:
        variable_depended_on_by_difference_to_backward_propagation_function[subtrahend] = lambda d_minimization_target_over_d_difference: d_minimization_target_over_d_difference
    difference_variable = Variable(difference, variable_depended_on_by_difference_to_backward_propagation_function)
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
        raise ValueError(f'The parameters {[repr(kwarg_name) for kwarg_name in kwargs.keys()]} are not supported for {Variable.__qualname__}.')
    variable_depended_on_by_power_to_backward_propagation_function = {}
    if base_is_variable:
        variable_depended_on_by_power_to_backward_propagation_function[base] = lambda d_minimization_target_over_d_power: d_minimization_target_over_d_power * exponent_data * np_float_power(base_data, exponent_data-1)
    if exponent_is_variable:
        variable_depended_on_by_power_to_backward_propagation_function[b] = lambda d_minimization_target_over_d_power: d_minimization_target_over_d_power * power.data*np.log(base_data)
    power_variable = Variable(power, variable_depended_on_by_power_to_backward_propagation_function)
    return power_variable

# @todo support np.power
# @todo support __eq__ and other comparators
