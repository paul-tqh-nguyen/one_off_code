
'''

'''

# @todo fill in docstring

###########
# Imports #
###########

import forbiddenfruit
from itertools import chain
from functools import reduce
from collections import defaultdict
import numpy as np
from typing import List, Set, Tuple, DefaultDict, Dict, Callable, Union, Generator, Optional

from .misc_utilities import *

# @todo verify these imports are used

####################
# Variable Classes #
####################

# @todo we use ValueError in places where NotImplementedError would be more appropriate

VariableOperand = Union[int, float, bool, np.number, np.ndarray, 'Variable']

# @todo add transpose method
# @todo add slicing via the __getitem__ method that returns variables

class Variable:

    ##############
    # Decorators #
    ##############

    # numpy_replacement Decorator
    
    @staticmethod
    def _numpy_replacement_extract_inputs(internally_used_name_to_np_path_specification: Dict[str, Union[List[str], str]]) -> Tuple[str, List[str], List[Callable]]:
        if len(internally_used_name_to_np_path_specification) != 1:
            raise ValueError(f'Only one internally used name can be specified. {len(internally_used_name_to_np_path_specification)} were given.')
        
        internally_used_name, np_path_specification = only_one(internally_used_name_to_np_path_specification.items())
        
        if isinstance(np_path_specification, str):
            np_paths = [np_path_specification]
        elif isinstance(np_path_specification, list):
            np_paths = np_path_specification
        else:
            raise ValueError(f'{np_path_specification} does not specify a numpy callable.')

        if len(np_paths) == 0:
            raise ValueError('No numpy callable specified to be replaced.')
        
        if not internally_used_name.isidentifier():
            raise ValueError(f'"{internally_used_name}" is not a vaild identifier name.')

        replaced_callables: List[Callable] = []
        for np_path in np_paths:

            if not isinstance(np_path, str):
                raise ValueError(f'{np_path_specification} does not specify a numpy callable.')
            
            replaced_callable_parent_attribute = np
            np_path_sub_attributes = np_path.split('.')
    
            if globals().get(np_path_sub_attributes[0]) != np:
                raise ValueError(f'"{np_path}" does not specify a numpy callable.')
            
            for np_path_sub_attribute_index, np_path_sub_attribute in enumerate(np_path_sub_attributes[1:-1], start=1):
                if not hasattr(replaced_callable_parent_attribute, np_path_sub_attribute):
                    raise ValueError(f'"{".".join(np_path_sub_attributes[:np_path_sub_attribute_index+1])}" does not exist.')
                replaced_callable_parent_attribute = getattr(replaced_callable_parent_attribute, np_path_sub_attribute)
    
            if not hasattr(replaced_callable_parent_attribute, np_path_sub_attributes[-1]):
                raise ValueError(f'"{np_path}" does not specify a numpy callable.')
    
            replaced_callable = getattr(replaced_callable_parent_attribute, np_path_sub_attributes[-1])
            
            replaced_callables.append(replaced_callable)
    
        return internally_used_name, np_paths, replaced_callables
    
    @staticmethod
    def _replace_numpy_method(np_path: str, replacement_function: Callable) -> None:
        np_path_sub_attributes = np_path.split('.')
        module = np
        for np_path_sub_attribute in np_path_sub_attributes[1:-1]:
            module = getattr(module, np_path_sub_attribute)
        try:
            setattr(module, np_path_sub_attributes[-1], replacement_function)
        except TypeError as error:
            if len(error.args) == 1 and 'can\'t set attributes of built-in/extension type' in error.args[0]:
                # @todo warn when this case is used stating that it's dangerous.
                forbiddenfruit.curse(module, np_path_sub_attributes[-1], replacement_function)
            else:
                raise error
        return
    
    @classmethod
    def numpy_replacement(cls, **internally_used_name_to_np_path_specification: Dict[str, Union[List[str], str]]) -> Callable:
        '''Replaces numpy methods via monkey patching. The replaced methods are assumed to be behavioraly equivalent.'''
        internally_used_name, np_paths, replaced_callables = cls._numpy_replacement_extract_inputs(internally_used_name_to_np_path_specification)
        def decorator(func: Callable) -> Callable:
            for np_path, replaced_callable in zip(np_paths, replaced_callables):
                def decorated_function(*args, **kwargs):
                    assert internally_used_name not in kwargs.keys()
                    kwargs[internally_used_name] = replaced_callable
                    return func(*args, **kwargs)
                decorated_function.replaced_callable = replaced_callable
                if isinstance(replaced_callable, np.ufunc): # @todo test this with both True and False
                    # @todo instead of passing in is_ufunc, check if the replaced_callable is an instance of ufunc
                    # @todo abstract this out
                    # @todo support other ufunc methods
                    def reducer(array, axis, dtype, out, **reducer_kwargs): # @todo add type hints
                        if axis is not None: # @todo test this
                            raise ValueError(f'The parameter "axis" is not supported for {np_path}.reduce.')
                        if dtype is not None: # @todo test this
                            raise ValueError(f'The parameter "dtype" is not supported for {np_path}.reduce.')
                        if out is not None: # @todo test this
                            raise ValueError(f'The parameter "out" is not supported for {np_path}.reduce.')
                        if len(reducer_kwargs) > 0: # @todo test this
                            raise ValueError(f'The parameters {[repr(kwarg_name) for kwarg_name in reducer_kwargs.keys()]} are not supported for {np_path}.reduce.')
                        reduction = replaced_callable.reduce(array, axis, dtype, out)
                        return reduction
                    decorated_function.reduce = reducer
                decorated_function.__name__ = func.__name__
                cls._replace_numpy_method(np_path, decorated_function)
            return decorated_function
        return decorator
    
    # new_method Decorator
    
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
    def new_method(cls, *method_names: List[str]) -> Callable:
        return cls._DifferentiableMethodDecorator(method_names)

    ###########
    # Methods #
    ###########
    
    def __init__(self, data: np.ndarray, directly_depended_on_variable_to_backward_propagation_functions: Dict['Variable', List[Callable]] = dict()):
        '''
        For self.directly_depended_on_variable_to_backward_propagation_functions, 
            - Each key is a variable (let's call it var)
            - Each value is a list of functions that take in d_minimization_target_variable_over_d_self (i.e. the gradient for self) and returns d_minimization_target_variable_over_d_var (i.e. the gradient for var).
                - When all of these functions are executed, they perform one step of backward propagation along one edge in the computation graph.
                - There need to be multiple functions since self can direcltly depend on depended_on_variable in multiple ways, e.g. y = x**2 + x**3
        '''
        self.data = data
        self.directly_depended_on_variable_to_backward_propagation_functions = directly_depended_on_variable_to_backward_propagation_functions
        return

    @property
    def directly_depended_on_variables(self):
        return self.directly_depended_on_variable_to_backward_propagation_functions.keys()

    @property
    def shape(self):
        return self.data.shape
    
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
    
    def directly_backward_propagate_gradient(self, d_minimization_target_variable_over_d_self: Union[int, float, np.number, np.ndarray]) -> Dict['Variable', Union[int, float, np.number, np.ndarray]]:
        '''
        Backward propagates the gradient (i.e. d_minimization_target_variable_over_d_self) to variables that self directly 
        relies on, i.e. is a direct function of (does not include variables it is transitively or indirectly dependent on).
        
        Returns a dictionary mapping where:
            Each entry's key is a variable (let's call it var) that self directly relies on.
            Each entry's value is gradient for var (i.e. d_minimization_target_variable_over_d_var).
        '''
        directly_depended_on_variable_to_gradient = defaultdict(lambda: 0)
        for depended_on_variable, backward_propagation_functions in self.directly_depended_on_variable_to_backward_propagation_functions.items():
            for calculate_depended_on_variable_gradient in backward_propagation_functions:
                gradient = calculate_depended_on_variable_gradient(d_minimization_target_variable_over_d_self)
                directly_depended_on_variable_to_gradient[depended_on_variable] += gradient
        directly_depended_on_variable_to_gradient = dict(directly_depended_on_variable_to_gradient)
        return directly_depended_on_variable_to_gradient

    def __repr__(self):
        if isinstance(self.data, np.ndarray):
            old_prefix = 'array'
            new_prefix = self.__class__.__name__
            representation = repr(self.data).replace('\n'+' '*len(old_prefix), '\n'+' '*len(new_prefix)).replace(old_prefix, new_prefix)
        else:
            representation = f'{self.__class__.__name__}({self.data})'
        return representation
    
    def all(self, **kwargs) -> Union[bool, np.ndarray]:
        return self.data.all(**kwargs) if isinstance(self.data, np.ndarray) else bool(self.data)
    
    def any(self, **kwargs) -> Union[bool, np.ndarray]:
        return self.data.any(**kwargs) if isinstance(self.data, np.ndarray) else bool(self.data)
    
    def sum(self, **kwargs) -> 'Variable':
        summation = self.data.sum(**kwargs)
        if kwargs.get('axis', UNIQUE_BOGUS_RESULT_IDENTIFIER) == None:
            del kwargs['axis']
        if kwargs.get('out', UNIQUE_BOGUS_RESULT_IDENTIFIER) == None:
            del kwargs['out']
        if len(kwargs) > 0:
            raise ValueError(f'The parameters {[repr(kwarg_name) for kwarg_name in kwargs.keys()]} are not supported for {Variable.__qualname__}.')
        variable_depended_on_by_summation_to_backward_propagation_functions = defaultdict(list)
        variable_depended_on_by_summation_to_backward_propagation_functions[self].append(lambda d_minimization_target_over_d_summation: d_minimization_target_over_d_summation * np.ones(self.shape))
        summation_variable = Variable(summation, dict(variable_depended_on_by_summation_to_backward_propagation_functions))
        return summation_variable
        
    def __abs__(self, **kwargs) -> 'Variable':
        return self.abs(**kwargs)
    
    def abs(self, **kwargs) -> 'Variable':
        absolute_value = np.abs(self.data, **kwargs)
        if len(kwargs) > 0:
            raise ValueError(f'The parameters {[repr(kwarg_name) for kwarg_name in kwargs.keys()]} are not supported for {Variable.__qualname__}.')
        variable_depended_on_by_absolute_value_to_backward_propagation_functions = defaultdict(list)
        d_absolute_value_over_d_self = np.ones(self.shape, dtype=float)
        d_absolute_value_over_d_self[self.data==0] = 0
        d_absolute_value_over_d_self[self.data<0] = -1
        variable_depended_on_by_absolute_value_to_backward_propagation_functions[self].append(lambda d_minimization_target_over_d_absolute_value: d_minimization_target_over_d_absolute_value * d_absolute_value_over_d_self)
        absolute_value_variable = Variable(absolute_value, dict(variable_depended_on_by_absolute_value_to_backward_propagation_functions))
        return absolute_value_variable

    def __radd__(self, summand: VariableOperand) -> VariableOperand:
        return self.add(summand) # @todo test this

    def __rsub__(self, minuend: VariableOperand) -> VariableOperand:
        return self.neg().add(minuend) # @todo test this

    def __rmul__(self, factor: VariableOperand) -> VariableOperand:
        return self.multiply(factor) # @todo test this

    def __rtruediv__(self, dividend: VariableOperand) -> VariableOperand:
        return self.pow(-1).multiply(dividend) # @todo test this
    
    def sigmoid(self) -> 'Variable':
        return 1 / (np.exp(self.neg())+1) # @todo test this

    def squared_error(self, target: VariableOperand) -> 'Variable': # @todo test this
        return self.subtract(target).pow(2.0)

    def bce_loss(self, target: VariableOperand,  epsilon: float = 1e-16) -> 'Variable': # @todo test this
        prediction = self.add(epsilon) if self.eq(0).all() else self
        loss = -(target*np.log(prediction) + (1-target)*np.log(1-prediction))
        return loss

##########################################
# Variable Non-Differentiable Operations #
##########################################

# @todo lots of boiler plate ; abstract it out

@Variable.new_method('round', 'around') # @todo test this
@Variable.numpy_replacement(np_around='np.around')
def around(operand: VariableOperand, np_around: Callable, **kwargs) -> VariableOperand:
    if kwargs.get('out') is not None:
        raise ValueError(f'The parameter {repr("out")} is not supported for {Variable.__qualname__}.')
    operand_is_variable = isinstance(operand, Variable)
    operand_data = operand.data if operand_is_variable else operand
    return np_around(operand_data, **kwargs)

@Variable.numpy_replacement(np_any='np.any') # @todo retry using @Variable.new_method here
def any(operand: VariableOperand, np_any: Callable, **kwargs) -> Union[bool, np.ndarray]:
    operand_is_variable = isinstance(operand, Variable)
    operand_data = operand.data if operand_is_variable else operand
    return np_any(operand_data, **kwargs)

@Variable.numpy_replacement(np_all='np.all') # @todo retry using @Variable.new_method here
def all(operand: VariableOperand, np_all: Callable, **kwargs) -> Union[bool, np.ndarray]:
    operand_is_variable = isinstance(operand, Variable)
    operand_data = operand.data if operand_is_variable else operand
    return np_all(operand_data, **kwargs)

@Variable.new_method('isclose')
@Variable.numpy_replacement(np_isclose='np.isclose')
def isclose(a: VariableOperand, b: VariableOperand, np_isclose: Callable, **kwargs) -> VariableOperand: # @todo review this return type
    a_is_variable = isinstance(a, Variable)
    b_is_variable = isinstance(b, Variable)
    a_data = a.data if a_is_variable else a
    b_data = b.data if b_is_variable else b
    return np_isclose(a_data, b_data, **kwargs)

@Variable.new_method('equal', 'eq', '__eq__')
@Variable.numpy_replacement(np_equal='np.equal')
def equal(a: VariableOperand, b: VariableOperand, np_equal: Callable, **kwargs) -> VariableOperand: # @todo review this return type
    a_is_variable = isinstance(a, Variable)
    b_is_variable = isinstance(b, Variable)
    a_data = a.data if a_is_variable else a
    b_data = b.data if b_is_variable else b
    return np_equal(a_data, b_data, **kwargs)

@Variable.new_method('not_equal', 'neq', 'ne', '__ne__')
@Variable.numpy_replacement(np_not_equal='np.not_equal')
def not_equal(a: VariableOperand, b: VariableOperand, np_not_equal: Callable, **kwargs) -> VariableOperand: # @todo review this return type
    a_is_variable = isinstance(a, Variable)
    b_is_variable = isinstance(b, Variable)
    a_data = a.data if a_is_variable else a
    b_data = b.data if b_is_variable else b
    return np_not_equal(a_data, b_data, **kwargs)

@Variable.new_method('greater', 'greater_than', 'gt', '__gt__')
@Variable.numpy_replacement(np_greater='np.greater')
def greater(a: VariableOperand, b: VariableOperand, np_greater: Callable, **kwargs) -> VariableOperand: # @todo review this return type
    a_is_variable = isinstance(a, Variable)
    b_is_variable = isinstance(b, Variable)
    a_data = a.data if a_is_variable else a
    b_data = b.data if b_is_variable else b
    return np_greater(a_data, b_data, **kwargs)

@Variable.new_method('greater_equal', 'greater_than_equal', 'ge', 'gte', '__ge__')
@Variable.numpy_replacement(np_greater_equal='np.greater_equal')
def greater_equal(a: VariableOperand, b: VariableOperand, np_greater_equal: Callable, **kwargs) -> VariableOperand: # @todo review this return type
    a_is_variable = isinstance(a, Variable)
    b_is_variable = isinstance(b, Variable)
    a_data = a.data if a_is_variable else a
    b_data = b.data if b_is_variable else b
    return np_greater_equal(a_data, b_data, **kwargs)

@Variable.new_method('less', 'less_than', 'lt', '__lt__')
@Variable.numpy_replacement(np_less='np.less')
def less(a: VariableOperand, b: VariableOperand, np_less: Callable, **kwargs) -> VariableOperand: # @todo review this return type
    a_is_variable = isinstance(a, Variable)
    b_is_variable = isinstance(b, Variable)
    a_data = a.data if a_is_variable else a
    b_data = b.data if b_is_variable else b
    return np_less(a_data, b_data, **kwargs)

@Variable.new_method('less_equal', 'less_than_equal', 'le', 'lte', '__le__')
@Variable.numpy_replacement(np_less_equal='np.less_equal')
def less_equal(a: VariableOperand, b: VariableOperand, np_less_equal: Callable, **kwargs) -> VariableOperand: # @todo review this return type
    a_is_variable = isinstance(a, Variable)
    b_is_variable = isinstance(b, Variable)
    a_data = a.data if a_is_variable else a
    b_data = b.data if b_is_variable else b
    return np_less_equal(a_data, b_data, **kwargs)

######################################
# Variable Differentiable Operations #
######################################

# @todo lots of boiler plate ; abstract it out.

# @todo support int, float, and all the np types of various sizes for each operation

# @todo support np.squeeze
# @todo support np.power

@Variable.new_method()
@Variable.numpy_replacement(np_dot=['np.dot', 'np.ndarray.dot'])
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
    variable_depended_on_by_dot_product_to_backward_propagation_functions = defaultdict(list)
    if a_is_variable:
        variable_depended_on_by_dot_product_to_backward_propagation_functions[a].append(lambda d_minimization_target_over_d_dot_product: d_minimization_target_over_d_dot_product * b_data)
    if b_is_variable:
        variable_depended_on_by_dot_product_to_backward_propagation_functions[b].append(lambda d_minimization_target_over_d_dot_product: d_minimization_target_over_d_dot_product * a_data)
    dot_product_variable = Variable(dot_product, dict(variable_depended_on_by_dot_product_to_backward_propagation_functions))
    return dot_product_variable

@Variable.new_method('power', 'pow', '__pow__')
@Variable.numpy_replacement(np_float_power='np.float_power')
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
    variable_depended_on_by_power_to_backward_propagation_functions = defaultdict(list)
    if base_is_variable:
        variable_depended_on_by_power_to_backward_propagation_functions[base].append(lambda d_minimization_target_over_d_power: d_minimization_target_over_d_power * exponent_data * np_float_power(base_data, exponent_data-1))
    if exponent_is_variable:
        variable_depended_on_by_power_to_backward_propagation_functions[exponent].append(lambda d_minimization_target_over_d_power: d_minimization_target_over_d_power * power.data * np.log(base_data))
    power_variable = Variable(power, dict(variable_depended_on_by_power_to_backward_propagation_functions))
    return power_variable

@Variable.new_method('multiply', '__mul__')
@Variable.numpy_replacement(np_multiply='np.multiply')
def multiply(a: VariableOperand, b: VariableOperand, np_multiply: Callable, **kwargs) -> VariableOperand:
    a_is_variable = isinstance(a, Variable)
    b_is_variable = isinstance(b, Variable)
    a_data = a.data if a_is_variable else a
    b_data = b.data if b_is_variable else b
    product = np_multiply(a_data, b_data, **kwargs)
    if not a_is_variable and not b_is_variable:
        return product
    if len(kwargs) > 0:
        raise ValueError(f'The parameters {[repr(kwarg_name) for kwarg_name in kwargs.keys()]} are not supported for {Variable.__qualname__}.')
    variable_depended_on_by_product_to_backward_propagation_functions = defaultdict(list)
    if a_is_variable:
        variable_depended_on_by_product_to_backward_propagation_functions[a].append(lambda d_minimization_target_over_d_product: d_minimization_target_over_d_product * b_data)
    if b_is_variable:
        variable_depended_on_by_product_to_backward_propagation_functions[b].append(lambda d_minimization_target_over_d_product: d_minimization_target_over_d_product * a_data)
    product_variable = Variable(product, dict(variable_depended_on_by_product_to_backward_propagation_functions))
    return product_variable

@Variable.new_method('divide', '__truediv__')
@Variable.numpy_replacement(np_divide='np.divide')
def divide(dividend: VariableOperand, divisor: VariableOperand, np_divide: Callable, **kwargs) -> VariableOperand:
    dividend_is_variable = isinstance(dividend, Variable)
    divisor_is_variable = isinstance(divisor, Variable)
    dividend_data = dividend.data if dividend_is_variable else dividend
    divisor_data = divisor.data if divisor_is_variable else divisor
    quotient = np_divide(dividend_data, divisor_data, **kwargs)
    if not dividend_is_variable and not divisor_is_variable:
        return quotient
    if len(kwargs) > 0:
        raise ValueError(f'The parameters {[repr(kwarg_name) for kwarg_name in kwargs.keys()]} are not supported for {Variable.__qualname__}.')
    variable_depended_on_by_quotient_to_backward_propagation_functions = defaultdict(list)
    if dividend_is_variable:
        variable_depended_on_by_quotient_to_backward_propagation_functions[dividend].append(lambda d_minimization_target_over_d_quotient: d_minimization_target_over_d_quotient / divisor_data)
    if divisor_is_variable:
        variable_depended_on_by_quotient_to_backward_propagation_functions[divisor].append(lambda d_minimization_target_over_d_quotient: d_minimization_target_over_d_quotient * -dividend_data * (divisor_data ** -2.0))
    quotient_variable = Variable(quotient, dict(variable_depended_on_by_quotient_to_backward_propagation_functions))
    return quotient_variable

@Variable.new_method('subtract', '__sub__')
@Variable.numpy_replacement(np_subtract='np.subtract')
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
    variable_depended_on_by_difference_to_backward_propagation_functions = defaultdict(list)
    if minuend_is_variable:
        variable_depended_on_by_difference_to_backward_propagation_functions[minuend].append(lambda d_minimization_target_over_d_difference: d_minimization_target_over_d_difference)
    if subtrahend_is_variable:
        variable_depended_on_by_difference_to_backward_propagation_functions[subtrahend].append(lambda d_minimization_target_over_d_difference: -d_minimization_target_over_d_difference)
    difference_variable = Variable(difference, dict(variable_depended_on_by_difference_to_backward_propagation_functions))
    return difference_variable

@Variable.new_method('add', '__add__')
@Variable.numpy_replacement(np_add='np.add')
def add(a: VariableOperand, b: VariableOperand, np_add: Callable, **kwargs) -> VariableOperand:
    a_is_variable = isinstance(a, Variable)
    b_is_variable = isinstance(b, Variable)
    a_data = a.data if a_is_variable else a
    b_data = b.data if b_is_variable else b
    summation = np_add(a_data, b_data, **kwargs)
    if not a_is_variable and not b_is_variable:
        return summation
    if len(kwargs) > 0:
        raise ValueError(f'The parameters {[repr(kwarg_name) for kwarg_name in kwargs.keys()]} are not supported for {Variable.__qualname__}.')
    variable_depended_on_by_summation_to_backward_propagation_functions = defaultdict(list)
    if a_is_variable:
        variable_depended_on_by_summation_to_backward_propagation_functions[a].append(lambda d_minimization_target_over_d_summation: d_minimization_target_over_d_summation)
    if b_is_variable:
        variable_depended_on_by_summation_to_backward_propagation_functions[b].append(lambda d_minimization_target_over_d_summation: d_minimization_target_over_d_summation)
    summation_variable = Variable(summation, dict(variable_depended_on_by_summation_to_backward_propagation_functions))
    return summation_variable

# @todo support np.mean

@Variable.numpy_replacement(np_sum='np.sum') # @todo retry using @Variable.new_method here
def sum(operand: VariableOperand, np_sum: Callable, **kwargs) -> VariableOperand:
    operand_is_variable = isinstance(operand, Variable)
    operand_data = operand.data if operand_is_variable else operand
    summation = np_sum(operand_data, **kwargs)
    if not operand_is_variable:
        return summation
    if len(kwargs) > 0:
        raise ValueError(f'The parameters {[repr(kwarg_name) for kwarg_name in kwargs.keys()]} are not supported for {Variable.__qualname__}.')
    variable_depended_on_by_summation_to_backward_propagation_functions = defaultdict(list)
    variable_depended_on_by_summation_to_backward_propagation_functions[operand].append(lambda d_minimization_target_over_d_summation: d_minimization_target_over_d_summation * np.ones(operand.shape))
    summation_variable = Variable(summation, dict(variable_depended_on_by_summation_to_backward_propagation_functions))
    return summation_variable

@Variable.new_method('log', 'natural_log')
@Variable.numpy_replacement(np_log='np.log')
def log(operand: VariableOperand, np_log: Callable, **kwargs) -> VariableOperand:
    operand_is_variable = isinstance(operand, Variable)
    operand_data = operand.data if operand_is_variable else operand
    log_result = np_log(operand_data, **kwargs)
    if not operand_is_variable:
        return log_result
    if len(kwargs) > 0:
        raise ValueError(f'The parameters {[repr(kwarg_name) for kwarg_name in kwargs.keys()]} are not supported for {Variable.__qualname__}.')
    variable_depended_on_by_log_result_to_backward_propagation_functions = defaultdict(list)
    variable_depended_on_by_log_result_to_backward_propagation_functions[operand].append(lambda d_minimization_target_over_d_log_result: d_minimization_target_over_d_log_result * np.float_power(operand_data, -1))
    log_result_variable = Variable(log_result, dict(variable_depended_on_by_log_result_to_backward_propagation_functions))
    return log_result_variable

@Variable.numpy_replacement(np_abs=['np.abs', 'np.absolute'])
def abs(operand: VariableOperand, np_abs: Callable, **kwargs) -> VariableOperand:
    operand_is_variable = isinstance(operand, Variable)
    operand_data = operand.data if operand_is_variable else operand
    absolute_value = np_abs(operand_data, **kwargs)
    if not operand_is_variable:
        return absolute_value
    if len(kwargs) > 0:
        raise ValueError(f'The parameters {[repr(kwarg_name) for kwarg_name in kwargs.keys()]} are not supported for {Variable.__qualname__}.')
    variable_depended_on_by_absolute_value_to_backward_propagation_functions = defaultdict(list)
    variable_depended_on_by_absolute_value_to_backward_propagation_functions[operand].append(lambda d_minimization_target_over_d_absolute_value: d_minimization_target_over_d_absolute_value * np.ones(operand.shape))
    absolute_value_variable = Variable(absolute_value, dict(variable_depended_on_by_absolute_value_to_backward_propagation_functions))
    return absolute_value_variable

@Variable.new_method('matmul', '__matmul__')
@Variable.numpy_replacement(np_matmul='np.matmul')
def matmul(a: VariableOperand, b: VariableOperand, np_matmul: Callable, **kwargs) -> VariableOperand:
    a_is_variable = isinstance(a, Variable)
    b_is_variable = isinstance(b, Variable)
    a_data = a.data if a_is_variable else a
    b_data = b.data if b_is_variable else b
    matrix_product = np_matmul(a_data, b_data, **kwargs)
    if not a_is_variable and not b_is_variable:
        return matrix_product
    if len(kwargs) > 0:
        raise ValueError(f'The parameters {[repr(kwarg_name) for kwarg_name in kwargs.keys()]} are not supported for {Variable.__qualname__}.')
    variable_depended_on_by_matrix_product_to_backward_propagation_functions = defaultdict(list)
    if a_is_variable:
        variable_depended_on_by_matrix_product_to_backward_propagation_functions[a].append(lambda d_minimization_target_over_d_matrix_product: np_matmul(d_minimization_target_over_d_matrix_product, b_data.T))
    if b_is_variable:
        variable_depended_on_by_matrix_product_to_backward_propagation_functions[b].append(lambda d_minimization_target_over_d_matrix_product: np_matmul(a_data.T, d_minimization_target_over_d_matrix_product))
    matrix_product_variable = Variable(matrix_product, dict(variable_depended_on_by_matrix_product_to_backward_propagation_functions))
    return matrix_product_variable

@Variable.new_method('expand_dims')
@Variable.numpy_replacement(np_expand_dims='np.expand_dims')
def expand_dims(operand: VariableOperand, axis: Union[Tuple[int], int], np_expand_dims: Callable, **kwargs) -> VariableOperand:
    '''For variables, this returns a copy instead of a view.'''
    operand_is_variable = isinstance(operand, Variable)
    operand_data = operand.data if operand_is_variable else operand
    expanded_operand = np_expand_dims(operand_data, axis, **kwargs)
    if not operand_is_variable:
        return expanded_operand
    if len(kwargs) > 0:
        raise ValueError(f'The parameters {[repr(kwarg_name) for kwarg_name in kwargs.keys()]} are not supported for {Variable.__qualname__}.')
    variable_depended_on_by_expanded_operand_to_backward_propagation_functions = defaultdict(list)
    # @todo test that we don't have infinite recursion when backgpropagating through a series of np.exapnd_dims calls and np.squeeze calls
    variable_depended_on_by_expanded_operand_to_backward_propagation_functions[operand].append(lambda d_minimization_target_over_d_expanded_operand: d_minimization_target_over_d_expanded_operand.squeeze(axis))
    expanded_operand_variable = Variable(expanded_operand.copy(), dict(variable_depended_on_by_expanded_operand_to_backward_propagation_functions))
    return expanded_operand_variable

@Variable.new_method('exp')
@Variable.numpy_replacement(np_exp='np.exp')
def exp(operand: VariableOperand, np_exp: Callable, **kwargs) -> VariableOperand:
    del np_exp
    return np.float_power(np.e, operand, **kwargs)

@Variable.new_method('__neg__', 'neg', 'negative', 'negate')
@Variable.numpy_replacement(np_negative='np.negative')
def neg(operand: VariableOperand, np_negative: Callable, **kwargs) -> VariableOperand:
    del np_negative
    return np.multiply(-1.0, operand, **kwargs)
