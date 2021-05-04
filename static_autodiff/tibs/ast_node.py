
'''
'''

# TODO fill in doc string

###########
# Imports #
###########

import decorator
import inspect
import uuid
import typing
import typing_extensions
from abc import ABC, abstractmethod
import inspect
import pyparsing
import operator
import itertools
from functools import reduce

from .parser_utilities import (
    ParseError,
    SemanticError,
    BASE_TYPES,
    BaseTypeName,
    sanity_check_base_types,
    TensorTypeParserElementBaseTypeTracker,
    AtomicLiteralParserElementBaseTypeTracker,
    LiteralASTNodeClassBaseTypeTracker
)
from .misc_utilities import *

# TODO make sure imports are used
# TODO make sure these imports are ordered in some way

###############
# AST Helpers #
###############

# type hint checkers

def is_instance_type_hint(instance: typing.Any, type_declaration: typing.Union[type, typing._Final]) -> bool:
    is_instance = BOGUS_TOKEN
    if isinstance(type_declaration, type):
        is_instance = isinstance(instance, type_declaration)
    elif type_declaration.__origin__ is typing_extensions.Literal:
        is_instance = instance in type_declaration.__args__
    elif type_declaration.__origin__ is list:
        element_type_declaration = only_one(type_declaration.__args__)
        is_instance = isinstance(instance, list) and all(is_instance_type_hint(element, element_type_declaration) for element in instance)
    elif type_declaration.__origin__ is tuple:
        is_instance = isinstance(instance, tuple) and all(is_instance_type_hint(instance_element, element_annotation) for instance_element, element_annotation in zip(instance, type_declaration.__args__))
    elif type_declaration.__origin__ is typing.Union:
        is_instance = any(is_instance_type_hint(instance, possible_type) for possible_type in type_declaration.__args__)
    else:
        breakpoint() # TODO remove this
        raise NotImplementedError(f'Checking membership of {type_declaration} not currently supported.')
    assert is_instance is not BOGUS_TOKEN
    return is_instance

# AST Node copy helpers

def _copy_function_for_typing_type_hint(type_declaration: typing._Final) -> typing.Callable[[typing.Any], typing.Any] :
    if type_declaration.__origin__ is list:
        element_annotation = only_one(type_declaration.__args__)
        element_copy_func = _copy_function_for_type_declaration(element_annotation)
        def copy_func(value: type_declaration) -> type_declaration:
            ASSERT.SemanticError(is_instance_type_hint(value, type_declaration), f'{value} expected to be an instance of {type_declaration}.') 
            return [element_copy_func(element) for element in value]
    elif type_declaration.__origin__ is tuple:
        element_copy_funcs = [_copy_function_for_type_declaration(element_annotation) for element_annotation in type_declaration.__args__]
        def copy_func(value: type_declaration) -> type_declaration:
            ASSERT.SemanticError(is_instance_type_hint(value, type_declaration), f'{value} expected to be an instance of {type_declaration}.') 
            return tuple(element_copy_func(element) for element, element_copy_func in zip(value, element_copy_funcs))
    elif type_declaration.__origin__ is typing.Union:
        copy_func_to_type: typing.Tuple[typing.Callable[[typing.Union[typing._Final, type]], typing.Callable[[typing.Any], typing.Any]], typing.Union[typing._Final, type]] = tuple(
            (_copy_function_for_type_declaration(annotation), annotation)
            for annotation in type_declaration.__args__
        )
        def copy_func(value: type_declaration) -> type_declaration:
            ASSERT.SemanticError(is_instance_type_hint(value, type_declaration), f'{value} expected to be an instance of {type_declaration}.') 
            relevant_copy_function = next(copy_func for copy_func, potential_type in copy_func_to_type if is_instance_type_hint(value, potential_type))
            return relevant_copy_function(value)
    elif type_declaration.__origin__ is typing_extensions.Literal:
        literal_value_types = tuple(map(type, type_declaration.__args__))
        type_declaration_equivalent_wrt_copying = typing.Union[literal_value_types]
        union_copy_func = _copy_function_for_type_declaration(type_declaration_equivalent_wrt_copying)
        def copy_func(value: type_declaration) -> type_declaration:
            ASSERT.SemanticError(value in type_declaration.__args__, f'{value} expected to be one of {type_declaration.__args__}.')
            return union_copy_func(value)
    else:
        breakpoint() # TODO remove this
        raise NotImplementedError(f'Copying instances {type_declaration} not currently supported.')    
    return copy_func

def _copy_function_for_type_declaration(type_declaration: typing.Union[type, typing._Final]) -> typing.Callable[[typing.Any], typing.Any]:
    '''Traversal of the type declaration (when a type hint and not a class) happens eagerly (and not at runtime when copy_func is called).'''
    if isinstance(type_declaration, type) and issubclass(type_declaration, ASTNode):
        def copy_func(value: ASTNode) -> ASTNode:
            ASSERT.SemanticError(isinstance(value, ASTNode), f'{value} expected to be an instance of {ASTNode.__qualname__}.') 
            return value.copy()
    elif isinstance(type_declaration, typing._Final):
        copy_func = _copy_function_for_typing_type_hint(type_declaration)
    elif type_declaration is NoneType:
        def copy_func(none_value: NoneType) -> None:
            ASSERT.SemanticError(none_value is None, f'{none_value} expected to be {None}.')
            return 
    elif type_declaration in (str, int, float, bool):
        def copy_func(value: type_declaration) -> type_declaration:
            ASSERT.SemanticError(isinstance(value, type_declaration), f'{value} expected to be an instance of {type_declaration.__qualname__}.') 
            return type_declaration(value)
    else:
        breakpoint() # TODO remove this
        raise NotImplementedError(f'Copying instances {type_declaration} not currently supported.')    
    return copy_func

# Decorators

def rename_function_args(**old_to_new_arg_names):
    
    def result_decorator(old_function):
        parameters = inspect.signature(old_function).parameters.values()

        eval_dict = {f'annotation_{i}': parameter.annotation for i, parameter in enumerate(parameters) if parameter.annotation is not inspect.Parameter.empty}
        eval_dict.update({'old_function': old_function})

        new_arg_names = tuple(old_to_new_arg_names.get(parameter.name, parameter.name) for parameter in parameters)
        input_string_to_old_function = ', '.join(new_arg_names)
        body_string = f'return old_function({input_string_to_old_function})'
        
        old_function_name = old_function.__name__ if old_function.__name__.isidentifier() else f'anonymous_function_{uuid.uuid4().int}'
        new_signature_string = ', '.join(
            new_arg_names[i] + ('' if parameter.annotation is inspect.Parameter.empty else f': annotation_{i}')
            for i, parameter in enumerate(parameters)
        )
        name_and_signature_string = f'{old_function_name}({new_signature_string})'

        new_parameters = (
            parameter.replace(name=old_to_new_arg_names.get(parameter.name, parameter.name))
            for parameter in inspect.signature(old_function).parameters.values()
        )
        new_signature = inspect.signature(old_function).replace(parameters=new_parameters)
        
        decorated_function = decorator.FunctionMaker.create(
            name_and_signature_string,
            body_string,
            eval_dict,
            defaults=old_function.__defaults__,
            addsource=True,
            __wrapped__=old_function,
            __signature__=new_signature
        )
        return decorated_function
    
    return result_decorator

#####################
# AST Functionality #
#####################

class ASTNode(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def parse_action(cls, s: str, loc: int, tokens: pyparsing.ParseResults) -> 'ASTNode':
        raise NotImplementedError

    def copy(self) -> 'ASTNode':
        '''Raises an exception when an ill-formed AST node is encountered.'''
        init_kwargs = {}
        parameters = list(inspect.signature(self.__init__).parameters.values())
        assert (len(parameters) and parameters[0].name == 'self' and not inspect.ismethod(self.__init__)) ^ (all(parameter.name != 'self' for parameter in parameters) and inspect.ismethod(self.__init__))
        parameters = filter(lambda parameter: parameter != 'self', parameters)
        for parameter in parameters:
            assert parameter.annotation is not inspect.Parameter.empty
            parameter_value = getattr(self, parameter.name)
            copy_func = _copy_function_for_type_declaration(parameter.annotation)
            init_kwargs[parameter.name] = copy_func(parameter_value)
        return self.__class__(**init_kwargs)

    @abstractmethod
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

    @abstractmethod
    def traverse(self) -> typing.Generator['ASTNode', None, None]:
        '''This method will yield itself.'''
        raise NotImplementedError
    
    def __repr__(self) -> str:
        attributes_string = ', '.join(f'{k}={repr(self.__dict__[k])}' for k in sorted(self.__dict__.keys()))
        return f'{self.__class__.__name__}({attributes_string})'

    def __eq__(self, other: 'ASTNode') -> bool:
        '''Determines syntactic equivalence, not semantic equivalence or canonicality.'''
        raise NotImplementedError

class AtomASTNodeType(type):
    
    base_ast_node_class = ASTNode
    
    def __new__(meta, class_name: str, bases: typing.Tuple[type, ...], attributes: dict) -> type:
        method_names = ('__init__', 'parse_action', '__eq__')
        for method_name in method_names:
            assert method_name not in attributes.keys(), f'{method_name} already defined for class {class_name}'
        
        updated_attributes = dict(attributes)
        # Required Declarations
        value_type: type = updated_attributes.pop('value_type')
        token_checker: typing.Callable[[typing.Any], bool] = updated_attributes.pop('token_checker')
        # Optional Declarations
        value_attribute_name: str = updated_attributes.pop('value_attribute_name', 'value')
        dwimming_func: typing.Callable[[typing.Any], value_type] = updated_attributes.pop('dwimming_func', lambda token: token)
        
        @rename_function_args(value=value_attribute_name)
        def __init__(self, value: value_type) -> None:
            assert isinstance(value, value_type)
            setattr(self, value_attribute_name, value)
            return
        
        @classmethod
        def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> class_name:
            token = only_one(tokens)
            assert token_checker(token)
            value: value_type = dwimming_func(token)
            node_instance: cls = cls(value)
            return node_instance
        
        def __eq__(self, other: ASTNode) -> bool:
            other_value = getattr(other, value_attribute_name, BOGUS_TOKEN)
            same_type = type(self) is type(other)
            
            self_value = getattr(self, value_attribute_name)
            same_value_type = type(self_value) is type(other_value)
            same_value = self_value == other_value
            return same_type and same_value_type and same_value
        
        def traverse(self) -> typing.Generator[class_name, None, None]:
            yield self
            return
        
        updated_attributes['__init__'] = __init__
        updated_attributes['parse_action'] = parse_action
        updated_attributes['__eq__'] = __eq__
        updated_attributes['traverse'] = traverse
        
        result_class = type(class_name, bases+(meta.base_ast_node_class,), updated_attributes)
        assert all(hasattr(result_class, method_name) for method_name in method_names)
        return result_class

class StatementASTNode(ASTNode):
    # TODO should this be an abstract class?
    pass

class ExpressionASTNode(StatementASTNode):
    # TODO should this be an abstract class?
    pass

class ExpressionAtomASTNodeType(AtomASTNodeType):
    '''The atomic bases that make up expressions.'''
    base_ast_node_class = ExpressionASTNode

class BinaryOperationExpressionASTNode(ExpressionASTNode):
    # TODO should this be an abstract class?

    def __init__(self, left_arg: ExpressionASTNode, right_arg: ExpressionASTNode) -> None:
        self.left_arg: ExpressionASTNode = left_arg
        self.right_arg: ExpressionASTNode = right_arg
        return

    @classmethod
    def parse_action(cls, _s: str, _loc: int, group_tokens: pyparsing.ParseResults) -> 'BinaryOperationExpressionASTNode':
        # TODO is this used? do we need this?
        tokens = only_one(group_tokens)
        assert len(tokens) >= 2
        node_instance: cls = reduce(cls, tokens)
        return node_instance
    
    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.left_arg == other.left_arg and \
            self.right_arg == other.right_arg
    
    def traverse(self) -> typing.Generator[ExpressionASTNode, None, None]:
        yield self
        yield from self.left_arg.traverse()
        yield from self.right_arg.traverse()
        return
        
class UnaryOperationExpressionASTNode(ExpressionASTNode):
    # TODO should this be an abstract class?

    def __init__(self, arg: ExpressionASTNode) -> None:
        self.arg: ExpressionASTNode = arg
        return

    @classmethod
    def parse_action(cls, _s: str, _loc: int, group_tokens: pyparsing.ParseResults) -> 'UnaryOperationExpressionASTNode':
        # TODO is this used? do we need this?
        tokens = only_one(group_tokens)
        token = only_one(tokens)
        node_instance: cls = cls(token)
        return node_instance
    
    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and self.arg == other.arg
    
    def traverse(self) -> typing.Generator[ExpressionASTNode, None, None]:
        yield self
        yield from self.arg.traverse()
        return
    
class ArithmeticExpressionASTNode(ExpressionASTNode):
    # TODO should this be an abstract class?
    pass

class ArithmeticExpressionAtomASTNodeType(ExpressionAtomASTNodeType):
    base_ast_node_class = ArithmeticExpressionASTNode

class BooleanExpressionASTNode(ExpressionASTNode):
    # TODO should this be an abstract class?
    pass

class BooleanExpressionAtomASTNodeType(ExpressionAtomASTNodeType):
    base_ast_node_class = BooleanExpressionASTNode

class ComparisonExpressionASTNode(BinaryOperationExpressionASTNode, BooleanExpressionASTNode):
    # TODO should this be an abstract class?
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, group_tokens: pyparsing.ParseResults) -> 'ComparisonExpressionASTNode':
        tokens = only_one(group_tokens).asList()
        left_arg, right_arg = tokens
        node_instance: cls = cls(left_arg, right_arg)
        return node_instance

class StringExpressionASTNode(ExpressionASTNode):
    # TODO should this be an abstract class?
    pass

class StringExpressionAtomASTNodeType(ExpressionAtomASTNodeType):
    base_ast_node_class = StringExpressionASTNode

# Literal Node Generation

@LiteralASTNodeClassBaseTypeTracker.Boolean
class BooleanLiteralASTNode(metaclass=BooleanExpressionAtomASTNodeType):
    value_type = bool
    token_checker = lambda token: token in ('True', 'False')
    dwimming_func = lambda token: True if token == 'True' else False
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

@LiteralASTNodeClassBaseTypeTracker.Integer
class IntegerLiteralASTNode(metaclass=ArithmeticExpressionAtomASTNodeType):
    value_type = int
    token_checker = lambda token: isinstance(token, int)
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

@LiteralASTNodeClassBaseTypeTracker.Float
class FloatLiteralASTNode(metaclass=ArithmeticExpressionAtomASTNodeType):
    value_type = float
    token_checker = lambda token: isinstance(token, float)
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

@LiteralASTNodeClassBaseTypeTracker.String
class StringLiteralASTNode(metaclass=StringExpressionAtomASTNodeType):
    value_type = str
    token_checker = lambda token: isinstance(token, str)
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

@LiteralASTNodeClassBaseTypeTracker.NothingType
class NothingTypeLiteralASTNode(ExpressionASTNode):
    
    def __init__(self) -> None:
        '''
        An instance of this AST Node represents a "nothing type literal", 
        i.e. the tibs constant "Nothing", not the "nothing type".
        '''
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'NothingTypeLiteralASTNode':
        assert len(tokens) is 0
        node_instance = cls()
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other)
    
    def traverse(self) -> typing.Generator['NothingTypeLiteralASTNode', None, None]:
        yield self
        return

    def emit_mlir(self) -> 'str':
        raise NotImplementedError

# Identifier / Variable Node Generation

class VariableASTNode(metaclass=ExpressionAtomASTNodeType):
    value_attribute_name = 'name'
    value_type = str
    token_checker = lambda token: isinstance(token, str)
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

# Arithmetic Expression Node Generation

class NegativeExpressionASTNode(UnaryOperationExpressionASTNode, ArithmeticExpressionASTNode):
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

class ExponentExpressionASTNode(BinaryOperationExpressionASTNode, ArithmeticExpressionASTNode):
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

class MultiplicationExpressionASTNode(BinaryOperationExpressionASTNode, ArithmeticExpressionASTNode):
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

class DivisionExpressionASTNode(BinaryOperationExpressionASTNode, ArithmeticExpressionASTNode):
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

class AdditionExpressionASTNode(BinaryOperationExpressionASTNode, ArithmeticExpressionASTNode):
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

class SubtractionExpressionASTNode(BinaryOperationExpressionASTNode, ArithmeticExpressionASTNode):
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

def parse_multiplication_or_division_expression_pe(_s: str, _loc: int, group_tokens: pyparsing.ParseResults) -> typing.Union[MultiplicationExpressionASTNode, DivisionExpressionASTNode]:
    tokens = only_one(group_tokens).asList()
    node_instance = tokens[0]
    remaining_operation_strings = tokens[1::2]
    remaining_node_instances = tokens[2::2]
    for remaining_operation_string, remaining_node_instance in zip(remaining_operation_strings, remaining_node_instances):
        if remaining_operation_string is '*':
            node_instance = MultiplicationExpressionASTNode(node_instance, remaining_node_instance)
        elif remaining_operation_string is '/':
            node_instance = DivisionExpressionASTNode(node_instance, remaining_node_instance)
        else:
            raise ValueError(f'Unrecognized operation {repr(remaining_operation_string)}')
    return node_instance

def parse_addition_or_subtraction_expression_pe(_s: str, _loc: int, group_tokens: pyparsing.ParseResults) -> typing.Union[AdditionExpressionASTNode, SubtractionExpressionASTNode]:
    tokens = only_one(group_tokens).asList()
    node_instance = tokens[0]
    remaining_operation_strings = tokens[1::2]
    remaining_node_instances = tokens[2::2]
    for remaining_operation_string, remaining_node_instance in zip(remaining_operation_strings, remaining_node_instances):
        if remaining_operation_string is '+':
            node_instance = AdditionExpressionASTNode(node_instance, remaining_node_instance)
        elif remaining_operation_string is '-':
            node_instance = SubtractionExpressionASTNode(node_instance, remaining_node_instance)
        else:
            raise ValueError(f'Unrecognized operation {repr(remaining_operation_string)}')
    return node_instance

# Boolean Expression Node Generation

class NotExpressionASTNode(UnaryOperationExpressionASTNode, BooleanExpressionASTNode):
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

class AndExpressionASTNode(BinaryOperationExpressionASTNode, BooleanExpressionASTNode):
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

class XorExpressionASTNode(BinaryOperationExpressionASTNode, BooleanExpressionASTNode):
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

class OrExpressionASTNode(BinaryOperationExpressionASTNode, BooleanExpressionASTNode):
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

# Comparison Expression Node Generation

class GreaterThanExpressionASTNode(ComparisonExpressionASTNode):
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

class GreaterThanOrEqualToExpressionASTNode(ComparisonExpressionASTNode):
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

class LessThanExpressionASTNode(ComparisonExpressionASTNode):
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

class LessThanOrEqualToExpressionASTNode(ComparisonExpressionASTNode):
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

class EqualToExpressionASTNode(ComparisonExpressionASTNode):
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

class NotEqualToExpressionASTNode(ComparisonExpressionASTNode):
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

comparator_to_ast_node_class = {
    '>': GreaterThanExpressionASTNode,
    '>=': GreaterThanOrEqualToExpressionASTNode,
    '<': LessThanExpressionASTNode,
    '<=': LessThanOrEqualToExpressionASTNode,
    '==': EqualToExpressionASTNode,
    '!=': NotEqualToExpressionASTNode,
}

assert set(comparator_to_ast_node_class.values()) == set(child_classes(ComparisonExpressionASTNode))

def parse_comparison_expression_pe(_s: str, _loc: int, tokens: pyparsing.ParseResults) -> ComparisonExpressionASTNode:
    tokens = tokens.asList()
    assert len(tokens) >= 3
    assert len(tokens) % 2 == 1
    node_instance = comparator_to_ast_node_class[tokens[1]](left_arg=tokens[0], right_arg=tokens[2])
    prev_operand = tokens[2]
    comparators = tokens[3::2]
    operands = tokens[4::2]
    for comparator, operand in zip(comparators, operands):
        cls = comparator_to_ast_node_class[comparator]
        comparison_ast_node = cls(prev_operand, operand)
        node_instance = AndExpressionASTNode(left_arg=node_instance, right_arg=comparison_ast_node)
        prev_operand =  operand
    return node_instance

# String Expression Node Generation

class StringConcatenationExpressionASTNode(BinaryOperationExpressionASTNode, StringExpressionASTNode):
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

# Function Call Node Generation

class FunctionCallExpressionASTNode(ExpressionASTNode):
    
    def __init__(self, function_name: str, arg_bindings: typing.List[typing.Tuple[VariableASTNode, ExpressionASTNode]]) -> None:
        self.function_name = function_name
        self.arg_bindings = arg_bindings
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'FunctionCallExpressionASTNode':
        assert len(tokens) is 2
        function_name = tokens[0]
        arg_bindings = eager_map(tuple, map(pyparsing.ParseResults.asList, tokens[1]))
        ASSERT.SemanticError(len({variable_ast_node.name for variable_ast_node, _ in arg_bindings}) == len(arg_bindings), f'{function_name} called with redundantly defined parameters.')
        node_instance = cls(function_name, arg_bindings)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.function_name == other.function_name and \
            self.arg_bindings == other.arg_bindings
    
    def traverse(self) -> typing.Generator[ASTNode, None, None]:
        yield self
        yield from itertools.chain(node.traverse() for node in sum(self.arg_bindings, ()))
        return

    def emit_mlir(self) -> 'str':
        raise NotImplementedError

# Vector Node Generation

class VectorExpressionASTNode(ExpressionASTNode):
    
    def __init__(self, values: typing.List[ExpressionASTNode]) -> None:
        self.values = values
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'VectorExpressionASTNode':
        values = tokens.asList()
        node_instance = cls(values)
        return node_instance
    
    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            len(self.values) == len(other.values) and \
            all(
                self_value == other_value
                for self_value, other_value
                in zip(self.values, other.values)
            )
    
    def traverse(self) -> typing.Generator[ASTNode, None, None]:
        yield self
        yield from itertools.chain(node.traverse() for node in self.values)
        return
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

# Tensor Type Node Generation

class TensorTypeASTNode(ASTNode):

    def __init__(self, base_type_name: typing.Optional[BaseTypeName], shape: typing.Optional[typing.List[typing.Optional[int]]]) -> None:
        '''
        shape == [1, 2, 3] means this tensor has 3 dimensions with the given sizes
        shape == [None, 2] means this tensor has 2 dimensions with an unspecified size for the first dimension size and a fixed size for the second dimension 
        shape == [] means this tensor has 0 dimensions, i.e. is a scalar
        shape == None means this tensor could have any arbitrary number of dimensions
        '''
        assert implies(base_type_name is None, shape is None)
        self.base_type_name = base_type_name
        self.shape = shape
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'TensorTypeASTNode':
        assert len(tokens) in (1, 2)
        base_type_name = tokens[0]
        if len(tokens) is 2:
            shape = [None if e=='?' else e for e in tokens[1].asList()]
            if shape == ['???']:
                shape = None
        else:
            shape = []
        node_instance = cls(base_type_name, shape)
        return node_instance
    
    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.base_type_name is other.base_type_name and \
            self.shape == other.shape

    def traverse(self) -> typing.Generator[ASTNode, None, None]:
        yield self
        return
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

EMPTY_TENSOR_TYPE_AST_NODE = TensorTypeASTNode(None, None)

# Assignment Node Generation

def parse_variable_type_declaration_pe(_s: str, _loc: int, type_label_tokens: pyparsing.ParseResults) -> TensorTypeASTNode:
    assert len(type_label_tokens) in (0, 1)
    if len(type_label_tokens) is 0:
        node_instance: TensorTypeASTNode = EMPTY_TENSOR_TYPE_AST_NODE
    elif len(type_label_tokens) is 1:
        node_instance: TensorTypeASTNode = only_one(type_label_tokens)
    else:
        raise ValueError(f'Unexpected type label tokens {type_label_tokens}')
    assert isinstance(node_instance, TensorTypeASTNode)
    return node_instance

class AssignmentASTNode(StatementASTNode):
    
    def __init__(self, variable_type_pairs: typing.List[typing.Tuple[VariableASTNode, TensorTypeASTNode]], value: ExpressionASTNode) -> None:
        self.variable_type_pairs = variable_type_pairs
        self.value = value
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'AssignmentASTNode':
        assert len(tokens) is 2
        variable_type_pairs: typing.List[typing.Tuple[VariableASTNode, TensorTypeASTNode]] = eager_map(tuple, map(pyparsing.ParseResults.asList, tokens[0]))
        value = tokens[1]
        node_instance = cls(variable_type_pairs, value)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.variable_type_pairs == other.variable_type_pairs and \
            self.value == other.value

    def traverse(self) -> typing.Generator[ASTNode, None, None]:
        yield self
        yield from itertools.chain(node.traverse() for node in sum(self.variable_type_pairs, ()))
        yield from self.value.traverse()
        return
    
    def emit_mlir(self) -> 'str':
        assert implies(len(self.variable_type_pairs) != 1, isinstance(self.value, FunctionCallExpressionASTNode))
        raise NotImplementedError

# Return Statement Node Generation

class ReturnStatementASTNode(StatementASTNode):
    
    def __init__(self, return_values: typing.List[ExpressionASTNode]) -> None:
        self.return_values = return_values
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'ReturnStatementASTNode':
        return_values = tokens.asList()
        if len(return_values) is 0:
            return_values = [NothingTypeLiteralASTNode()]
        node_instance = cls(return_values)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.return_values == other.return_values

    def traverse(self) -> typing.Generator[ASTNode, None, None]:
        yield self
        yield from itertools.chain(node.traverse() for node in self.return_values)
        return
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

# Print Statement Node Generation

class PrintStatementASTNode(StatementASTNode):
    
    def __init__(self, values_to_print: typing.List[ExpressionASTNode]) -> None:
        self.values_to_print = values_to_print
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'PrintStatementASTNode':
        values_to_print = tokens.asList()
        node_instance = cls(values_to_print)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.values_to_print == other.values_to_print

    def traverse(self) -> typing.Generator[ASTNode, None, None]:
        yield self
        yield from itertools.chain(node.traverse() for node in self.values_to_print)
        return
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

# Scoped Statement Sequence Node Generation

class ScopedStatementSequenceASTNode(StatementASTNode):
    
    def __init__(self, statements: typing.List[StatementASTNode]) -> None:
        self.statements = statements
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'ScopedStatementSequenceASTNode':
        statements = tokens.asList()
        node_instance = cls(statements)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.statements == other.statements

    def traverse(self) -> typing.Generator[ASTNode, None, None]:
        yield self
        yield from itertools.chain(node.traverse() for node in self.statements)
        return
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

# Function Definition Node Generation

class FunctionDefinitionASTNode(StatementASTNode):
    
    def __init__(
            self,
            function_name: str,
            function_signature: typing.List[typing.Tuple[VariableASTNode, TensorTypeASTNode]],
            function_return_types: typing.List[TensorTypeASTNode], 
            function_body: StatementASTNode
    ) -> None:
        self.function_name = function_name
        self.function_signature = function_signature
        self.function_return_types = function_return_types
        self.function_body = function_body
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'FunctionDefinitionASTNode':
        function_name, function_signature, function_return_types, function_body = tokens.asList()
        function_signature = eager_map(tuple, function_signature)
        ASSERT.SemanticError(len({variable_ast_node.name for variable_ast_node, _ in function_signature}) == len(function_signature), f'{function_name}  defined with redundantly defined parameters.')
        node_instance = cls(function_name, function_signature, function_return_types, function_body)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.function_name == other.function_name and \
            self.function_signature == other.function_signature and \
            self.function_return_types == other.function_return_types and \
            self.function_body == other.function_body 

    def traverse(self) -> typing.Generator[ASTNode, None, None]:
        yield self
        yield from itertools.chain(node.traverse() for node in sum(self.function_signature, ()))
        yield from itertools.chain(node.traverse() for node in self.function_return_types)
        yield from self.function_body.traverse()
        return
   
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

# For Loop Node Generation

class ForLoopASTNode(StatementASTNode):
    
    def __init__(
            self,
            iterator_variable_name: str,
            minimum: typing.Union[VariableASTNode, ArithmeticExpressionASTNode],
            supremum: typing.Union[VariableASTNode, ArithmeticExpressionASTNode],
            delta: typing.Union[VariableASTNode, ArithmeticExpressionASTNode],
            body: StatementASTNode
    ) -> None:
        self.iterator_variable_name = iterator_variable_name
        self.minimum = minimum
        self.supremum = supremum
        self.delta = delta
        self.body = body
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'ForLoopASTNode':
        iterator_variable_name, range_specification, body = tokens.asList()
        assert len(range_specification) in (2, 3)
        if len(range_specification) is 2:
            minimum, supremum = range_specification
            delta = IntegerLiteralASTNode(value=1)
        elif len(range_specification) is 3:
            minimum, supremum, delta = range_specification
        node_instance = cls(iterator_variable_name, minimum, supremum, delta, body)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.iterator_variable_name == other.iterator_variable_name and \
            self.minimum == other.minimum and \
            self.supremum == other.supremum and \
            self.delta == other.delta and \
            self.body == other.body

    def traverse(self) -> typing.Generator[ASTNode, None, None]:
        yield self
        yield from self.minimum.traverse()
        yield from self.supremum.traverse()
        yield from self.delta.traverse()
        yield from self.body.traverse()
        return
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

# While Loop Node Generation

class WhileLoopASTNode(StatementASTNode):
    
    def __init__(
            self,
            condition: typing.Union[VariableASTNode, BooleanExpressionASTNode],
            body: StatementASTNode
    ) -> None:
        self.condition = condition
        self.body = body
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'WhileLoopASTNode':
        condition, body = tokens.asList()
        node_instance = cls(condition, body)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.condition == other.condition and \
            self.body == other.body

    def traverse(self) -> typing.Generator[ASTNode, None, None]:
        yield self
        yield from self.condition.traverse()
        yield from self.body.traverse()
        return
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

# Conditional Node Generation

class ConditionalASTNode(StatementASTNode):
    
    def __init__(
            self,
            condition: typing.Union[VariableASTNode, BooleanExpressionASTNode],
            then_body: StatementASTNode,
            else_body: typing.Optional[StatementASTNode],
    ) -> None:
        self.condition = condition
        self.then_body = then_body
        self.else_body = else_body
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'ConditionalASTNode':
        tokens = tokens.asList()
        assert len(tokens) in (2,3)
        if len(tokens) is 2:
            condition, then_body = tokens
            else_body = None
        elif len(tokens) is 3:
            condition, then_body, else_body = tokens
        node_instance = cls(condition, then_body, else_body)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.condition == other.condition and \
            self.then_body == other.then_body and \
            self.else_body == other.else_body

    def traverse(self) -> typing.Generator[ASTNode, None, None]:
        yield self
        yield from self.condition.traverse()
        yield from self.then_body.traverse()
        if self.else_body is not None:
            yield from self.else_body.traverse()
        return
    
    def emit_mlir(self) -> 'str':
        raise NotImplementedError

# Module Node Generation

class ModuleASTNode(ASTNode):
    
    def __init__(self, statements: typing.List[StatementASTNode]) -> None:
        self.statements: typing.List[StatementASTNode] = statements
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'ModuleASTNode':
        statement_ast_nodes = tokens.asList()
        node_instance = cls(statement_ast_nodes)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        '''Order of statements matters here since this method determines syntactic equivalence.'''
        return type(self) is type(other) and \
            len(self.statements) == len(other.statements) and \
            self.statements == other.statements

    def traverse(self) -> typing.Generator[ASTNode, None, None]:
        yield self
        yield from itertools.chain(node.traverse() for node in self.statements)
        return
    
    def emit_mlir(self) -> 'str':
        mlir_text = '\n'.join(statement.emit_mlir() for statement in self.statements)
        return mlir_text

##########
# Driver #
##########

if __name__ == '__main__':
    print("TODO add something here")
