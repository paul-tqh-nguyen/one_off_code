
'''
'''

# TODO fill in doc string

###########
# Imports #
###########

import typing
import typing_extensions
import weakref
import pyparsing
from pyparsing import pyparsing_common as ppc
from pyparsing import (
    Word,
    Literal,
    Forward,
    Empty,
    Optional,
    Combine,
    NotAny,
    Regex,
    ZeroOrMore,
    Suppress,
    Group,
    FollowedBy,
    OneOrMore,
)
from pyparsing import (
    oneOf,
    infixNotation,
    opAssoc,
    delimitedList,
    pythonStyleComment,
)
from abc import ABC, abstractmethod
from functools import reduce
from collections import defaultdict
import operator

from .misc_utilities import *

# TODO make usure imports are used

###########
# Globals #
###########

BOGUS_TOKEN = object()

# pyparsing.ParserElement.enablePackrat() # TODO consider enabling this

#############################
# Sanity Checking Utilities #
#############################

BASE_TYPES = ('Boolean', 'Integer', 'Float', 'NothingType')

BaseTypeName = operator.getitem(typing_extensions.Literal, BASE_TYPES)

class BaseTypeTrackerType(type):
    
    instantiated_base_type_tracker_classes: typing.List[weakref.ref] = []
    # TODO track weak references to instances of this metaclass ; use that to sanity check later on ; also, assert that these weak instances are still valid at sanity-checking-time
    
    def __new__(meta, class_name: str, bases: typing.Tuple[type, ...], attributes: dict):
        
        updated_attributes = dict(attributes)
        assert 'tracked_type' in updated_attributes
        updated_attributes['base_type_to_value'] = {}
        result_class = type.__new__(meta, class_name, bases, updated_attributes)
        
        result_class_weakref = weakref.ref(result_class)
        meta.instantiated_base_type_tracker_classes.append(result_class_weakref)
        
        return result_class
    
    def __getitem__(cls, base_type_name: BaseTypeName) -> typing.Callable[[typing.Any], typing.Any]:
        def note_value(value: typing.Any) -> typing.Any:
            assert base_type_name not in cls.base_type_to_value
            assert isinstance(value, cls.tracked_type), f'{value} is not an instance of the tracked type {cls.tracked_type}'
            cls.base_type_to_value[base_type_name] = value
            return value
        return note_value
    
    def __getattr__(cls, base_type_name: BaseTypeName) -> typing.Callable[[typing.Any], typing.Any]:
        return cls[base_type_name]
    
    @classmethod
    def vaildate_base_types(meta) -> None:
        for instantiated_base_type_tracker_class_weakref in meta.instantiated_base_type_tracker_classes:
            instantiated_base_type_tracker_class = instantiated_base_type_tracker_class_weakref()
            instantiated_base_type_tracker_class_is_alive = instantiated_base_type_tracker_class is not None
            assert instantiated_base_type_tracker_class_is_alive
            assert len(BASE_TYPES) == len(instantiated_base_type_tracker_class.base_type_to_value.keys())
            assert all(key in BASE_TYPES for key in instantiated_base_type_tracker_class.base_type_to_value.keys())
            assert all(base_type in instantiated_base_type_tracker_class.base_type_to_value.keys() for base_type in BASE_TYPES)
        return

def sanity_check_base_types() -> None:
    BaseTypeTrackerType.vaildate_base_types()
    return 

class TensorTypeParserElementBaseTypeTracker(metaclass=BaseTypeTrackerType):
    tracked_type: type = pyparsing.ParserElement

class AtomicLiteralParserElementBaseTypeTracker(metaclass=BaseTypeTrackerType):
    tracked_type: type = pyparsing.ParserElement

class LiteralASTNodeClassBaseTypeTracker(metaclass=BaseTypeTrackerType):
    tracked_type: type = type

#####################
# AST Functionality #
#####################

def _trace_parse(s: str, loc: int, tokens: pyparsing.ParseResults): # TODO remove this
    print()
    print(f"s {repr(s)}")
    print(f"loc {repr(loc)}")
    print(f"tokens {repr(tokens)}")

class ASTNode(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def parse_action(cls, s: str, loc: int, tokens: pyparsing.ParseResults):
        raise NotImplementedError

    def __repr__(self):
        attributes_string = ', '.join(f'{k}={repr(self.__dict__[k])}' for k in sorted(self.__dict__.keys()))
        return f'{self.__class__.__name__}({attributes_string})'

    # def is_equivalent(self, other: 'ASTNode') -> bool: # TODO Enable this
    def __eq__(self, other: 'ASTNode') -> bool:
        '''Determines syntactic equivalence, not semantic equivalence or canonicality.'''
        raise NotImplementedError

class AtomASTNodeType(type):

    # TODO is this needed or is ExpressionAtomASTNodeType sufficient?

    base_ast_node_class = ASTNode
    
    def __new__(meta, class_name: str, bases: typing.Tuple[type, ...], attributes: dict):
        method_names = ('__init__', 'parse_action', '__eq__') # TODO replace '__eq__' with 'is_equivalent'
        for method_name in method_names:
            assert method_name not in attributes.keys(), f'{method_name} already defined for class {class_name}'

        updated_attributes = dict(attributes)
        # Required Declarations
        value_type: type = updated_attributes.pop('value_type')
        token_checker: typing.Callable[[typing.Any], bool] = updated_attributes.pop('token_checker')
        # Optional Declarations
        value_attribute_name: str = updated_attributes.pop('value_attribute_name', 'value')
        dwimming_func: typing.Callable[[typing.Any], value_type] = updated_attributes.pop('dwimming_func', lambda token: token)
        
        def __init__(self, *args, **kwargs) -> None:
            assert sum(map(len, (args, kwargs))) == 1
            value: value_type = args[0] if len(args) == 1 else kwargs[value_attribute_name]
            assert isinstance(value, value_type)
            setattr(self, value_attribute_name, value)

        @classmethod
        def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> class_name:
            token = only_one(tokens)
            assert token_checker(token)
            value: value_type = dwimming_func(token)
            node_instance: cls = cls(value)
            return node_instance
        
        def __eq__(self, other: ASTNode) -> bool: # TODO replace '__eq__' with 'is_equivalent'
            other_value = getattr(other, value_attribute_name, BOGUS_TOKEN)
            same_type = type(self) is type(other)
            
            self_value = getattr(self, value_attribute_name)
            same_value_type = type(self_value) is type(other_value)
            same_value = self_value == other_value
            return same_type and same_value_type and same_value
        
        updated_attributes['__init__'] = __init__
        updated_attributes['parse_action'] = parse_action
        updated_attributes['__eq__'] = __eq__ # TODO replace '__eq__' with 'is_equivalent'
        
        result_class = type(class_name, bases+(meta.base_ast_node_class,), updated_attributes)
        assert all(hasattr(result_class, method_name) for method_name in method_names)
        return result_class

class FunctionStatementASTNode(ASTNode):
    pass

class ModuleStatementASTNode(FunctionStatementASTNode):
    pass
    
class ExpressionASTNode(ModuleStatementASTNode):
    pass

class ExpressionAtomASTNodeType(AtomASTNodeType):
    '''The atomic bases that make up expressions.'''

    base_ast_node_class = ExpressionASTNode

class BinaryOperationExpressionASTNode(ExpressionASTNode):

    def __init__(self, left_arg: ExpressionASTNode, right_arg: ExpressionASTNode) -> None:
        self.left_arg: ExpressionASTNode = left_arg
        self.right_arg: ExpressionASTNode = right_arg

    @classmethod
    def parse_action(cls, _s: str, _loc: int, group_tokens: pyparsing.ParseResults) -> 'BinaryOperationExpressionASTNode':
        tokens = only_one(group_tokens)
        assert len(tokens) >= 2
        node_instance: cls = reduce(cls, tokens)
        return node_instance
    
    def __eq__(self, other: ASTNode) -> bool: # TODO replace '__eq__' and '==' with 'is_equivalent'
        return type(self) is type(other) and \
            self.left_arg == other.left_arg and \
            self.right_arg == other.right_arg 

class UnaryOperationExpressionASTNode(ExpressionASTNode):

    def __init__(self, arg: ExpressionASTNode) -> None:
        self.arg: ExpressionASTNode = arg

    @classmethod
    def parse_action(cls, _s: str, _loc: int, group_tokens: pyparsing.ParseResults) -> 'BinaryOperationExpressionASTNode':
        tokens = only_one(group_tokens)
        token = only_one(tokens)
        node_instance: cls = cls(token)
        return node_instance
    
    def __eq__(self, other: ASTNode) -> bool: # TODO replace '__eq__' and '==' with 'is_equivalent'
        return type(self) is type(other) and self.arg == other.arg

# Literal Node Generation

@LiteralASTNodeClassBaseTypeTracker.Boolean
class BooleanLiteralASTNode(metaclass=ExpressionAtomASTNodeType):
    value_type = bool
    token_checker = lambda token: token in ('True', 'False')
    dwimming_func = lambda token: True if token == 'True' else False

@LiteralASTNodeClassBaseTypeTracker.Integer
class IntegerLiteralASTNode(metaclass=ExpressionAtomASTNodeType):
    value_type = int
    token_checker = lambda token: isinstance(token, int)

@LiteralASTNodeClassBaseTypeTracker.Float
class FloatLiteralASTNode(metaclass=ExpressionAtomASTNodeType):
    value_type = float
    token_checker = lambda token: isinstance(token, float)

@LiteralASTNodeClassBaseTypeTracker.NothingType
class NothingTypeLiteralASTNode(ExpressionASTNode):
    
    def __init__(self) -> None:
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'NothingTypeLiteralASTNode':
        assert len(tokens) is 0
        node_instance = cls()
        return node_instance

    # def is_equivalent(self, other: 'ASTNode') -> bool: # TODO Enable this
    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other)

# Identifier / Variable Node Generation

class VariableASTNode(metaclass=ExpressionAtomASTNodeType):
    value_attribute_name = 'name'
    value_type = str
    token_checker = lambda token: isinstance(token, str)

# Arithmetic Expression Node Generation

class NegativeExpressionASTNode(UnaryOperationExpressionASTNode):
    pass

class ExponentExpressionASTNode(BinaryOperationExpressionASTNode):
    pass

class MultiplicationExpressionASTNode(BinaryOperationExpressionASTNode):
    pass

class DivisionExpressionASTNode(BinaryOperationExpressionASTNode):
    pass

class AdditionExpressionASTNode(BinaryOperationExpressionASTNode):
    pass

class SubtractionExpressionASTNode(BinaryOperationExpressionASTNode):
    pass

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

class NotExpressionASTNode(UnaryOperationExpressionASTNode):
    pass

class AndExpressionASTNode(BinaryOperationExpressionASTNode):
    pass

class XorExpressionASTNode(BinaryOperationExpressionASTNode):
    pass

class OrExpressionASTNode(BinaryOperationExpressionASTNode):
    pass

# Function Call Node Generation

class FunctionCallExpressionASTNode(ExpressionASTNode):
    
    def __init__(self, function_name: str, arg_bindings: typing.List[typing.Tuple[VariableASTNode, ExpressionASTNode]]) -> None:
        self.function_name = function_name
        self.arg_bindings = arg_bindings
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'FunctionCallExpressionASTNode':
        assert len(tokens) is 2
        function_name = tokens[0]
        arg_bindings = eager_map(tuple, map(pyparsing.ParseResults.asList, tokens[1]))
        node_instance = cls(function_name, arg_bindings)
        return node_instance

    # def is_equivalent(self, other: 'ASTNode') -> bool: # TODO Enable this
    def __eq__(self, other: ASTNode) -> bool:
        # TODO make the below use is_equivalent instead of "=="
        return type(self) is type(other) and \
            self.function_name == other.function_name and \
            all(
                self_arg_binding == other_arg_binding
                for self_arg_binding, other_arg_binding
                in zip(self.arg_bindings, other.arg_bindings)
            )

# Vector Node Generation

class VectorExpressionASTNode(ExpressionASTNode):
    
    def __init__(self, values: typing.List[ExpressionASTNode]) -> None:
        self.values = values
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'VectorExpressionASTNode':
        values = tokens.asList()
        node_instance = cls(values)
        return node_instance

    # def is_equivalent(self, other: 'ASTNode') -> bool: # TODO Enable this
    def __eq__(self, other: ASTNode) -> bool:
        # TODO make the below use is_equivalent instead of "=="
        return type(self) is type(other) and \
            all(
                self_value == other_value
                for self_value, other_value
                in zip(self.values, other.values)
            )

# Tensor Type Node Generation

class TensorTypeASTNode(ASTNode):

    def __init__(self, base_type_name: typing.Optional[BaseTypeName], shape: typing.Optional[typing.List[int]]) -> None:
        '''
        shape == [IntegerLiteralASTNode(value=1), IntegerLiteralASTNode(value=2), IntegerLiteralASTNode(value=3)] means this tensor has 3 dimensions with the given sizes
        shape == [None, IntegerLiteralASTNode(value=2)] means this tensor has 2 dimensions with an unspecified size for the first dimension size and a fixed size for the second dimension 
        shape == [] means this tensor has 0 dimensions, i.e. is a scalar
        shape == None means this tensor could have any arbitrary number of dimensions
        '''
        assert implies(base_type_name is None, shape is None)
        self.base_type_name = base_type_name
        self.shape = shape
    
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

    # def is_equivalent(self, other: 'ASTNode') -> bool: # TODO Enable this
    def __eq__(self, other: ASTNode) -> bool:
        # TODO make the below use is_equivalent instead of "=="
        return type(self) is type(other) and \
            self.base_type_name is other.base_type_name and \
            self.shape == other.shape

# Assignment Node Generation

def parse_variable_type_declaration_pe(_s: str, _loc: int, type_label_tokens: pyparsing.ParseResults) -> TensorTypeASTNode:
    assert len(type_label_tokens) in (0, 1)
    if len(type_label_tokens) is 0:
        node_instance: TensorTypeASTNode = TensorTypeASTNode(None, None)
    elif len(type_label_tokens) is 1:
        node_instance: TensorTypeASTNode = only_one(type_label_tokens)
    else:
        raise ValueError(f'Unexpected type label tokens {type_label_tokens}')
    assert isinstance(node_instance, TensorTypeASTNode)
    return node_instance

class AssignmentASTNode(ModuleStatementASTNode):
    
    def __init__(self, identifier: VariableASTNode, identifier_type: TensorTypeASTNode, value: ExpressionASTNode) -> None:
        self.identifier = identifier
        self.identifier_type = identifier_type
        self.value = value
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'AssignmentASTNode':
        assert len(tokens) is 3
        node_instance = cls(tokens[0], tokens[1], tokens[2])
        return node_instance

    # def is_equivalent(self, other: 'ASTNode') -> bool: # TODO Enable this
    def __eq__(self, other: ASTNode) -> bool:
        # TODO make the below use is_equivalent instead of "=="
        return type(self) is type(other) and \
            self.identifier == other.identifier and \
            self.identifier_type == other.identifier_type and \
            self.value == other.value

# Return Statement Node Generation

class ReturnStatementASTNode(FunctionStatementASTNode):
    
    def __init__(self, return_values: typing.List[ExpressionASTNode]) -> None:
        self.return_values = return_values
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'AssignmentASTNode':
        return_values = tokens.asList()
        if len(return_values) is 0:
            return_values = [NothingTypeLiteralASTNode()]
        node_instance = cls(return_values)
        return node_instance

    # def is_equivalent(self, other: 'ASTNode') -> bool: # TODO Enable this
    def __eq__(self, other: ASTNode) -> bool:
        # TODO make the below use is_equivalent instead of "=="
        return type(self) is type(other) and \
            all(
                self_return_value == other_return_value
                for self_return_value, other_return_value
                in zip(self.return_values, other.return_values)
            )

# Function Definition Node Generation

class FunctionDefinitionExpressionASTNode(ModuleStatementASTNode):
    
    def __init__(
            self,
            function_name: str,
            function_signature: typing.List[typing.Tuple[VariableASTNode, TensorTypeASTNode]],
            function_return_type: TensorTypeASTNode, 
            function_body_statements: typing.List[FunctionStatementASTNode]
    ) -> None:
        self.function_name = function_name
        self.function_signature = function_signature
        self.function_return_type = function_return_type
        self.function_body_statements = function_body_statements
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'FunctionDefinitionExpressionASTNode':
        function_name, function_signature, function_return_type, function_body_statements = tokens.asList()
        function_signature = eager_map(tuple, function_signature)
        node_instance = cls(function_name, function_signature, function_return_type, function_body_statements)
        return node_instance

    # def is_equivalent(self, other: 'ASTNode') -> bool: # TODO Enable this
    def __eq__(self, other: ASTNode) -> bool:
        # TODO make the below use is_equivalent instead of "=="
        return type(self) is type(other) and \
            self.function_name == other.function_name and \
            self.function_signature == other.function_signature and \
            self.function_return_type == other.function_return_type and \
            self.function_body_statements == other.function_body_statements

# Module Node Generation

class ModuleASTNode(ASTNode):
    
    def __init__(self, statements: typing.List[ModuleStatementASTNode]) -> None:
        self.statements: typing.List[ModuleStatementASTNode] = statements
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'ModuleASTNode':
        statement_ast_nodes = tokens.asList()
        node_instance = cls(statement_ast_nodes)
        return node_instance

    # def is_equivalent(self, other: 'ASTNode') -> bool: # TODO Enable this
    def __eq__(self, other: ASTNode) -> bool:
        '''Order of statements matters here since this method determines syntactic equivalence.'''
        # TODO make the below use is_equivalent instead of "=="
        return type(self) is type(other) and \
            len(self.statements) == len(other.statements) and \
            all(
                self_statement == other_statement
                for self_statement, other_statement
                in zip(self.statements, other.statements)
            )

# TODO use is_equivalent in the tests as well

###########
# Grammar #
###########

# Convention: The "_pe" suffix indicates an instance of pyparsing.ParserElement

pyparsing.ParserElement.setDefaultWhitespaceChars(' \t')

# Type Parser Elements

tensor_dimension_declaration_pe = Suppress('<') + Group(Optional(delimitedList(Literal('???') | Literal('?') | ppc.integer))) + Suppress('>')

boolean_tensor_type_pe = TensorTypeParserElementBaseTypeTracker['Boolean'](Literal('Boolean') + Optional(tensor_dimension_declaration_pe)).setName('boolean tensor type')
integer_tensor_type_pe = TensorTypeParserElementBaseTypeTracker['Integer'](Literal('Integer') + Optional(tensor_dimension_declaration_pe)).setName('integer tensor type')
float_tensor_type_pe = TensorTypeParserElementBaseTypeTracker['Float'](Literal('Float') + Optional(tensor_dimension_declaration_pe)).setName('float tensor type')
nothing_tensor_type_pe = TensorTypeParserElementBaseTypeTracker['NothingType'](Literal('NothingType') + Optional(tensor_dimension_declaration_pe)).setName('nothingg tensor type')

tensor_type_pe = (boolean_tensor_type_pe | integer_tensor_type_pe | float_tensor_type_pe | nothing_tensor_type_pe).setParseAction(TensorTypeASTNode.parse_action)

# Literal Parser Elements

boolean_pe = AtomicLiteralParserElementBaseTypeTracker['Boolean'](oneOf('True False')).setName('boolean').setParseAction(BooleanLiteralASTNode.parse_action)

integer_pe = AtomicLiteralParserElementBaseTypeTracker['Integer'](ppc.integer).setName('unsigned integer').setParseAction(ppc.convertToInteger, IntegerLiteralASTNode.parse_action)

float_pe = AtomicLiteralParserElementBaseTypeTracker['Float'](ppc.real | ppc.sci_real).setName('floating point number').setParseAction(FloatLiteralASTNode.parse_action)

nothing_pe = AtomicLiteralParserElementBaseTypeTracker['NothingType'](Suppress('Nothing')).setName('nothing type').setParseAction(NothingTypeLiteralASTNode.parse_action)

# Arithmetic Operation Parser Elements

negative_operation_pe = Suppress('-').setName('negative operation')
exponent_operation_pe = Suppress(oneOf('^ **')).setName('exponent operation')
multiplication_operation_pe = Literal('*').setName('multiplication operation')
division_operation_pe = Literal('/').setName('division operation')
addition_operation_pe = Literal('+').setName('addition operation')
subtraction_operation_pe = Literal('-').setName('subtraction operation')

# Boolean Operation Parser Elements

not_operation_pe = Suppress('not').setName('not operation')
and_operation_pe = Suppress('and').setName('and operation')
xor_operation_pe = Suppress('xor').setName('xor operation')
or_operation_pe = Suppress('or').setName('or operation')

boolean_operation_pe = not_operation_pe | and_operation_pe | xor_operation_pe | or_operation_pe

# Identifier Parser Elements

# identifiers are not variables as identifiers can also be used as function names. Function names are not variables since functions are not first class citizens.

return_statement_pe = Forward()

function_definition_keyword_pe = Suppress('function')

# TODO using return_statement_pe here is more general than necessary since we only need to capture the emtpy return statement
not_reserved_keyword_pe = reduce(operator.add, map(NotAny, map(Suppress, BASE_TYPES))) + \
     ~nothing_pe + \
     ~boolean_operation_pe + \
     ~boolean_pe + \
     ~function_definition_keyword_pe + \
     ~return_statement_pe


identifier_pe = (not_reserved_keyword_pe + ppc.identifier)

# Atom Parser Elements

variable_pe = identifier_pe.copy().setName('identifier').setParseAction(VariableASTNode.parse_action)

atom_pe = (variable_pe | float_pe | integer_pe | boolean_pe | nothing_pe).setName('atom')

# Expression Parser Elements

expression_pe = Forward()

arithmetic_expression_pe = infixNotation(
    float_pe | integer_pe | variable_pe,
    [
        (negative_operation_pe, 1, opAssoc.RIGHT, NegativeExpressionASTNode.parse_action),
        (exponent_operation_pe, 2, opAssoc.RIGHT, ExponentExpressionASTNode.parse_action),
        (multiplication_operation_pe | division_operation_pe, 2, opAssoc.LEFT, parse_multiplication_or_division_expression_pe),
        (addition_operation_pe | subtraction_operation_pe, 2, opAssoc.LEFT, parse_addition_or_subtraction_expression_pe),
    ],
).setName('arithmetic expression')

boolean_expression_pe = infixNotation(
    boolean_pe | variable_pe,
    [
        (not_operation_pe, 1, opAssoc.RIGHT, NotExpressionASTNode.parse_action),
        (and_operation_pe, 2, opAssoc.LEFT, AndExpressionASTNode.parse_action),
        (xor_operation_pe, 2, opAssoc.LEFT, XorExpressionASTNode.parse_action),
        (or_operation_pe, 2, opAssoc.LEFT, OrExpressionASTNode.parse_action),
    ],
).setName('boolean expression')

function_variable_binding_pe = Group(variable_pe + Suppress(':=') + expression_pe)
function_variable_bindings_pe = Group(Optional(delimitedList(function_variable_binding_pe)))
function_call_expression_pe = (identifier_pe + Suppress('(') + function_variable_bindings_pe + Suppress(')')).setName('function call').setParseAction(FunctionCallExpressionASTNode.parse_action)

vector_pe = Forward()
expression_pe <<= (function_call_expression_pe | arithmetic_expression_pe | boolean_expression_pe | atom_pe | vector_pe).setName('expression')

# Vector Parser Elements

vector_pe <<= (Suppress('[') + delimitedList(expression_pe, delim=',') + Suppress(']')).setName('tensor').setParseAction(VectorExpressionASTNode.parse_action)

# Assignment Parser Elements

variable_type_declaration_pe = Optional(Suppress(':') + tensor_type_pe).setParseAction(parse_variable_type_declaration_pe)
assignment_value_declaration_pe = Suppress('=') + expression_pe
assignment_pe = (variable_pe + variable_type_declaration_pe + assignment_value_declaration_pe).setParseAction(AssignmentASTNode.parse_action).setName('assignment')

# Module & Misc. Parser Elements

comment_pe = Regex(r"#(?:\\\n|[^\n])*").setName('comment')

function_definition_pe = Forward()
required_atomic_module_statement_pe = function_definition_pe | assignment_pe | expression_pe
atomic_module_statement_pe = Optional(required_atomic_module_statement_pe)

non_atomic_module_statement_pe = atomic_module_statement_pe + Suppress(';') + delimitedList(atomic_module_statement_pe, delim=';')

module_statement_pe = (non_atomic_module_statement_pe | atomic_module_statement_pe).setName('module statement')

module_pe = delimitedList(module_statement_pe, delim='\n').ignore(comment_pe).setParseAction(ModuleASTNode.parse_action)

# Return Statement Parser Elements

return_statement_pe <<= (Suppress('return') + Optional(delimitedList(expression_pe))).setParseAction( ReturnStatementASTNode.parse_action).setName('return statetment')

# Function Definition Parser Elements

atomic_function_statement_pe = Optional(required_atomic_module_statement_pe | return_statement_pe)
non_atomic_function_statement_pe = atomic_function_statement_pe + Suppress(';') + delimitedList(atomic_function_statement_pe, delim=';')
function_statement_pe = (non_atomic_function_statement_pe | atomic_function_statement_pe).setName('function statement')

function_signature_pe = Suppress('(') + Group(Optional(delimitedList(variable_pe + variable_type_declaration_pe))) + Suppress(')')
function_return_type_pe = (Suppress('->') + tensor_type_pe).setParseAction(TensorTypeASTNode.parse_action)
function_body_pe = Suppress('{') + Group(Optional(delimitedList(function_statement_pe, delim='\n'))).setWhitespaceChars(' \t\n') + Suppress('}')

function_definition_pe <<= (function_definition_keyword_pe + identifier_pe + function_signature_pe + function_return_type_pe + function_body_pe).ignore(comment_pe).setParseAction(FunctionDefinitionExpressionASTNode.parse_action)

################
# Entry Points #
################

class ParseError(Exception):

    def __init__(self, original_text, problematic_text, problem_column_number) -> None:
        self.original_text = original_text
        self.problematic_text = problematic_text
        self.problem_column_number = problem_column_number
        super().__init__(f'''Could not parse the following:

    {self.problematic_text}
    {(' '*(self.problem_column_number - 1))}^
''')

def parseSourceCode(input_string: str) -> ModuleASTNode:
    try:
        result = only_one(module_pe.parseString(input_string, parseAll=True))
    except pyparsing.ParseException as exception:
        raise ParseError(input_string, exception.line, exception.col)
    return result

# TODO enable this
# __all__ = [
#     'parseSourceCode'
# ]

##########
# Driver #
##########

sanity_check_base_types()

if __name__ == '__main__':
    print("TODO add something here")
