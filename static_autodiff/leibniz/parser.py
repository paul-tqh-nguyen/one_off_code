
'''
'''

# TODO fill in doc string

###########
# Imports #
###########

import typing
import pyparsing
from pyparsing import pyparsing_common as ppc
from pyparsing import (
    Word,
    Literal,
    Forward,
    Empty,
    Optional,
    Combine,
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

from .misc_utilities import *

# TODO make usure imports are used

###########
# Globals #
###########

# pyparsing.ParserElement.enablePackrat() # TODO consider enabling this

#####################
# AST Functionality #
#####################

class ASTNode(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
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
    
class LiteralASTNodeType(type):

    base_ast_node_class = ASTNode
    
    def __new__(meta, class_name: str, bases: typing.Tuple[type, ...], attributes: dict):
        method_names = ('__init__', 'parse_action', '__eq__') # TODO replace '__eq__' with 'is_equivalent'
        for method_name in method_names:
            assert method_name not in attributes.keys(), f'{method_name} already defined for class {class_name}'

        updated_attributes = dict(attributes)
        value_type: type = updated_attributes.pop('value_type')
        token_checker: typing.Callable[[typing.Any], bool] = updated_attributes.pop('token_checker')
        dwimming_func: typing.Callable[[typing.Any], value_type] = updated_attributes.pop('dwimming_func')
        
        def __init__(self, value: value_type):
            self.value: value_type = value

        @classmethod
        def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> class_name:
            token = only_one(tokens)
            assert token_checker(token)
            value: value_type = dwimming_func(token)
            node_instance: cls = cls(value)
            return node_instance
        
        def __eq__(self, other: ASTNode) -> bool: # TODO replace '__eq__' with 'is_equivalent'
            return type(self) is type(other) and self.value is other.value
        
        updated_attributes['__init__'] = __init__
        updated_attributes['parse_action'] = parse_action
        updated_attributes['__eq__'] = __eq__ # TODO replace '__eq__' with 'is_equivalent'
        
        result_class = type(class_name, bases+(meta.base_ast_node_class,), updated_attributes)
        assert all(hasattr(result_class, method_name) for method_name in method_names)
        return result_class

class ExpressionASTNode(ASTNode):
    pass

class ExpressionLiteralASTNodeType(LiteralASTNodeType):
    '''Though "ExpressionLiteral" might sound like an oxymoron, literals are just the atomic elements of an expression, but they are also expressions (albeit trivial expressions).'''

    base_ast_node_class = ExpressionASTNode

class BinaryOperationExpressionASTNode(ExpressionASTNode):

    def __init__(self, left_arg: ExpressionASTNode, right_arg: ExpressionASTNode):
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

    def __init__(self, arg: ExpressionASTNode):
        self.arg: ExpressionASTNode = arg

    @classmethod
    def parse_action(cls, _s: str, _loc: int, group_tokens: pyparsing.ParseResults) -> 'BinaryOperationExpressionASTNode':
        tokens = only_one(group_tokens)
        token = only_one(tokens)
        node_instance: cls = cls(token)
        return node_instance
    
    def __eq__(self, other: ASTNode) -> bool: # TODO replace '__eq__' and '==' with 'is_equivalent'
        return type(self) is type(other) and self.arg == other.arg

# Type Node

class TypeASTNode(metaclass=LiteralASTNodeType):
    value_type = typing.Optional[str]
    token_checker = lambda token: token in (None, 'Boolean', 'Integer', 'Real')
    dwimming_func = lambda token: token

# Literal Node Generation

class BooleanLiteralASTNode(metaclass=ExpressionLiteralASTNodeType):
    value_type = bool
    token_checker = lambda token: token in ('True', 'False')
    dwimming_func = lambda token: True if token == 'True' else False

class RealLiteralASTNode(metaclass=ExpressionLiteralASTNodeType):
    value_type = bool
    token_checker = lambda token: isinstance(token, (float, int))
    dwimming_func = lambda token: token

# Expression Node Generation

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

# Declaration Node Generation

def parse_atomic_declaration_type_label_pe(_s: str, _loc: int, type_label_tokens: pyparsing.ParseResults) -> TypeASTNode:
    assert len(type_label_tokens) in (0, 1)
    if len(type_label_tokens) is 0:
        node_instance: TypeASTNode = TypeASTNode(None)
    elif len(type_label_tokens) is 1:
        node_instance: TypeASTNode = only_one(type_label_tokens)
    else:
        raise ValueError(f'Unexpected type label tokens {type_label_tokens}')
    assert isinstance(node_instance, TypeASTNode)
    return node_instance

class AtomicDeclarationASTNode(ASTNode):
    
    def __init__(self, identifier: str, identifier_type: TypeASTNode, value: typing.Optional[typing.Any]): # TODO update the type for "value"
        self.identifier: str = identifier
        self.identifier_type: TypeASTNode = identifier_type
        self.value: typing.Optional[typing.Any] = value # TODO update the type for self.value
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'AtomicDeclarationASTNode':
        assert len(tokens) in (2, 3)
        if len(tokens) is 2:
            node_instance = cls(tokens[0], tokens[1], None)
        elif len(tokens) is 3:
            node_instance = cls(tokens[0], tokens[1], tokens[2])
        else:
            raise ValueError(f'Unexpected declaration tokens {tokens}')
        return node_instance

    # def is_equivalent(self, other: 'ASTNode') -> bool: # TODO Enable this
    def __eq__(self, other: 'ASTNode') -> bool:
        # TODO make the below use is_equivalent instead of "=="
        return type(self) is type(other) and \
            self.identifier is other.identifier and \
            self.identifier_type == other.identifier_type and \
            self.value == other.value

# Subtheory Node Generation

class SubtheoryASTNode(ASTNode):
    
    def __init__(self, declarations: typing.List[AtomicDeclarationASTNode]):
        self.declarations: typing.List[AtomicDeclarationASTNode] = declarations
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'SubtheoryASTNode':
        atomic_declaration_ast_node_lists  = tokens.asList()
        atomic_declaration_ast_nodes = sum(atomic_declaration_ast_node_lists, [])
        node_instance = cls(atomic_declaration_ast_nodes)
        return node_instance

    # def is_equivalent(self, other: 'ASTNode') -> bool: # TODO Enable this
    def __eq__(self, other: 'ASTNode') -> bool:
        '''Order of declarations matters here since this method determines syntactic equivalence.'''
        # TODO make the below use is_equivalent instead of "=="
        return type(self) is type(other) and \
            len(self.declarations) == len(other.declarations) and \
            all(
                self_declaration == other_declaration
                for self_declaration, other_declaration
                in zip(self.declarations, other.declarations)
            )

# TODO use is_equivalent in the tests as well

###########
# Grammar #
###########

# Convention: The "_pe" suffix indicates an instance of pyparsing.ParserElement

pyparsing.ParserElement.setDefaultWhitespaceChars(' \t')

# Type Parser Elements

boolean_type_pe = Literal('Boolean').setName('boolean type')
integer_type_pe = Literal('Integer').setName('integer type')
real_type_pe = Literal('Real').setName('real type')

type_pe = (boolean_type_pe | integer_type_pe | real_type_pe).setParseAction(TypeASTNode.parse_action)

# Literal Parser Elements

boolean_pe = oneOf('True False').setName('boolean').setParseAction(BooleanLiteralASTNode.parse_action)

integer_pe = ppc.integer
float_pe = ppc.real | ppc.sci_real
real_pe = (float_pe | integer_pe).setName('real number').setParseAction(RealLiteralASTNode.parse_action)

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

# Atom Parser Elements

identifier_pe = ~boolean_operation_pe + ~boolean_pe + ppc.identifier.setName('identifier')

atom_pe = (identifier_pe | real_pe | boolean_pe).setName('atom')

# Expression Parser Elements

expression_pe = Forward()

arithmetic_expression_pe = infixNotation(
    real_pe | identifier_pe,
    [
        (negative_operation_pe, 1, opAssoc.RIGHT, NegativeExpressionASTNode.parse_action),
        (exponent_operation_pe, 2, opAssoc.RIGHT, ExponentExpressionASTNode.parse_action),
        (multiplication_operation_pe | division_operation_pe, 2, opAssoc.LEFT, parse_multiplication_or_division_expression_pe),
        (addition_operation_pe | subtraction_operation_pe, 2, opAssoc.LEFT, parse_addition_or_subtraction_expression_pe),
    ],
).setName('arithmetic expression')

boolean_expression_pe = infixNotation(
    boolean_pe | identifier_pe,
    [
        (not_operation_pe, 1, opAssoc.RIGHT),
        (and_operation_pe, 2, opAssoc.LEFT),
        (xor_operation_pe, 2, opAssoc.LEFT),
        (or_operation_pe, 2, opAssoc.LEFT),
    ],
).setName('boolean expression')

function_variable_binding_pe = identifier_pe + ':=' + expression_pe
function_variable_bindings_pe = Optional(delimitedList(function_variable_binding_pe))
function_call_pe = (identifier_pe + Literal('(') + function_variable_bindings_pe + Literal(')')).setName('function call')

expression_pe <<= (function_call_pe | arithmetic_expression_pe | boolean_expression_pe | atom_pe).setName('expression')

# Declaration Parser Elements

atomic_declaration_type_label_pe = Optional(Suppress(':') + type_pe).setParseAction(parse_atomic_declaration_type_label_pe)
atomic_declaration_value_label_pe = Optional(Suppress('=') + expression_pe)
atomic_declaration_pe = (identifier_pe + atomic_declaration_type_label_pe + atomic_declaration_value_label_pe).setParseAction(AtomicDeclarationASTNode.parse_action)

non_atomic_declaration_pe = atomic_declaration_pe + Suppress(';') + delimitedList(atomic_declaration_pe, delim=';')

declaration_pe = (non_atomic_declaration_pe | atomic_declaration_pe).setName('expression').setName('declaration')

# Subtheory & Misc. Parser Elements

comment_pe = Regex(r"#(?:\\\n|[^\n])*").setName('end-of-line comment')

line_of_code_pe = Group(Optional(declaration_pe)) # TODO update this with imperative code as well

subtheory_pe = delimitedList(line_of_code_pe, delim='\n').ignore(comment_pe).setParseAction(SubtheoryASTNode.parse_action)

################
# Entry Points #
################

def parseSourceCode(input_string: str): # TODO add return type
    return subtheory_pe.parseString(input_string, parseAll=True)

# TODO enable this
# __all__ = [
#     'parseSourceCode'
# ]

##########
# Driver #
##########

if __name__ == '__main__':
    print("TODO add something here")

