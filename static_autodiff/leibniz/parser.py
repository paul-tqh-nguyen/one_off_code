
'''
'''

# TODO fill in doc string

###########
# Imports #
###########

import typing
import pyparsing
from pyparsing import pyparsing_common as ppc
from pyparsing import Word, Literal, Forward, Empty, Optional, Combine, Regex, ZeroOrMore, Suppress, Group
from pyparsing import oneOf, infixNotation, opAssoc, delimitedList, pythonStyleComment

from abc import ABC, abstractmethod

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
        raise NotImplementedError

class ASTLiteralNodeType(type):

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
            node_instance = cls(value)
            return node_instance
        
        def __eq__(self, other: ASTNode) -> bool: # TODO replace '__eq__' with 'is_equivalent'
            return type(self) is type(other) and self.value is other.value
        
        updated_attributes['__init__'] = __init__
        updated_attributes['parse_action'] = parse_action
        updated_attributes['__eq__'] = __eq__ # TODO replace '__eq__' with 'is_equivalent'
        
        result_class = type(class_name, bases+(ASTNode,), updated_attributes)
        assert all(hasattr(result_class, method_name) for method_name in method_names)
        return result_class

# Type Node

class TypeASTNode(metaclass=ASTLiteralNodeType):
    value_type = typing.Optional[str]
    token_checker = lambda token: token in (None, 'Boolean', 'Integer', 'Real')
    dwimming_func = lambda token: token

# Literal Node Generation

class BooleanLiteralASTNode(metaclass=ASTLiteralNodeType):
    value_type = bool
    token_checker = lambda token: token in ('True', 'False')
    dwimming_func = lambda token: True if token == 'True' else False

class RealLiteralASTNode(metaclass=ASTLiteralNodeType):
    value_type = bool
    token_checker = lambda token: isinstance(token, (float, int))
    dwimming_func = lambda token: token

# Declaration Node Generation

def parse_type_label_pe(s: str, loc: int, type_label_group_tokens: pyparsing.ParseResults) -> TypeASTNode:
    type_label_tokens: pyparsing.ParserElement = only_one(type_label_group_tokens)
    assert len(type_label_tokens) in (0, 2)
    if len(type_label_tokens) is 0:
        node_instance: TypeASTNode = TypeASTNode(None)
    elif len(type_label_tokens) is 2:
        assert type_label_tokens[0] == ':'
        node_instance: TypeASTNode = type_label_tokens[1]
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
        assert len(tokens) in (2, 4)
        if len(tokens) is 2:
            node_instance = cls(tokens[0], tokens[1], None)
        elif len(tokens) is 4:
            assert tokens[2] is '='
            node_instance = cls(tokens[0], tokens[1], tokens[3])
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

negative_operation_pe = Literal('-').setName('negative operation')
exponent_operation_pe = oneOf('^ **').setName('power operation')
multiplication_operation_pe = Literal('*').setName('multiplication operation')
division_operation_pe = Literal('/').setName('division operation')
addition_operation_pe = Literal('+').setName('addition operation')
subtraction_operation_pe = Literal('-').setName('subtraction operation')

# Boolean Operation Parser Elements

not_operation_pe = Literal('not').setName('not operation')
and_operation_pe = Literal('and').setName('and operation')
xor_operation_pe = Literal('xor').setName('xor operation')

or_operation_pe = Literal('or').setName('or operation')

boolean_operation_pe = not_operation_pe | and_operation_pe | xor_operation_pe | or_operation_pe

# Atom Parser Elements

identifier_pe = ~boolean_operation_pe + ~boolean_pe + ppc.identifier.setName('identifier')

atom_pe = (identifier_pe | real_pe | boolean_pe).setName('atom')

# Expression Parser Elements

expression_pe = Forward()

arithmetic_expression_pe = infixNotation(
    real_pe | identifier_pe,
    [
        (negative_operation_pe, 1, opAssoc.RIGHT),
        (exponent_operation_pe, 2, opAssoc.RIGHT),
        (multiplication_operation_pe | division_operation_pe, 2, opAssoc.LEFT),
        (addition_operation_pe | subtraction_operation_pe, 2, opAssoc.LEFT),
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

type_label_pe = Group(Optional(Literal(':') + type_pe)).setParseAction(parse_type_label_pe)

value_declaration_pe = (identifier_pe + type_label_pe + "=" + expression_pe).setName('value declaration')
type_declaration_pe = (identifier_pe + type_label_pe).setName('type declaration')
name_declaration_pe = (identifier_pe).setName('name declaration')

atomic_declaration_pe = (value_declaration_pe | type_declaration_pe | name_declaration_pe).setParseAction(AtomicDeclarationASTNode.parse_action)
non_atomic_declaration_pe = atomic_declaration_pe + Suppress(';') + delimitedList(atomic_declaration_pe, delim=';')
declaration_pe = (non_atomic_declaration_pe | atomic_declaration_pe).setName('expression').setName('declaration')

# Grammar

comment_pe = Regex(r"#(?:\\\n|[^\n])*").setName('end-of-line comment')

line_of_code_pe = Group(Optional(declaration_pe)) # TODO update this with imperative code as well

grammar_pe = delimitedList(line_of_code_pe, delim='\n').ignore(comment_pe)

################
# Entry Points #
################

def parseSourceCode(input_string: str): # TODO add return type
    return grammar_pe.parseString(input_string, parseAll=True)

# TODO enable this
# __all__ = [
#     'parseSourceCode'
# ]

##########
# Driver #
##########

if __name__ == '__main__':
    print("TODO add something here")

