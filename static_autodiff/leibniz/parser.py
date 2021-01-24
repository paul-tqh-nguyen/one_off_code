
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

def _trace_parse(s: str, loc: int, tokens: pyparsing.ParseResults): # TODO remove this
    print()
    print(f"s {repr(s)}")
    print(f"loc {repr(loc)}")
    print(f"tokens {repr(tokens)}")
    

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
    
class AtomASTNodeType(type):

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
        
        def __init__(self, *args, **kwargs):
            assert sum(map(len, (args, kwargs))) == 1
            value: value_type = args[0] if len(args) == 1 else kwargs[value_attribute_name]
            setattr(self, value_attribute_name, value)

        @classmethod
        def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> class_name:
            token = only_one(tokens)
            assert token_checker(token)
            value: value_type = dwimming_func(token)
            node_instance: cls = cls(value)
            return node_instance
        
        def __eq__(self, other: ASTNode) -> bool: # TODO replace '__eq__' with 'is_equivalent'
            other_value = getattr(other, value_attribute_name)
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

class StatementASTNode(ASTNode):
    pass
    
class ExpressionASTNode(StatementASTNode):
    pass

class ExpressionAtomASTNodeType(AtomASTNodeType):
    '''The atomic bases that make up expressions.'''

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

class TypeASTNode(metaclass=AtomASTNodeType):
    value_type = typing.Optional[str]
    token_checker = lambda token: token in (None, 'Boolean', 'Integer', 'Real')

# Literal Node Generation

class BooleanLiteralASTNode(metaclass=ExpressionAtomASTNodeType):
    value_type = bool
    token_checker = lambda token: token in ('True', 'False')
    dwimming_func = lambda token: True if token == 'True' else False

class RealLiteralASTNode(metaclass=ExpressionAtomASTNodeType):
    value_type = bool
    token_checker = lambda token: isinstance(token, (float, int))

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
    
    def __init__(self, function_name: VariableASTNode, arg_bindings: typing.List[typing.Tuple[VariableASTNode, ExpressionASTNode]]):
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
    def __eq__(self, other: 'ASTNode') -> bool:
        # TODO make the below use is_equivalent instead of "=="
        return type(self) is type(other) and \
            self.function_name == other.function_name and \
            all(
                self_arg_binding == other_arg_binding
                for self_arg_binding, other_arg_binding
                in zip(self.arg_bindings, other.arg_bindings)
            )

# Assignment Node Generation

def parse_assignment_type_declaration_pe(_s: str, _loc: int, type_label_tokens: pyparsing.ParseResults) -> TypeASTNode:
    assert len(type_label_tokens) in (0, 1)
    if len(type_label_tokens) is 0:
        node_instance: TypeASTNode = TypeASTNode(None)
    elif len(type_label_tokens) is 1:
        node_instance: TypeASTNode = only_one(type_label_tokens)
    else:
        raise ValueError(f'Unexpected type label tokens {type_label_tokens}')
    assert isinstance(node_instance, TypeASTNode)
    return node_instance

class AssignmentASTNode(StatementASTNode):
    
    def __init__(self, identifier: VariableASTNode, identifier_type: TypeASTNode, value: ExpressionASTNode):
        self.identifier = identifier
        self.identifier_type = identifier_type
        self.value = value
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'AssignmentASTNode':
        assert len(tokens) is 3
        node_instance = cls(tokens[0], tokens[1], tokens[2])
        return node_instance

    # def is_equivalent(self, other: 'ASTNode') -> bool: # TODO Enable this
    def __eq__(self, other: 'ASTNode') -> bool:
        # TODO make the below use is_equivalent instead of "=="
        return type(self) is type(other) and \
            self.identifier == other.identifier and \
            self.identifier_type == other.identifier_type and \
            self.value == other.value

# Module Node Generation

class ModuleASTNode(ASTNode):
    
    def __init__(self, statements: typing.List[StatementASTNode]):
        self.statements: typing.List[StatementASTNode] = statements
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'ModuleASTNode':
        statement_ast_node_lists  = tokens.asList()
        statement_ast_nodes = sum(statement_ast_node_lists, [])
        node_instance = cls(statement_ast_nodes)
        return node_instance

    # def is_equivalent(self, other: 'ASTNode') -> bool: # TODO Enable this
    def __eq__(self, other: 'ASTNode') -> bool:
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

identifier_pe = (~boolean_operation_pe + ~boolean_pe + ppc.identifier).setName('identifier').setParseAction(VariableASTNode.parse_action)

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
        (not_operation_pe, 1, opAssoc.RIGHT, NotExpressionASTNode.parse_action),
        (and_operation_pe, 2, opAssoc.LEFT, AndExpressionASTNode.parse_action),
        (xor_operation_pe, 2, opAssoc.LEFT, XorExpressionASTNode.parse_action),
        (or_operation_pe, 2, opAssoc.LEFT, OrExpressionASTNode.parse_action),
    ],
).setName('boolean expression')

function_variable_binding_pe = Group(identifier_pe + Suppress(':=') + expression_pe)
function_variable_bindings_pe = Group(Optional(delimitedList(function_variable_binding_pe)))
function_call_expression_pe = (identifier_pe + Suppress('(') + function_variable_bindings_pe + Suppress(')')).setName('function call').setParseAction(FunctionCallExpressionASTNode.parse_action)

expression_pe <<= (function_call_expression_pe | arithmetic_expression_pe | boolean_expression_pe | atom_pe).setName('expression')

# Assignment Parser Elements

assignment_type_declaration_pe = Optional(Suppress(':') + type_pe).setParseAction(parse_assignment_type_declaration_pe)
assignment_value_declaration_pe = Suppress('=') + expression_pe
assignment_pe = (identifier_pe + assignment_type_declaration_pe + assignment_value_declaration_pe).setParseAction(AssignmentASTNode.parse_action).setName('assignment')

# Module & Misc. Parser Elements

comment_pe = Regex(r"#(?:\\\n|[^\n])*").setName('comment')

atomic_statement_pe = Group(Optional(assignment_pe | expression_pe))

non_atomic_statement_pe = atomic_statement_pe + Suppress(';') + delimitedList(atomic_statement_pe, delim=';')

statement_pe = (non_atomic_statement_pe | atomic_statement_pe).setName('statement')

module_pe = delimitedList(statement_pe, delim='\n').ignore(comment_pe).setParseAction(ModuleASTNode.parse_action)

################
# Entry Points #
################

class ParseError(Exception):

    def __init__(self, original_text, problematic_text, problem_column_number):
        self.original_text = original_text
        self.problematic_text = problematic_text
        self.problem_column_number = problem_column_number
        super().__init__(f'''Could not parse the following:

    {self.problematic_text}
    {(' '*(self.problem_column_number - 1))}^
''')

def parseSourceCode(input_string: str): # TODO add return type
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

if __name__ == '__main__':
    print("TODO add something here")

