
'''
'''

# TODO fill in doc string

###########
# Imports #
###########

import typing_extensions
import inspect
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
    QuotedString,
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
from functools import reduce, lru_cache
from collections import defaultdict
import operator

from .parser_utilities import (
    BASE_TYPES,
    BaseTypeName,
    sanity_check_base_types,
    TensorTypeParserElementBaseTypeTracker,
    AtomicLiteralParserElementBaseTypeTracker,
    LiteralASTNodeClassBaseTypeTracker
)
from .ast_node import (
    ASTNode,
    AtomASTNodeType,
    StatementASTNode,
    ExpressionASTNode,
    ExpressionAtomASTNodeType,
    BinaryOperationExpressionASTNode,
    UnaryOperationExpressionASTNode,
    ArithmeticExpressionASTNode,
    BooleanExpressionASTNode,
    ComparisonExpressionASTNode,
    BooleanLiteralASTNode,
    IntegerLiteralASTNode,
    FloatLiteralASTNode,
    NothingTypeLiteralASTNode,
    VariableASTNode,
    NegativeExpressionASTNode,
    ExponentExpressionASTNode,
    MultiplicationExpressionASTNode,
    DivisionExpressionASTNode,
    AdditionExpressionASTNode,
    SubtractionExpressionASTNode,
    NotExpressionASTNode,
    AndExpressionASTNode,
    XorExpressionASTNode,
    OrExpressionASTNode,
    GreaterThanExpressionASTNode,
    GreaterThanOrEqualToExpressionASTNode,
    LessThanExpressionASTNode,
    LessThanOrEqualToExpressionASTNode,
    EqualToExpressionASTNode,
    NotEqualToExpressionASTNode,
    FunctionCallExpressionASTNode,
    VectorExpressionASTNode,
    TensorTypeASTNode,
    AssignmentASTNode,
    ReturnStatementASTNode,
    PrintStatementASTNode,
    ScopedStatementSequenceASTNode,
    FunctionDefinitionASTNode,
    ForLoopASTNode,
    WhileLoopASTNode,
    ConditionalASTNode,
    ModuleASTNode,
)
from .misc_utilities import *

# TODO make sure imports are used
# TODO make sure these imports are ordered in some way

###########
# Globals #
###########

BOGUS_TOKEN = object()

pyparsing.ParserElement.enablePackrat()

###########
# Grammar #
###########

def _trace_parse(name): # TODO remove this
    def func(s: str, loc: int, tokens: pyparsing.ParseResults):
        print()
        print(f"name {repr(name)}")
        print(f"s {repr(s)}")
        print(f"loc {repr(loc)}")
        print(f"tokens {repr(tokens)}")
    return func

# Convention: The "_pe" suffix indicates an instance of pyparsing.ParserElement

pyparsing.ParserElement.setDefaultWhitespaceChars(' \t')

# Type Parser Elements

tensor_dimension_declaration_pe = Suppress('<') + Group(Optional(Literal('???') | delimitedList(Literal('?') | ppc.integer))) + Suppress('>')

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

# Comparison Operation Parser Elements

greater_than_comparator_keyword_pe = Literal('>').setName('greater than operation')
greater_than_or_equal_to_comparator_keyword_pe = Literal('>=').setName('greater than or equal to operation')
less_than_comparator_keyword_pe = Literal('<').setName('less than operation')
less_than_or_equal_to_comparator_keyword_pe = Literal('<=').setName('less than or equal to operation')
equal_to_comparator_keyword_pe = Literal('==').setName('equal to operation')
not_equal_to_comparator_keyword_pe = Literal('!=').setName('not equal to operation')

comparator_pe = greater_than_or_equal_to_comparator_keyword_pe | greater_than_comparator_keyword_pe | less_than_or_equal_to_comparator_keyword_pe | less_than_comparator_keyword_pe | equal_to_comparator_keyword_pe | not_equal_to_comparator_keyword_pe

# Identifier Parser Elements

# identifiers are not variables as identifiers can also be used as function names. Function names are not variables since functions are not first class citizens.

print_keyword_pe = Suppress('print')

return_keyword_pe = Suppress('return')

if_keyword_pe = Suppress('if')

else_keyword_pe = Suppress('else')

while_loop_keyword_pe = Suppress('while')

for_loop_keyword_pe = Suppress('for')

function_definition_keyword_pe = Suppress('function')

not_reserved_keyword_pe = reduce(operator.add, map(NotAny, map(Suppress, BASE_TYPES))) + (
    ~comparator_pe +
    ~nothing_pe +
    ~boolean_operation_pe +
    ~boolean_pe +
    ~print_keyword_pe +
    ~while_loop_keyword_pe +
    ~for_loop_keyword_pe +
    ~if_keyword_pe +
    ~else_keyword_pe +
    ~function_definition_keyword_pe +
    ~return_keyword_pe
)

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

comparison_expression_pe = (
    arithmetic_expression_pe +
    OneOrMore(comparator_pe + arithmetic_expression_pe)
).setName('comparison expression').setParseAction(parse_comparison_expression_pe)

function_variable_binding_pe = Group(variable_pe + Suppress(':=') + expression_pe)
function_variable_bindings_pe = Group(Optional(delimitedList(function_variable_binding_pe)))
function_call_expression_pe = (identifier_pe + Suppress('(') + function_variable_bindings_pe + Suppress(')')).setName('function call').setParseAction(FunctionCallExpressionASTNode.parse_action)

vector_pe = Forward()
expression_pe <<= (function_call_expression_pe | (comparison_expression_pe ^ arithmetic_expression_pe) | boolean_expression_pe | atom_pe | vector_pe).setName('expression')

# Vector Parser Elements

vector_pe <<= (Suppress('[') + delimitedList(expression_pe, delim=',') + Suppress(']')).setName('tensor').setParseAction(VectorExpressionASTNode.parse_action)

# Assignment Parser Elements

variable_type_declaration_pe = Optional(Suppress(':') + tensor_type_pe).setParseAction(parse_variable_type_declaration_pe)
assignment_pe = (
    Group(delimitedList(Group(variable_pe + variable_type_declaration_pe))) + 
    Suppress('=') +
    expression_pe
).setParseAction(AssignmentASTNode.parse_action).setName('assignment')

# Ignorable Parser Elements

comment_pe = Regex(r"#(?:\\\n|[^\n])*").setName('comment')
newline_escapes = Suppress('\\\n')

ignorable_pe = comment_pe | newline_escapes

# Return Statement Parser Elements

return_statement_pe = (return_keyword_pe + Optional(delimitedList(expression_pe))).setParseAction(ReturnStatementASTNode.parse_action).setName('return statetment')

# Print Statement Parser Elements

print_statement_pe = (print_keyword_pe + Suppress('(') + Optional(delimitedList(
    QuotedString('"', escChar='\\', multiline=True) | expression_pe
)) + Suppress(')')).setParseAction(PrintStatementASTNode.parse_action).setName('print statetment')

# Statement Parser Elements

conditional_pe = Forward()
while_loop_pe = Forward()
for_loop_pe = Forward()
function_definition_pe = Forward()
scoped_statement_sequence_pe = Forward()

required_atomic_statement_pe = scoped_statement_sequence_pe | conditional_pe | while_loop_pe | for_loop_pe | function_definition_pe | return_statement_pe | print_statement_pe | assignment_pe | expression_pe

atomic_statement_pe = Optional(required_atomic_statement_pe)

non_atomic_statement_pe = atomic_statement_pe + Suppress(';') + delimitedList(atomic_statement_pe, delim=';')

statement_pe = (non_atomic_statement_pe | atomic_statement_pe).setName('statement')

# Scoped Statement Sequence Parser Elements

statement_sequence_pe = Optional(delimitedList(statement_pe, delim='\n').ignore(ignorable_pe))

scoped_statement_sequence_pe <<= (Suppress('{') + statement_sequence_pe + Suppress('}')).setParseAction(ScopedStatementSequenceASTNode.parse_action)


# Function Definition Parser Elements

function_signature_pe = Suppress('(') + Group(Optional(delimitedList(Group(variable_pe + variable_type_declaration_pe)))) + Suppress(')')
function_return_type_pe = (Suppress('->') + tensor_type_pe).setParseAction(TensorTypeASTNode.parse_action)

function_definition_pe <<= (
    function_definition_keyword_pe +
    identifier_pe +
    function_signature_pe +
    function_return_type_pe +
    required_atomic_statement_pe
).ignore(ignorable_pe).setParseAction(FunctionDefinitionASTNode.parse_action)

# For Loop Parser Elements

for_loop_pe <<= (
    for_loop_keyword_pe +
    identifier_pe +
    Suppress(':') +
    Group(
        Suppress('(') +
        arithmetic_expression_pe +
        Suppress(',') +
        arithmetic_expression_pe +
        Optional(Suppress(',') + arithmetic_expression_pe) +
        Suppress(')')
    ) + 
    required_atomic_statement_pe
).ignore(ignorable_pe).setParseAction(ForLoopASTNode.parse_action)

# While Loop Parser Elements

while_loop_pe <<= (
    while_loop_keyword_pe +
    boolean_expression_pe +
    required_atomic_statement_pe
).ignore(ignorable_pe).setParseAction(WhileLoopASTNode.parse_action)

# Conditional Parser Elements

conditional_pe <<= (
    if_keyword_pe +
    boolean_expression_pe +
    required_atomic_statement_pe +
    Optional(else_keyword_pe + required_atomic_statement_pe)
).ignore(ignorable_pe).setParseAction(ConditionalASTNode.parse_action)

# Module & Misc. Parser Elements

module_pe = statement_sequence_pe.copy().setParseAction(ModuleASTNode.parse_action)

################
# Entry Points #
################

class ParseError(Exception):

    def __init__(self, original_text: str, problematic_text: str, problem_column_number: int) -> None:
        self.original_text = original_text
        self.problematic_text = problematic_text
        self.problem_column_number = problem_column_number
        super().__init__(f'''Could not parse the following:

    {self.problematic_text}
    {(' '*(self.problem_column_number - 1))}^
''')
        return

def parseSourceCode(input_string: str) -> ModuleASTNode:
    '''The returned ModuleASTNode may contain the same identical-in-memory nodes due to caching.'''
    try:
        result = only_one(module_pe.parseString(input_string, parseAll=True))
    except pyparsing.ParseException as exception:
        raise ParseError(input_string, exception.line, exception.col)
    return result

# TODO enable this
# __all__ = [
#     'parseSourceCode'
# ]

#############################
# Sanity Checking Utilities #
#############################

def sanity_check_concrete_ast_node_subclass_method_annotations() -> None:
    for ast_node_class in child_classes(ASTNode):

        # __init__ method
        init_method = ast_node_class.__init__
        assert inspect.signature(init_method).return_annotation is None

        # __eq__ method
        eq_method = ast_node_class.__eq__
        assert inspect.signature(eq_method).return_annotation is bool
        
        # parse_action method
        parse_action_method = ast_node_class.parse_action
        if not getattr(parse_action_method, '__isabstractmethod__', False):
            return_annotation = inspect.signature(parse_action_method).return_annotation
            if issubclass(ast_node_class, ComparisonExpressionASTNode):
                assert return_annotation == 'ComparisonExpressionASTNode'
            elif issubclass(ast_node_class, (ArithmeticExpressionASTNode, BooleanExpressionASTNode)):
                if issubclass(ast_node_class, UnaryOperationExpressionASTNode):
                    assert return_annotation == 'UnaryOperationExpressionASTNode'
                elif issubclass(ast_node_class, BinaryOperationExpressionASTNode):
                    assert return_annotation == 'BinaryOperationExpressionASTNode'
            else:
                assert return_annotation == ast_node_class.__qualname__, f'{ast_node_class.__qualname__}.parse_action is not declared to return a {ast_node_class.__qualname__}'
    return

@lru_cache(maxsize=128)
def perform_sanity_check() -> None:
    sanity_check_base_types()
    sanity_check_concrete_ast_node_subclass_method_annotations()
    return

perform_sanity_check()

##########
# Driver #
##########

if __name__ == '__main__':
    print("TODO add something here")
