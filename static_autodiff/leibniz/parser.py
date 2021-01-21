
'''
'''

# TODO fill in doc string

###########
# Imports #
###########

import pyparsing
from pyparsing import pyparsing_common as ppc
from pyparsing import Word, Literal, Forward, Empty, Optional, Combine, Regex, ZeroOrMore, Suppress
from pyparsing import oneOf, infixNotation, opAssoc, delimitedList, pythonStyleComment

from .misc_utilities import *

# TODO make usure imports are used

###########
# Globals #
###########

# pyparsing.ParserElement.enablePackrat() # TODO consider enabling this

###########
# Grammar #
###########

# Convention: The "_pe" suffix indicates an instance of pyparsing.ParserElement

pyparsing.ParserElement.setDefaultWhitespaceChars(' \t')

# Type Parser Elements

boolean_type_pe = Literal('Boolean').setName('boolean type')
integer_type_pe = Literal('Integer').setName('integer type')
real_type_pe = Literal('Real').setName('real type')

type_pe = boolean_type_pe | integer_type_pe | real_type_pe

def _parse_tracer(s: str, loc: int, tokens: pyparsing.ParseResults): # TODO Remove this
    # .addParseAction(_parse_tracer)
    print(f"s {repr(s)}")
    print(f"loc {repr(loc)}")
    print(f"tokens {repr(tokens)}")    

# Literal Parser Elements

def parse_boolean(_s: str, _loc: int, tokens: pyparsing.ParseResults):
    token = only_one(tokens)
    assert token in ('True', 'False')
    bool_value = True if token == 'True' else False
    return bool_value

boolean_pe = oneOf('True False').setName('boolean').setParseAction(parse_boolean)

integer_pe = ppc.integer.setName('integer')
_float_pe = ppc.real | ppc.sci_real
real_pe = (_float_pe | integer_pe).setName('real number')

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

_boolean_operation_pe = not_operation_pe | and_operation_pe | xor_operation_pe | or_operation_pe

# Atom Parser Elements

identifier_pe = ~_boolean_operation_pe + ~boolean_pe + ppc.identifier.setName('identifier')

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
).setName('arithmetic expression').addParseAction(_parse_tracer)

boolean_expression_pe = infixNotation(
    boolean_pe | identifier_pe,
    [
        (not_operation_pe, 1, opAssoc.RIGHT),
        (and_operation_pe, 2, opAssoc.LEFT),
        (xor_operation_pe, 2, opAssoc.LEFT),
        (or_operation_pe, 2, opAssoc.LEFT),
    ],
).setName('boolean expression')

_function_variable_binding_pe = identifier_pe + ':=' + expression_pe
expr = _function_variable_binding_pe
_function_variable_bindings_pe = Optional(delimitedList(_function_variable_binding_pe))
function_call_pe = (identifier_pe + Literal('(') + _function_variable_bindings_pe + Literal(')')).setName('function call')

expression_pe <<= (function_call_pe | arithmetic_expression_pe | boolean_expression_pe | atom_pe).setName('expression')

# Declaration Parser Elements

value_declaration_pe = (identifier_pe + Optional(Literal(':') + type_pe) + "=" + expression_pe).setName('value declaration')
type_declaration_pe = (identifier_pe + Literal(':') + type_pe).setName('type declaration')
name_declaration_pe = (identifier_pe).setName('name declaration')

_atomic_declaration_pe = (value_declaration_pe | type_declaration_pe | name_declaration_pe)
declaration_pe = delimitedList(_atomic_declaration_pe, delim=';').setName('expression').setName('declaration')

# Grammar

comment_pe = Regex(r"#(?:\\\n|[^\n])*").setName('end-of-line comment')

line_of_code_pe = Optional(declaration_pe) # TODO update this with imperative code as well

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

