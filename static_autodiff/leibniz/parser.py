
'''
'''

# TODO fill in doc string

###########
# Imports #
###########

import pyparsing
from pyparsing import pyparsing_common as ppc
from pyparsing import Word, Literal
from pyparsing import oneOf, infixNotation, opAssoc

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

# ParserElement.setDefaultWhitespaceChars(' \t') # TODO enable this

# Atom Parser Elements

def parse_boolean(_s: str, _loc: int, tokens: pyparsing.ParseResults):
    assert len(tokens) == 1
    token = tokens.pop()
    assert token in ('True', 'False')
    bool_token = True if token == 'True' else False
    tokens.append(bool_token)

boolean_pe = oneOf('True False').setName('boolean').setParseAction(parse_boolean)

integer_pe = ppc.signed_integer.setName('integer')
_float_pe = ppc.real | ppc.sci_real
real_pe = (_float_pe | integer_pe).setName('real number')

identifier_pe = ppc.identifier.setName('identifier')

atom_pe = (boolean_pe | real_pe | identifier_pe).setName('atom')

# Arithmetic Operation Parser Elements

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

# Expression Parser Elements

arithmetic_expression_pe = infixNotation(
    real_pe | identifier_pe,
    [
        (exponent_operation_pe, 2, opAssoc.RIGHT),
        (multiplication_operation_pe | division_operation_pe, 2, opAssoc.LEFT),
        (addition_operation_pe | subtraction_operation_pe, 2, opAssoc.LEFT),
    ],
).setName('arithmetic expression')

boolean_expression_pe = infixNotation(
    boolean_pe | identifier_pe,
    [
        (not_operation_pe, 2, opAssoc.RIGHT),
        (and_operation_pe, 2, opAssoc.LEFT),
        (xor_operation_pe, 2, opAssoc.LEFT),
        (or_operation_pe, 2, opAssoc.LEFT),
    ],
).setName('boolean expression')

expression_pe = (atom_pe | arithmetic_expression_pe | boolean_expression_pe).setName('expression')

# Declaration Parser Elements

declaration_pe = (identifier_pe + "=" + expression_pe).setName('declaration')

# Grammar

grammar_pe = declaration_pe # TODO update this

################
# Entry Points #
################

def parseSourceCode(input_string: str): # TODO add return type
    return grammar_pe.parseString(input_string, parseAll=True)

##########
# Driver #
##########

if __name__ == '__main__':
    print("TODO add something here")
