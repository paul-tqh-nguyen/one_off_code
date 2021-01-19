
'''
'''

# TODO fill in doc string

###########
# Imports #
###########

import pyparsing
from pyparsing import pyparsing_common as ppc
from pyparsing import Word

from misc_utilities import *

# TODO make usure imports are used

###########
# Globals #
###########

# pyparsing.ParserElement.enablePackrat() # TODO consider enabling this

###########
# Grammar #
###########

# ParserElement.setDefaultWhitespaceChars(' \t') # TODO enable this

# Convention: The "_pe" suffix indicates an instance of pyparsing.ParserElement

integer_pe = ppc.signed_integer

_float_pe = ppc.real | ppc.sci_real

real_pe = _float_pe | integer_pe

grammar_pe = real_pe # TODO update this

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
