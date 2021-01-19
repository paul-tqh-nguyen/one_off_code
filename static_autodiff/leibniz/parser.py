
'''
'''

###########
# Imports #
###########

import pyparsing
from pyparsing import pyparsing_common as ppc
from pyparsing import Word

from misc_utilities import *

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

##########
# Driver #
##########

# @debug_on_error
# def main() -> None:
#     input_string = '''
# 0.3213 1.2 -123 1e2 3
# '''
#     parse_result = grammar_pe.parseString(input_string, parseAll=True)
#     print(f"parse_result {repr(parse_result)}")
#     breakpoint()
#     return

if __name__ == '__main__':
    main()
