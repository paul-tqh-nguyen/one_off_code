#!/usr/bin/python3
'#!/usr/bin/python3 -OO'

"""
"""

# @todo fill this in
# @todo verify all the imports are used here and elsewhere

###########
# Imports #
###########

import pandas as pd
from typing import Tuple

###########################
# Preprocessing Utilities #
###########################

def preprocess_article_text(input_string: str) -> Tuple[str, pd.DataFrame]:
    preprocessed_input_string = input_string
    preprocessed_input_string = preprocessed_input_string.lower()
    
    return (preprocessed_input_string, preprocessed_input_string_position_df)

###############
# Main Driver #
###############

if __name__ == '__main__':
    print() # @todo fill this in
