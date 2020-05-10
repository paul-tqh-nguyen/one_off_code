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
from typing import Tuple, List

from misc_utilities import *

import torchtext

###########
# Globals #
###########

SPACY_TOKENIZER = torchtext.data.get_tokenizer('spacy')

TRAINING_DATA_CSV_FILE = './data/train.csv'
PREPROCESSED_TRAINING_DATA_JSON_FILE = './data/preprocessed_train.json'

###########################
# Preprocessing Utilities #
###########################

def pervasively_replace(input_string: str, old: str, new: str) -> str:
    while old in input_string:
        input_string = input_string.replace(old, new)
    return input_string

def normalize_input_string(input_string: str) -> str:
    normalized_input_string = input_string
    normalized_input_string = pervasively_replace(normalized_input_string, '\t', ' ')
    normalized_input_string = pervasively_replace(normalized_input_string, '\n', ' ')
    normalized_input_string = pervasively_replace(normalized_input_string, '  ',' ')
    normalized_input_string = normalized_input_string.strip()
    return normalized_input_string

def preprocess_token(token: str) -> str:
    preprocessed_token = token.lower()
    return preprocessed_token

def preprocess_tweet(input_string: str) -> Tuple[str, List[dict]]:
    current_input_string_position = 0
    token_index_to_position_info_map = {}
    normalized_input_string = normalize_input_string(input_string)
    input_string_tokens = SPACY_TOKENIZER(normalized_input_string)
    preprocessed_tokens = eager_map(preprocess_token, input_string_tokens)
    for token_index, (token, preprocessed_token) in enumerate(zip(input_string_tokens, preprocessed_tokens)):
        for _ in range(len(input_string)):
            if input_string[current_input_string_position] != token[0]:
                current_input_string_position +=1
            else:
                break
        token_start = current_input_string_position
        current_input_string_position += len(token)
        token_end = current_input_string_position
        assert token[0] == input_string[token_start]
        assert token[-1] == input_string[token_end-1]
        assert current_input_string_position <= len(input_string)
        token_index_to_position_info_map[token_index] = {'token_original_start_position': token_start, 'token_original_end_position': token_end, 'original_token': token, 'preprocessed_token': preprocessed_token}
    preprocessed_input_string = ' '.join(preprocessed_tokens)
    return (preprocessed_input_string, token_index_to_position_info_map)

###############
# Main Driver #
###############

@debug_on_error
def preprocess_data() -> None:
    training_data_df = pd.read_csv(TRAINING_DATA_CSV_FILE)
    training_data_df[['text', 'selected_text']] = training_data_df[['text', 'selected_text']].fillna(value='')
    text_series = training_data_df.text
    text_series_preprocessed = text_series.map(preprocess_tweet)
    preprocessed_input_string_series = text_series_preprocessed.map(lambda pair: pair[0])
    position_dicts_series = text_series_preprocessed.map(lambda pair: pair[1])
    training_data_df['preprocessed_input_string'] = preprocessed_input_string_series
    training_data_df['position_dicts'] = position_dicts_series
    training_data_df = training_data_df.set_index('textID')
    training_data_df.to_json(PREPROCESSED_TRAINING_DATA_JSON_FILE, orient='index')
    return

if __name__ == '__main__':
    preprocess_data()
