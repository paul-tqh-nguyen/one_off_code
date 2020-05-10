#!/usr/bin/python3
'#!/usr/bin/python3 -OO'

"""
"""

# @todo fill this in
# @todo verify all the imports are used here and elsewhere

###########
# Imports #
###########

import re
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

def normalize_string_white_space(input_string: str) -> str:
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
    normalized_input_string = normalize_string_white_space(input_string)
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

def numericalize_selected_text(input_string: str, selected_text: str) -> List[int]:
    assert isinstance(input_string, str)
    assert isinstance(selected_text, str)
    normalized_input_string = normalize_string_white_space(input_string)
    normalized_selected_text = normalize_string_white_space(selected_text)
    assert normalized_selected_text in normalized_input_string
    input_string_tokens = SPACY_TOKENIZER(normalized_input_string)
    normalized_selected_text_match = next(re.finditer(re.escape(normalized_selected_text), normalized_input_string))
    normalized_selected_text_start_position, normalized_selected_text_end_position = (normalized_selected_text_match.start(), normalized_selected_text_match.end())
    assert normalized_input_string[normalized_selected_text_start_position:normalized_selected_text_end_position] == normalized_selected_text
    current_sentence_position = 0
    numericalized_text = [0]*len(input_string_tokens)
    for token_index, input_string_token in enumerate(input_string_tokens):
        if current_sentence_position >= normalized_selected_text_start_position:
            assert current_sentence_position == normalized_selected_text_start_position or numericalized_text[token_index-1]==1 or \
                normalized_input_string[normalized_selected_text_start_position-1] != ' ' # for mislabelled data / inter-word cutoff
            numericalized_text[token_index] = 1
        current_sentence_position += len(input_string_token)
        if current_sentence_position<len(normalized_input_string):
            if normalized_input_string[current_sentence_position] == ' ':
                current_sentence_position += 1
        if current_sentence_position >= normalized_selected_text_end_position:
            assert current_sentence_position == normalized_selected_text_end_position or \
                (normalized_selected_text_end_position == current_sentence_position-1 and normalized_input_string[current_sentence_position-1] == ' ') or \
                (normalized_selected_text_end_position<len(normalized_input_string) and normalized_input_string[normalized_selected_text_end_position] != ' ') # for mislabelled data / inter-word cutoff
            assert numericalized_text[token_index] == 1 or \
                (normalized_input_string[normalized_selected_text_start_position-1] != ' ' or normalized_input_string[normalized_selected_text_end_position] != ' ') # for mislabelled data / inter-word cutoff
            break
    assert len(list(uniq(numericalized_text))) <= 3
    return numericalized_text

###############
# Main Driver #
###############

@debug_on_error
def preprocess_data() -> None:
    training_data_df = pd.read_csv(TRAINING_DATA_CSV_FILE)
    training_data_df[['text', 'selected_text']] = training_data_df[['text', 'selected_text']].fillna(value='')
    text_series_preprocessed = training_data_df.text.map(preprocess_tweet)
    preprocessed_input_string_series = text_series_preprocessed.map(lambda pair: pair[0])
    token_index_to_position_info_map_series = text_series_preprocessed.map(lambda pair: pair[1])
    training_data_df['preprocessed_input_string'] = preprocessed_input_string_series
    training_data_df['token_index_to_position_info_map'] = token_index_to_position_info_map_series
    numericalize_selected_text_series = training_data_df[['text', 'selected_text']].apply(lambda row: numericalize_selected_text(row[0], row[1]), axis=1)
    assert isinstance(numericalize_selected_text_series, pd.Series)
    training_data_df['numericalize_selected_text'] = numericalize_selected_text_series
    if __debug__:
        print(f"{len(training_data_df[training_data_df.numericalize_selected_text.map(sum) == 0])} out of {len(training_data_df)} unhandled cases likely due to selected text starting or ending within a word.")
    training_data_df = training_data_df[training_data_df.numericalize_selected_text.map(sum) > 0]
    training_data_df.to_json(PREPROCESSED_TRAINING_DATA_JSON_FILE, orient='records', lines=True)
    return

if __name__ == '__main__':
    preprocess_data()
