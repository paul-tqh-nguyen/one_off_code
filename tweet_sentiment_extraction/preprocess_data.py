 #!/usr/bin/python3
'#!/usr/bin/python3 -OO'

'''
'''

# @todo fill this in
# @todo verify all the imports are used here and elsewhere

###########
# Imports #
###########

import re
import spacy
import string
import unicodedata
import tqdm
import pandas as pd
import multiprocessing as mp
import numpy as np
import concurrent.futures
from typing import Tuple, List, Callable
from spacy.tokenizer import Tokenizer

from model_utilities import *
from misc_utilities import *

import torchtext

###########
# Globals #
###########

PREPROCESS_TEXT_IN_PARALLEL = False

SIMPLE_URL_RE = re.compile(r'''(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))''')

def is_url(input_string: str) -> bool:
    match = SIMPLE_URL_RE.match(input_string)
    if match: # Adding '$' to the end of the regexp causes severe slowdowns
        start_index, end_index = match.span()
        return start_index == 0 and end_index == len(input_string)
    return False

def initialize_tokenizer() -> Callable:
    nlp = spacy.load('en')
    prefix_re = re.compile(r'''^['''+re.escape(string.punctuation)+r''']''')
    suffix_re = re.compile(r'''['''+re.escape(string.punctuation)+r''']$''')
    infix_re = re.compile(r'''['''+re.escape(string.punctuation)+r''']''')
    nlp.tokenizer = Tokenizer(nlp.vocab,
                              prefix_search=prefix_re.search,
                              suffix_search=suffix_re.search,
                              infix_finditer=infix_re.finditer,
                              token_match=is_url)
    tokenizer = lambda input_string: [t.text for t in nlp(input_string)]
    return tokenizer
TOKENIZER = initialize_tokenizer()

URL_SPECIAL_TOKEN = 'URLLINK'

SPECIAL_TOKENS = {
    URL_SPECIAL_TOKEN,
}

SPECIAL_CHARACTER_SENTINEL = '9'

SPECIAL_CHARACTER_TO_NORMALIZED_VALUE_MAP = {
    '″': '"',
    '´': "'",
    '′': "'",
    '”': '"',
    '“': '"',
    '’': "'",
    '‘': "'",
    '—': '-',
    '–': '-',
    'Å': 'A',
    'Â': 'A',
    'à': 'a',
    'â': 'a',
    'å': 'a',
    'ā': 'a',
    'ç': 'c',
    'è': 'e',
    'é': 'e',
    'ê': 'e',
    'ë': 'e',
    'ì': 'i',
    'ï': 'i',
    '¡': 'i',
    'ö': 'o',
    'Ø': 'o',
    'ñ': 'n',
    'ü': 'u',
    '×': 'x',
    # Exotic characters
    '¿': SPECIAL_CHARACTER_SENTINEL,
    '©': SPECIAL_CHARACTER_SENTINEL,
    '®': SPECIAL_CHARACTER_SENTINEL,
    '»': SPECIAL_CHARACTER_SENTINEL,
    '«': SPECIAL_CHARACTER_SENTINEL,
    '…': SPECIAL_CHARACTER_SENTINEL,
    '½': SPECIAL_CHARACTER_SENTINEL,
}

#############################
# Sanity Checking Utilities #
#############################

def sanity_check_training_data_json_file() -> bool:
    if __debug__:
        import json
        with open(PREPROCESSED_TRAINING_DATA_JSON_FILE, 'r') as json_file_handle:
            for row_text in tqdm.tqdm(json_file_handle.readlines()):
                row_data = json.loads(row_text)
                original_text = row_data['text']
                selected_text = row_data['selected_text']
                token_index_to_position_info_map_json = row_data['token_index_to_position_info_map']
                token_index_to_position_info_map = {int(token_index_as_string): position_info for token_index_as_string, position_info in token_index_to_position_info_map_json.items()}
                numericalized_selected_text = row_data['numericalized_selected_text']
                selected_token_indices = [token_index for token_index, is_selected in enumerate(numericalized_selected_text) if is_selected == '1']
                reconstructed_selected_text = reconstruct_selected_text_from_token_indices(selected_token_indices, original_text, token_index_to_position_info_map)
                reconstructed_selected_text_sufficiently_matches_selected_text = jaccard_index_over_strings(reconstructed_selected_text, selected_text) == 1
                
                preprocessed_input_string, _ = preprocess_tweet(original_text)
                preprocessed_selected_text, _ = preprocess_tweet(selected_text)
                selected_text_is_bogus = any(selected_text_position_validity(preprocessed_input_string, preprocessed_selected_text))

                # @todo handle the cases revealed by this section
                preprocessed_selected_text_tokens = preprocessed_selected_text.split()
                preprocessed_selected_text_start_token = preprocessed_selected_text_tokens[0]
                preprocessed_selected_text_end_token = preprocessed_selected_text_tokens[-1]
                selected_text_is_possibly_bogus = (
                    (not reconstructed_selected_text_sufficiently_matches_selected_text) and \
                    (selected_text_is_bogus or SPECIAL_TOKENS.intersection({preprocessed_selected_text_start_token, preprocessed_selected_text_end_token}))
                )
                selected_text_needs_manual_review = selected_text_is_possibly_bogus and not selected_text_is_bogus and not reconstructed_selected_text_sufficiently_matches_selected_text
                if selected_text_needs_manual_review:
                    print(f'''

Please manually review that the reconstructed text {repr(reconstructed_selected_text)}
is sufficiently similar to the selected text       {repr(selected_text)}

''')
                
                assert (reconstructed_selected_text_sufficiently_matches_selected_text or selected_text_is_bogus or selected_text_needs_manual_review), \
                    f'\nReconstructed {repr(reconstructed_selected_text)}\nas opposed to {repr(selected_text)}\nfrom          {repr(original_text)}'
    return True

############################
# Interpretation Utilities #
############################

def reconstruct_selected_text_from_token_indices(token_indices: List[int], original_text: str, token_index_to_position_info_map: dict) -> str:
    assert token_indices == sorted(token_indices)
    selected_text_substrings: List[str] = []
    previous_end_position = None
    for token_index in token_indices:
        token_position_data = token_index_to_position_info_map[token_index]
        token_original_start_position = token_position_data['token_original_start_position']
        token_original_end_position = token_position_data['token_original_end_position']
        selected_text_substring = original_text[token_original_start_position:token_original_end_position]
        assert selected_text_substring == token_position_data['original_token']
        if previous_end_position is not None:
            selected_text_substrings.append(original_text[previous_end_position:token_original_start_position])
        selected_text_substrings.append(selected_text_substring)
        previous_end_position = token_original_end_position
    selected_text = ''.join(selected_text_substrings)
    return selected_text

###########################
# Preprocessing Utilities #
###########################

def pervasively_replace(input_string: str, old: str, new: str) -> str:
    while old in input_string:
        input_string = input_string.replace(old, new)
    return input_string

def normalize_string_white_space(input_string: str) -> str:
    normalized_input_string = input_string
    normalized_input_string = normalized_input_string.replace('\xa0',' ')
    normalized_input_string = pervasively_replace(normalized_input_string, '\t', ' ')
    normalized_input_string = pervasively_replace(normalized_input_string, '\n', ' ')
    normalized_input_string = pervasively_replace(normalized_input_string, '  ',' ')
    normalized_input_string = normalized_input_string.strip()
    return normalized_input_string

def normalize_special_characters(input_string: str) -> str:
    output_string = input_string
    for special_character, normalized_character in SPECIAL_CHARACTER_TO_NORMALIZED_VALUE_MAP.items():
        output_string = output_string.replace(special_character, normalized_character)
    return output_string

def preprocess_token(input_token: str) -> str:
    preprocessed_token = input_token
    if is_url(preprocessed_token):
        preprocessed_token = URL_SPECIAL_TOKEN
    return preprocessed_token

def is_ascii(input_string: str) -> bool:
    return all(ord(character) < 128 for character in input_string)

def preprocess_and_tokenize_string(input_string: str) -> str:
    preprocessed_input_string = input_string
    preprocessed_input_string = preprocessed_input_string.lower()
    preprocessed_input_string = normalize_string_white_space(preprocessed_input_string)
    preprocessed_input_string = normalize_special_characters(preprocessed_input_string)
    assert is_ascii(preprocessed_input_string)
    preprocessed_tokens = TOKENIZER(preprocessed_input_string)
    preprocessed_tokens = eager_map(preprocess_token, preprocessed_tokens)
    return preprocessed_tokens

def preprocess_tweet(input_string: str) -> Tuple[str, List[dict]]:
    current_input_string_position = 0
    token_index_to_position_info_map = {}
    white_space_normalized_input_string = normalize_string_white_space(input_string)
    special_character_normalized_input_string = normalize_special_characters(input_string)
    normalized_input_string = normalize_special_characters(white_space_normalized_input_string)
    assert len(white_space_normalized_input_string) == len(normalized_input_string)
    normalized_input_string_tokens = TOKENIZER(normalized_input_string)
    preprocessed_tokens = preprocess_and_tokenize_string(input_string)
    assert len(normalized_input_string_tokens) == len(preprocessed_tokens)
    for token_index, (normalized_token, preprocessed_token) in enumerate(zip(normalized_input_string_tokens, preprocessed_tokens)):
        for _ in range(len(input_string)):
            if special_character_normalized_input_string[current_input_string_position] != normalized_token[0]:
                current_input_string_position +=1
            else:
                break
        token_start = current_input_string_position
        current_input_string_position += len(normalized_token)
        token_end = current_input_string_position
        assert normalized_token[0] == normalize_special_characters(input_string[token_start])
        assert normalized_token[-1] == normalize_special_characters(input_string[token_end-1])
        assert normalized_token == special_character_normalized_input_string[token_start:token_end]
        assert current_input_string_position <= len(input_string)
        original_token = input_string[token_start:token_end]
        token_index_to_position_info_map[token_index] = {'token_original_start_position': token_start, 'token_original_end_position': token_end, 'original_token': original_token, 'preprocessed_token': preprocessed_token}
    preprocessed_input_string = ' '.join(preprocessed_tokens)
    return preprocessed_input_string, token_index_to_position_info_map

###################################
# Text Numericalization Utilities #
###################################

def selected_text_position_validity(preprocessed_input_string: str, preprocessed_selected_text: str) -> Tuple[bool, bool]:
    assert '  ' not in preprocessed_input_string
    assert '  ' not in preprocessed_selected_text
    preprocessed_selected_text_tokens = preprocessed_selected_text.split()
    assert ' '.join(preprocessed_selected_text_tokens) == preprocessed_selected_text
    if len(preprocessed_selected_text_tokens) == 1:
        preprocessed_input_string_tokens = preprocessed_input_string.split()
        assert ' '.join(preprocessed_input_string_tokens) == preprocessed_input_string
        preprocessed_selected_text_token = only_one(preprocessed_selected_text_tokens)
        selected_text_starts_in_middle_of_word = preprocessed_selected_text_token not in preprocessed_input_string_tokens
        selected_text_ends_in_middle_of_word = selected_text_starts_in_middle_of_word
    else:
        # @todo this doesn't properly handle the case where the end cuts off in the middle of a URL or other special token
        selected_text_starts_in_middle_of_word = ' '+preprocessed_selected_text not in preprocessed_input_string and preprocessed_selected_text != preprocessed_input_string[:len(preprocessed_selected_text)]
         # @todo this doesn't handle the case where the start cuts off in the middle of a url.
         # @todo this doesn't handle the case where the end cuts off in the middle of a url.
        selected_text_ends_in_middle_of_word = preprocessed_selected_text+' ' not in preprocessed_input_string and preprocessed_selected_text != preprocessed_input_string[-len(preprocessed_selected_text):]
    return selected_text_starts_in_middle_of_word, selected_text_ends_in_middle_of_word

def numericalize_selected_text(preprocessed_input_string: str, selected_text: str) -> str:
    assert isinstance(preprocessed_input_string, str)
    assert isinstance(selected_text, str)
    preprocessed_selected_text, _ = preprocess_tweet(selected_text)
    selected_text_starts_in_middle_of_word, selected_text_ends_in_middle_of_word = selected_text_position_validity(preprocessed_input_string, preprocessed_selected_text)
    if selected_text_starts_in_middle_of_word:
        preprocessed_selected_text = ' '.join(preprocessed_selected_text.split()[1:])
    if selected_text_ends_in_middle_of_word:
        preprocessed_selected_text = ' '.join(preprocessed_selected_text.split()[:-1])
    assert preprocessed_selected_text in preprocessed_input_string
    preprocessed_selected_text_tokens = preprocessed_selected_text.split()
    preprocessed_input_string_tokens = preprocessed_input_string.split()
    numericalized_text = [0]*len(preprocessed_input_string_tokens)
    for preprocessed_token_index in range(len(preprocessed_input_string_tokens)):
        if preprocessed_selected_text_tokens == preprocessed_input_string_tokens[preprocessed_token_index:preprocessed_token_index+len(preprocessed_selected_text_tokens)]:
            for numericalized_text_index in range(preprocessed_token_index,preprocessed_token_index+len(preprocessed_selected_text_tokens)):
                numericalized_text[numericalized_text_index] = 1
            break
    assert len(list(uniq(numericalized_text))) <= 3
    numericalized_text_as_string = ''.join(map(str, numericalized_text))
    assert len(numericalized_text_as_string) == len(preprocessed_input_string_tokens)
    return numericalized_text_as_string

###############
# Main Driver #
###############

@debug_on_error
def preprocess_data() -> None:
    training_data_df = pd.read_csv(TRAINING_DATA_CSV_FILE)
    training_data_df.selected_text = training_data_df.selected_text.replace(np.nan, '', regex=True)
    training_data_df[['text', 'selected_text']] = training_data_df[['text', 'selected_text']].fillna(value='')
    print()
    print('Preprocessing tweets...')
    if PREPROCESS_TEXT_IN_PARALLEL:
        with concurrent.futures.ProcessPoolExecutor(mp.cpu_count()) as pool:
            text_series_preprocessed = pd.Series(pool.map(preprocess_tweet, training_data_df.text, chunksize=1000))
    else:
        text_series_preprocessed = training_data_df.text.progress_map(preprocess_tweet)
    preprocessed_input_string_series = text_series_preprocessed.progress_map(lambda pair: pair[0])
    token_index_to_position_info_map_series = text_series_preprocessed.progress_map(lambda pair: pair[1])
    assert all(preprocessed_input_string_series.map(lambda x: isinstance(x, str)))
    assert all(token_index_to_position_info_map_series.map(lambda x: isinstance(x, dict)))
    training_data_df['preprocessed_input_string'] = preprocessed_input_string_series
    training_data_df['token_index_to_position_info_map'] = token_index_to_position_info_map_series
    print()
    print('Numericalizing selected texts...')
    numericalized_selected_text_series = training_data_df[['preprocessed_input_string', 'selected_text']].progress_apply(lambda row: numericalize_selected_text(row[0], row[1]), axis=1)
    assert isinstance(numericalized_selected_text_series, pd.Series)
    training_data_df['numericalized_selected_text'] = numericalized_selected_text_series
    print(f'''{len(training_data_df[training_data_df.numericalized_selected_text.map(lambda x: 0 if x=='' else int(x)) == 0])} out of {len(training_data_df)} '''
          '''entirely unhandled cases likely due to selected text starting or ending within a word.''')
    training_data_df = training_data_df[training_data_df.numericalized_selected_text.str.contains('1')]
    training_data_df.to_json(PREPROCESSED_TRAINING_DATA_JSON_FILE, orient='records', lines=True)
    print()
    print(f'Preprocessed data saved to {PREPROCESSED_TRAINING_DATA_JSON_FILE} with {len(training_data_df)} entries.')
    assert sanity_check_training_data_json_file()
    return

if __name__ == '__main__':
    preprocess_data()
