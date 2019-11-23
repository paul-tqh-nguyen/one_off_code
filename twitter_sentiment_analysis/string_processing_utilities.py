#!/usr/bin/python3 -O

"""

String processsing utilities for text processing.

Owner : paul-tqh-nguyen

Created : 11/15/2019

File Name : string_processing_utilities.py

File Organization:
* Imports
* Misc. Utilities
* Meaningful Character Sequence Utilities
* Named Entity Handling
* Contraction Expansion
* Unknown Word DWIMming Utilities
* Misc. String Utilities

"""

###########
# Imports #
###########

import string
import html
import re
import spellchecker
import unicodedata
from functools import lru_cache
from word2vec_utilities import WORD2VEC_MODEL
from named_entity_recognition_via_wikidata import string_corresponding_wikidata_term_type_pairs
from typing import List, Tuple, Callable

###################
# Misc. Utilities #
###################

def false(*args, **kwargs):
    return False

PLACEHOLDER_PREFIX = "place0holder0token0with0id"

def word_string_resembles_meaningful_special_character_sequence_placeholder(word_string: str) -> bool:
    return bool(re.findall(r"^"+PLACEHOLDER_PREFIX+r".+$", word_string))

def common_word_missing_from_word2vec_model(word_string: str) -> bool:
    return word_string.lower() in ['a','to', 'and', 'of']

def unknown_word_worth_dwimming(word_string: str) -> bool:
    return not word_string.isnumeric() and \
        not common_word_missing_from_word2vec_model(word_string) and \
        word_string not in PUNCTUATION_SET and \
        not word_string_resembles_meaningful_special_character_sequence_placeholder(word_string) and \
        word_string not in WORD2VEC_MODEL

def _correct_words_via_subsequence_substitutions(text_string: str, old_subsequence: str, new_subsequence: str, word_exception_checker: Callable[[str], bool]=false) -> str:
    updated_text_string = text_string
    word_match_iterator = re.finditer(r"\b\w+\b", text_string)
    for word_match in word_match_iterator:
        word_string = word_match.group()
        if not word_exception_checker(word_string):
            if unknown_word_worth_dwimming(word_string):
                word_string_normalized = word_string.lower()
                if old_subsequence in word_string_normalized:
                    corrected_word = word_string_normalized.replace(old_subsequence, new_subsequence)
                    if corrected_word in WORD2VEC_MODEL:
                        updated_text_string = re.sub(r"\b"+word_string+r"\b", corrected_word, updated_text_string, 1)
    return updated_text_string

###########################################
# Meaningful Character Sequence Utilities #
###########################################

EMOTICONS = '''
:‑D :D 8‑D 8D x‑D X‑D =D =3 B^D :‑( :( :‑c :c :‑< :< :‑[ :[ :-|| >:[ :{ :@ >:( :-)) :'‑( :'( :'‑) :') D‑': D:< D: D8 D; D= :‑O :O :‑o :o :-0 8‑0 >:O :-* :* :× ;‑) ;) *-) *) ;‑] ;] ;^) :‑, ;D :‑P :P X‑P x‑p :‑p :p :‑Þ :Þ :‑þ :þ :‑b :b d: =p >:P :‑/ :/ :‑. >:\ >:/ :\ =/ =\ :L =L :S :‑| :| :$ ://) ://3 :‑X :X :‑# :# :‑& :& O:‑) O:) 0:‑3 0:3 0:‑) 0:) 0;^) >:‑) >:) }:‑) }:) 3:‑) 3:) >;) >:3 >;3 |;‑) |‑O :‑J #‑) %‑) %) <:‑| ',:-| ',:-l :-| T_T @-) 
'''.strip().split(' ')

EMOTICON_TO_PLACE_HOLDER_MAP = {meaningful_special_character_sequence : PLACEHOLDER_PREFIX+'0emoticon'+str(index) \
                                for index, meaningful_special_character_sequence in \
                                enumerate(EMOTICONS)}

ELIPSIS_PLACE_HOLDER = PLACEHOLDER_PREFIX+'0elipsis'

def replace_meaningful_special_character_sequence_with_placeholder_token(text_string: str) -> str:
    text_string_with_replacements = text_string
    text_string_with_replacements = simplify_elipsis_sequences(text_string_with_replacements)
    for emoticon, placeholder in EMOTICON_TO_PLACE_HOLDER_MAP.items():
        text_string_with_replacements = re.sub(r"\b"+re.escape(emoticon)+r"\b", placeholder, text_string_with_replacements, 0)
    text_string_with_replacements = text_string_with_replacements.replace('...', ELIPSIS_PLACE_HOLDER)
    return text_string_with_replacements

#########################
# Named Entity Handling #
#########################

NAMED_ENTITY_PLACEHOLDER = PLACEHOLDER_PREFIX+"0named0entity"

# @todo also see if we can handle the "Also known as" lexical information we get if we click on the Wikidata search suggestions; it would handle cases like "SYTYCD"
def replace_well_known_named_entities_with_placeholder_token(text_string: str) -> str:
    text_string_with_replacements = text_string
    word_strings = re.findall(r"\b\w+\b", text_string)
    for word_string in word_strings: # @todo handle multi-word cases
        if unknown_word_worth_dwimming(word_string):
            # @todo consider doing something more robust with the extra semantic information
            word_string_is_well_known_named_entity_via_wikidata = bool(string_corresponding_wikidata_term_type_pairs(word_string))
            if word_string_is_well_known_named_entity_via_wikidata:
                text_string_with_replacements = re.sub(r"\b"+word_string+r"\b", NAMED_ENTITY_PLACEHOLDER, text_string_with_replacements, 1)
    return text_string_with_replacements

#########################
# Contraction Expansion #
#########################

CONTRACTION_EXPANSION_MAP = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

CONTRACTION_EXPANSION_PAIRS_SORTED_BIGGEST_FIRST = sorted(CONTRACTION_EXPANSION_MAP.items(), key=lambda x: len(x[0]), reverse=True)

def expand_contractions(text_string: str) -> str:
    updated_text_string = text_string
    for contraction, expansion in CONTRACTION_EXPANSION_PAIRS_SORTED_BIGGEST_FIRST:
        updated_text_string = re.sub(r"\b"+contraction+r"\b", expansion, updated_text_string, 0, re.IGNORECASE)
    return updated_text_string

###################################
# Unknown Word DWIMming Utilities #
###################################

def omg_star_expand(text_string: str) -> str:
    expanded_text_string = text_string
    word_replacement = "omg"
    assert word_replacement in WORD2VEC_MODEL
    expanded_text_string = re.sub(r"\bomg\w+\b", word_replacement, expanded_text_string, 0, re.IGNORECASE)
    return expanded_text_string

def number_word_concatenation_expand(text_string: str) -> str:
    expanded_text_string = text_string
    matches = re.findall(r"\b[0-9]+[a-z]+\b", text_string, re.IGNORECASE)
    for match in matches:
        if unknown_word_worth_dwimming(match):
            numeric_half_matches = re.findall(r"\b[0-9]+", match)
            assert len(numeric_half_matches) == 1
            numeric_half_match = numeric_half_matches[0]
            alphabetic_half = match.replace(numeric_half_match, "")
            replacement = numeric_half_match+' '+alphabetic_half
            expanded_text_string = re.sub(r"\b"+match+r"\b", replacement, expanded_text_string, 1)
    return expanded_text_string

def aw_star_expand(text_string: str) -> str:
    expanded_text_string = text_string
    matches = re.findall(r"\baw[w|a|e|i|o|u|h|\!]*\b", text_string, re.IGNORECASE)
    word_replacement = "aw"
    assert word_replacement in WORD2VEC_MODEL
    for match in matches:
        if match.lower() != 'awe':
            expanded_text_string = expanded_text_string.replace(match, word_replacement)
    return expanded_text_string

def two_word_concatenation_expand(text_string: str) -> str:
    updated_text_string = text_string
    word_match_iterator = re.finditer(r"\b\w+\b", text_string)
    for word_match in word_match_iterator:
        word = word_match.group()
        if unknown_word_worth_dwimming(word):
            word_length = len(word)
            min_first_word_length = 5
            min_second_word_length = 3
            if word_length > min_first_word_length+min_second_word_length:
                split_index_supremum = word_length-(min_second_word_length-1)
                for split_index in range(min_first_word_length, split_index_supremum):
                    first_sub_word = word[:split_index]
                    second_sub_word = word[split_index:]
                    if first_sub_word in WORD2VEC_MODEL and second_sub_word in WORD2VEC_MODEL:
                        split_words_combined = first_sub_word+' '+second_sub_word
                        updated_text_string = re.sub(r"\b"+word+r"\b", split_words_combined, updated_text_string, 1)
                        break
    return updated_text_string

def _starts_with(big: str, small: str) -> bool:
    len_small = len(small)
    return len(big) >= len_small and big[:len_small] == small

def _word_string_corresponds_to_laugh(word_string: str) -> bool:
    word_string_corresponds_to_laugh = False
    word_string = word_string.lower()
    acceptable_prefixes = "mw", "mu", "h", "g", "b", "bw"
    prefix_starts_word_string_test = lambda acceptable_starting_string: _starts_with(word_string, acceptable_starting_string)
    relevant_prefixes = list(filter(prefix_starts_word_string_test, acceptable_prefixes))
    if len(relevant_prefixes) != 0:
        relevant_prefix = max(relevant_prefixes, key=len)
        relevant_prefix_len = len(relevant_prefix)
        remaining_word_string = word_string[relevant_prefix_len:]
        non_laugh_characters = re.sub(r"[hga]+", '', remaining_word_string, 0, re.IGNORECASE)
        word_string_corresponds_to_laugh = len(non_laugh_characters)==0
    return word_string_corresponds_to_laugh

def laughing_expand(text_string: str) -> str:
    text_string_with_replacements = text_string
    word_match_iterator = re.finditer(r"\b\w+\b", text_string, re.IGNORECASE)
    for word_match in word_match_iterator:
        word = word_match.group()
        if unknown_word_worth_dwimming(word):
            if _word_string_corresponds_to_laugh(word):
                text_string_with_replacements = re.sub(r"\b"+word+r"\b", 'haha', text_string_with_replacements, 1)
    return text_string_with_replacements

VOWELS = {'a','e','i','o','u'}
SPELL_CHECKER = spellchecker.SpellChecker()

def duplicate_letters_exaggeration_expand(text_string: str) -> str:
    updated_text_string = text_string
    word_match_iterator = re.finditer(r"\b\w+\b", text_string)
    for word_match in word_match_iterator:
        word_string = word_match.group()
        if unknown_word_worth_dwimming(word_string):
            reduced_word = word_string.lower()
            reduced_word_is_known = False
            letters = set(reduced_word)
            for _ in word_string:
                no_change_happened = True
                for letter in letters:
                    letter_probable_upper_limit = 2 if letter in VOWELS else 1
                    disallowed_letter_duplicate_sequence = letter*(letter_probable_upper_limit+1)
                    disallowed_letter_duplicate_sequence_replacement = letter*letter_probable_upper_limit
                    if disallowed_letter_duplicate_sequence in reduced_word:
                        no_change_happened = False
                        reduced_word = reduced_word.replace(disallowed_letter_duplicate_sequence, disallowed_letter_duplicate_sequence_replacement)
                        reduced_word_is_known = reduced_word in WORD2VEC_MODEL
                        if reduced_word_is_known:
                            break
                if no_change_happened or reduced_word_is_known:
                    break
            if not reduced_word_is_known:
                candidate_words_via_spell_checker = SPELL_CHECKER.candidates(reduced_word)
                candidate_words_that_dont_introduce_new_characters = filter(lambda word: set(word)==letters, candidate_words_via_spell_checker)
                for candidate_word in candidate_words_that_dont_introduce_new_characters:
                    if candidate_word in WORD2VEC_MODEL:
                        reduced_word = candidate_word
                        reduced_word_is_known = True
                        break
            if not reduced_word_is_known:
                if len(candidate_words_via_spell_checker) == 1:
                    reduced_word = tuple(candidate_words_via_spell_checker)[0]
                    reduced_word_is_known = reduced_word in WORD2VEC_MODEL
            if reduced_word_is_known:
                updated_text_string = re.sub(r"\b"+word_string+r"\b", reduced_word, updated_text_string, 1)
    return updated_text_string

SLANG_WORD_DICTIONARY = {
    "ily2" : "I love you too",
    "sowwy" : "sorry",
    "luvly" : "lovely",
    "ilysfm" : "I love you so fucking much",
    "fu2" : "fuck you too",
}

def slang_word_expand(text_string: str) -> str:
    updated_text_string = text_string
    word_match_iterator = re.finditer(r"\b\w+\b", text_string)
    for word_match in word_match_iterator:
        word_string = word_match.group()
        if unknown_word_worth_dwimming(word_string):
            word_string_canonicalized_for_lookup = word_string.lower()
            if word_string_canonicalized_for_lookup in SLANG_WORD_DICTIONARY:
                expanded_text = SLANG_WORD_DICTIONARY[word_string_canonicalized_for_lookup]
                updated_text_string = re.sub(r"\b"+word_string+r"\b", expanded_text, updated_text_string, 1)
    return updated_text_string

def our_or_british_sland_correction_expand(text_string: str) -> str:
    word_is_our_exactly = lambda word: word.lower() == 'our'
    return _correct_words_via_subsequence_substitutions(text_string, 'our', 'or', word_is_our_exactly)

def q_g_slang_correction_expand(text_string: str) -> bool:
    return _correct_words_via_subsequence_substitutions(text_string, 'q', 'g')

def f_ph_slang_correction_expand(text_string: str) -> bool:
    return _correct_words_via_subsequence_substitutions(text_string, 'f', 'ph')

def g_dropping_correction_expand(text_string: str) -> bool:
    return _correct_words_via_subsequence_substitutions(text_string, 'in', 'ing')

DWIMMING_EXPAND_FUNCTIONS = [
    omg_star_expand,
    number_word_concatenation_expand,
    aw_star_expand,
    two_word_concatenation_expand,
    g_dropping_correction_expand,
    q_g_slang_correction_expand,
    f_ph_slang_correction_expand,
    our_or_british_sland_correction_expand,
    laughing_expand,
    slang_word_expand,
    duplicate_letters_exaggeration_expand,
]

def perform_single_pass_to_dwim_unknown_words(text_string: str) -> str:
    updated_text_string = text_string
    for expand_function in DWIMMING_EXPAND_FUNCTIONS:
        expanded_result = expand_function(updated_text_string)
        if expanded_result != updated_text_string:
            updated_text_string = expanded_result
            break
    return updated_text_string

def possibly_dwim_unknown_words(text_string: str) -> str:
    current_text_string = text_string
    premature_exit = False
    for _ in text_string:
        updated_text_string = perform_single_pass_to_dwim_unknown_words(current_text_string)
        if current_text_string == updated_text_string:
            premature_exit = True
            break
        else:
            current_text_string = updated_text_string
    assert premature_exit, "Unknown word DWIMming did not process until quiescence."
    return current_text_string

##########################
# Misc. String Utilities #
##########################

def replace_exotic_character(character: str) -> str:
    normalized_character = unicodedata.normalize('NFKD', character).encode('ascii', 'ignore').decode('utf-8')
    if normalized_character != character:
        normalized_character = " "+normalized_character+" "
    return normalized_character

def replace_exotic_characters(text_string: str) -> str:
    text_string_with_replacements = text_string
    text_string_with_replacements = ''.join(map(replace_exotic_character, text_string))
    return text_string_with_replacements

TAGGED_USER_PLACEHOLDER = PLACEHOLDER_PREFIX+"0tagged0user"

def replace_tagged_users_with_placeholder_token(text_string: str) -> str:
    updated_text_string = text_string
    tagged_users = re.findall(r"@\w+", text_string)
    for tagged_user in tagged_users:
        updated_text_string = updated_text_string.replace(tagged_user, ' '+TAGGED_USER_PLACEHOLDER+' ')
    return updated_text_string

URL_PLACEHOLDER = PLACEHOLDER_PREFIX+"0url0link"
URL_REGEX_PATTERN = "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"

def replace_urls_with_placeholder_token(text_string: str) -> str:
    updated_text_string = text_string
    urls = re.findall(URL_REGEX_PATTERN, text_string)
    for url in urls:
        updated_text_string = updated_text_string.replace(url, URL_PLACEHOLDER)
    return updated_text_string

HASH_TAG_REGEX_PATTERN = "#\w+"

def replace_hash_tags_with_placeholder_token(text_string: str) -> str:
    updated_text_string = text_string
    hash_tags = re.findall(HASH_TAG_REGEX_PATTERN, text_string)
    for hash_tag in hash_tags:
        updated_text_string = updated_text_string.replace(hash_tag, PLACEHOLDER_PREFIX+'0hash0tag')
    return updated_text_string

def lower_case_unknown_words(text_string: str) -> str:
    text_string_with_replacements = text_string
    word_strings = re.findall(r"\b\w+\b", text_string)
    for word_string in word_strings:
        if unknown_word_worth_dwimming(word_string):
            replacement = word_string.lower()
            text_string_with_replacements = re.sub(r"\b"+word_string+r"\b", replacement, text_string_with_replacements, 1)
    return text_string_with_replacements

def simplify_substrings_until_quiescence(old: str, new: str, input_string: str) -> str:
    simplified_string = input_string
    for _ in simplified_string:
        if old not in simplified_string:
            break
        else:
            simplified_string = simplified_string.replace(old, new)
    return simplified_string

def simplify_elipsis_sequences(input_string: str) -> str:
    simplified_sequence = input_string
    simplified_sequence = input_string.replace('..','...')
    simplified_sequence = simplify_substrings_until_quiescence('....','...',simplified_sequence)
    return simplified_sequence

PUNCTUATION_SET = set(string.punctuation)

def simplify_spaces(input_string: str) -> str:
    return simplify_substrings_until_quiescence('  ',' ',input_string).strip()

def separate_punctuation(text_string: str) -> str:
    final_text_string = text_string
    for punctuation_character in PUNCTUATION_SET:
        final_text_string = final_text_string.replace(punctuation_character, " "+punctuation_character+" ")
    final_text_string = simplify_spaces(final_text_string)
    return final_text_string

def normalized_words_from_text_string(text_string: str) -> List[str]: 
    normalized_text_string = text_string
    normalized_text_string = html.unescape(normalized_text_string)
    normalized_text_string = replace_tagged_users_with_placeholder_token(normalized_text_string)
    normalized_text_string = replace_urls_with_placeholder_token(normalized_text_string)
    normalized_text_string = replace_hash_tags_with_placeholder_token(normalized_text_string)
    normalized_text_string = replace_exotic_characters(normalized_text_string)
    print(normalized_text_string)
    normalized_text_string = replace_meaningful_special_character_sequence_with_placeholder_token(normalized_text_string)
    normalized_text_string = expand_contractions(normalized_text_string)
    normalized_text_string = separate_punctuation(normalized_text_string)
    normalized_text_string = possibly_dwim_unknown_words(normalized_text_string)
    normalized_text_string = replace_well_known_named_entities_with_placeholder_token(normalized_text_string)
    normalized_text_string = lower_case_unknown_words(normalized_text_string)
    normalized_words = normalized_text_string.split(' ')
    return normalized_words

def main():
    print("This module contains string normalization utilities for sentiment analysis on Twitter data.")

if __name__ == '__main__':
    main()
