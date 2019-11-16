#!/usr/bin/python3 -O

"""

String processsing utilities for text processing.

Owner : paul-tqh-nguyen

Created : 11/15/2019

File Name : string_processing_utilities.py

File Organization:
* Imports
* Unknown Word DWIMming Utilities
* Meaningful Character Sequence Utilities
* Misc. String Utilities

"""

###########
# Imports #
###########

import string
import html
import re
from spellchecker import SpellChecker
from word2vec_utilities import WORD2VEC_MODEL
from typing import List, Tuple

###################################
# Unknown Word DWIMming Utilities #
###################################

def omg_star_applicability(word_string: str) -> bool:
    return re.findall("^omg.+$", word_string.lower())

def omg_star_expand(word_string: str) -> str:
    return ["omg"]

def num_word_concatenation_applicability(word_string: str) -> bool:
    return re.findall("^[0-9]+[a-z]+$", word_string.lower())
    
def num_word_concatenation_expand(word_string: str) -> str:
    number_matches = re.findall("^[0-9]+", word_string)
    assert len(number_matches)==1
    number_substring = number_matches[0]
    character_substring = word_string.replace(number_substring, '')
    return [number_substring, character_substring]

VOWELS = {'a','e','i','o','u'}
SPELL_CHECKER = SpellChecker()

def possibly_dwim_duplicate_letters_exaggeration(word_string: str) -> Tuple[bool,str]:
    reduced_word = word_string.lower()
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
        if no_change_happened:
            break
    candidate_words_via_spell_checker = SPELL_CHECKER.candidates(reduced_word)
    candidate_words_that_dont_introduce_new_characters = filter(lambda word: set(word)==letters, candidate_words_via_spell_checker)
    for candidate_word in candidate_words_that_dont_introduce_new_characters:
        if candidate_word in WORD2VEC_MODEL:
            reduced_word = candidate_word
            reduced_word_is_known = True
            break
    return reduced_word_is_known, reduced_word

def duplicate_letters_exaggeration_applicability(word_string: str) -> bool:
    corrected_word_is_known, _ = possibly_dwim_duplicate_letters_exaggeration(word_string)
    return corrected_word_is_known

def duplicate_letters_exaggeration_expand(word_string: str) -> str:
    _, corrected_word = possibly_dwim_duplicate_letters_exaggeration(word_string)
    return [corrected_word]

DWIMMING_APPLICABILITY_FUNCTION_EXPAND_FUNCTION_PAIRS = [
    (omg_star_applicability, omg_star_expand),
    (num_word_concatenation_applicability, num_word_concatenation_expand),
    (duplicate_letters_exaggeration_applicability, duplicate_letters_exaggeration_expand),
]

def possibly_dwim_unknown_words(word_strings: List[str]) -> str:
    updated_word_strings = []
    for word_string in word_strings:
        final_strings = [word_string]
        if word_string not in WORD2VEC_MODEL:
            for applicability_function, expand_function in DWIMMING_APPLICABILITY_FUNCTION_EXPAND_FUNCTION_PAIRS:
                if applicability_function(word_string):
                    final_strings = expand_function(word_string)
                    break
        for final_string in final_strings:
            updated_word_strings.append(final_string)
    return updated_word_strings

###########################################
# Meaningful Character Sequence Utilities #
###########################################

EMOTICONS = '''
:‑D :D 8‑D 8D x‑D xD X‑D XD =D =3 B^D :‑( :( :‑c :c :‑< :< :‑[ :[ :-|| >:[ :{ :@ >:( :-)) :'‑( :'( :'‑) :') D‑': D:< D: D8 D; D= DX :‑O :O :‑o :o :-0 8‑0 >:O :-* :* :× ;‑) ;) *-) *) ;‑] ;] ;^) :‑, ;D :‑P :P X‑P XP x‑p xp :‑p :p :‑Þ :Þ :‑þ :þ :‑b :b d: =p >:P :‑/ :/ :‑. >:\ >:/ :\ =/ =\ :L =L :S :‑| :| :$ ://) ://3 :‑X :X :‑# :# :‑& :& O:‑) O:) 0:‑3 0:3 0:‑) 0:) 0;^) >:‑) >:) }:‑) }:) 3:‑) 3:) >;) >:3 >;3 |;‑) |‑O :‑J #‑) %‑) %) <:‑| ',:-| ',:-l :-| T_T @-) 
'''.strip().split(' ')

''' These emoticons cause trouble with treating ".." as a "..."
:‑###.. :###.. 
'''

MEANINGFUL_SPECIAL_CHARACTER_SEQUENCES = EMOTICONS+[
    "...",
]

PLACEHOLDER_PREFIX = "0meaningful0special0character0sequence0id"

MEANINGFUL_SPECIAL_CHARACTER_SEQUENCE_TO_PLACE_HOLDER_MAP = {meaningful_special_character_sequence : PLACEHOLDER_PREFIX+str(index) \
                                                             for index, meaningful_special_character_sequence in \
                                                             enumerate(MEANINGFUL_SPECIAL_CHARACTER_SEQUENCES)}

def replace_meaningful_special_character_sequence_with_placeholder_token(text_string: str) -> str:
    text_string_with_replacements = text_string
    text_string_with_replacements = simplify_elipsis_sequences(text_string_with_replacements)
    for meaningful_special_character_sequence, placeholder in MEANINGFUL_SPECIAL_CHARACTER_SEQUENCE_TO_PLACE_HOLDER_MAP.items():
        text_string_with_replacements = text_string_with_replacements.replace(meaningful_special_character_sequence, ' '+placeholder+' ')
    return text_string_with_replacements

def word_string_resembles_meaningful_special_character_sequence_placeholder(word_string: str) -> bool:
    return bool(re.findall("^0meaningful0special0character0sequence0id[0-9]*$", word_string))

##########################
# Misc. String Utilities #
##########################

def lower_case_string(input_string: str) -> str:
    return input_string.lower()

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

def simplify_spaces(input_string: str) -> str:
    return simplify_substrings_until_quiescence('  ',' ',input_string).strip()

PUNCTUATION_SET = set(string.punctuation)

def separate_punctuation(text_string: str) -> str:
    final_text_string = text_string
    for punctuation_character in PUNCTUATION_SET:
        final_text_string = final_text_string.replace(punctuation_character, " "+punctuation_character+" ")
    return final_text_string

def normalized_words_from_text_string(text_string: str) -> List[str]: 
    normalized_text_string = text_string
    normalized_text_string = html.unescape(normalized_text_string)
    normalized_text_string = replace_meaningful_special_character_sequence_with_placeholder_token(normalized_text_string)
    # @todo expand contractions
    normalized_text_string = separate_punctuation(normalized_text_string)
    normalized_text_string = simplify_spaces(normalized_text_string)
    normalized_words = normalized_text_string.split(' ')
    normalized_words = possibly_dwim_unknown_words(normalized_words)
    normalized_words = map(lower_case_string, normalized_words)
    return normalized_words

# @todo do we need this?
# def remove_punctuation(input_string:str) -> str:
#     return ''.join([char for char in input_string if char not in PUNCTUATION_SET])

# @todo figure out how we handle punctuation prior to enabling this step
# def possibly_replace_word_wrt_elisions(word_string: str) -> str:
#     final_string = word_string
#     word_is_known =  word_string in WORD2VEC_MODEL
#     if not word_is_known:
#         if final_string[-2:] == 'in':
#             final_string += 'g'
#     return final_string
