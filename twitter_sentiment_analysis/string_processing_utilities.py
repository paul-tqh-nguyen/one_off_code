#!/usr/bin/python3 -O

"""

String processsing utilities for text processing.

Owner : paul-tqh-nguyen

Created : 11/15/2019

File Name : string_processing_utilities.py

File Organization:
* Imports
* Contraction Expansion
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
import spellchecker
from word2vec_utilities import WORD2VEC_MODEL
from typing import List, Tuple

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
        updated_text_string = updated_text_string.replace(contraction, expansion)
    return updated_text_string

###################################
# Unknown Word DWIMming Utilities #
###################################

def omg_star_applicability(word_string: str) -> bool:
    return bool(re.findall("^omg.+$", word_string.lower()))

def omg_star_expand(word_string: str) -> List[str]:
    expanded_word = "omg"
    assert expanded_word in WORD2VEC_MODEL 
    return [expanded_word]

def number_word_concatenation_applicability(word_string: str) -> bool:
    return bool(re.findall("^[0-9]+[a-z]+$", word_string.lower()))
    
def number_word_concatenation_expand(word_string: str) -> List[str]:
    number_matches = re.findall("^[0-9]+", word_string)
    assert len(number_matches)==1
    number_substring = number_matches[0]
    character_substring = word_string.replace(number_substring, '')
    assert character_substring in WORD2VEC_MODEL
    return [number_substring, character_substring]

def aw_star_applicability(word_string: str) -> bool:
    return bool(re.findall("^aw[a|e|i|o|u|h|\!]*$", word_string.lower()))

def aw_star_expand(word_string: str) -> List[str]:
    expanded_word = "aw"
    assert expanded_word in WORD2VEC_MODEL 
    return [expanded_word]

VOWELS = {'a','e','i','o','u'}
CONSONANTS = {'b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','x','z','w','y'}
ALPHABETIC_CHARACTERS = VOWELS.union(CONSONANTS)
SPELL_CHECKER = spellchecker.SpellChecker()

def possibly_dwim_duplicate_letters_exaggeration(word_string: str) -> Tuple[bool,str]:
    reduced_word = word_string.lower()
    reduced_word_is_known = False
    letters = set(reduced_word)
    if letters.issubset(ALPHABETIC_CHARACTERS):
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
    return reduced_word_is_known, reduced_word

def duplicate_letters_exaggeration_applicability(word_string: str) -> bool:
    corrected_word_is_known, _ = possibly_dwim_duplicate_letters_exaggeration(word_string)
    return corrected_word_is_known

def duplicate_letters_exaggeration_expand(word_string: str) -> str:
    _, corrected_word = possibly_dwim_duplicate_letters_exaggeration(word_string)
    assert corrected_word in WORD2VEC_MODEL
    return [corrected_word]

DWIMMING_APPLICABILITY_FUNCTION_EXPAND_FUNCTION_PAIRS = [
    (omg_star_applicability, omg_star_expand),
    (number_word_concatenation_applicability, number_word_concatenation_expand),
    (aw_star_applicability, aw_star_expand),
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
:‑D :D 8‑D 8D x‑D X‑D =D =3 B^D :‑( :( :‑c :c :‑< :< :‑[ :[ :-|| >:[ :{ :@ >:( :-)) :'‑( :'( :'‑) :') D‑': D:< D: D8 D; D= :‑O :O :‑o :o :-0 8‑0 >:O :-* :* :× ;‑) ;) *-) *) ;‑] ;] ;^) :‑, ;D :‑P :P X‑P x‑p :‑p :p :‑Þ :Þ :‑þ :þ :‑b :b d: =p >:P :‑/ :/ :‑. >:\ >:/ :\ =/ =\ :L =L :S :‑| :| :$ ://) ://3 :‑X :X :‑# :# :‑& :& O:‑) O:) 0:‑3 0:3 0:‑) 0:) 0;^) >:‑) >:) }:‑) }:) 3:‑) 3:) >;) >:3 >;3 |;‑) |‑O :‑J #‑) %‑) %) <:‑| ',:-| ',:-l :-| T_T @-) 
'''.strip().split(' ')

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
    return bool(re.findall("^0meaningful0special0character0sequence0id.+$", word_string))

##########################
# Misc. String Utilities #
##########################

def is_tagged_user_handle(word_string: str) -> bool:
    return bool(re.findall("^@.*$", word_string)) and " " not in word_string

TAGGED_USER_PLACEHOLDER = PLACEHOLDER_PREFIX+"0tagged0user"

def replace_tagged_users_with_placeholder_token(text_string: str) -> str:
    updated_text_string = ' '.join([TAGGED_USER_PLACEHOLDER if is_tagged_user_handle(word_string) else word_string for word_string in text_string.split(' ')])
    return updated_text_string

URL_PLACEHOLDER = PLACEHOLDER_PREFIX+"0url0link"
URL_REGEX_PATTERN = "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"

def replace_tagged_urls_with_placeholder_token(text_string: str) -> str:
    updated_text_string = text_string
    urls = re.findall(URL_REGEX_PATTERN, text_string)
    for url in urls:
        updated_text_string = updated_text_string.replace(url, URL_PLACEHOLDER)
    return updated_text_string

HASH_TAG_REGEX_PATTERN = "B(\#[a-zA-Z]+\b)"

def replace_hash_tags_with_placeholder_token(text_string: str) -> str:
    # @todo is it better to learn what the hash tags mean? Or to have a global token for the hash tag
    updated_text_string = text_string
    hash_tags = re.findall(HASH_TAG_REGEX_PATTERN, text_string)
    for hash_tag in hash_tags:
        updated_text_string = updated_text_string.replace(hash_tag, PLACEHOLDER_PREFIX+''.join([char for char in hash_tag if char in ALPHABETIC_CHARACTERS]))
    return updated_text_string

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
    normalized_text_string = replace_tagged_users_with_placeholder_token(normalized_text_string)
    normalized_text_string = replace_tagged_urls_with_placeholder_token(normalized_text_string)
    normalized_text_string = replace_hash_tags_with_placeholder_token(normalized_text_string)
    normalized_text_string = replace_meaningful_special_character_sequence_with_placeholder_token(normalized_text_string)
    # @todo expand contractions
    normalized_text_string = separate_punctuation(normalized_text_string)
    normalized_text_string = simplify_spaces(normalized_text_string)
    normalized_words = normalized_text_string.split(' ')
    normalized_words = possibly_dwim_unknown_words(normalized_words)
    normalized_words = map(lower_case_string, normalized_words)
    return normalized_words
