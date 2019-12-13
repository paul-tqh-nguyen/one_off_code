#!/usr/bin/python3 -O

"""

String processing utilities for text processing.

Owner : paul-tqh-nguyen

Created : 11/15/2019

File Name : string_processing_utilities.py

File Organization:
* Imports
* Misc. Utilities
* Meaningful Character Sequence Utilities
* Shorthand with Special Characters & Contraction Expansion
* Unknown Word DWIMming Utilities
* Misc. String Utilities

"""

###########
# Imports #
###########

import string
import html
import re
import unicodedata
import time
import warnings
from itertools import chain, combinations
from contextlib import contextmanager
from functools import lru_cache
from typing import List, Tuple, Callable, Generator, Set
from word2vec_utilities import WORD2VEC_MODEL

###################
# Misc. Utilities #
###################

def implies(antecedent, consequent) -> bool:
    return not antecedent or consequent

# @todo use pervasively
def quiescently_replace_subsequence(old_subsequence: str, new_subsequence: str, text_string: str) -> str:
    updated_text_string = text_string
    for _ in text_string:
        if old_subsequence in updated_text_string:
            updated_text_string = updated_text_string.replace(old_subsequence, new_subsequence)
        else:
            break
    return updated_text_string

UNIQUE_BOGUS_RESULT_IDENTIFIER = (lambda x: x)

def uniq(iterator):
    previous = UNIQUE_BOGUS_RESULT_IDENTIFIER
    for value in iterator:
        if previous != value:
            yield value
            previous = value

def powerset(iterable):
    items = list(iterable)
    number_of_items = len(items)
    subset_iterable = chain.from_iterable(combinations(items, length) for length in range(1, number_of_items+1))
    return subset_iterable

@contextmanager
def timeout(time, functionToExecuteOnTimeout=None):
    """NB: This cannot be nested."""
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(time)
    try:
        yield
    except TimeoutError:
        if functionToExecuteOnTimeout is not None:
            functionToExecuteOnTimeout()
    finally:
        signal.signal(signal.SIGALRM, signal.SIG_IGN)

@contextmanager
def timer(section_name=None, exitCallback=None):
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    if exitCallback != None:
        exitCallback(elapsed_time)
    elif section_name:
        print('Execution of "{section_name}" took {elapsed_time} seconds.'.format(section_name=section_name, elapsed_time=elapsed_time))
    else:
        print('Execution took {elapsed_time} seconds.'.format(elapsed_time=elapsed_time))

def false(*args, **kwargs):
    return False

PLACEHOLDER_PREFIX = "place0holder0token0with0id"

def word_string_resembles_meaningful_special_character_sequence_placeholder(word_string: str) -> bool:
    return bool(re.findall(r"^"+PLACEHOLDER_PREFIX+r".+$", word_string))

def unknown_word_worth_dwimming(word_string: str) -> bool:
    return not word_string.isnumeric() and \
        word_string not in PUNCTUATION_SET and \
        not word_string_resembles_meaningful_special_character_sequence_placeholder(word_string) and \
        word_string not in WORD2VEC_MODEL

def _correct_words_via_subsequence_substitutions(text_string: str, old_subsequence_0: str, new_subsequence_0: str, word_exception_checker: Callable[[str], bool]=false) -> str:
    old_subsequence = old_subsequence_0.lower()
    new_subsequence = new_subsequence_0.lower()
    updated_text_string = text_string
    word_match_iterator = re.finditer(r"\b\w+\b", text_string)
    for word_match in word_match_iterator:
        word_string = word_match.group()
        if not word_exception_checker(word_string):
            if unknown_word_worth_dwimming(word_string):
                word_string_normalized = word_string.lower()
                old_subsequence_match_iterator = re.finditer(old_subsequence, word_string)
                old_subsequence_span_specifications = list(map(lambda match: match.span(), old_subsequence_match_iterator))
                assert old_subsequence_span_specifications == sorted(old_subsequence_span_specifications, key=lambda pair: pair[0])
                corrected_words = []
                if len(old_subsequence_span_specifications) < 10:
                    for old_subsequence_span_specifications_subset in powerset(old_subsequence_span_specifications):
                        corrected_word = word_string
                        character_offset_count_per_span = len(new_subsequence)-len(old_subsequence)
                        for span_index, (start_index, end_index) in enumerate(old_subsequence_span_specifications_subset):
                            offset = character_offset_count_per_span*span_index
                            corrected_word = corrected_word[:start_index+offset]+new_subsequence+corrected_word[end_index+offset:]
                        corrected_words.append(corrected_word)
                else:
                    warnings.warn("Got an explosive number of correction to try when replacing '{old_subsequence}' with '{new_subsequence}' in '{word_string}'.".format(old_subsequence=old_subsequence, new_subsequence=new_subsequence, word_string=word_string))
                    corrected_word = word_string_normalized.replace(old_subsequence, new_subsequence)
                    corrected_words.append(corrected_word)
                for corrected_word in corrected_words:
                    if corrected_word in WORD2VEC_MODEL:
                        updated_text_string = re.sub(r"\b"+word_string+r"\b", corrected_word, updated_text_string, 1)
                        break
    return updated_text_string

ALPHABETIC_CHARACTERS = set('abcdefghijklmnopqrstuvwxyz') # @todo do we use this?
VOWELS = {'a','e','i','o','u'}
NUMERIC_CHARACTERS = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

def _word_splits(word_string: str) -> List[Tuple[str,str]]:
    return [(word_string[:i], word_string[i:]) for i in range(len(word_string) + 1)]

def _words_with_distance_1(word_string: str, character_set: Set[str],
                           allow_deletes: bool=True,
                           allow_transposes: bool=True,
                           allow_replacement: bool=True,
                           allow_inserts: bool=True) -> Generator[str, None, None]:
    '''Can yield duplicates.'''
    splits = _word_splits(word_string)
    inserts = (L + c + R for L, R in splits for c in character_set) if allow_inserts else []
    transposes = (L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1) if allow_transposes else []
    replaces = (L + c + R[1:] for L, R in splits if R for c in character_set) if allow_replacement else []
    deletes = (L + R[1:] for L, R in splits if R) if allow_deletes else []
    return chain(deletes, transposes, replaces, inserts)

def _search_words_with_distance_n(word_string: str, character_set: Set[str], n: int, word_validity_checkers_sorted_by_importance: List[Callable[[str], bool]],
                                  allow_deletes: bool=True,
                                  allow_transposes: bool=True,
                                  allow_replacement: bool=True,
                                  allow_inserts: bool=True) -> str:
    '''
    print("_search_words_with_distance_n")
    print("word_string {}".format(word_string))
    print("allow_deletes {}".format(allow_deletes))
    print("allow_transposes {}".format(allow_transposes))
    print("allow_replacement {}".format(allow_replacement))
    print("allow_inserts {}".format(allow_inserts))
    #'''
    for word_validity_checker in word_validity_checkers_sorted_by_importance:
        if word_validity_checker(word_string):
            return word_string
    words_yielded_from_most_recently_completed_n = set([word_string])
    for i in range(n):
        words_yielded_from_current_n = set()
        for word_yielded_from_most_recently_completed_n in words_yielded_from_most_recently_completed_n:
            words_1_distance_from_word_yielded_from_most_recently_completed_n = set(_words_with_distance_1(word_yielded_from_most_recently_completed_n, character_set,
                                                                                                           allow_deletes,
                                                                                                           allow_transposes,
                                                                                                           allow_replacement,
                                                                                                           allow_inserts))
            words_yielded_from_current_n.update(words_1_distance_from_word_yielded_from_most_recently_completed_n)
            for word_validity_checker in word_validity_checkers_sorted_by_importance:
                for word_yielded_from_current_n in words_1_distance_from_word_yielded_from_most_recently_completed_n:
                    #print("word_yielded_from_current_n {}".format(word_yielded_from_current_n))
                    if word_validity_checker(word_yielded_from_current_n):
                        '''
                        print("i {}".format(i))
                        print("word_yielded_from_current_n {}".format(word_yielded_from_current_n))
                        #'''
                        return word_yielded_from_current_n
        words_yielded_from_most_recently_completed_n = words_yielded_from_current_n
    return word_string

def _word_string_minimized_for_search(word_string: str) -> str:
    word_string_minimized_for_search = ''
    most_recent_duplicated_vowel = UNIQUE_BOGUS_RESULT_IDENTIFIER
    previous_character = UNIQUE_BOGUS_RESULT_IDENTIFIER
    vowels_likely_to_have_duplicates = {'e','o'}
    for character in word_string:
        if character in NUMERIC_CHARACTERS or previous_character != character:
            word_string_minimized_for_search += character
            previous_character = character
            most_recent_duplicated_vowel = UNIQUE_BOGUS_RESULT_IDENTIFIER
        elif character in VOWELS and most_recent_duplicated_vowel != character:
            word_string_minimized_for_search += character
            previous_character = character
            most_recent_duplicated_vowel = character
    return word_string_minimized_for_search

def _possibly_correct_word_via_edit_distance_search_using_strictly_vowel_insertion_or_transposes(word_string: str) -> str:
    relevant_characters = VOWELS
    word_string_without_consecutive_duplicate_characters = _remove_consecutive_duplciate_characters(word_string)
    word_validity_checkers_sorted_by_importance = [
        lambda candidate_word: candidate_word in WORD2VEC_MODEL and sorted(word_string.lower()) != sorted(candidate_word.lower())
    ]
    word_string_minimized_for_search = _word_string_minimized_for_search(word_string)
    #print("_possibly_correct_word_via_edit_distance_search_using_strictly_vowel_insertion_or_transposes")
    corrected_word = _search_words_with_distance_n(word_string_minimized_for_search, relevant_characters, 2, word_validity_checkers_sorted_by_importance,
                                                   allow_deletes=False, allow_replacement=False)
    if not corrected_word in WORD2VEC_MODEL:
        corrected_word = word_string
    return corrected_word

def _possibly_correct_word_via_edit_distance_search_using_no_new_characters(word_string: str, edit_distance: int) -> str:
    relevant_characters = set(word_string)
    word_string_without_consecutive_duplicate_characters = _remove_consecutive_duplciate_characters(word_string)
    word_validity_checkers_sorted_by_importance = [
        lambda candidate_word: candidate_word in WORD2VEC_MODEL and _remove_consecutive_duplciate_characters(candidate_word) == word_string_without_consecutive_duplicate_characters,
        lambda candidate_word: candidate_word in WORD2VEC_MODEL and set(candidate_word) == relevant_characters,
        lambda candidate_word: candidate_word in WORD2VEC_MODEL and set(candidate_word).issubset(relevant_characters),
    ]
    word_string_minimized_for_search = _word_string_minimized_for_search(word_string)
    '''
    print("_possibly_correct_word_via_edit_distance_search_using_no_new_characters")
    print("word_string {}".format(word_string))
    #'''
    corrected_word = _search_words_with_distance_n(word_string_minimized_for_search, relevant_characters, edit_distance, word_validity_checkers_sorted_by_importance)
    if not corrected_word in WORD2VEC_MODEL:
        corrected_word = word_string
    return corrected_word

def _correct_words_via_suffix_substitutions(text_string: str, old_suffix: str, new_suffix: str, word_exception_checker: Callable[[str], bool]=false) -> str:
    updated_text_string = text_string
    word_match_iterator = re.finditer(r"\b\w+\b", text_string)
    old_suffix_len = len(old_suffix)
    for word_match in word_match_iterator:
        word_string = word_match.group()
        word_is_an_exception_and_should_not_be_corrected = word_exception_checker(word_string)
        if not word_is_an_exception_and_should_not_be_corrected:
            if unknown_word_worth_dwimming(word_string):
                word_string_normalized = word_string.lower()
                if re.match(r'^\w+'+re.escape(old_suffix)+r'$', word_string_normalized):
                    base_word = word_string_normalized[:-old_suffix_len]
                    corrected_word = base_word+new_suffix
                    if corrected_word in WORD2VEC_MODEL:
                        updated_text_string = re.sub(r"\b"+word_string+r"\b", corrected_word, updated_text_string, 1)
                    else:
                        '''
                        print("_correct_words_via_suffix_substitutions")
                        print("old_suffix {}".format(old_suffix))
                        print("new_suffix {}".format(new_suffix))
                        #'''
                        corrected_word = _possibly_correct_word_via_edit_distance_search_using_no_new_characters(corrected_word, 1)
                        if corrected_word in WORD2VEC_MODEL:
                            updated_text_string = re.sub(r"\b"+word_string+r"\b", corrected_word, updated_text_string, 1)
                            break
    return updated_text_string

def _remove_consecutive_duplciate_characters(text_string: str) -> str:
    final_text_string = ''.join(uniq(text_string))
    return final_text_string

###########################################
# Meaningful Character Sequence Utilities #
###########################################

EMOTICONS = '''
:‑D :D 8‑D 8D x‑D X‑D =D =3 B^D :‑( :( :‑c :c :‑< :< :‑[ :[ :-|| >:[ :{ :@ >:( :-)) :'‑( :'( :'‑) :') D‑': D:< D: D8 D; D= :‑O :O :‑o :o :-0 8‑0 >:O :-* :* :× ;‑) ;) *-) *) ;‑] ;] ;^) :‑, ;D :‑P :P X‑P x‑p :‑p :p :‑Þ :Þ :‑þ :þ :‑b :b d: =p >:P :‑/ :/ :‑. >:\ >:/ :\ =/ =\ :L =L :S :‑| :| :$ ://) ://3 :‑X :X :‑# :# :‑& :& O:‑) O:) 0:‑3 0:3 0:‑) 0:) 0;^) >:‑) >:) }:‑) }:) 3:‑) 3:) >;) >:3 >;3 |;‑) |‑O :‑J #‑) %‑) %) <:‑| ',:-| ',:-l :-| T_T @-) 
'''.strip().split(' ')

EMOTICON_TO_PLACEHOLDER_MAP = {meaningful_special_character_sequence : PLACEHOLDER_PREFIX+'0emoticon'+str(index) \
                                for index, meaningful_special_character_sequence in \
                                enumerate(EMOTICONS)}

ELIPSIS_PLACEHOLDER = PLACEHOLDER_PREFIX+'0elipsis'

def replace_meaningful_special_character_sequence_with_placeholder_token(text_string: str) -> str:
    text_string_with_replacements = text_string
    text_string_with_replacements = simplify_elipsis_sequences(text_string_with_replacements)
    for emoticon, placeholder in EMOTICON_TO_PLACEHOLDER_MAP.items():
        text_string_with_replacements = re.sub(r"\b"+re.escape(emoticon)+r"\b", ' '+placeholder+' ', text_string_with_replacements, 0)
    text_string_with_replacements = text_string_with_replacements.replace('...', ' '+ELIPSIS_PLACEHOLDER+' ')
    return text_string_with_replacements

#############################################################
# Shorthand with Special Characters & Contraction Expansion #
#############################################################

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
    "you've": "you have",
}

CONTRACTION_EXPANSION_PAIRS_SORTED_BIGGEST_FIRST = sorted(CONTRACTION_EXPANSION_MAP.items(), key=lambda x: len(x[0]), reverse=True)

SHORTHAND_WITH_SPECIAL_CHARACTERS_EXPANSION_MAP = {
    "w/": "with",
    "w/o": "without",
}

SHORTHAND_WITH_SPECIAL_CHARACTERS_EXPANSION_PAIRS_SORTED_BIGGEST_FIRST = sorted(SHORTHAND_WITH_SPECIAL_CHARACTERS_EXPANSION_MAP.items(), key=lambda x: len(x[0]), reverse=True)

def expand_contractions_and_shorthand_words_with_special_characters(text_string: str) -> str:
    updated_text_string = text_string
    for contraction, expansion in CONTRACTION_EXPANSION_PAIRS_SORTED_BIGGEST_FIRST:
        updated_text_string = re.sub(r"\b"+contraction+r"\b", expansion, updated_text_string, 0, re.IGNORECASE)
        updated_text_string = re.sub(r"\b"+contraction.replace("'", "")+r"\b", expansion, updated_text_string, 0, re.IGNORECASE)
    for shorthand, expansion in SHORTHAND_WITH_SPECIAL_CHARACTERS_EXPANSION_PAIRS_SORTED_BIGGEST_FIRST:
        updated_text_string = ' '.join([expansion if word.lower() == shorthand else word for word in updated_text_string.split()])
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

def ugh_star_expand(text_string: str) -> str:
    expanded_text_string = text_string
    word_replacement = "ugh"
    assert word_replacement in WORD2VEC_MODEL
    expanded_text_string = re.sub(r"\bugh\w+\b", word_replacement, expanded_text_string, 0, re.IGNORECASE)
    return expanded_text_string

def mhm_expand(text_string: str) -> str:
    expanded_text_string = text_string
    word_replacement = "sure"
    assert word_replacement in WORD2VEC_MODEL
    expanded_text_string = re.sub(r"\bm+h+m+\w+\b", word_replacement, expanded_text_string, 0, re.IGNORECASE)
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


def word_number_concatenation_expand(text_string: str) -> str:
    expanded_text_string = text_string
    matches = re.findall(r"\b[a-z]+[0-9]+\b", text_string, re.IGNORECASE)
    for match in matches:
        if unknown_word_worth_dwimming(match):
            numeric_half_matches = re.findall(r"[0-9]+", match)
            assert len(numeric_half_matches) == 1
            numeric_half_match = numeric_half_matches[0]
            alphabetic_half = match.replace(numeric_half_match, "")
            replacement = alphabetic_half+' '+numeric_half_match
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
            min_first_word_length = 4
            min_second_word_length = 3
            if word_length >= min_first_word_length+min_second_word_length:
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
    acceptable_prefixes = "mw", "mu", "muw", "h", "g", "b", "bw", "ahah", "he", 
    prefix_starts_word_string_test = lambda acceptable_starting_string: _starts_with(word_string, acceptable_starting_string)
    relevant_prefixes = list(filter(prefix_starts_word_string_test, acceptable_prefixes))
    if len(relevant_prefixes) != 0:
        relevant_prefix = max(relevant_prefixes, key=len)
        relevant_prefix_len = len(relevant_prefix)
        remaining_word_string = word_string[relevant_prefix_len:]
        non_laugh_characters = re.sub(r"[hgae]+", '', remaining_word_string, 0, re.IGNORECASE)
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

def yay_star_expand(text_string: str) -> str:
    text_string_with_replacements = text_string
    word_match_iterator = re.finditer(r"\b\w+\b", text_string, re.IGNORECASE)
    corrected_word = 'yay'
    assert corrected_word in WORD2VEC_MODEL
    for word_match in word_match_iterator:
        word = word_match.group()
        if unknown_word_worth_dwimming(word):
            dwimmed_word = word
            dwimmed_word = quiescently_replace_subsequence('aa', 'a', dwimmed_word)
            dwimmed_word = quiescently_replace_subsequence('yy', 'y', dwimmed_word)
            if re.findall('^yay.*$',dwimmed_word, re.IGNORECASE):
                text_string_with_replacements = re.sub(r"\b"+word+r"\b", corrected_word, text_string_with_replacements, 0, re.IGNORECASE)
    return text_string_with_replacements

def correct_words_via_edit_distance_search_using_no_new_characters_expand(text_string: str) -> str:
    word_strings = text_string.split(' ')
    possibly_corrected_word_strings = map(lambda word_string: word_string if not unknown_word_worth_dwimming(word_string) else
                                          _possibly_correct_word_via_edit_distance_search_using_no_new_characters(word_string, 2), word_strings)
    updated_text_string = ' '.join(possibly_corrected_word_strings)
    return updated_text_string

def correct_words_via_edit_distance_search_using_strictly_vowel_insertion_or_transposes(text_string: str) -> str:
    word_strings = text_string.split(' ')
    possibly_corrected_word_strings = map(lambda word_string: word_string if not unknown_word_worth_dwimming(word_string) else
                                          _possibly_correct_word_via_edit_distance_search_using_strictly_vowel_insertion_or_transposes(word_string), word_strings)
    updated_text_string = ' '.join(possibly_corrected_word_strings)
    return updated_text_string

SLANG_WORD_DICTIONARY = {
    "bday" : "birthday",
    "beeyatch" : "biatch",
    "fu2" : "fuck you too",
    "hungy" : "hungry",
    "hvnt" : "have not",
    "idunno" : "I do not know",
    "ilym" : "I love you more",
    "ilysfm" : "I love you so fucking much",
    "ily2" : "I love you too",
    "inorite" : "I know right",
    "lmbo" : "lmao",
    "lul" : "lol",
    "luvly" : "lovely",
    "muah" : "me",
    "nvmd" : "nevermind",
    "rlly" : "really",
    "smthg" : "something",
    "sowwy" : "sorry",
    "woots" : "woot",
    "yestie" : "yesterday",
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

def ne_any_slang_correction_expand(text_string: str) -> bool:
    return _correct_words_via_subsequence_substitutions(text_string, 'ne', 'any')

def q_g_slang_correction_expand(text_string: str) -> bool:
    return _correct_words_via_subsequence_substitutions(text_string, 'q', 'g')

def f_ph_slang_correction_expand(text_string: str) -> bool:
    return _correct_words_via_subsequence_substitutions(text_string, 'f', 'ph')

def ee_y_slang_correction_expand(text_string: str) -> bool:
    corrected_text_string = text_string
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, 'ee', 'y')
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, 'ie', 'y')
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, 'ey', 'y')
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, 'i', 'y')
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, 'y', 'ee')
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, 'y', 'ie')
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, 'y', 'ey')
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, 'y', 'i')
    return corrected_text_string

def z_s_slang_correction_expand(text_string: str) -> bool:
    corrected_text_string = text_string
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, 's', 'z')
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, 'z', 's')
    return corrected_text_string

def ce_se_slang_correction_expand(text_string: str) -> bool:
    corrected_text_string = text_string
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, 'ce', 'se')
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, 'se', 'ce')
    return corrected_text_string

def f_th_slang_correction_expand(text_string: str) -> bool:
    return _correct_words_via_subsequence_substitutions(text_string, 'f', 'th')

def d_t_slang_correction_expand(text_string: str) -> bool:
    return _correct_words_via_subsequence_substitutions(text_string, 'd', 't')

def leet_slang_correction_expand(text_string: str) -> bool:
    corrected_text_string = text_string
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, '0', 'o')
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, '1', 'i')
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, '1', 'l')
    return corrected_text_string

def eight_ate_slang_correction_expand(text_string: str) -> bool:
    corrected_text_string = text_string
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, '8', 'at')
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, '8', 'ate')
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, '8', 'eight')
    return corrected_text_string

def oo_u_slang_correction_expand(text_string: str) -> bool:
    corrected_text_string = text_string
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, 'oo', 'u')
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, 'u', 'oo')
    return corrected_text_string

def ya_you_slang_correction_expand(text_string: str) -> bool:
    corrected_text_string = text_string
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, 'ya', 'you')
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, 'ya', 'your')
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, 'cha', 'you')
    corrected_text_string = _correct_words_via_subsequence_substitutions(corrected_text_string, 'cha', 'your')
    return corrected_text_string

def irregular_past_tense_dwimming_expand(text_string: str) -> bool:
    updated_text_string = text_string
    updated_text_string = _correct_words_via_suffix_substitutions(updated_text_string, 't', 'ed',
                                                                  lambda word_string: len(word_string)>2 and word_string[-1].lower() == 't' and word_string[-2].lower() in VOWELS.union({'t'}))
    updated_text_string = _correct_words_via_suffix_substitutions(updated_text_string, 'ed', 't')
    return updated_text_string

def ies_suffix_expand(text_string: str) -> bool:
    updated_text_string = text_string
    updated_text_string = _correct_words_via_suffix_substitutions(updated_text_string, 'ies', 's')
    return updated_text_string

def a_er_suffix_expand(text_string: str) -> bool:
    updated_text_string = text_string
    updated_text_string = _correct_words_via_suffix_substitutions(updated_text_string, 'ah', 'er')
    updated_text_string = _correct_words_via_suffix_substitutions(updated_text_string, 'a', 'er')
    return updated_text_string

def r_er_suffix_expand(text_string: str) -> bool:
    updated_text_string = text_string
    updated_text_string = _correct_words_via_suffix_substitutions(updated_text_string, 'r', 'er')
    return updated_text_string

def y_suffix_removal_expand(text_string: str) -> bool:
    updated_text_string = text_string
    updated_text_string = _correct_words_via_suffix_substitutions(updated_text_string, 'y', '')
    return updated_text_string

def g_dropping_suffix_expand(text_string: str) -> bool:
    updated_text_string = text_string
    updated_text_string = _correct_words_via_suffix_substitutions(updated_text_string, 'in', 'ing')
    updated_text_string = _correct_words_via_suffix_substitutions(updated_text_string, 'n', 'ing')
    return updated_text_string

DWIMMING_EXPAND_FUNCTIONS = [
    # Dictionary Based Correction
    slang_word_expand,
    
    # Missing Vowel Injection
    correct_words_via_edit_distance_search_using_strictly_vowel_insertion_or_transposes,
    
    # Simple Duplicate Letter Corection
    omg_star_expand,
    ugh_star_expand,
    mhm_expand,
    aw_star_expand,
    laughing_expand,
    yay_star_expand,
    
    # Subsequence Correction
    q_g_slang_correction_expand,
    f_ph_slang_correction_expand,
    ee_y_slang_correction_expand,
    z_s_slang_correction_expand,
    ce_se_slang_correction_expand,
    f_th_slang_correction_expand,
    d_t_slang_correction_expand,
    leet_slang_correction_expand,
    eight_ate_slang_correction_expand,
    oo_u_slang_correction_expand,
    ya_you_slang_correction_expand,
    our_or_british_sland_correction_expand,
    ne_any_slang_correction_expand,
    
    # Suffix Correction
    irregular_past_tense_dwimming_expand,
    ies_suffix_expand,
    a_er_suffix_expand,
    y_suffix_removal_expand,
    r_er_suffix_expand,
    g_dropping_suffix_expand,
    
    # Word Splitting Correction
    number_word_concatenation_expand,
    word_number_concatenation_expand,
    two_word_concatenation_expand,

    # Spell Correction via Character Edits
    correct_words_via_edit_distance_search_using_no_new_characters_expand,
]

def perform_single_pass_to_dwim_unknown_words(text_string: str) -> str:
    updated_text_string = text_string
    for expand_function in DWIMMING_EXPAND_FUNCTIONS:
        expanded_result = expand_function(updated_text_string)
        '''
        print()
        print("expand_function {}".format(expand_function))
        print("updated_text_string {}".format(updated_text_string))
        print("expanded_result {}".format(expanded_result))
        #'''
        if expanded_result != updated_text_string:
            before = updated_text_string.split()
            after = expanded_result.split()
            start_diff_index = None
            for i,e in enumerate(before):
                if i<len(after):
                    if e != after[i]:
                        start_diff_index = i
                        break
            end_diff_index = None
            for i in range(len(before)):
                index_from_end = -(i+1)
                if abs(index_from_end)<=len(after):
                    if before[index_from_end] != after[index_from_end]:
                        end_diff_index = index_from_end
                        break
            inclusive_end_diff_index = None if end_diff_index == -1 else end_diff_index+1
            before_string = ' '.join(before[start_diff_index:inclusive_end_diff_index])
            after_string = ' '.join(after[start_diff_index:inclusive_end_diff_index])
            print()
            print(text_string)
            print("""'{before_string}': ['{after_string}'],""".format(before_string=before_string, after_string=after_string))
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
    assert implies(text_string, premature_exit), "Unknown word DWIMming did not process until quiescence as we processed \"{text_string}\" into \"{current_text_string}\".".format(
        text_string=text_string, 
        current_text_string=current_text_string)
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
        updated_text_string = updated_text_string.replace(url, ' '+URL_PLACEHOLDER+' ')
    return updated_text_string

HASH_TAG_REGEX_PATTERN = "#\w+"
HASH_TAG_PLACEHOLDER = PLACEHOLDER_PREFIX+'0hash0tag'

def replace_hash_tags_with_placeholder_token(text_string: str) -> str:
    updated_text_string = text_string
    hash_tags = re.findall(HASH_TAG_REGEX_PATTERN, text_string)
    for hash_tag in hash_tags:
        updated_text_string = updated_text_string.replace(hash_tag, ' '+HASH_TAG_PLACEHOLDER+' ')
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

#@profile
def normalized_words_from_text_string(text_string: str) -> List[str]:
    normalized_text_string = text_string
    normalized_text_string = html.unescape(normalized_text_string)
    normalized_text_string = replace_tagged_users_with_placeholder_token(normalized_text_string)
    normalized_text_string = replace_urls_with_placeholder_token(normalized_text_string)
    normalized_text_string = replace_hash_tags_with_placeholder_token(normalized_text_string)
    normalized_text_string = replace_exotic_characters(normalized_text_string)
    normalized_text_string = replace_meaningful_special_character_sequence_with_placeholder_token(normalized_text_string)
    normalized_text_string = expand_contractions_and_shorthand_words_with_special_characters(normalized_text_string)
    normalized_text_string = separate_punctuation(normalized_text_string)
    normalized_text_string = possibly_dwim_unknown_words(normalized_text_string)
    normalized_text_string = lower_case_unknown_words(normalized_text_string)
    normalized_words = normalized_text_string.split(' ')
    return normalized_words

def main():
    print("This module contains string normalization utilities for sentiment analysis on Twitter data.")

if __name__ == '__main__':
    main()
