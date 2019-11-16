#!/usr/bin/python3

"""

Tests for String processsing utilities.

Owner : paul-tqh-nguyen

Created : 11/15/2019

File Name : string_processing_tests.py

File Organization:
* Imports
* Testing Utilities
* Tests

"""

###########
# Imports #
###########

import unittest
import csv
import re
import random
from word2vec_utilities import WORD2VEC_MODEL
from string_processing_utilities import word_string_resembles_meaningful_special_character_sequence_placeholder, normalized_words_from_text_string, PUNCTUATION_SET
from sentiment_analysis import TRAINING_DATA_LOCATION, TEST_DATA_LOCATION

#####################
# Testing Utilities #
#####################

COMMONLY_USED_MISSING_WORD2VEC_WORDS = [
    # stop words
    'a', 'to', 'and', 'of',
    # new words worth learning during training
    'blowd', 'yipee', 'momacita',
    # Proper Nouns @todo handle named entities
    'hifa', 'edeka', 'swartz', "semisonic", 'sytycd', 'christy','pavel','safina','eddings',
    # @todo to figure out
    'uplanders',
    # non-sense
    'leysh', 't9ar5',
    # words we don't care to learn
    'tda' # today
]

def string_corresponds_to_number(word_string: str) -> str:
    return bool(re.findall("^[0-9]+$", word_string))

def questionable_normalized_words_from_text_string(text_string: str) -> bool:
    normalized_words = normalized_words_from_text_string(text_string)
    unknown_words_worth_mentioning = filter(lambda word: word not in WORD2VEC_MODEL, normalized_words)
    unknown_words_worth_mentioning = filter(lambda word: word not in COMMONLY_USED_MISSING_WORD2VEC_WORDS, unknown_words_worth_mentioning)
    unknown_words_worth_mentioning = filter(lambda word: word not in PUNCTUATION_SET, unknown_words_worth_mentioning)
    unknown_words_worth_mentioning = filter(lambda word: not word_string_resembles_meaningful_special_character_sequence_placeholder(word), unknown_words_worth_mentioning)
    unknown_words_worth_mentioning = filter(lambda word: not string_corresponds_to_number(word), unknown_words_worth_mentioning)
    return list(unknown_words_worth_mentioning)

#########
# Tests #
#########

class testTextStringNormalizationViaData(unittest.TestCase):
    def testTextStringNormalizationViaData(self):
        csv_file_locations = [TRAINING_DATA_LOCATION, TEST_DATA_LOCATION]
        for csv_file_location in csv_file_locations:
            with open(csv_file_location, encoding='ISO-8859-1') as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter=',')
                row_dicts = list(csv_reader)
                #random.shuffle(row_dicts)
                failed_string_to_questionable_normalized_words_map = dict()
                for row_dict in row_dicts:
                    sentiment_text = row_dict['SentimentText']
                    questionable_normalized_words = questionable_normalized_words_from_text_string(sentiment_text)
                    if len(questionable_normalized_words)!=0:
                        failed_string_to_questionable_normalized_words_map[sentiment_text] = questionable_normalized_words
                self.assertTrue(len(failed_string_to_questionable_normalized_words_map)==0,
                                msg="We failed to process the following: \n{bad_pairs_printout}".format(
                                    bad_pairs_printout = ''.join(['"{0}" : {1}'.format(sentiment_text, questionable_normalized_words)
                                                                  for sentiment_text, questionable_normalized_words in failed_string_to_questionable_normalized_words_map.items()])))

def run_all_tests():
    print()
    print("Running our test suite.")
    print()
    loader = unittest.TestLoader()
    tests = [
        loader.loadTestsFromTestCase(testTextStringNormalizationViaData),
    ]
    suite = unittest.TestSuite(tests)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
    print()
    print("Test run complete.")
    print()

if __name__ == '__main__':
    run_all_tests()
