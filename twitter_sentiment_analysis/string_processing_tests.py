#!/usr/bin/python3

"""

Tests for String processsing utilities.

Owner : paul-tqh-nguyen

Created : 11/15/2019

File Name : string_processing_tests.py

File Organization:
* Imports
* Misc. Utilities
* Testing Utilities
* Tests

"""

###########
# Imports #
###########

import unittest
import tqdm
from contextlib import contextmanager
import csv
from datetime import datetime
from sentiment_analysis import TRAINING_DATA_LOCATION
from string_processing_utilities import unknown_word_worth_dwimming, normalized_words_from_text_string, PUNCTUATION_SET, timer

###################
# Misc. Utilities #
###################

@profile
def identity(args):
    return args

#####################
# Testing Utilities #
#####################

COMMONLY_USED_MISSING_WORD2VEC_WORDS = [
    # stop words
    'a', 'to', 'and', 'of',
]

@profile
def questionable_normalized_words_from_text_string(text_string: str) -> bool:
    normalized_words = normalized_words_from_text_string(text_string)
    unknown_words_worth_mentioning = normalized_words
    unknown_words_worth_mentioning = filter(lambda word: word not in COMMONLY_USED_MISSING_WORD2VEC_WORDS, unknown_words_worth_mentioning)
    unknown_words_worth_mentioning = filter(lambda word: unknown_word_worth_dwimming(word), unknown_words_worth_mentioning)
    return list(unknown_words_worth_mentioning)

#########
# Tests #
#########

LOGGING_FILE = "/home/pnguyen/Desktop/log_from_tests.txt"

def _logging_print(input_string='') -> None:
    with open(LOGGING_FILE, 'a') as f:
        f.write(input_string+'\n')
    print(input_string)
    return None

class testTextStringNormalizationViaData(unittest.TestCase):
    @profile
    def testTextStringNormalizationViaData(self):
        _logging_print()
        latest_failed_string = None
        with open(TRAINING_DATA_LOCATION, encoding='ISO-8859-1') as training_data_csv_file:
            training_data_csv_reader = csv.DictReader(training_data_csv_file, delimiter=',')
            log_progress = False
            possibly_tqdm = tqdm.tqdm if log_progress else identity
            for iteration_index, row_dict in possibly_tqdm(enumerate(training_data_csv_reader)):
                if iteration_index < 50169:
                    continue
                sentiment_text = row_dict['SentimentText']
                notes_worth_printing = []
                questionable_normalized_words_determination_timing_results = dict()
                def note_questionable_normalized_words_determination_timing_results(time):
                    questionable_normalized_words_determination_timing_results['total_time'] = time
                with timer(exitCallback=note_questionable_normalized_words_determination_timing_results):
                    questionable_normalized_words = questionable_normalized_words_from_text_string(sentiment_text)
                questionable_normalized_words_determination_time = questionable_normalized_words_determination_timing_results['total_time']
                max_tolerable_number_of_seconds_for_processing = 1.0 # @todo lower this
                if questionable_normalized_words_determination_time > max_tolerable_number_of_seconds_for_processing:
                    notes_worth_printing.append("Processing the following string took {questionable_normalized_words_determination_time} to process:\n{sentiment_text}\n".format(
                        questionable_normalized_words_determination_time=questionable_normalized_words_determination_time,
                        sentiment_text=sentiment_text))
                if len(questionable_normalized_words) != 0:
                    latest_failed_string = sentiment_text
                    notes_worth_printing.append("\nWe encountered these unhandled words: {questionable_normalized_words}".format(questionable_normalized_words=questionable_normalized_words))
                    notes_worth_printing.append("")
                if len(notes_worth_printing) != 0 or True:
                    _logging_print()
                    _logging_print("==============================================================================================")
                    _logging_print("Timestamp: {timestamp}".format(timestamp=datetime.now()))
                    _logging_print("Current Iteration: {iteration_index}".format(iteration_index=iteration_index))
                    _logging_print("Current Sentence Being Processed:\n{sentiment_text}\n".format(sentiment_text=sentiment_text))
                    for note in notes_worth_printing:
                        _logging_print(note)
                    _logging_print("==============================================================================================")
                    _logging_print()
        self.assertTrue(latest_failed_string is None, msg="We failed to process the following string (among possibly many): \n{bad_string}".format(bad_string=latest_failed_string))

@profile
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
