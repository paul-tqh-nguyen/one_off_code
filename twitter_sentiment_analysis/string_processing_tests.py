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
import tqdm
import csv
from datetime import datetime
from sentiment_analysis import RAW_TRAINING_DATA_LOCATION, RAW_TEST_DATA_LOCATION
from string_processing_utilities import unknown_word_worth_dwimming, normalized_words_from_text_string, PUNCTUATION_SET, timer
from unit_test_data import WORD_TO_ACCEPTABLE_CORRECTIONS_MAP
from misc_utilities import *

#####################
# Testing Utilities #
#####################

def questionable_normalized_words_from_text_string(text_string: str) -> bool:
    normalized_words = normalized_words_from_text_string(text_string)
    unknown_words_worth_mentioning = normalized_words
    unknown_words_worth_mentioning = filter(lambda word: unknown_word_worth_dwimming(word), unknown_words_worth_mentioning)
    return list(unknown_words_worth_mentioning)

#########
# Tests #
#########

class testTextStringNormalizationTestCases(unittest.TestCase):
    def testTextStringNormalizationTestCases(self):
        bad_results = dict()
        for word, acceptable_corrections in WORD_TO_ACCEPTABLE_CORRECTIONS_MAP.items():
            normalized_words = normalized_words_from_text_string(word)
            normalized_string = ' '.join(normalized_words)
            for normalized_word in normalized_words:
                if unknown_word_worth_dwimming(normalized_word):
                    bad_results[word] = acceptable_corrections
                    logging_print()
                    logging_print("{normalized_word} is not known.".format(normalized_word=normalized_word))
                    normalized_string = ' '.join(normalized_words)
            if not normalized_string in acceptable_corrections:
                bad_results[word] = acceptable_corrections
                logging_print()
                logging_print("{word} was expected to normalized into one of {acceptable_corrections} but was normalized to '{normalized_string}'".format(
                    word=word,
                    acceptable_corrections=acceptable_corrections,
                    normalized_string=normalized_string))
        total_number_of_cases = len(WORD_TO_ACCEPTABLE_CORRECTIONS_MAP)
        number_of_base_cases = len(bad_results)
        percent_of_bad_cases = 100*number_of_base_cases/total_number_of_cases
        logging_print("Percent of failure cases is {percent_of_bad_cases}%.".format(percent_of_bad_cases=percent_of_bad_cases))
        self.assertTrue(percent_of_bad_cases<10, msg="Percent of failure cases is too high at {percent_of_bad_cases}%.".format(percent_of_bad_cases=percent_of_bad_cases))

class testTextStringNormalizationViaTrainingData(unittest.TestCase):
    def testTextStringNormalizationViaTrainingData(self):
        logging_print()
        latest_failed_string = None
        for csv_file_location in (RAW_TRAINING_DATA_LOCATION, RAW_TEST_DATA_LOCATION):
            with open(csv_file_location, encoding='ISO-8859-1') as csv_file:
                training_data_csv_reader = csv.DictReader(csv_file, delimiter=',')
                log_progress = False
                possibly_tqdm = tqdm.tqdm if log_progress else identity
                for iteration_index, row_dict in possibly_tqdm(enumerate(training_data_csv_reader)):
                    sentiment_text = row_dict['SentimentText']
                    notes_worth_printing = []
                    questionable_normalized_words_determination_timing_results = dict()
                    def note_questionable_normalized_words_determination_timing_results(time):
                        questionable_normalized_words_determination_timing_results['total_time'] = time
                    with timer(exitCallback=note_questionable_normalized_words_determination_timing_results):
                        questionable_normalized_words = questionable_normalized_words_from_text_string(sentiment_text)
                    questionable_normalized_words_determination_time = questionable_normalized_words_determination_timing_results['total_time']
                    max_tolerable_number_of_seconds_for_processing = 1.0 
                    if questionable_normalized_words_determination_time > max_tolerable_number_of_seconds_for_processing:
                        notes_worth_printing.append("Processing the following string took {questionable_normalized_words_determination_time} to process:\n{sentiment_text}\n".format(
                            questionable_normalized_words_determination_time=questionable_normalized_words_determination_time,
                            sentiment_text=sentiment_text))
                    if len(questionable_normalized_words) != 0:
                        latest_failed_string = sentiment_text
                        notes_worth_printing.append("\nWe encountered these unhandled words: {questionable_normalized_words}".format(questionable_normalized_words=questionable_normalized_words))
                        notes_worth_printing.append("")
                    if len(notes_worth_printing) != 0:
                        logging_print()
                        logging_print("==============================================================================================")
                        logging_print("Timestamp: {timestamp}".format(timestamp=datetime.now()))
                        logging_print("Current Iteration: {iteration_index}".format(iteration_index=iteration_index))
                        logging_print("Current Sentence Being Processed:\n{sentiment_text}\n".format(sentiment_text=sentiment_text))
                        for note in notes_worth_printing:
                            logging_print(note)
                        logging_print("==============================================================================================")
                        logging_print()
        self.assertTrue(latest_failed_string is None, msg="We failed to process the following string (among possibly many): \n{bad_string}".format(bad_string=latest_failed_string))

def run_all_tests():
    print()
    print("Running our test suite.")
    print()
    loader = unittest.TestLoader()
    tests = [
        loader.loadTestsFromTestCase(testTextStringNormalizationTestCases),
        loader.loadTestsFromTestCase(testTextStringNormalizationViaTrainingData),
    ]
    suite = unittest.TestSuite(tests)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
    print()
    print("Test run complete.")
    print()

if __name__ == '__main__':
    run_all_tests()
