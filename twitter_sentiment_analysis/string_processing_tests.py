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

#@profile
def identity(args):
    return args

#####################
# Testing Utilities #
#####################

#@profile
def questionable_normalized_words_from_text_string(text_string: str) -> bool:
    normalized_words = normalized_words_from_text_string(text_string)
    unknown_words_worth_mentioning = normalized_words
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

class testTextStringNormalizationTestCases(unittest.TestCase):
    def testTextStringNormalizationTestCases(self):
        word_to_acceptable_corrections_map = {
            'yayayayayayayay': ['yay'],
            'yesssssssssssssssssssss': ['yes','yesss'],
            'woooooooooooooooooooooo': ['woo','woooooooooo'],
            'rlly': ['rly', 'really', 'rolly',],
            'ddnt': ['did not', 'didnt', 'dnt'],
            'hihihi': ['hi', 'hihi'],
            'yaaaaa': ['ya', 'yes', 'yaa'],
            'onnnnnnn': ['on', 'onn'],
            'aanswwe': ['answe', 'answer'],
            'vox': ['vox', 'ing'],
            'loooool': ['lol', 'lool'],
            'iiiiittt': ['it', 'iit'],
            'wrkshp': ['workshop', 'workship'],
            'heheeheh': ['hehehehe', 'haha'],
            'hotellllll': ['hotell', 'hotel'],
            'yeyyyyyyyy': ['yey'],
            'thnkz': ['thanks', 'thnks'],
            'thnkx': ['thanks', 'thnk', 'thnx'],
            'tomrowo': ['tomorrow', 'tomorow', 'tomrrow'],
            'meannnnn': ['mean'],
            'youuuu': ['you', 'youu'],
            'ayoyo': ['ayoo'],
            'tonighttttt': ['tonight'],
            'looooooooooooooooove': ['loooooove', 'looove'],
            'bdayyyyyy': ['bday'],
            'sweeeeeeeet': ['sweeeet', 'swete', 'sweet'],
            'arrghh': ['arrgh', 'argh'],
            'heyyyyyy': ['hey'],
            'wompppppp': ['womp'],
            'yezzzz': ['yessss', 'yez'],
            'hinnnnt': ['hint'],
            'yepyep': ['yep yep', 'yeye'],
            'zelwegger': ['zellwegger', 'zellweger'],
            'huhuhu': ['huhu'],
            'ooooppsss': ['ooops', 'ops', 'oops'],
            'isumthing': ['sumthing'],
            'drewww': ['drew'],
            'unforcenatly': ['unfortunatly'],
            'wehehehehe': ['wehe hehehe'],
            'annnddd': ['and'],
            'urrrghhhh': ['urgh'],
            'rref': ['ref'],
            'yeahhhhhh': ['yeah'],
            'loool': ['lol', 'lool'],
            'hiiiiiiiiiii': ['hi', 'hii'],
            'aghhh': ['agh'],
            'wrks': ['works', 'werks', 'warks'],
            'realllllly': ['realy'],
            'ppppppplease': ['please'],
            'wwwwwwhhatt': ['what'],
            'brkfast': ['breakfast'],
            'eeeeeekkkkkkkk': ['ek', 'eek'],
            'sooooooooooooooooooooooooooooooooooooo': ['so','soo'],
            'againnnnnnnnnnnn': ['again'],
            'yaywyyyyyy': ['yay'],
            'nawwww': ['naw'],
            'iloveyou': ['ilove you'],
            'loooooooooove': ['love', 'looove'],
            'amazinggggggggggg': ['amazing'],
            'luhhh': ['luh'],
            'ppffttt': ['pft'],
            'yooooo': ['yo', 'yoo'],
            'tehehehe': ['hehehehe'],
            'outtttt': ['out'],
            'ppppppfffffftttttttt': ['pft'],
            'suuuxxx': ['sux'],
            'bestiee': ['bestie'],
            'eddipuss': ['oedipus'],
            'Omgaga': ['omg'],
            '30mins': ['30 mins'],
            'Juuuuuuuuuuuuuuuuussssst': ['Just', 'Juuust'],
            'wompppp wompp': ['womp womp'],
            'eveer': ['ever'],
            '120gb': ['120 gb'],
            'Tigersfan': ['Tigers fan'],
            'Aww': ['aw'],
            'Winona4ever': ['Winona 4ever'],
            'whyy': ['why'],
            'waahhh': ['waah'],
            'mmmaybe': ['maybe'],
            'timessss': ['times'],
            'Goodniqht': ['Goodnight'],
            'wantt': ['want'],
            'celulite': ['cellulite'],
            'Uprizing': ['Uprising'],
            'Grac': ['Grace', 'Garc'],
            'rooooooooomies': ['roomies'],
            'LMBO': ['lmao'],
            'sowwy': ['sorry'],
            'fuuuuck': ['fuuck', 'fuck'],
            'fuuck': ['fuck'],
            'Awwwwwh': ['Awh', 'aw'],
            'Awh': ['aw'],
            'mowin': ['mowing'],
            'Aiigght': ['Aiight'],
            '18th': ['18 th'],
            'lazzzzyyyy': ['lazy'],
            'Huhuhu': ['Huhu'],
            'NOOOOOOOOOO': ['NO'],
            'yippeeeee': ['yipee'],
            'UGHHHHHHHHHHHHHHHHHHHHHHHHHHHHH': ['UGH'],
            'realy2': ['realy 2'],
            'GRRRRRRRRREAT': ['GREAT'],
            'Kateyy': ['Katey'],
            'hahahahahahahahaha': ['haha'],
            'TWATLIGHT': ['TWAT LIGHT'],
            'woooohooo': ['woohoo'],
            'WOOOOOOOOOOOOOOOOOOOOOO': ['WOO', 'WO'],
            'PrintChick': ['Print Chick'],
            'Headbangin': ['headbanging'],
            'tweetaholics': ['tweet aholics'],
            'softsynths': ['soft synths'],
            'wellll': ['wel'],
            'wazzza': ['waza'],
            'yoooou': ['yoou', 'you', 'youu'],
            'yoou': ['you', 'yuu'],
            'Feeeling': ['Feeling'],
            'foreverr': ['forever'],
            'sooooon': ['soon'],
            '400th': ['400 th'],
            '11th': ['11 th'],
            'fucccckkkkkkkkkk': ['fuck'],
            'noseeyyy': ['nosey'],
            'yayyyyyyyy': ['yay'],
            'iiiiii': ['ii'],
            'pleeeeeeeeeeeeeeeeeeeeeeeeeease': ['pleeease'],
            'ppoooo': ['poo'],
            'lolllllll': ['lol'],
            'nebody': ['anybody'],
            'workkkkkkkkk': ['work'],
            'merrrrrrrr': ['mer'],
            'booom': ['boom'],
            'bestttttttttttttttttttt': ['best'],
            'swooooon': ['swoon'],
            'citysearchbelle': ['citysearch belle'],
            'gratissssssssssss': ['gratis'],
            'causee': ['causse'],
            'ugghhh': ['ugh'],
            'clarieey': ['clarie'],
            'eheheh': ['heheh'],
            'linderbug': ['linder bug'],
            'anyonefeeling': ['anyone feeling'],
            'goiing': ['gooing'],
            'Aiqht': ['Aight'],
            'arghgh': ['arghhh'],
            'bleehh': ['bleh'],
            'Fightiin': ['fighting'],
            'echolink': ['echo link'],
            '2at': ['2 at'],
            'loviie': ['lovie'],
            'easyer': ['easier'],
            'aaaaaaaaaaaaaaah': ['aah'],
            'narniaaaaaaaaa': ['narnia'],
            'chubbchubbs': ['chubb chubbs'],
            'birthdayyyyyyyy': ['birthday'],
            'aaaaaahhhhhhhh': ['aah'],
            'reallllllly': ['realy'],
            'hotttttt': ['hot'],
            
            # @todo get these working
            #'iidk': ['idk'],
            #'youget': ['you get'],
            #'blankeys': ['blankies'],
            #'nausia': ['nausea'],
            #'toooooootally': ['totaly'],
            #'wiit': ['with'],
            #'Waiit': ['Wait'],
            #'awhhe': ['aw'],
            #'Fawken': ['fucking'],
            #'Mmyeah': ['yeah'],
            #'enlish': ['english'],
            #'pleeeez': ['please'],
            #'Hospitol': ['Hospital'],
            #'greeting': ['greeting'],
            #'glich': ['glitch'],
            #'baithing': ['baithing'],
            #'dayjob': ['day job'],
            #'youtubee': ['youtube'],
            #'bbygrl': ['babygirl'],
            #'wutsupppppp': ['wut sup'],
            #'thanxs': ['thanx'],
            #'': [''],
        }
        bad_results = dict()
        for word, acceptable_corrections in word_to_acceptable_corrections_map.items():
            normalized_words = normalized_words_from_text_string(word)
            normalized_string = ' '.join(normalized_words)
            for normalized_word in normalized_words:
                if unknown_word_worth_dwimming(normalized_word):
                    bad_results[word] = acceptable_corrections
                    print()
                    print("{normalized_word} is not known.".format(normalized_word=normalized_word))
                    normalized_string = ' '.join(normalized_words)
            if not normalized_string in acceptable_corrections:
                bad_results[word] = acceptable_corrections
                print()
                print("{word} was expected to normalized into one of {acceptable_corrections} but was normalized to '{normalized_string}'".format(
                    word=word,
                    acceptable_corrections=acceptable_corrections,
                    normalized_string=normalized_string))
        for word, acceptable_corrections in bad_results.items():
            normalized_words = normalized_words_from_text_string(word)
            for normalized_word in normalized_words:
                self.assertFalse(unknown_word_worth_dwimming(normalized_word),msg="{normalized_word} is not known.".format(normalized_word=normalized_word))
                normalized_string = ' '.join(normalized_words)
                self.assertTrue(normalized_string in acceptable_corrections, msg="{word} was expected to normalized into one of {acceptable_corrections} but was normalized to '{normalized_string}'".format(
                    word=word,
                    acceptable_corrections=acceptable_corrections,
                    normalized_string=normalized_string))

class testTextStringNormalizationViaTrainingData(unittest.TestCase):
    #@profile
    def testTextStringNormalizationViaTrainingData(self):
        _logging_print()
        latest_failed_string = None
        with open(TRAINING_DATA_LOCATION, encoding='ISO-8859-1') as training_data_csv_file:
            training_data_csv_reader = csv.DictReader(training_data_csv_file, delimiter=',')
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
                max_tolerable_number_of_seconds_for_processing = 1.0 # @todo lower this
                if questionable_normalized_words_determination_time > max_tolerable_number_of_seconds_for_processing:
                    notes_worth_printing.append("Processing the following string took {questionable_normalized_words_determination_time} to process:\n{sentiment_text}\n".format(
                        questionable_normalized_words_determination_time=questionable_normalized_words_determination_time,
                        sentiment_text=sentiment_text))
                if len(questionable_normalized_words) != 0:
                    latest_failed_string = sentiment_text
                    notes_worth_printing.append("\nWe encountered these unhandled words: {questionable_normalized_words}".format(questionable_normalized_words=questionable_normalized_words))
                    notes_worth_printing.append("")
                if len(notes_worth_printing) != 0:
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

#@profile
def run_all_tests():
    print()
    print("Running our test suite.")
    print()
    loader = unittest.TestLoader()
    tests = [
        loader.loadTestsFromTestCase(testTextStringNormalizationTestCases),
        # loader.loadTestsFromTestCase(testTextStringNormalizationViaTrainingData),
    ]
    suite = unittest.TestSuite(tests)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
    print()
    print("Test run complete.")
    print()

if __name__ == '__main__':
    run_all_tests()
