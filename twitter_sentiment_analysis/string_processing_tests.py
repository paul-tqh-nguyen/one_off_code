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
import re
import tqdm
import time
from contextlib import contextmanager
from torch.utils import data
from word2vec_utilities import WORD2VEC_MODEL
from string_processing_utilities import unknown_word_worth_dwimming, normalized_words_from_text_string, PUNCTUATION_SET
from sentiment_analysis import determine_training_and_validation_datasets

###################
# Misc. Utilities #
###################

def identity(args):
    return args

@contextmanager
def timer(section_name=None, exitCallback=None):
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    if bool(exitCallback):
        exitCallback(elapsed_time)
    elif section_name:
        print('Execution of "{section_name}" took {elapsed_time} seconds.'.format(section_name=section_name, elapsed_time=elapsed_time))
    else:
        print('Execution took {elapsed_time} seconds.'.format(elapsed_time=elapsed_time))

#####################
# Testing Utilities #
#####################

COMMONLY_USED_MISSING_WORD2VEC_WORDS = [
    # Proper Nouns @todo handle named entities
    # 'hifa', 'edeka', 'swartz', "semisonic", 'sytycd', 'christy', 'pavel', 'safina', 'eddings', 'sannesias', 'winona', 'trae', 'tombre', 'rishabh', 'paramore', 'coldplay', 'neena', 'jlew', 
    # 'taylorrhicks', 'grac', 'seville', 'drexel', 'voltron', 'win7rc', 'lagerfeld', 'ahmier', 'zoro', 'rinitis', 'gongwer', 'aiden', 'jerrys', 'voltrons', 'hedo',
    # 'kimmy', 'nkotb', 'da70mm', 'minaj', 'f16', 'kallis', 'uat', 'pman', 'canaveral', 'imal', 'ohac', 'tirthankar', 'ankie','smf','dinara', 'garros','manee','anyer','burswood',
    # 'pixma', 'mx310','alistair','landin','ayonna','robsten','farrah', 'fawcett','kerrang','bizkit','rsl','paley','bjork','rb2','palmolive','supertramp','bcd', 'nodaji','trackle',
    # 'standfield','roberson','mgonewild','sadie','jbarsodmg','pbmall','farrah','aidan','avett','thames','horton','kokomo','brandice','bertolucci','lalalauren', 'weikert', 'hartley', 'cardona',
    # 'mmva','villarreal', 'leland', 'enigk','epsom','doodadoo','foxtell','bisante','tommi', 'oulu','farrah', 'fawcett','poirot','clopin','westwick','birtney', 'ciaraaaa','marah',
    # 'drunvalo','melchizedek','paraguay','irissa','stavros','shottas','karmada','btvs','lautner','bebot', 'lagerfeld', 'imodium',
    # stop words
    'a', 'to', 'and', 'of',
    # @todo figure out how to handle these
    'quh', 'uqhh', 'qotta', 'qet','aiqht', # replace G's with Q's
    'blowd', 'yipee', 'momacita', 'beautifulylost', 'scoobs','muah','huhuhu', 'tlk', 'mowin', 'loviie','steezy','sowwy', 'lmbo', 'really2','longlelong','st0mach', 'g0od','fallatio','nsfw',
    'trippn', 'ontd','quizzage','woots','neighbours', 'mhmm','1920x1080', '1280x1024','blankeys','sleepies','goodbassplayer','spilt','fonebook','boyfie','luvly','ilysfm','thelovelybones',
    'lalalalala','killah','ticketsandpassports', 'lufflies','kandie','bahahaha','doesnt','tmr','imy','loveeupeople','seeester','nuu','eckkky','littleboxofevil','tew','jajajaja','iming',
    'wasnt', 'honour','goodmorning','frisby','seeya','icant','fxcking','fcked','nitey','bouta','bahaa','hosptal','doesnt','blesh','asdfghjkl'
    # non-sense
    'leysh', 't9ar5','cmf','oilipo','drizzy','nbeem','hourst','twittz','nessa','trvs','kqs','ar47','t20',
    # words we don't care to learn
    'tda', # means "today"
    'huz', # random slang    
    'throwbie', # rare word
    'quorn', # rare word
    'mysapce', # typo
    'fu2', # not clear what this means in context
    'anoron', 'reedcourty', # part of bad link
    # spanish
    'menno', 'chootaa', 'chhoota', 'morrea', 'faltam', 'talvez', 'seja', 'nfase', 'aten','madres',
    'perfekta', # foreign language
]

def questionable_normalized_words_from_text_string(text_string: str) -> bool:
    normalized_words = normalized_words_from_text_string(text_string)
    unknown_words_worth_mentioning = normalized_words
    unknown_words_worth_mentioning = filter(lambda word: word not in COMMONLY_USED_MISSING_WORD2VEC_WORDS, unknown_words_worth_mentioning)
    unknown_words_worth_mentioning = filter(lambda word: unknown_word_worth_dwimming(word), unknown_words_worth_mentioning)
    return list(unknown_words_worth_mentioning)

def failed_string_to_questionable_normalized_words_map_repr(failed_string_to_questionable_normalized_words_map: dict) -> None:
    return '\n'+''.join(['"{0}" : {1}\n'.format(sentiment_text, questionable_normalized_words)
                         for sentiment_text, questionable_normalized_words in failed_string_to_questionable_normalized_words_map.items()])

#########
# Tests #
#########

class testTextStringNormalizationViaData(unittest.TestCase):
    def testTextStringNormalizationViaData(self):
        print()
        training_set, validation_set = determine_training_and_validation_datasets()
        training_generator = data.DataLoader(training_set, batch_size=1, shuffle=False)
        validation_generator = data.DataLoader(validation_set, batch_size=1, shuffle=False)
        failed_string_to_questionable_normalized_words_map = dict()
        log_progress = False
        possibly_tqdm = tqdm.tqdm if log_progress else identity
        for iteration_index, (input_batch, _) in possibly_tqdm(enumerate(training_generator)):
            notes_worth_printing = []
            assert len(input_batch)==1
            sentiment_text = input_batch[0]
            questionable_normalized_words_determination_timing_results = dict()
            def note_questionable_normalized_words_determination_timing_results(time):
                questionable_normalized_words_determination_timing_results['total_time'] = time
            with timer(exitCallback=note_questionable_normalized_words_determination_timing_results):
                questionable_normalized_words = questionable_normalized_words_from_text_string(sentiment_text)
            questionable_normalized_words_determination_time = questionable_normalized_words_determination_timing_results['total_time']
            max_tolerable_number_of_seconds_for_processing = 0.01
            if questionable_normalized_words_determination_time > max_tolerable_number_of_seconds_for_processing:
                notes_worth_printing.append("Processing the following string took {questionable_normalized_words_determination_time} to process:\n{sentiment_text}\n".format(
                    questionable_normalized_words_determination_time=questionable_normalized_words_determination_time,
                    sentiment_text=sentiment_text))
            if len(questionable_normalized_words)!=0:
                failed_string_to_questionable_normalized_words_map[sentiment_text] = questionable_normalized_words
                notes_worth_printing.append("\nWe encountered these unhandled words: {questionable_normalized_words}".format(questionable_normalized_words=questionable_normalized_words))
                notes_worth_printing.append("")
            if len(notes_worth_printing) != 0:
                # print()
                # print("==============================================================================================")
                # print("Current Iteration: {iteration_index}".format(iteration_index=iteration_index))
                print("Current Sentence Being Processed:\n{sentiment_text}\n".format(sentiment_text=sentiment_text))
                # for note in notes_worth_printing:
                #     print(note)
                # print("==============================================================================================")
                # print()
        self.assertTrue(len(failed_string_to_questionable_normalized_words_map)==0,
                        msg="We failed to process the following: \n{bad_pairs_printout}".format(
                            bad_pairs_printout=failed_string_to_questionable_normalized_words_map_repr(failed_string_to_questionable_normalized_words_map)))
            
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
