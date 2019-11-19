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
import tqdm
from word2vec_utilities import WORD2VEC_MODEL
from string_processing_utilities import word_string_resembles_meaningful_special_character_sequence_placeholder, normalized_words_from_text_string, PUNCTUATION_SET
from sentiment_analysis import TRAINING_DATA_LOCATION, TEST_DATA_LOCATION

#####################
# Testing Utilities #
#####################

COMMONLY_USED_MISSING_WORD2VEC_WORDS = [
    # Proper Nouns @todo handle named entities
    'hifa', 'edeka', 'swartz', "semisonic", 'sytycd', 'christy', 'pavel', 'safina', 'eddings', 'sannesias', 'winona', 'trae', 'tombre', 'rishabh', 'paramore', 'coldplay', 'neena', 'jlew', 
    'taylorrhicks', 'grac', 'seville', 'drexel', 'voltron', 'win7rc', 'lagerfeld', 'ahmier', 'zoro', 'rinitis', 'gongwer', 'aiden', 'jerrys', 'voltrons', 'hedo', 'bebot', 'lagerfeld', 'imodium', 
    'kimmy', 'nkotb', 'da70mm', 'minaj', 'f16', 'kallis', 'uat', 'pman', 'canaveral', 'imal', 'ohac', 'tirthankar', 'ankie','smf','dinara', 'garros','manee','anyer','burswood','shottas','karmada',
    'pixma', 'mx310','alistair','landin','ayonna','robsten','farrah', 'fawcett','kerrang','bizkit','rsl','paley','bjork','rb2','palmolive','supertramp','bcd', 'nodaji','trackle','btvs','lautner',
    'standfield','roberson','mgonewild','sadie','jbarsodmg','pbmall','farrah','aidan','avett','thames','horton','kokomo','brandice','bertolucci','lalalauren', 'weikert', 'hartley', 'cardona',
    'mmva','villarreal', 'leland', 'enigk','epsom','doodadoo','foxtell','bisante','tommi', 'oulu','farrah', 'fawcett','poirot','clopin','westwick','birtney', 'ciaraaaa','marah','irissa','stavros',
    'drunvalo','melchizedek','paraguay',
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

def failed_string_to_questionable_normalized_words_map_repr(failed_string_to_questionable_normalized_words_map: dict) -> None:
    return '\n'+''.join(['"{0}" : {1}\n'.format(sentiment_text, questionable_normalized_words)
                         for sentiment_text, questionable_normalized_words in failed_string_to_questionable_normalized_words_map.items()])

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
                for row_dict_index, row_dict in tqdm.tqdm(list(enumerate(row_dicts))):
                    sentiment_text = row_dict['SentimentText']
                    questionable_normalized_words = questionable_normalized_words_from_text_string(sentiment_text)
                    if len(questionable_normalized_words)!=0:
                        failed_string_to_questionable_normalized_words_map[sentiment_text] = questionable_normalized_words
                        print("{} : {}".format(sentiment_text, questionable_normalized_words))
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
