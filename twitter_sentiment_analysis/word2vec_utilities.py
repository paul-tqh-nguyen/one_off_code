#!/usr/bin/python3 -O

"""

This is a module to that imports the WROD2VEC model.

Owner : paul-tqh-nguyen

Created : 11/15/2019

File Name : word2vec_utilities.py

"""

from gensim.models import KeyedVectors
import numpy as np

WORD2VEC_BIN_LOCATION = '/home/pnguyen/code/datasets/GoogleNews-vectors-negative300.bin'
WORD2VEC_MODEL_FROM_FILE = KeyedVectors.load_word2vec_format(WORD2VEC_BIN_LOCATION, binary=True)
WORD2VEC_VECTOR_LENGTH = len(WORD2VEC_MODEL_FROM_FILE['queen'])

COMMON_WORDS_MISSING_FROM_WORD2VEC_MODEL_TO_KNOWN_SYNONYMS_MAP = {
    'a': ['an',],
    'to': ['into', 'toward', 'before', 'on', 'over', 'upon', 'till', 'until', 'so'],
    'and': ['also', 'furthermore', 'including', 'moreover', 'plus', 'et', 'additionally'],
    'of': ['about', 'concerning', 'regarding', 'containing', 'wrt', 'referencing', 'like', 'from'],
}

MISC_WORDS_MISSING_FROM_WORD2VEC_MODEL_TO_KNOWN_SYNONYMS_MAP = {
}

SUPPLEMENTAL_WORD2VEC_DWIMMED_ENTRIES_VIA_SYNONYMS = dict()

WORD_TO_SYNONYMS_MAPS = (
    COMMON_WORDS_MISSING_FROM_WORD2VEC_MODEL_TO_KNOWN_SYNONYMS_MAP,
    MISC_WORDS_MISSING_FROM_WORD2VEC_MODEL_TO_KNOWN_SYNONYMS_MAP
)

for word_to_synonyms_map in WORD_TO_SYNONYMS_MAPS:
    for word, synonyms in word_to_synonyms_map.items():
        vectors_for_synonyms = [WORD2VEC_MODEL_FROM_FILE[synonyms] for synonym in synonyms]
        vectors_for_synonyms_as_array = np.array(vectors_for_synonyms)
        mean_vector = np.mean(vectors_for_synonyms_as_array, axis=0)
        SUPPLEMENTAL_WORD2VEC_DWIMMED_ENTRIES_VIA_SYNONYMS[word] = mean_vector

def common_word_missing_from_word2vec_model(word_string: str) -> bool:
    return word_string.lower() in COMMON_WORDS_MISSING_FROM_WORD2VEC_MODEL_TO_KNOWN_SYNONYMS_MAP

class SupplementedWord2VecModel():
    def __init__(self):
        self.word2vec_model_from_file = WORD2VEC_MODEL_FROM_FILE
        self.supplemental_word2vec_dwimmed_entries_via_synonyms = SUPPLEMENTAL_WORD2VEC_DWIMMED_ENTRIES_VIA_SYNONYMS
    
    def __contains__(self, word: str):
        return word in self.word2vec_model_from_file or word in self.supplemental_word2vec_dwimmed_entries_via_synonyms
    
    def __getitem__(self, word: str):
        item = None
        if word in self.word2vec_model_from_file:
            item = self.word2vec_model_from_file[word]
        elif word in self.supplemental_word2vec_dwimmed_entries_via_synonyms:
            item = self.supplemental_word2vec_dwimmed_entries_via_synonyms[word]
        else:
            raise AttributeError('{word} is not in the vocabulary.'.format(word=word))
        return item

WORD2VEC_MODEL = SupplementedWord2VecModel()

def main():
    print("This module contains WORD2VEC importing utilities.")

if __name__ == '__main__':
    main()
