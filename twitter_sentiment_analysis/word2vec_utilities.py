#!/usr/bin/python3 -O

"""

This is a module to that imports the WROD2VEC model.

Owner : paul-tqh-nguyen

Created : 11/15/2019

File Name : word2vec_utilities.py

"""

from gensim.models import KeyedVectors

WORD2VEC_BIN_LOCATION = '/home/pnguyen/code/datasets/GoogleNews-vectors-negative300.bin'
WORD2VEC_MODEL = KeyedVectors.load_word2vec_format(WORD2VEC_BIN_LOCATION, binary=True)
WORD2VEC_VECTOR_LENGTH = len(WORD2VEC_MODEL['queen'])

def main():
    print("This module contains WORD2VEC importing utilities.")

if __name__ == '__main__':
    main()
