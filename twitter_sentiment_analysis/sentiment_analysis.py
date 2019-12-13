#!/usr/bin/python3 -O

"""

This file contains the main driver for a neural network based sentiment analyzer for Twitter data. 

Owner : paul-tqh-nguyen

Created : 10/30/2019

File Name : sentiment_analysis.py

File Organization:
* Imports
* Main Runner

"""

###########
# Imports #
###########

import argparse
import string_processing_tests

###############
# Main Runner #
###############

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-run-tests', action='store_true', help="To run all of the tests.")
    args = parser.parse_args()
    print(vars(args))
    return None
    
if __name__ == '__main__':
    main()
