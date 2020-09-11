'#!/usr/bin/python3 -OO' # @todo use this

'''
'''

# @todo update doc string

###########
# Imports #
###########

import os
import logging
import nvgpu
import multiprocessing as mp

from misc_utilities import *

# @todo make sure these imports are used

###########
# Logging #
###########

LOGGER_NAME = 'anime_collaborative_filtering_logger'
LOGGER = logging.getLogger(LOGGER_NAME)
LOGGER_OUTPUT_FILE = './tuning_logs.txt'
LOGGER_STREAM_HANDLER = logging.StreamHandler()

def _initialize_logger() -> None:
    LOGGER.setLevel(logging.INFO)
    logging_formatter = logging.Formatter('{asctime} - pid: {process} - threadid: {thread} - func: {funcName} - {levelname}: {message}', style='{')
    logging_file_handler = logging.FileHandler(LOGGER_OUTPUT_FILE)
    logging_file_handler.setFormatter(logging_formatter)
    LOGGER.addHandler(logging_file_handler)
    LOGGER.addHandler(LOGGER_STREAM_HANDLER)
    return

_initialize_logger()

###########
# Globals #
###########

DB_URL = 'sqlite:///collaborative_filtering.db'
STUDY_NAME = 'collaborative-filtering'

HYPERPARAMETER_SEARCH_IS_DISTRIBUTED = True
NUMBER_OF_HYPERPARAMETER_SEARCH_TRIALS = 200
NUMBER_OF_BEST_HYPERPARAMETER_RESULTS_TO_DISPLAY = 5

CPU_COUNT = mp.cpu_count()
if not HYPERPARAMETER_SEARCH_IS_DISTRIBUTED:
    pandarallel.initialize(nb_workers=CPU_COUNT, progress_bar=False, verbose=0)

GPU_IDS = eager_map(int, nvgpu.available_gpus())
DEFAULT_GPU = GPU_IDS[0]

NUM_WORKERS = 0 if HYPERPARAMETER_SEARCH_IS_DISTRIBUTED else 2

if not os.path.isdir('./checkpoints'):
    os.makedirs('./checkpoints')

# https://www.kaggle.com/CooperUnion/anime-recommendations-database
# ANIME_CSV_FILE_LOCATION = './data/anime.csv' # @todo use this
RATING_CSV_FILE_LOCATION = './data/rating.csv'

PROCESSED_DATA_CSV_FILE_LOCATION = './data/processed_data.csv'

RATING_HISTORGRAM_PNG_FILE_LOCATION = './data/rating_histogram.png'

TRAINING_LABEL, VALIDATION_LABEL, TESTING_LABEL = 0, 1, 2

TRAINING_PORTION = 0.65
VALIDATION_PORTION = 0.15
TESTING_PORTION = 0.20

MINIMUM_NUMBER_OF_RATINGS_PER_ANIME = 100
MINIMUM_NUMBER_OF_RATINGS_PER_USER = 100

# @todo make sure these globals are used
