#!/usr/bin/python3 -OO

'''

This module contains global initializations and global values.

Sections:
* Imports
* Logging
* Globals

'''

###########
# Imports #
###########

import os
import logging
import nvgpu
import multiprocessing as mp
from pandarallel import pandarallel

from misc_utilities import *

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

HYPERPARAMETER_SEARCH_IS_DISTRIBUTED = True

CPU_COUNT = mp.cpu_count()
pandarallel.initialize(nb_workers=CPU_COUNT, progress_bar=False, verbose=0)

GPU_IDS = eager_map(int, nvgpu.available_gpus())
DEFAULT_GPU = GPU_IDS[0]

NUM_WORKERS = 0 if HYPERPARAMETER_SEARCH_IS_DISTRIBUTED else 2

if not os.path.isdir('./checkpoints'):
    os.makedirs('./checkpoints')

if __name__ == '__main__':
    print('This module contains global initializations and global values.')
