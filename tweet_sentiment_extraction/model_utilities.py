#!/usr/bin/python3
'#!/usr/bin/python3 -OO'

'''
'''

# @todo update doc string

###########
# Imports #
###########

import tqdm
import math
import random
import numpy as np

from misc_utilities import *

import torch

###################
# Initializations #
###################

with warnings_suppressed():
    tqdm.tqdm.pandas()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return

SEED = 1234 if __debug__ else os.getpid()
set_seed(SEED)
torch.backends.cudnn.deterministic = __debug__
torch.backends.cudnn.benchmark = not __debug__

##########
# Device #
##########

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUMBER_OF_DATALOADER_WORKERS = 8
DEVICE_ID = None if DEVICE == 'cpu' else torch.cuda.current_device()

def set_global_device_id(global_device_id: int) -> None:
    assert DEVICE.type == 'cuda'
    assert global_device_id < torch.cuda.device_count()
    global DEVICE_ID
    DEVICE_ID = global_device_id
    torch.cuda.set_device(DEVICE_ID)
    return

###############
# Convergence #
###############

NUMBER_OF_RELEVANT_RECENT_ITERATIONS = 1_000
MIN_NUMBER_OF_RELEVANT_RECENT_EPOCHS = 5
MAX_NUMBER_OF_RELEVANT_RECENT_EPOCHS = 10

def jaccard_sufficiently_high_for_epoch(jaccard_score: float, epoch_index: int) -> bool:
    jaccard_thresholds_for_number_of_epochs = [
        (0.5, 25),
        (0.6, 40),
    ]
    jaccard_sufficiently_high = all(implies(epoch_index >= number_of_epochs_for_threshold, jaccard_score > jaccard_threshold) for jaccard_threshold, number_of_epochs_for_threshold in jaccard_thresholds_for_number_of_epochs)
    return jaccard_sufficiently_high

def number_of_relevant_recent_epochs_for_data_size_and_batch_size(data_size: int, batch_size: int) -> int:
    number_of_iterations_per_epoch = data_size / batch_size
    number_of_epochs_per_iteration = number_of_iterations_per_epoch ** -1
    number_of_relevant_recent_epochs = math.ceil(number_of_epochs_per_iteration * NUMBER_OF_RELEVANT_RECENT_ITERATIONS)
    number_of_relevant_recent_epochs = max(MIN_NUMBER_OF_RELEVANT_RECENT_EPOCHS, number_of_relevant_recent_epochs)
    number_of_relevant_recent_epochs = min(MAX_NUMBER_OF_RELEVANT_RECENT_EPOCHS, number_of_relevant_recent_epochs)
    return number_of_relevant_recent_epochs

########
# Data #
########

TRAINING_DATA_CSV_FILE = './data/train.csv'
TESTING_DATA_CSV_FILE = './data/test.csv'
PREPROCESSED_TRAINING_DATA_JSON_FILE = './data/preprocessed_train.json'

FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME = 'final_model_score.json'
GLOBAL_BEST_MODEL_SCORE_JSON_FILE_LOCATION = 'global_best_model_score.json'
CROSS_VALIDATION_RESULTS_CSV_FILE_LOCATION_BASE_NAME = 'cross_validation.csv'
TESTING_RESULTS_CSV_FILE_LOCATION_BASE_NAME = "test_results.csv"
SUBMISSION_CSV_FILE_LOCATION_BASE_NAME = "submission.csv"

#########
# Misc. #
#########

SENTIMENTS = ['positive', 'negative', 'neutral']

NON_TRAINING_BATCH_SIZE = 1024

OUTPUT_DIR = './default_output'
NUMBER_OF_FOLDS = 5
NUMBER_OF_EPOCHS = 100

NUMBER_OF_EXAMPLES_TO_DEMONSTRATE = 30
JACCARD_INDEX_GOOD_SCORE_THRESHOLD = 0.5

def jaccard_index_over_strings(str1: str, str2: str):
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return 1 if len(a) + len(b) == 0 else float(len(c)) / (len(a) + len(b) - len(c))

def remove_non_ascii_characters(text: str) -> str:
    return ''.join(eager_filter(is_ascii, text))

def is_nan(obj) -> bool:
    return obj != obj

##########
# Driver #
##########

if __name__ == '__main__':
    print('This module contains globals and utlities to be used by our models.')
 
