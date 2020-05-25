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
import numpy as np

from misc_utilities import *

import torch

###################
# Initializations #
###################

with warnings_suppressed():
    tqdm.tqdm.pandas()

SEED = 1234 if __debug__ else os.getpid()
np.random.seed(SEED)
torch.manual_seed(SEED)
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
MIN_NUMBER_OF_RELEVANT_RECENT_EPOCHS = 10
MAX_NUMBER_OF_RELEVANT_RECENT_EPOCHS = 30

def number_of_relevant_recent_epochs_for_data_size_and_batch_size(data_size: int, batch_size: int) -> int:
    number_of_iterations_per_epoch = data_size / batch_size
    number_of_epochs_per_iteration = number_of_iterations_per_epoch ** -1
    number_of_relevant_recent_epochs = math.ceil(number_of_epochs_per_iteration * NUMBER_OF_RELEVANT_RECENT_ITERATIONS)
    number_of_relevant_recent_epochs = max(MIN_NUMBER_OF_RELEVANT_RECENT_EPOCHS, number_of_relevant_recent_epochs)
    number_of_relevant_recent_epochs = min(MAX_NUMBER_OF_RELEVANT_RECENT_EPOCHS, number_of_relevant_recent_epochs)
    return number_of_relevant_recent_epochs

#########
# Misc. #
#########

FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME = 'final_model_score.json'
GLOBAL_BEST_MODEL_SCORE_JSON_FILE_LOCATION = 'global_best_model_score.json'

SENTIMENTS = ['positive', 'negative', 'neutral']

NON_TRAINING_BATCH_SIZE = 1024

OUTPUT_DIR = './default_output'
TRAIN_PORTION = 0.80
VALIDATION_PORTION = 1-TRAIN_PORTION
NUMBER_OF_EPOCHS = 100

NUMBER_OF_EXAMPLES_TO_DEMONSTRATE = 30

def jaccard_index_over_strings(str1: str, str2: str): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

##########
# Driver #
##########

if __name__ == '__main__':
    print('This module contains globals and utlities to be used by our models.')
 
