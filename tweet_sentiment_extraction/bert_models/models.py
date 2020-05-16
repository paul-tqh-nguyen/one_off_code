#!/usr/bin/python3
'#!/usr/bin/python3 -OO'

"""
"""

# @todo update doc string

###########
# Imports #
###########

import json
import pandas as pd
from typing import Tuple, Iterable, List, Any

import sys ; sys.path.append("..")
from misc_utilities import *
from word_selector_models.abstract_predictor import DEVICE, SENTIMENTS
import preprocess_data

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig

###########
# Globals #
###########

SEED = 1234 if __debug__ else os.getpid()
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = __debug__
torch.backends.cudnn.benchmark = not __debug__

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SENTIMENTS = ['positive', 'negative', 'neutral']

OUTPUT_DIR = './default_output'
TRAIN_PORTION = 0.75
VALIDATION_PORTION = 1-TRAIN_PORTION
NUMBER_OF_EPOCHS = 100

BATCH_SIZE = 32

TRANSFORMERS_MODEL_SPEC = 'roberta-base'
TRANSFORMERS_TOKENIZER = RobertaTokenizer.from_pretrained(TRANSFORMERS_MODEL_SPEC)
CLS_TOKEN = TRANSFORMERS_TOKENIZER.cls_token
SEP_TOKEN = TRANSFORMERS_TOKENIZER.sep_token
MAX_SEQUENCE_LENGTH = 512

#############
# Load Data #
#############

def generate_batches(inputs: Iterable[Any]) -> List[List[Any]]:
    inputs = list(inputs)
    number_of_full_batches, last_batch_size = divmod(len(inputs), BATCH_SIZE)
    batches: List[List[Any]] = []
    for batch_index in range(number_of_full_batches):
        current_batch_size = last_batch_size if batch_index+1 == number_of_full_batches else BATCH_SIZE
        batch = inputs[batch_index*current_batch_size: (batch_index+1)*current_batch_size]
        batches.append(batch)
    return batches

def model_input_from_row(row: dict) -> Tuple[List[int], List[int]]:
    tokens = TRANSFORMERS_TOKENIZER.tokenize(row['preprocessed_input_string'])
    if len(tokens) > MAX_SEQUENCE_LENGTH-2:
        tokens = tokens[:MAX_SEQUENCE_LENGTH-2]
    tokens = [CLS_TOKEN]+tokens+[SEP_TOKEN]
    ids = TRANSFORMERS_TOKENIZER.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(tokens)
    return ids, attention_mask

def model_input_batches_from_rows(rows: List[dict]) -> List[Tuple[torch.LongTensor, List[List[int]]]]:
    model_inputs = map(model_input_from_row, rows)
    model_input_batches = generate_batches(model_inputs)
    coerce_model_input_batch_ids_to_tensor = lambda batch: (torch.LongTensor(batch[0]), batch[1])
    model_input_batches = eager_map(coerce_model_input_batch_ids_to_tensor, model_input_batches)
    return model_input_batches

def load_data() -> Tuple[List, List]:
    with open(preprocess_data.PREPROCESSED_TRAINING_DATA_JSON_FILE) as file_handle:
        rows = [json.loads(line) for line in file_handle.readlines()]
    x = model_input_batches_from_rows(rows)
    y = generate_batches((eager_map(int, row['numericalized_selected_text']) for row in rows))
    return x, y

###############
# Main Driver #
###############

@debug_on_error
def train_model() -> None:
    x, y = load_data()
    # load model
    transformers_model_config = RobertaConfig.from_pretrained(TRANSFORMERS_MODEL_SPEC)
    model = RobertaForMaskedLM(transformers_model_config)
    model.to(DEVICE)
    return 

if __name__ == '__main__':
    train_model()
