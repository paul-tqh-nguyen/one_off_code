#!/usr/bin/python3
'#!/usr/bin/python3 -OO'

"""
"""

# @todo update doc string

###########
# Imports #
###########

import json
import random
import pandas as pd
from typing import Tuple, Iterable, List, Any

import sys ; sys.path.append("..")
from misc_utilities import *
from word_selector_models.abstract_predictor import DEVICE, SENTIMENTS
import preprocess_data

import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaTokenizer, RobertaForTokenClassification, RobertaConfig

###########
# Globals #
###########

SEED = 1234 if __debug__ else os.getpid()
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = __debug__
torch.backends.cudnn.benchmark = not __debug__

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUMBER_OF_DATALOADER_WORKERS = 8

SENTIMENTS = ['positive', 'negative', 'neutral']

OUTPUT_DIR = './default_output'
TRAIN_PORTION = 0.75
VALIDATION_PORTION = 1-TRAIN_PORTION
NUMBER_OF_EPOCHS = 100

BATCH_SIZE = 32
INITIAL_LEARNING_RATE = 1e-5

TRANSFORMERS_MODEL_SPEC = 'roberta-base'
TRANSFORMERS_TOKENIZER = RobertaTokenizer.from_pretrained(TRANSFORMERS_MODEL_SPEC)
CLS_TOKEN = TRANSFORMERS_TOKENIZER.cls_token
SEP_TOKEN = TRANSFORMERS_TOKENIZER.sep_token
MAX_SEQUENCE_LENGTH = 512
NEW_WORD_PREFIX = chr(288)

###################
# Sanity Checking #
###################

def sanity_check_model_forward_pass(model: nn.modules.module.Module, dataloader: data.DataLoader) -> bool:
    random_index = random.randint(0, len(dataloader.dataset))
    x, y = dataloader.dataset[random_index]
    y_hat = only_one(model(x))
    assert y_hat.shape == y.shape
    return True

#############
# Load Data #
#############

class TweetSentimentSelectionDataset(data.Dataset):
    def __init__(self, rows: List[dict]):
        self.rows = rows
        self.x = eager_map(self._model_input_from_row, rows)
        self.y = eager_map(self._model_output_from_row, rows)
        assert eager_map(lambda x: x.shape[1], self.x) == eager_map(lambda y: y.shape[1], self.y)

    def _model_input_from_row(self, row: dict) -> torch.LongTensor:
        preprocessed_input_string = row['preprocessed_input_string']
        sentiment = row['sentiment']
        ids = TRANSFORMERS_TOKENIZER.encode(preprocessed_input_string, sentiment)
        id_tensor = torch.LongTensor(ids).to(DEVICE)
        id_batch = id_tensor.unsqueeze(0)
        return id_batch

    def _model_output_from_row(self, row: dict) -> torch.LongTensor:
        preprocessed_input_string = row['preprocessed_input_string']
        words = preprocessed_input_string.split()
        numericalized_selected_text = row['numericalized_selected_text']
        assert set(numericalized_selected_text).issubset('01')
        is_selected_output_value = [0,1]
        not_selected_output_value = [1,0]
        output_values: List[List[int]] = [not_selected_output_value]
        assert len(words) == len(numericalized_selected_text)
        for word, result_as_char in zip(words, numericalized_selected_text):
            token_output_value = is_selected_output_value if result_as_char == '1' else not_selected_output_value
            number_of_tokens_for_word = len(TRANSFORMERS_TOKENIZER.encode(word)[1:-1])
            for _ in range(number_of_tokens_for_word):
                output_values.append(token_output_value)
        output_values = output_values + [not_selected_output_value]*4
        assert len(output_values) == len(TRANSFORMERS_TOKENIZER.encode(preprocessed_input_string, row['sentiment']))
        output_tensor = torch.LongTensor(output_values)
        output_batch = output_tensor.unsqueeze(0)
        return output_batch
            
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)

def load_data() -> Tuple[data.DataLoader, data.DataLoader]:
    with open(preprocess_data.PREPROCESSED_TRAINING_DATA_JSON_FILE) as file_handle:
        rows = [json.loads(line) for line in file_handle.readlines()]
    random.shuffle(rows)
    number_of_training_examples = round(TRAIN_PORTION*len(rows))
    training_dataset = TweetSentimentSelectionDataset(rows[:number_of_training_examples])
    validation_dataset = TweetSentimentSelectionDataset(rows[number_of_training_examples:])
    assert len(validation_dataset) == round(VALIDATION_PORTION*len(rows))
    training_data_loader = data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=NUMBER_OF_DATALOADER_WORKERS)
    validation_data_loader = data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUMBER_OF_DATALOADER_WORKERS)
    return training_data_loader, validation_data_loader

###############
# Main Driver #
###############

@debug_on_error
def train_model() -> None:
    training_data_loader, validation_data_loader = load_data()
    # load model
    transformers_model_config = RobertaConfig.from_pretrained(TRANSFORMERS_MODEL_SPEC)
    model = RobertaForTokenClassification(transformers_model_config)
    model.to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=INITIAL_LEARNING_RATE)
    assert sanity_check_model_forward_pass(model, validation_data_loader)
    
    return 

if __name__ == '__main__':
    train_model()
