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
with warnings_suppressed():
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
IS_SELECTED_OUTPUT_VALUE = [0,1]
NOT_SELECTED_OUTPUT_VALUE = [1,0]

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
PAD_IDX = TRANSFORMERS_TOKENIZER.pad_token_id
MAX_SEQUENCE_LENGTH = 512
NEW_WORD_PREFIX = chr(288)

###################
# Sanity Checking #
###################

def sanity_check_model_forward_pass(model: nn.modules.module.Module, dataloader: data.DataLoader) -> bool:
    random_index = random.randint(0, len(dataloader.dataset)-1)
    x, y = dataloader.dataset[random_index]
    x_batch = x.unsqueeze(0).to(DEVICE)
    y_hat_batch = only_one(model(x_batch))
    y_hat = y_hat_batch.squeeze(0)
    assert y_hat.shape == y.shape
    return True

#############
# Load Data #
#############

class TweetSentimentSelectionDataset(data.Dataset):
    def __init__(self, rows: List[dict]):
        self.rows = rows
        self.x = eager_map(self._model_input_from_row, tqdm_with_message(rows, post_yield_message_func = lambda index: f'Loading Input Example {index}', ))
        self.y = eager_map(self._model_output_from_row, tqdm_with_message(rows, post_yield_message_func = lambda index: f'Loading Output Example {index}', ))
        assert eager_map(lambda x: x.shape[0], self.x) == eager_map(lambda y: y.shape[0], self.y)

    def _model_input_from_row(self, row: dict) -> torch.LongTensor:
        preprocessed_input_string = row['preprocessed_input_string']
        sentiment = row['sentiment']
        ids = TRANSFORMERS_TOKENIZER.encode(preprocessed_input_string, sentiment)
        id_tensor = torch.LongTensor(ids)
        return id_tensor

    def _model_output_from_row(self, row: dict) -> torch.LongTensor:
        preprocessed_input_string = row['preprocessed_input_string']
        words = preprocessed_input_string.split()
        numericalized_selected_text = row['numericalized_selected_text']
        assert set(numericalized_selected_text).issubset('01')
        output_values: List[List[int]] = [NOT_SELECTED_OUTPUT_VALUE]
        assert len(words) == len(numericalized_selected_text)
        for word, result_as_char in zip(words, numericalized_selected_text):
            token_output_value = IS_SELECTED_OUTPUT_VALUE if result_as_char == '1' else NOT_SELECTED_OUTPUT_VALUE
            number_of_tokens_for_word = len(TRANSFORMERS_TOKENIZER.encode(word)[1:-1])
            for _ in range(number_of_tokens_for_word):
                output_values.append(token_output_value)
        output_values = output_values + [NOT_SELECTED_OUTPUT_VALUE]*4
        assert len(output_values) == len(TRANSFORMERS_TOKENIZER.encode(preprocessed_input_string, row['sentiment']))
        output_tensor = torch.LongTensor(output_values)
        return output_tensor
            
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)

def collate(input_output_pairs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    input_tensors, output_tensors = zip(*input_output_pairs)
    assert len(input_tensors) == len( output_tensors)
    if __debug__:
        batch_size = len(input_tensors)
    assert {len(input_tensor.shape) for input_tensor in input_tensors} == {1}
    assert {len(output_tensor.shape) for output_tensor in output_tensors} == {2}
    max_sequence_length = max(input_tensor.shape[0] for input_tensor in input_tensors)
    assert max_sequence_length == max(output_tensor.shape[0] for output_tensor in output_tensors)
    padded_input_tensors = [torch.cat([input_tensor, torch.ones(max_sequence_length-len(input_tensor), dtype=int)*PAD_IDX]) for input_tensor in input_tensors]
    padded_input_tensor_batch = torch.stack(padded_input_tensors)
    print(f"(batch_size, max_sequence_length) {repr((batch_size, max_sequence_length))}")
    print(f"padded_input_tensor_batch.shape {repr(padded_input_tensor_batch.shape)}")
    assert tuple(padded_input_tensor_batch.shape) == (batch_size, max_sequence_length)
    padded_output_tensors = [torch.cat([output_tensor, torch.LongTensor([NOT_SELECTED_OUTPUT_VALUE]*(max_sequence_length-len(output_tensor)))]) for output_tensor in output_tensors]
    padded_output_tensor_batch = torch.stack(padded_output_tensors)
    assert tuple(padded_output_tensor_batch.shape) == (batch_size, max_sequence_length, 2)
    return padded_input_tensor_batch, padded_output_tensor_batch

def load_data() -> Tuple[data.DataLoader, data.DataLoader]:
    with open(preprocess_data.PREPROCESSED_TRAINING_DATA_JSON_FILE) as file_handle:
        rows = [json.loads(line) for line in file_handle.readlines()]
    rows = rows[:100] # @todo remove this
    random.shuffle(rows)
    number_of_training_examples = round(TRAIN_PORTION*len(rows))
    print()
    print('Loading Training Data...')
    training_dataset = TweetSentimentSelectionDataset(rows[:number_of_training_examples])
    print()
    print('Loading Validation Data...')
    validation_dataset = TweetSentimentSelectionDataset(rows[number_of_training_examples:])
    assert len(validation_dataset) == round(VALIDATION_PORTION*len(rows))
    training_data_loader = data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=NUMBER_OF_DATALOADER_WORKERS, collate_fn=collate)
    validation_data_loader = data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUMBER_OF_DATALOADER_WORKERS, collate_fn=collate)
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
    # train model
    for epoch_index in range(NUMBER_OF_EPOCHS):
        print(f'Epoch {repr(epoch_index)}')
        epoch_jaccard = 0
        optimizer.zero_grad()
        for batch_index, (text_batch, labels) in tqdm_with_message(enumerate(training_data_loader), post_yield_message_func = lambda index: f'Training Jaccard {epoch_jaccard/(index+1):.8f}', ):
            text_batch = text_batch.to(DEVICE)
            labels = labels.to(DEVICE)
            predicted_labels = model.forward(text_batch)
            loss = loss_function(text_batch, labels)
            loss.backward()
            optimizer.step()
    return 
    
if __name__ == '__main__':
    train_model()
