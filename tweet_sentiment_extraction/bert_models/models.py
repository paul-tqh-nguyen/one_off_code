#!/usr/bin/python3
'#!/usr/bin/python3 -OO'

"""
"""

# @todo update doc string

###########
# Imports #
###########

import os
import json
import random
import math
import pandas as pd
from typing import Tuple, Iterable, List, Any

import sys ; sys.path.append("..")
from misc_utilities import *
import preprocess_data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
with warnings_suppressed():
    from transformers import RobertaTokenizer, RobertaForTokenClassification, RobertaConfig

###########
# Globals #
###########

# @todo unify these globals with those in the abstract_predictor.py file

SEED = 1234 if __debug__ else os.getpid()
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = __debug__
torch.backends.cudnn.benchmark = not __debug__

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

FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME = 'final_model_score.json'
GLOBAL_BEST_MODEL_SCORE_JSON_FILE_LOCATION = 'global_best_model_score.json'

SENTIMENTS = ['positive', 'negative', 'neutral']
IS_SELECTED_OUTPUT_VALUE = [0,1]
NOT_SELECTED_OUTPUT_VALUE = [1,0]

NUMBER_OF_RELEVANT_RECENT_ITERATIONS = 1_000
MIN_NUMBER_OF_RELEVANT_RECENT_EPOCHS = 5

OUTPUT_DIR = './default_output'
TRAIN_PORTION = 0.75
VALIDATION_PORTION = 1-TRAIN_PORTION
NUMBER_OF_EPOCHS = 100

BATCH_SIZE = 32
NON_TRAINING_BATCH_SIZE = 128
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

    def _model_output_from_row(self, row: dict) -> torch.FloatTensor:
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
        output_tensor = torch.FloatTensor(output_values)
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
    assert tuple(padded_input_tensor_batch.shape) == (batch_size, max_sequence_length)
    padded_output_tensors = [torch.cat([output_tensor, torch.FloatTensor([NOT_SELECTED_OUTPUT_VALUE]*(max_sequence_length-len(output_tensor)))]) for output_tensor in output_tensors]
    padded_output_tensor_batch = torch.stack(padded_output_tensors)
    assert tuple(padded_output_tensor_batch.shape) == (batch_size, max_sequence_length, 2)
    return padded_input_tensor_batch, padded_output_tensor_batch

def load_data() -> Tuple[data.DataLoader, data.DataLoader]:
    with open(preprocess_data.PREPROCESSED_TRAINING_DATA_JSON_FILE) as file_handle:
        rows = [json.loads(line) for line in file_handle.readlines()]
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
    validation_data_loader = data.DataLoader(validation_dataset, batch_size=NON_TRAINING_BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUMBER_OF_DATALOADER_WORKERS, collate_fn=collate)
    return training_data_loader, validation_data_loader

##############
# Predictors #
##############

class BERTPredictor():
    def __init__(self, output_directory: str, number_of_epochs: int, batch_size: int, model: nn.modules.module.Module, loss_function: Callable, optimizer: optim.Optimizer):
        self.best_valid_jaccard = -1
        
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        self.best_saved_model_location = os.path.join(self.output_directory, 'best_model')
        if not os.path.exists(self.best_saved_model_location):
            os.makedirs(self.best_saved_model_location)
        
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        
        self.training_data_loader, self.validation_data_loader = load_data() # @todo this uses the global batch_size ; make this a method
        
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        assert sanity_check_model_forward_pass(self.model, self.validation_data_loader)

        self.jaccard_threshold = 0.5 # @todo optimize this

    @property
    def number_of_relevant_recent_epochs(self) -> int:
        number_of_iterations_per_epoch = len(self.training_data_loader) / self.batch_size
        number_of_epochs_per_iteration = number_of_iterations_per_epoch ** -1
        number_of_relevant_recent_epochs = math.ceil(number_of_epochs_per_iteration * NUMBER_OF_RELEVANT_RECENT_ITERATIONS)
        number_of_relevant_recent_epochs = max(MIN_NUMBER_OF_RELEVANT_RECENT_EPOCHS, number_of_relevant_recent_epochs)
        return number_of_relevant_recent_epochs
    
    def train_one_epoch(self) -> Tuple[float, float]:
        epoch_loss = 0
        epoch_jaccard = 0
        self.optimizer.zero_grad()
        for batch_index, (text_batch, labels) in tqdm_with_message(enumerate(self.training_data_loader),
                                                                   post_yield_message_func = lambda index: f'Training Jaccard {epoch_jaccard/(index+1):.8f}',
                                                                   total=len(self.training_data_loader)):
            text_batch = text_batch.to(DEVICE)
            labels = labels.to(DEVICE)
            pre_softmax_labels = only_one(self.model.forward(text_batch))
            predicted_labels = F.softmax(pre_softmax_labels, dim=2)
            if __debug__:
                batch_size = text_batch.shape[0]
                max_sequence_length = text_batch.shape[1]
            assert tuple(predicted_labels.shape) == (batch_size, max_sequence_length, 2)
            loss = self.loss_function(predicted_labels, labels)
            epoch_loss += loss.item()
            epoch_jaccard += self.scores_of_discretized_values(predicted_labels, labels)
            loss.backward()
            self.optimizer.step()
        epoch_loss /= len(self.validation_data_loader)
        epoch_jaccard /= len(self.validation_data_loader)
        return epoch_loss, epoch_jaccard
    
    def evaluate(self) -> Tuple[float, float]:
        epoch_loss = 0
        epoch_jaccard = 0
        self.model.eval()
        # self.optimize_jaccard_threshold() # @todo implement this
        with torch.no_grad():
            for text_batch, labels in tqdm_with_message(self.validation_data_loader,
                                                        post_yield_message_func = lambda index: f'Validation Jaccard {epoch_jaccard/(index+1):.8f}',
                                                        total=len(self.validation_data_loader)):
                text_batch = text_batch.to(DEVICE)
                labels = labels.to(DEVICE)
                pre_softmax_labels = only_one(self.model.forward(text_batch))
                predicted_labels = F.softmax(pre_softmax_labels, dim=2)
                if __debug__:
                    batch_size = text_batch.shape[0]
                    max_sequence_length = text_batch.shape[1]
                assert predicted_labels.shape[0] <= NON_TRAINING_BATCH_SIZE
                assert predicted_labels.shape[0] == batch_size
                assert predicted_labels.shape == labels.shape
                loss = self.loss_function(predicted_labels, labels)
                epoch_loss += loss.item()
                epoch_jaccard += self.scores_of_discretized_values(predicted_labels, labels)
        # self.reset_jaccard_threshold() # @todo implement this
        epoch_loss /= len(self.validation_data_loader)
        epoch_jaccard /= len(self.validation_data_loader)
        return epoch_loss, epoch_jaccard
    
    def validate(self, epoch_index: int, result_is_from_final_run: bool) -> Tuple[float, float]:
        valid_loss, valid_jaccard = self.evaluate()
        if not os.path.isfile(GLOBAL_BEST_MODEL_SCORE_JSON_FILE_LOCATION):
            log_current_model_as_best = True
        else:
            with open(GLOBAL_BEST_MODEL_SCORE_JSON_FILE_LOCATION, 'r') as current_global_best_model_score_json_file:
                current_global_best_model_score_dict = json.load(current_global_best_model_score_json_file)
                current_global_best_model_jaccard: float = current_global_best_model_score_dict['valid_jaccard']
                log_current_model_as_best = current_global_best_model_jaccard < valid_jaccard
        self_score_dict = {
            'predictor_type': self.__class__.__name__,
            'valid_jaccard': valid_jaccard,
            'valid_loss': valid_loss,
            'best_valid_jaccard': self.best_valid_jaccard,
            'number_of_epochs': self.number_of_epochs,
            'most_recently_completed_epoch_index': epoch_index,
            'number_of_relevant_recent_epochs': self.number_of_relevant_recent_epochs,
            'batch_size': self.batch_size,
            'train_portion': TRAIN_PORTION, # @todo make this an attribute
            'validation_portion': VALIDATION_PORTION, # @todo make this an attribute
            'number_of_parameters': self.count_parameters(),
        }
        if log_current_model_as_best:
            with open(GLOBAL_BEST_MODEL_SCORE_JSON_FILE_LOCATION, 'w') as outfile:
                json.dump(self_score_dict, outfile)
        with open(self.latest_model_score_location, 'w') as outfile:
            json.dump(self_score_dict, outfile)
        if result_is_from_final_run:
            with open(self.final_model_score_location, 'w') as outfile:
                json.dump(self_score_dict, outfile)
        if valid_jaccard > self.best_valid_jaccard:
            self.best_valid_jaccard = valid_jaccard
            self.save_parameters(self.best_saved_model_location)
            print(f'Best model so far saved to {self.best_saved_model_location}')
        return valid_loss, valid_jaccard
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    @property
    def latest_model_score_location(self) -> str:
        return os.path.join(self.output_directory, 'latest_model_score.json')

    @property
    def final_model_score_location(self) -> str:
        return os.path.join(self.output_directory, FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME)
    
    def save_parameters(self, parameter_directory_location: str) -> None:
        self.model.save_pretrained(parameter_directory_location)
        return
    
    def load_parameters(self, parameter_directory_location: str) -> None:
        self.model.from_pretrained(parameter_directory_location)
        return
    
    def train(self) -> None:
        self.print_hyperparameters()
        most_recent_validation_jaccard_scores = [0]*self.number_of_relevant_recent_epochs
        print(f'Starting training')
        for epoch_index in range(self.number_of_epochs):
            print("\n")
            print(f"Epoch {epoch_index}")
            with timer(section_name=f"Epoch {epoch_index}"):
                train_loss, train_jaccard = self.train_one_epoch()
                valid_loss, valid_jaccard = self.validate(epoch_index, False)
                print(f'\t   Training Jaccard: {train_jaccard:.8f} |   Training Loss: {train_loss:.8f}')
                print(f'\t Validation Jaccard: {valid_jaccard:.8f} | Validation Loss: {valid_loss:.8f}')
            print("\n")
            if any(valid_jaccard > previous_jaccard for previous_jaccard in most_recent_validation_jaccard_scores):
                most_recent_validation_jaccard_scores.pop(0)
                most_recent_validation_jaccard_scores.append(valid_jaccard)
            else:
                print()
                print(f"Validation is not better than any of the {self.number_of_relevant_recent_epochs} recent epochs, so training is ending early due to apparent convergence.")
                print()
                break
        self.load_parameters(self.best_saved_model_location)
        self.validate(epoch_index, True)
        return
    
    def scores_of_discretized_values(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        epsilon = 1e-5
        assert (y-y.int()).sum()==0
        assert y_hat.shape == y.shape
        batch_size, max_sequence_length, number_of_token_classes = y.shape
        assert number_of_token_classes == 2
        y_hat_discretized = (y_hat.detach() > self.jaccard_threshold)
        intersection_count = (y_hat_discretized[:,:,1] & y[:,:,1].bool()).sum(dim=1)
        assert tuple(intersection_count.shape) == (batch_size,)
        union_count = y_hat_discretized[:,:,1].sum(dim=1) + y[:,:,1].sum(dim=1) - intersection_count
        assert tuple(union_count.shape) == (batch_size,)
        jaccard_index = intersection_count / (union_count + epsilon)
        assert tuple(jaccard_index.shape) == (batch_size,)
        mean_jaccard_index = jaccard_index.mean().item()
        assert isinstance(mean_jaccard_index, float)
        assert mean_jaccard_index == 0.0 or 0.0 not in (intersection_count.sum(), union_count.sum())
        return mean_jaccard_index

    def print_hyperparameters(self) -> None:
        print()
        print(f"Model hyperparameters are:")
        print(f'        predictor_type: {self.__class__.__name__}')
        print(f'        number_of_epochs: {self.number_of_epochs}')
        print(f'        number_of_relevant_recent_epochs: {self.number_of_relevant_recent_epochs}')
        print(f'        batch_size: {self.batch_size}')
        print(f'        output_directory: {self.output_directory}')
        for config_key, config_value in sorted(self.model.config.to_dict().items()):
            print(f'        {config_key}: {repr(config_value)}')
        print()
        print(f'The model has {self.count_parameters():,} trainable parameters.')
        print(f"This processes's PID is {os.getpid()}.")
        if DEVICE.type == 'cuda':
            print(f'The CUDA device being used is {torch.cuda.get_device_name(DEVICE_ID)}')
            print(f'The CUDA device ID being used is {DEVICE_ID}')
        print()

        return

###############
# Main Driver #
###############

@debug_on_error
def train_model() -> None:
    transformers_model_config = RobertaConfig.from_pretrained(TRANSFORMERS_MODEL_SPEC)
    model = RobertaForTokenClassification(transformers_model_config).to(DEVICE)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=INITIAL_LEARNING_RATE)
    predictor = BERTPredictor(OUTPUT_DIR, NUMBER_OF_EPOCHS, BATCH_SIZE, model, loss_function, optimizer)
    predictor.train()
    return 

if __name__ == '__main__':
    train_model()
