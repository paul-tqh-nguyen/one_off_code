#!/usr/bin/python3
'#!/usr/bin/python3 -OO'

'''
'''

# @todo update doc string

###########
# Imports #
###########

import os
import json
import random
import math
import tqdm
import numpy as np
import pandas as pd
from statistics import mean
from functools import reduce
from typing import Tuple, Iterable, List

import sys ; sys.path.append('..')
from model_utilities import *
from misc_utilities import *
import preprocess_data

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
with warnings_suppressed():
    from transformers import RobertaTokenizer, RobertaForTokenClassification, RobertaConfig

###########
# Globals #
###########

BATCH_SIZE = 32
NON_TRAINING_BATCH_SIZE = 256
INITIAL_LEARNING_RATE = 1e-5
GRADIENT_CLIPPING_THRESHOLD = 30

TRANSFORMERS_MODEL_SPEC = 'roberta-base'
TRANSFORMERS_TOKENIZER = RobertaTokenizer.from_pretrained(TRANSFORMERS_MODEL_SPEC)
CLS_TOKEN = TRANSFORMERS_TOKENIZER.cls_token
SEP_TOKEN = TRANSFORMERS_TOKENIZER.sep_token
PAD_IDX = TRANSFORMERS_TOKENIZER.pad_token_id
NEW_WORD_PREFIX = chr(288)

WEIRD_NON_ASCII_SEQUENCE = 'ï¿½'
WEIRD_NON_ASCII_SEQUENCE_PLACE_HOLDER = chr(0)

#############
# Load Data #
#############

class  TweetSentimentSelectionDataset(data.Dataset):
    def __init__(self, indices: Iterable[int], x: pd.Series, y: pd.Series):
        self.x = [x.iloc[index] for index in indices]
        self.y = [y.iloc[index] for index in indices]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)

def normalize_text(input_string: str) -> str:
    input_string_normalized = ' '+' '.join(input_string.split())
    input_string_normalized = input_string_normalized.replace(WEIRD_NON_ASCII_SEQUENCE, WEIRD_NON_ASCII_SEQUENCE_PLACE_HOLDER)
    return input_string_normalized

def denormalize_text(input_string: str) -> str:
    input_string_denormalized = input_string_normalized.replace(WEIRD_NON_ASCII_SEQUENCE_PLACE_HOLDER, WEIRD_NON_ASCII_SEQUENCE)
    return input_string_denormalized

def model_input_from_row(text: str, sentiment: str) -> torch.LongTensor:
    assert WEIRD_NON_ASCII_SEQUENCE_PLACE_HOLDER not in text
    text_normalized = normalize_text(text)
    ids = TRANSFORMERS_TOKENIZER.encode(text_normalized, sentiment)
    id_tensor = torch.LongTensor(ids)
    return id_tensor

def _normalize_selected_text(text: str, selected_text: str) -> str:
    selected_text_start_position = text.find(selected_text.strip())
    selected_text_end_position = selected_text_start_position + len(selected_text)
    assert selected_text_start_position >= 0
    assert selected_text_end_position <= len(text)
    while selected_text_end_position != len(text) and text[selected_text_end_position] != ' ':
        selected_text_end_position += 1
    while selected_text_start_position != 0 and text[selected_text_start_position-1] != ' ':
        selected_text_start_position -= 1
    assert selected_text_start_position >= 0
    assert selected_text_end_position <= len(text)
    selected_text_normalized = text[selected_text_start_position:selected_text_end_position]
    selected_text_normalized = normalize_text(selected_text_normalized)
    selected_text_normalized = selected_text_normalized.strip()
    assert len(set(WEIRD_NON_ASCII_SEQUENCE).intersection(selected_text_normalized)) == 0
    return selected_text_normalized

def model_output_from_row(text: str, selected_text: str, sentiment: str) -> torch.FloatTensor:
    assert WEIRD_NON_ASCII_SEQUENCE_PLACE_HOLDER not in text+selected_text
    text_normalized = normalize_text(text)
    selected_text_normalized = _normalize_selected_text(text, selected_text)
    assert selected_text_normalized in text_normalized
    selected_text_start_position_in_text = text_normalized.find(selected_text_normalized)
    assert selected_text_start_position_in_text >=0
    selected_characters = [False] * len(text_normalized)
    for selected_text_position in range(selected_text_start_position_in_text, selected_text_start_position_in_text+len(selected_text_normalized)):
        selected_characters[selected_text_position] = True
    if text_normalized[selected_text_start_position_in_text-1] == ' ':
        selected_characters[selected_text_start_position_in_text-1] = True
    assert ''.join(eager_map(str, eager_map(int, uniq(selected_characters)))) in ['1','010','10','01']
    text_ids = TRANSFORMERS_TOKENIZER.encode(text_normalized)[1:-1]
    assert TRANSFORMERS_TOKENIZER.sep_token_id not in text_ids
    assert TRANSFORMERS_TOKENIZER.cls_token_id not in text_ids
    
    token_offsets: List[Tuple[int, int]] = []
    current_token_start_index = 0
    for text_id in text_ids:
        token = TRANSFORMERS_TOKENIZER.decode([text_id], clean_up_tokenization_spaces=False)
        start_and_end_index_pair = (current_token_start_index, current_token_start_index+len(token))
        assert text_normalized[start_and_end_index_pair[0]:start_and_end_index_pair[1]] == token
        token_offsets.append(start_and_end_index_pair)
        current_token_start_index += len(token)
    assert current_token_start_index == len(text_normalized)
    
    selected_token_indices: List[int] = []
    for token_index, (token_start_index, token_end_index) in enumerate(token_offsets):
        if any(selected_characters[token_start_index:token_end_index]):
            assert all(selected_characters[token_start_index:token_end_index])
            selected_token_indices.append(token_index)
    
    input_ids = TRANSFORMERS_TOKENIZER.encode(text_normalized, sentiment)
    assert input_ids[0] == TRANSFORMERS_TOKENIZER.cls_token_id
    assert input_ids[-4] == TRANSFORMERS_TOKENIZER.sep_token_id
    assert input_ids[-3] == TRANSFORMERS_TOKENIZER.sep_token_id
    assert input_ids[-1] == TRANSFORMERS_TOKENIZER.sep_token_id
    assert len(selected_token_indices) > 0
    
    output_tensor = torch.zeros([len(input_ids), 2])
    start_index = selected_token_indices[0]+1
    end_index = selected_token_indices[-1]+1
    output_tensor[start_index][0] = 1
    output_tensor[end_index][1] = 1
    
    assert selected_text_normalized.startswith(TRANSFORMERS_TOKENIZER.decode(input_ids[start_index], clean_up_tokenization_spaces=False)[1:])
    assert selected_text_normalized.endswith(TRANSFORMERS_TOKENIZER.decode(input_ids[end_index], clean_up_tokenization_spaces=False)[1:])
    
    return output_tensor

def pad_tensor(input_tensor: torch.Tensor, max_sequence_length: int) -> torch.Tensor:
    return torch.cat([input_tensor, torch.ones(max_sequence_length-len(input_tensor), dtype=int)*PAD_IDX])

def attention_mask_for_tensor_length(tensor_length: int, max_sequence_length: int) -> torch.Tensor:
    return torch.cat([torch.ones(tensor_length, dtype=int), torch.zeros(max_sequence_length-tensor_length, dtype=int)])

def collate(input_output_pairs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    input_tensors, output_tensors = zip(*input_output_pairs)
    assert len(input_tensors) == len(output_tensors)
    if __debug__:
        batch_size = len(input_tensors)
    assert {len(input_tensor.shape) for input_tensor in input_tensors} == {1}
    assert {len(output_tensor.shape) for output_tensor in output_tensors} == {2}
    max_sequence_length = max(input_tensor.shape[0] for input_tensor in input_tensors)
    assert max_sequence_length == max(output_tensor.shape[0] for output_tensor in output_tensors)
    assert all(input_tensor.shape[0]==output_tensor.shape[0] for input_tensor, output_tensor in zip(input_tensors, output_tensors))
    padded_input_tensors = [pad_tensor(input_tensor, max_sequence_length) for input_tensor in input_tensors]
    padded_input_tensor_batch = torch.stack(padded_input_tensors)
    attention_mask_tensors = [attention_mask_for_tensor_length(len(input_tensor), max_sequence_length) for input_tensor in input_tensors]
    attention_mask_batch = torch.stack(attention_mask_tensors)

    assert attention_mask_batch.sum() == sum(len(input_tensor) for input_tensor in input_tensors)
    assert tuple(padded_input_tensor_batch.shape) == (batch_size, max_sequence_length)
    input_token_type_id_batch = torch.zeros([batch_size, max_sequence_length], dtype=int)
    assert input_token_type_id_batch.sum() == 0
    padded_output_tensors = [torch.cat([output_tensor, torch.zeros([max_sequence_length-output_tensor.shape[0],2])]) for output_tensor in output_tensors]
    padded_output_tensor_batch = torch.stack(padded_output_tensors)
    assert tuple(padded_output_tensor_batch.shape) == (batch_size, max_sequence_length , 2)
    return padded_input_tensor_batch, attention_mask_batch, input_token_type_id_batch, padded_output_tensor_batch

##############
# Predictors #
##############

class BERTPredictor():
    def __init__(self, output_directory: str, number_of_epochs: int, batch_size: int, number_of_folds: int, gradient_clipping_threshold: float, initial_learning_rate: float, model_initializer: Callable, loss_function: Callable, optimizer_initializer: Callable):
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.number_of_folds = number_of_folds
        self.cross_validator = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=SEED)
        self.gradient_clipping_threshold = gradient_clipping_threshold
        self.initial_learning_rate = initial_learning_rate
        
        self.load_data()
        
        self.model_initializer = model_initializer
        self.loss_function = loss_function
        self.optimizer_initializer = optimizer_initializer
    
    def load_data(self) -> None:
        self.all_data_df = pd.read_csv(TRAINING_DATA_CSV_FILE)
        self.all_data_df = self.all_data_df[self.all_data_df['text'].notna()]
        print()
        print('Loading Input Data...')
        self.input_id_tensors = self.all_data_df[['text', 'sentiment']].progress_apply(lambda row: model_input_from_row(row[0], row[1]), axis=1)
        print()
        print('Loading Output Data...')
        self.output_tensors = self.all_data_df[['text', 'selected_text', 'sentiment']].progress_apply(lambda row: model_output_from_row(row[0], row[1], row[2]), axis=1)
        assert len(self.input_id_tensors) == len(self.output_tensors)
        assert eager_map(lambda x: x.shape[0], self.input_id_tensors) == eager_map(lambda y: y.shape[0], self.output_tensors)
        return

    def number_of_relevant_recent_epochs(self, training_data_loader: data.DataLoader) -> int:
        number_of_relevant_recent_epochs = number_of_relevant_recent_epochs_for_data_size_and_batch_size(len(training_data_loader), self.batch_size)
        return number_of_relevant_recent_epochs

    def best_saved_model_location_for_fold(self, fold_index: int) -> str:
        best_saved_model_location = os.path.join(self.output_directory, f'best_model_for_fold_{fold_index}')
        if not os.path.exists(best_saved_model_location):
            os.makedirs(best_saved_model_location)
        return best_saved_model_location
    
    def train_one_epoch(self, training_data_loader: data.DataLoader) -> Tuple[float, float]:
        epoch_loss = 0
        epoch_jaccard = 0
        for batch_index, (text_batch, attention_mask_batch, input_token_type_id_batch, labels) in tqdm_with_message(enumerate(training_data_loader),
                                                                                                                    post_yield_message_func = lambda index: f'Training Jaccard {epoch_jaccard/(index+1):.8f}',
                                                                                                                    total=len(training_data_loader)):
            self.optimizer.zero_grad()
            text_batch = text_batch.to(DEVICE)
            attention_mask_batch = attention_mask_batch.to(DEVICE)
            input_token_type_id_batch = input_token_type_id_batch.to(DEVICE)
            labels = labels.to(DEVICE)
            pre_softmax_labels = only_one(self.model.forward(input_ids=text_batch, attention_mask=attention_mask_batch, token_type_ids=input_token_type_id_batch))
            predicted_labels = F.softmax(pre_softmax_labels, dim=1)
            if __debug__:
                batch_size = text_batch.shape[0]
                max_sequence_length = text_batch.shape[1]
            assert tuple(predicted_labels.shape) == (batch_size, max_sequence_length, 2)
            loss = self.loss_function(predicted_labels, labels)
            epoch_loss += loss.item()
            epoch_jaccard += self.scores_of_discretized_values(predicted_labels, labels)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.gradient_clipping_threshold)
            self.optimizer.step()
        epoch_loss /= len(training_data_loader)
        epoch_jaccard /= len(training_data_loader)
        return epoch_loss, epoch_jaccard
    
    def evaluate(self, validation_data_loader: data.DataLoader) -> Tuple[float, float]:
        epoch_loss = 0
        epoch_jaccard = 0
        self.model.eval()
        with torch.no_grad():
            for text_batch, attention_mask_batch, input_token_type_id_batch, labels in tqdm_with_message(validation_data_loader,
                                                                                                         post_yield_message_func = lambda index: f'Validation Jaccard {epoch_jaccard/(index+1):.8f}',
                                                                                                         total=len(validation_data_loader)):
                text_batch = text_batch.to(DEVICE)
                attention_mask_batch = attention_mask_batch.to(DEVICE)
                input_token_type_id_batch = input_token_type_id_batch.to(DEVICE)
                labels = labels.to(DEVICE)
                pre_softmax_labels = only_one(self.model.forward(input_ids=text_batch, attention_mask=attention_mask_batch, token_type_ids=input_token_type_id_batch))
                predicted_labels = F.softmax(pre_softmax_labels, dim=1)
                if __debug__:
                    batch_size = text_batch.shape[0]
                    max_sequence_length = text_batch.shape[1]
                assert predicted_labels.shape[0] <= NON_TRAINING_BATCH_SIZE
                assert predicted_labels.shape[0] == batch_size
                assert predicted_labels.shape == labels.shape
                loss = self.loss_function(predicted_labels, labels)
                epoch_loss += loss.item()
                epoch_jaccard += self.scores_of_discretized_values(predicted_labels, labels)
        epoch_loss /= len(validation_data_loader)
        epoch_jaccard /= len(validation_data_loader)
        return epoch_loss, epoch_jaccard
    
    def _update_global_best_score(self, score_dict: dict) -> None:
        if not os.path.isfile(GLOBAL_BEST_MODEL_SCORE_JSON_FILE_LOCATION):
            log_current_model_as_best = True
        else:
            with open(GLOBAL_BEST_MODEL_SCORE_JSON_FILE_LOCATION, 'r') as current_global_best_model_score_json_file:
                current_global_best_model_score_dict = json.load(current_global_best_model_score_json_file)
                current_global_best_model_jaccard: float = current_global_best_model_score_dict['jaccard']
                log_current_model_as_best = current_global_best_model_jaccard < score_dict['jaccard']
        if log_current_model_as_best:
            with open(GLOBAL_BEST_MODEL_SCORE_JSON_FILE_LOCATION, 'w') as outfile:
                json.dump(score_dict, outfile)
        return
    
    def validate(self, fold_index: int, training_data_loader: data.DataLoader, validation_data_loader: data.DataLoader, epoch_index: int, result_is_from_final_run: bool) -> Tuple[float, float]:
        valid_loss, valid_jaccard = self.evaluate(validation_data_loader)
        if valid_jaccard > self.best_valid_jaccard_for_current_fold:
            self.best_valid_jaccard_for_current_fold = valid_jaccard
            best_saved_model_location = self.best_saved_model_location_for_fold(fold_index)
            self.save_parameters(best_saved_model_location)
            print(f'Best model so far saved to {best_saved_model_location}')
        self_score_dict = {
            'predictor_type': self.__class__.__name__,
            'fold_index': fold_index,
            'jaccard': valid_jaccard,
            'valid_jaccard': valid_jaccard,
            'valid_loss': valid_loss,
            'best_valid_jaccard': self.best_valid_jaccard_for_current_fold,
            'number_of_epochs': self.number_of_epochs,
            'most_recently_completed_epoch_index': epoch_index,
            'number_of_relevant_recent_epochs': self.number_of_relevant_recent_epochs(training_data_loader),
            'batch_size': self.batch_size,
            'number_of_folds': self.number_of_folds,
            'number_of_parameters': self.count_parameters(),
            'gradient_clipping_threshold': self.gradient_clipping_threshold,
            'output_directory': self.output_directory,
        }
        self._update_global_best_score(self_score_dict)
        with open(self.latest_model_score_location_for_fold(fold_index), 'w') as outfile:
            json.dump(self_score_dict, outfile)
        if result_is_from_final_run:
            with open(self.final_model_score_location_for_fold(fold_index), 'w') as outfile:
                json.dump(self_score_dict, outfile)
        return valid_loss, valid_jaccard
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def gradient_norm(self) -> float:
        return sum(torch.sum(p.grad.data**2).item() for p in self.model.parameters() if p.grad is not None) ** (1. / 2)
    
    def latest_model_score_location_for_fold(self, fold_index: int) -> str:
        return os.path.join(self.best_saved_model_location_for_fold(fold_index), 'latest_model_score.json')

    def final_model_score_location_for_fold(self, fold_index: int) -> str:
        return os.path.join(self.best_saved_model_location_for_fold(fold_index), FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME)
    
    def save_parameters(self, parameter_directory_location: str) -> None:
        self.model.save_pretrained(parameter_directory_location)
        return
    
    def load_parameters(self, parameter_directory_location: str) -> None:
        self.model.from_pretrained(parameter_directory_location)
        return
    
    def aggregate_score_over_all_folds(self, fold_index_to_splits: list) -> None:
        fold_index_to_validation_jaccard = [None] * self.number_of_folds
        for fold_index in range(self.number_of_folds):
            json_file_location_for_fold = self.final_model_score_location_for_fold(fold_index)
            with open(json_file_location_for_fold, 'r') as file_handle:
                fold_score_dict = json.load(file_handle)
                best_validation_jaccard_score = fold_score_dict['best_valid_jaccard']
                fold_index_to_validation_jaccard[fold_index] = best_validation_jaccard_score
        min_validation_jaccard_fold_index, min_validation_jaccard = min(((fold_index, validation_jaccard) for fold_index, validation_jaccard in enumerate(fold_index_to_validation_jaccard)), key = lambda x: x[1])
        max_validation_jaccard_fold_index, max_validation_jaccard = max(((fold_index, validation_jaccard) for fold_index, validation_jaccard in enumerate(fold_index_to_validation_jaccard)), key = lambda x: x[1])
        mean_validation_jaccard = mean(fold_index_to_validation_jaccard)
        aggregated_score_dict = {
            'min_validation_jaccard_fold_index': min_validation_jaccard_fold_index,
            'min_validation_jaccard': min_validation_jaccard,
            'max_validation_jaccard_fold_index': max_validation_jaccard_fold_index,
            'max_validation_jaccard': max_validation_jaccard,
            'mean_validation_jaccard': mean_validation_jaccard,
            'jaccard': mean_validation_jaccard,
            'predictor_type': self.__class__.__name__,
            'number_of_epochs': self.number_of_epochs,
            'batch_size': self.batch_size,
            'number_of_folds': self.number_of_folds,
            'number_of_parameters': self.count_parameters(),
            'gradient_clipping_threshold': self.gradient_clipping_threshold,
            'initial_learning_rate': self.initial_learning_rate,
            'output_directory': self.output_directory,

        }
        with open(os.path.join(self.output_directory, FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME), 'w') as outfile:
            json.dump(aggregated_score_dict, outfile)
        self._update_global_best_score(aggregated_score_dict)
        self.model.eval()
        training_evaluation_df = pd.read_csv(TRAINING_DATA_CSV_FILE)
        training_evaluation_df.text[training_evaluation_df.text != training_evaluation_df.text] = ''
        training_evaluation_df.selected_text[training_evaluation_df.selected_text != training_evaluation_df.selected_text] = ''
        training_evaluation_df = pd.concat([training_evaluation_df, pd.DataFrame(columns=['predicted_selected_text','jaccard', 'predicted_selected_text_start_index', 'predicted_selected_text_end_index', 'fold_index'])])
        def _start_and_end_word_indices(text: str, selected_text: str) -> Tuple[int, int]:
            start_index = None
            end_index = None
            text_words = text.split()
            selected_text_words = selected_text.split()
            for text_word_index, text_word in enumerate(text_words):
                possible_end_index = text_word_index + len(selected_text_words)
                if possible_end_index == len(text_words):
                    break
                if selected_text_words == text_words[text_word_index:possible_end_index]:
                    start_index = text_word_index
                    end_index = possible_end_index
                    break
            assert isinstance(start_index, int)
            assert isinstance(end_index, int)
            return start_index, end_index
        remainder_indices = set(training_evaluation_df.index) - reduce(set.union, (set(split[1]) for split in  fold_index_to_splits))
        assert len(remainder_indices) == len(training_evaluation_df) % self.number_of_folds
        for fold_index, (training_indices, validation_indices) in enumerate(fold_index_to_splits):
            if fold_index == self.number_of_folds-1:
                validation_indices = np.concatenate([validation_indices, list(remainder_indices)])
            self.load_parameters(self.best_saved_model_location_for_fold(fold_index))
            training_evaluation_df['fold_index'].iloc[validation_indices] = fold_index
            print()
            print(f'Predicting selected text for fold {fold_index} cross-validation.')
            training_evaluation_df['predicted_selected_text'].iloc[validation_indices] = training_evaluation_df[['text', 'sentiment']].iloc[validation_indices].progress_apply(lambda row: self.select_substring(row[0], row[1]), axis=1)
            assert not any(training_evaluation_df['predicted_selected_text'].iloc[validation_indices].isnull())
            print(f'Calculating Jaccard scores for fold {fold_index} cross-validation.')
            training_evaluation_df['jaccard'].iloc[validation_indices] = training_evaluation_df[['selected_text', 'predicted_selected_text']].iloc[validation_indices].progress_apply(lambda row: jaccard_index_over_strings(row[0], row[1]), axis=1)
            assert not any(training_evaluation_df['jaccard'].iloc[validation_indices].isnull())
            start_and_end_word_indices_series = training_evaluation_df[['text', 'predicted_selected_text']].iloc[validation_indices].progress_apply(lambda row: _start_and_end_word_indices(row[0], row[1]), axis=1)
            training_evaluation_df['predicted_selected_text_start_index'].iloc[validation_indices] = start_and_end_word_indices_series.progress_map(lambda pair: pair[0])
            training_evaluation_df['predicted_selected_text_end_index'].iloc[validation_indices] = start_and_end_word_indices_series.progress_map(lambda pair: pair[1])
        assert not any(training_evaluation_df['fold_index'].isnull())
        assert not any(training_evaluation_df['predicted_selected_text'].isnull())
        assert not any(training_evaluation_df['jaccard'].isnull())
        training_evaluation_df.to_csv(os.path.join(self.output_directory, CROSS_VALIDATION_RESULTS_CSV_FILE_LOCATION_BASE_NAME), index=False)
        print(f'Training set cross-validation Jaccard score is {training_evaluation_df.jaccard.mean()}.')
        return

    def generate_testing_data_predictions(self) -> None:
        self.model.eval()
        test_data_df = pd.read_csv(TESTING_DATA_CSV_FILE)
        test_data_df.text[test_data_df.text != test_data_df.text] = ''
        test_data_df.selected_text[test_data_df.selected_text != test_data_df.selected_text] = ''
        fold_columns = sum(
            [(f'start_index_fold_{fold_index}', f'start_index_score_fold_{fold_index}', f'end_index_fold_{fold_index}', f'end_index_score_fold_{fold_index}')
                 for fold_index in range(self.number_of_folds)],
            ())
        fold_columns += ['start_index', 'end_index', 'selected_text']
        test_data_df = pd.concat([test_data_df, pd.DataFrame(columns=fold_columns)])
        print()
        print('Predicting selected text for test data.')
        for fold_index in range(self.number_of_folds):
            self.load_parameters(self.best_saved_model_location_for_fold(fold_index))
            select_substring_tuple_series = test_data_df[['text', 'sentiment']].progress_apply(lambda row: self._select_substring(row[0], row[1]), axis=1)
            test_data_df[f'start_index_fold_{fold_index}'] = select_substring_tuple_series.progress_map(lambda tup: tup[1])
            test_data_df[f'start_index_score_fold_{fold_index}'] = select_substring_tuple_series.progress_map(lambda tup: tup[2])
            test_data_df[f'end_index_fold_{fold_index}'] = select_substring_tuple_series.progress_map(lambda tup: tup[3])
            test_data_df[f'end_index_score_fold_{fold_index}'] = select_substring_tuple_series.progress_map(lambda tup: tup[4])
        def _highest_scoring_start_and_end_indices(*args) -> Tuple[int, int]:
            assert divmod(len(args), 4) == (self.number_of_folds, 0)
            best_start_index = UNIQUE_BOGUS_RESULT_IDENTIFIER
            best_start_score = 0
            best_end_index = UNIQUE_BOGUS_RESULT_IDENTIFIER
            best_end_score = 0
            for fold_index in range(self.number_of_folds):
                start_index, start_index_score, end_index, end_index_score = args[fold_index*4:(fold_index+1)*4]
                if start_index_score > best_start_score:
                    best_start_index = start_index
                if end_index_score > best_end_score:
                    best_end_index = end_index
            assert UNIQUE_BOGUS_RESULT_IDENTIFIER not in (best_start_index, best_end_index)
            return best_start_index, best_end_index
        start_and_end_indices_series = test_data_df[fold_columns].progress_apply(lambda row: _highest_scoring_start_and_end_indices(*row), axis=1)
        test_data_df['start_index'] = start_and_end_indices_series.progress_map(lambda pair: pair[0])
        test_data_df['end_index'] = start_and_end_indices_series.progress_map(lambda pair: pair[1])
        def _extract_selected_text_via_indices(text: str, sentiment: str, start_index: int, end_index: int) -> str:
            if end_index < start_index:
                return text
            normalized_text = normalize_text(input_string)
            encoded_normalized_text = TRANSFORMERS_TOKENIZER.encode(normalized_text, sentiment)
            selected_ids = encoded_normalized_text[start_index:end_index+1]                
            selected_text = denormalize_text(TRANSFORMERS_TOKENIZER.decode(selected_ids, clean_up_tokenization_spaces=False))
            return selected_text
        test_data_df['selected_text'] = test_data_df[['text', 'sentiment', 'start_index', 'end_index']].progress_apply(lambda row: _extract_selected_text_via_indices(row[0], row[1], row[2], row[3]), axis=1)
        assert not any(any(test_data_df[fold_column].isnull()) for fold_column in fold_columns)
        test_data_df.to_csv(os.path.join(self.output_directory, TESTING_RESULTS_CSV_FILE_LOCATION_BASE_NAME), index=False)
        test_data_df.to_csv(os.path.join(self.output_directory, SUBMISSION_CSV_FILE_LOCATION_BASE_NAME), index=False)
        return 
    
    def train(self, only_train_one_fold: bool = False) -> None:
        assert self.cross_validator.get_n_splits() == self.number_of_folds
        fold_index_to_splits: List[Tuple[np.ndarray, np.ndarray]] = list(self.cross_validator.split(self.all_data_df.index, self.all_data_df.sentiment))
        for fold_index, (training_indices, validation_indices) in enumerate(fold_index_to_splits):
            assert implies(only_train_one_fold, fold_index == 0)
            self.model = self.model_initializer()
            self.optimizer = self.optimizer_initializer(self.model)
            self.best_valid_jaccard_for_current_fold = -1
            training_dataset = TweetSentimentSelectionDataset(training_indices, self.input_id_tensors, self.output_tensors)
            validation_dataset = TweetSentimentSelectionDataset(validation_indices, self.input_id_tensors, self.output_tensors)
            training_data_loader = data.DataLoader(training_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=NUMBER_OF_DATALOADER_WORKERS, collate_fn=collate)
            validation_data_loader = data.DataLoader(validation_dataset, batch_size=NON_TRAINING_BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUMBER_OF_DATALOADER_WORKERS, collate_fn=collate)
            most_recent_validation_jaccard_scores = [0] * self.number_of_relevant_recent_epochs(training_data_loader)
            print()
            self.print_hyperparameters(training_data_loader)
            print(f'Starting training for fold {fold_index}')
            for epoch_index in range(self.number_of_epochs):
                print('\n')
                print(f'Epoch {epoch_index}')
                with timer(section_name=f'Epoch {epoch_index}'):
                    train_loss, train_jaccard = self.train_one_epoch(training_data_loader)
                    valid_loss, valid_jaccard = self.validate(fold_index, training_data_loader, validation_data_loader, epoch_index, False)
                    print(f'\t   Training Jaccard: {train_jaccard:.8f} |   Training Loss: {train_loss:.8f}')
                    print(f'\t Validation Jaccard: {valid_jaccard:.8f} | Validation Loss: {valid_loss:.8f}')
                print('\n')
                if not jaccard_sufficiently_high_for_epoch(valid_jaccard, epoch_index):
                    print()
                    print(f'Validation is not sufficiently high for the number of epochs passed, so training is ending early due to poor performance.')
                    print()
                    break
                elif any(valid_jaccard > previous_jaccard for previous_jaccard in most_recent_validation_jaccard_scores):
                    most_recent_validation_jaccard_scores.pop(0)
                    most_recent_validation_jaccard_scores.append(valid_jaccard)
                else:
                    print()
                    print(f'Validation is not better than any of the {self.number_of_relevant_recent_epochs(training_data_loader)} recent epochs, so training is ending early due to apparent convergence.')
                    print()
                    break
            if not only_train_one_fold:
                self.load_parameters(self.best_saved_model_location_for_fold(fold_index))
                self.validate(fold_index, training_data_loader, validation_data_loader, epoch_index, True)
            else: 
                break
        if not only_train_one_fold:
            self.aggregate_score_over_all_folds(fold_index_to_splits)
        return
    
    def scores_of_discretized_values(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        epsilon = 1e-5
        assert (y-y.int()).sum()==0
        assert y_hat.shape == y.shape
        batch_size, max_sequence_length, number_of_token_classes = y.shape
        assert number_of_token_classes == 2
        y_hat_start_and_end_indices = y_hat.detach().argmax(dim=1)
        y_hat_start_indices = y_hat_start_and_end_indices[:,0]
        y_hat_end_indices = y_hat_start_and_end_indices[:,1]
        assert tuple(y_hat_start_and_end_indices.shape) == (batch_size, number_of_token_classes)
        assert tuple(y_hat_start_indices.shape) == (batch_size,)
        assert tuple(y_hat_end_indices.shape) == (batch_size,)
        y_start_and_end_indices = y.argmax(dim=1)
        y_start_indices = y_start_and_end_indices[:,0]
        y_end_indices = y_start_and_end_indices[:,1]
        assert tuple(y_start_and_end_indices.shape) == (batch_size, number_of_token_classes)
        assert tuple(y_start_indices.shape) == (batch_size,)
        assert tuple(y_end_indices.shape) == (batch_size,)
        y_hat_start_and_end_indices_sane_wrt_each_other = (y_hat_start_indices < y_hat_end_indices).int()
        y_hat_start_index_sane_wrt_y_end_index = (y_hat_start_indices < y_end_indices).int()
        y_hat_end_index_sane_wrt_y_start_index =  (y_hat_end_indices > y_start_indices).int()
        y_hat_values_sane = y_hat_start_and_end_indices_sane_wrt_each_other * y_hat_start_index_sane_wrt_y_end_index * y_hat_end_index_sane_wrt_y_start_index
        assert tuple(y_hat_start_and_end_indices_sane_wrt_each_other.shape) == (batch_size,)
        assert tuple(y_hat_start_index_sane_wrt_y_end_index.shape) == (batch_size,)
        assert tuple(y_hat_end_index_sane_wrt_y_start_index.shape) == (batch_size,)
        assert tuple(y_hat_values_sane.shape) == (batch_size,)
        y_selected_text_lengths = (y_end_indices - y_start_indices)
        y_selected_text_lengths = y_selected_text_lengths + 1
        y_hat_selected_text_lengths = (y_hat_end_indices - y_hat_start_indices) * y_hat_values_sane
        y_hat_selected_text_lengths = y_hat_selected_text_lengths + 1
        assert tuple(y_selected_text_lengths.shape) == (batch_size,)
        assert all(y_selected_text_lengths > 0)
        assert tuple(y_hat_selected_text_lengths.shape) == (batch_size,)
        assert all(y_hat_selected_text_lengths >= 0)
        intersection_count_lost_via_y_hat_start_indices = y_start_indices-y_hat_start_indices
        intersection_count_lost_via_y_hat_start_indices = intersection_count_lost_via_y_hat_start_indices * (intersection_count_lost_via_y_hat_start_indices < 0).int()
        intersection_count_lost_via_y_hat_start_indices = intersection_count_lost_via_y_hat_start_indices * y_hat_values_sane
        intersection_count_lost_via_y_hat_start_indices = intersection_count_lost_via_y_hat_start_indices.abs()
        intersection_count_lost_via_y_hat_end_indices = y_end_indices-y_hat_end_indices
        intersection_count_lost_via_y_hat_end_indices = intersection_count_lost_via_y_hat_end_indices * (intersection_count_lost_via_y_hat_end_indices > 0).int()
        intersection_count_lost_via_y_hat_end_indices = intersection_count_lost_via_y_hat_end_indices * y_hat_values_sane
        assert tuple(intersection_count_lost_via_y_hat_start_indices.shape) == (batch_size,)
        assert tuple(intersection_count_lost_via_y_hat_end_indices.shape) == (batch_size,)
        assert all(intersection_count_lost_via_y_hat_start_indices>=0)
        assert all(intersection_count_lost_via_y_hat_end_indices>=0)
        intersection_count = (y_selected_text_lengths * y_hat_values_sane) - intersection_count_lost_via_y_hat_start_indices - intersection_count_lost_via_y_hat_end_indices
        intersection_count = intersection_count * (intersection_count > 0).int()
        assert tuple(intersection_count.shape) == (batch_size,)
        assert all(intersection_count>=0)
        union_count = y_selected_text_lengths + y_hat_selected_text_lengths - intersection_count
        assert tuple(union_count.shape) == (batch_size,)
        jaccard_index = intersection_count / (union_count + epsilon)
        assert tuple(jaccard_index.shape) == (batch_size,)
        mean_jaccard_index = jaccard_index.mean().item()
        assert isinstance(mean_jaccard_index, float)
        assert mean_jaccard_index == 0.0 or 0.0 not in (intersection_count.sum(), union_count.sum())
        return mean_jaccard_index

    def print_hyperparameters(self, training_data_loader: data.DataLoader) -> None:
        print()
        print(f'Model hyperparameters are:')
        print(f'        predictor_type: {self.__class__.__name__}')
        print(f'        number_of_epochs: {self.number_of_epochs}')
        print(f'        gradient_clipping_threshold: {self.gradient_clipping_threshold}')
        print(f'        initial_learning_rate: {self.initial_learning_rate}')
        print(f'        number_of_relevant_recent_epochs: {self.number_of_relevant_recent_epochs(training_data_loader)}')
        print(f'        batch_size: {self.batch_size}')
        print(f'        output_directory: {self.output_directory}')
        print()
        print(f'The model has {self.count_parameters():,} trainable parameters.')
        print(f'This processes has PID {os.getpid()}.')
        if DEVICE.type == 'cuda':
            print(f'The CUDA device being used is {torch.cuda.get_device_name(DEVICE_ID)}')
            print(f'The CUDA device ID being used is {DEVICE_ID}')
        print()
        return

    def _select_substring(self, input_string: str, sentiment: str) -> Tuple[str, int, int, int, int]:
        assert sentiment in SENTIMENTS
        self.model.eval()
        id_tensor = model_input_from_row(input_string, sentiment)
        batch_size = 1
        id_tensor_length = id_tensor.shape[0]
        padded_id_tensor = pad_tensor(id_tensor, id_tensor_length)
        padded_id_tensor_batch = padded_id_tensor.unsqueeze(0)
        padded_id_tensor_batch = padded_id_tensor_batch.to(self.model.device)
        assert tuple(padded_id_tensor_batch.shape) == (batch_size, id_tensor_length)
        attention_mask = attention_mask_for_tensor_length(id_tensor_length, id_tensor_length)
        attention_mask_batch = attention_mask.unsqueeze(0)
        attention_mask_batch = attention_mask_batch.to(self.model.device)
        assert tuple(attention_mask_batch.shape) == (batch_size, id_tensor_length)
        assert attention_mask_batch.sum().item() == id_tensor_length
        input_token_type_id_batch = torch.zeros([batch_size, id_tensor_length], dtype=int).to(self.model.device)
        assert tuple(input_token_type_id_batch.shape) == (batch_size, id_tensor_length)
        assert input_token_type_id_batch.sum().item() == 0 
        pre_softmax_labels = only_one(self.model.forward(input_ids=padded_id_tensor_batch, attention_mask=attention_mask_batch, token_type_ids=input_token_type_id_batch))
        assert tuple(pre_softmax_labels.shape) == (batch_size, id_tensor_length, 2)
        predicted_labels = F.softmax(pre_softmax_labels, dim=1)
        assert tuple(predicted_labels.shape) == (batch_size, id_tensor_length, 2)
        predicted_label = predicted_labels.squeeze(0)
        assert tuple(predicted_label.shape) == (id_tensor_length, 2)
        normalized_text = normalize_text(input_string)
        encoded_normalized_text = TRANSFORMERS_TOKENIZER.encode(normalized_text)
        assert encoded_normalized_text[0] == TRANSFORMERS_TOKENIZER.cls_token_id
        assert encoded_normalized_text[-1] == TRANSFORMERS_TOKENIZER.sep_token_id
        start_score, start_index = torch.max(predicted_label[:,0], dim=0, out=None)
        end_score, end_index = torch.max(predicted_label[:,1], dim=0, out=None)
        selected_ids = encoded_normalized_text[start_index:end_index+1]        
        selected_text = denormalize_text(TRANSFORMERS_TOKENIZER.decode(selected_ids, clean_up_tokenization_spaces=False))
        assert selected_text in normalized_text, f'{repr(selected_text)} not in {repr(normalized_text)}'
        return selected_text, start_index, start_score, end_index, end_score
    
    def select_substring(self, input_string: str, sentiment: str) -> str:
        return self._select_substring(input_string, sentiment)[0]
    
    def _evaluate_example(self, example_input_string: str, example_selected_text: str, example_sentiment: str) -> Tuple[str, float]:
        predicted_substring = self.select_substring(example_input_string, example_sentiment)
        jaccard_score = jaccard_index_over_strings(example_selected_text, predicted_substring)
        return predicted_substring, jaccard_score

    def _random_example_for_sentiment(self, sentiment: str) -> Tuple[str, str, int]:
        assert sentiment in SENTIMENTS
        example_for_sentiment_found = False
        for _ in range(len(self.all_data_df)):
            example_index = random.randrange(len(self.all_data_df))
            example = self.all_data_df.iloc[example_index]
            example_input_string = example.text
            example_selected_text = example.selected_text
            example_sentiment = example.sentiment
            assert example_sentiment in SENTIMENTS
            if example_sentiment == sentiment:
                example_for_sentiment_found = True
                break
        if not example_for_sentiment_found:
            raise Exception(f"Could not find relevant {sentiment} example sufficiently quickly.")
        return example_input_string, example_selected_text, example_index

    def demonstrate_examples(self) -> None:
        print()
        approximate_number_of_examples_per_sentiment = math.ceil(NUMBER_OF_EXAMPLES_TO_DEMONSTRATE/len(SENTIMENTS))
        sentiment_to_sentiment_example_count = {sentiment: approximate_number_of_examples_per_sentiment for sentiment in SENTIMENTS}
        sentiment_to_sentiment_example_count['neutral'] = NUMBER_OF_EXAMPLES_TO_DEMONSTRATE - approximate_number_of_examples_per_sentiment*(len(SENTIMENTS)-1)
        for sentiment in SENTIMENTS:
            sentiment_example_count = sentiment_to_sentiment_example_count[sentiment]
            print('\n'*2)
            print(f'Examples for {sentiment} tweets.')
            print()
            for sentiment_example_index in range(sentiment_example_count):
                show_good_example = sentiment_example_index >= sentiment_example_count // 2
                show_bad_example = not show_good_example
                for _ in range(len(self.all_data_df)):
                    text, selected_text, example_index = self._random_example_for_sentiment(sentiment)
                    predicted_substring, jaccard_score = self._evaluate_example(text, selected_text, sentiment)
                    example_fits_score_quality = (show_good_example and jaccard_score > JACCARD_INDEX_GOOD_SCORE_THRESHOLD) or (show_bad_example and jaccard_score < JACCARD_INDEX_GOOD_SCORE_THRESHOLD)
                    if example_fits_score_quality:
                        break
                print(f'example_index       {repr(example_index)}')
                print(f'sentiment           {repr(sentiment)}')
                print(f'text                {repr(text)}')
                print(f'predicted_substring {repr(predicted_substring)}')
                print(f'true_substring      {repr(selected_text)}')
                print(f'jaccard_score       {repr(jaccard_score)}')
                print()
                assert example_fits_score_quality
        return

class RoBERTaPredictor(BERTPredictor):
    def __init__(self, output_directory: str, number_of_epochs: int, batch_size: int, number_of_folds: int, gradient_clipping_threshold: float, initial_learning_rate: float):
        transformers_model_config = RobertaConfig.from_pretrained(TRANSFORMERS_MODEL_SPEC)
        model_initializer = lambda: RobertaForTokenClassification(transformers_model_config).to(DEVICE)
        loss_function = nn.BCELoss()
        optimizer_initializer = lambda model: optim.Adam(params=model.parameters(), lr=initial_learning_rate)
        super().__init__(output_directory, number_of_epochs, batch_size, number_of_folds, gradient_clipping_threshold, initial_learning_rate, model_initializer, loss_function, optimizer_initializer)

###############
# Main Driver #
###############

@debug_on_error
def train_model() -> None:
    predictor = RoBERTaPredictor(OUTPUT_DIR, NUMBER_OF_EPOCHS, BATCH_SIZE, NUMBER_OF_FOLDS, GRADIENT_CLIPPING_THRESHOLD, INITIAL_LEARNING_RATE)
    predictor.train()
    for fold_index in range(predictor.number_of_folds):
        print()
        print(f'Demonstrating examples via model for fold {fold_index}')
        predictor.load_parameters(predictor.best_saved_model_location_for_fold(fold_index))
        predictor.demonstrate_examples()
    return 

if __name__ == '__main__':
    train_model()
