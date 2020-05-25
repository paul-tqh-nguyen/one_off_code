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
import pandas as pd
from typing import Tuple, Iterable, List, Any

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
from sklearn.model_selection import train_test_split
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
    def __init__(self, data_df: pd.DataFrame):
        self.data_df = data_df.reset_index()
        self.x = self.data_df[['text', 'sentiment']].progress_apply(lambda row: model_input_from_row(row[0], row[1]), axis=1)
        self.y = self.data_df[['text', 'selected_text', 'sentiment']].progress_apply(lambda row: model_output_from_row(row[0], row[1], row[2]), axis=1)
        assert len(self.x) == len(data_df)
        assert len(self.y) == len(data_df)
        assert eager_map(lambda x: x.shape[0], self.x) == eager_map(lambda y: y.shape[0], self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)

def normalize_text(input_string: str) -> str:
    input_string_normalized = ' '+' '.join(input_string.split()) # @todo do we need this space?
    return input_string_normalized

def model_input_from_row(text: str, sentiment: str) -> torch.LongTensor:
    text_normalized = normalize_text(text)
    ids = TRANSFORMERS_TOKENIZER.encode(text_normalized, sentiment)
    id_tensor = torch.LongTensor(ids)
    return id_tensor

def model_output_from_row(text: str, selected_text: str, sentiment: str) -> torch.FloatTensor:
    text_normalized = normalize_text(text)
    selected_text_normalized = ' '.join(selected_text.split())
    selected_text_start_position_in_text = text_normalized.find(selected_text_normalized)
    selected_characters = [False] * len(text_normalized)
    for selected_text_position in range(selected_text_start_position_in_text, selected_text_start_position_in_text+len(selected_text_normalized)):
        selected_characters[selected_text_position] = True
    if text_normalized[selected_text_start_position_in_text-1] == ' ':
        selected_characters[selected_text_start_position_in_text-1] = True
    text_ids = TRANSFORMERS_TOKENIZER.encode(text_normalized)
    
    token_offsets: List[Tuple[int, int]] = []
    current_token_start_index = 0
    for text_id in text_ids:
        token = TRANSFORMERS_TOKENIZER.decode([text_id])
        token_offsets.append((current_token_start_index, current_token_start_index+len(token)))
        current_token_start_index += len(token)
    
    selected_token_indices: List[int] = []
    for token_index, (token_start_index, token_end_index) in enumerate(token_offsets):
        if any(selected_characters[token_start_index:token_end_index]):
            selected_token_indices.append(token_index)
    
    sentiment_encoded = TRANSFORMERS_TOKENIZER.encode(sentiment)
    assert len(sentiment_encoded) == 3
    sentiment_id = sentiment_encoded[1]
    input_ids = text_ids + [2, sentiment_id, 2]
    assert input_ids == TRANSFORMERS_TOKENIZER.encode(text_normalized, sentiment)
    assert len(selected_token_indices) > 0
    
    output_tensor = torch.zeros([len(input_ids), 2])
    output_tensor[selected_token_indices[0]+1][0] = 1
    output_tensor[selected_token_indices[-1]+1][1] = 1
    
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
    def __init__(self, output_directory: str, number_of_epochs: int, batch_size: int, train_portion: int, validation_portion: int, gradient_clipping_threshold: float, model: nn.modules.module.Module, loss_function: Callable, optimizer: optim.Optimizer):
        self.best_valid_jaccard = -1
        
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        self.best_saved_model_location = os.path.join(self.output_directory, 'best_model')
        if not os.path.exists(self.best_saved_model_location):
            os.makedirs(self.best_saved_model_location)
        
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.train_portion = train_portion
        self.validation_portion = validation_portion
        self.gradient_clipping_threshold = gradient_clipping_threshold
        
        self.load_data()
        
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        assert sanity_check_model_forward_pass(self.model, self.validation_data_loader)
        
    def load_data(self) -> None:
        all_data_df = pd.read_csv(preprocess_data.TRAINING_DATA_CSV_FILE)
        all_data_df = all_data_df[all_data_df['text'].notna()]
        training_data_df, validation_data_df = train_test_split(all_data_df, test_size=VALIDATION_PORTION)
        print()
        print('Loading Training Data...')
        training_dataset = TweetSentimentSelectionDataset(training_data_df)
        print()
        print('Loading Validation Data...')
        validation_dataset = TweetSentimentSelectionDataset(validation_data_df)
        assert len(validation_dataset) == round(self.validation_portion*len(all_data_df))
        training_data_loader = data.DataLoader(training_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=NUMBER_OF_DATALOADER_WORKERS, collate_fn=collate)
        validation_data_loader = data.DataLoader(validation_dataset, batch_size=NON_TRAINING_BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUMBER_OF_DATALOADER_WORKERS, collate_fn=collate)
        self.training_data_loader, self.validation_data_loader = training_data_loader, validation_data_loader
        return
    
    @property
    def number_of_relevant_recent_epochs(self) -> int:
        number_of_relevant_recent_epochs = number_of_relevant_recent_epochs_for_data_size_and_batch_size(len(self.training_data_loader), self.batch_size)
        return number_of_relevant_recent_epochs
    
    def train_one_epoch(self) -> Tuple[float, float]:
        epoch_loss = 0
        epoch_jaccard = 0
        for batch_index, (text_batch, attention_mask_batch, input_token_type_id_batch, labels) in tqdm_with_message(enumerate(self.training_data_loader),
                                                                                                                    post_yield_message_func = lambda index: f'Training Jaccard {epoch_jaccard/(index+1):.8f}',
                                                                                                                    total=len(self.training_data_loader)):
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
        epoch_loss /= len(self.training_data_loader)
        epoch_jaccard /= len(self.training_data_loader)
        return epoch_loss, epoch_jaccard
    
    def evaluate(self) -> Tuple[float, float]:
        epoch_loss = 0
        epoch_jaccard = 0
        self.model.eval()
        with torch.no_grad():
            for text_batch, attention_mask_batch, input_token_type_id_batch, labels in tqdm_with_message(self.validation_data_loader,
                                                                                                         post_yield_message_func = lambda index: f'Validation Jaccard {epoch_jaccard/(index+1):.8f}',
                                                                                                         total=len(self.validation_data_loader)):
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
        if valid_jaccard > self.best_valid_jaccard:
            self.best_valid_jaccard = valid_jaccard
            self.save_parameters(self.best_saved_model_location)
            print(f'Best model so far saved to {self.best_saved_model_location}')
        self_score_dict = {
            'predictor_type': self.__class__.__name__,
            'valid_jaccard': valid_jaccard,
            'valid_loss': valid_loss,
            'best_valid_jaccard': self.best_valid_jaccard,
            'number_of_epochs': self.number_of_epochs,
            'most_recently_completed_epoch_index': epoch_index,
            'number_of_relevant_recent_epochs': self.number_of_relevant_recent_epochs,
            'batch_size': self.batch_size,
            'train_portion': self.train_portion,
            'validation_portion': self.validation_portion,
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
        return valid_loss, valid_jaccard
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def gradient_norm(self) -> float:
        return sum(torch.sum(p.grad.data**2).item() for p in self.model.parameters() if p.grad is not None) ** (1. / 2)
    
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
            print('\n')
            print(f'Epoch {epoch_index}')
            with timer(section_name=f'Epoch {epoch_index}'):
                train_loss, train_jaccard = self.train_one_epoch()
                valid_loss, valid_jaccard = self.validate(epoch_index, False)
                print(f'\t   Training Jaccard: {train_jaccard:.8f} |   Training Loss: {train_loss:.8f}')
                print(f'\t Validation Jaccard: {valid_jaccard:.8f} | Validation Loss: {valid_loss:.8f}')
            print('\n')
            if any(valid_jaccard > previous_jaccard for previous_jaccard in most_recent_validation_jaccard_scores):
                most_recent_validation_jaccard_scores.pop(0)
                most_recent_validation_jaccard_scores.append(valid_jaccard)
            else:
                print()
                print(f'Validation is not better than any of the {self.number_of_relevant_recent_epochs} recent epochs, so training is ending early due to apparent convergence.')
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

    def print_hyperparameters(self) -> None:
        print()
        print(f'Model hyperparameters are:')
        print(f'        predictor_type: {self.__class__.__name__}')
        print(f'        number_of_epochs: {self.number_of_epochs}')
        print(f'        number_of_relevant_recent_epochs: {self.number_of_relevant_recent_epochs}')
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
    
    def select_substring(self, input_string: str, sentiment: str) -> str:
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
        encoded_normalized_text = TRANSFORMERS_TOKENIZER.encode(normalized_text, sentiment)
        assert encoded_normalized_text[0] == TRANSFORMERS_TOKENIZER.cls_token_id
        assert encoded_normalized_text[-1] == TRANSFORMERS_TOKENIZER.sep_token_id
        assert encoded_normalized_text[-3] == TRANSFORMERS_TOKENIZER.sep_token_id
        assert encoded_normalized_text[-4] == TRANSFORMERS_TOKENIZER.sep_token_id
        start_index = predicted_label[:,0].argmax().item()
        end_index = predicted_label[:,1].argmax().item()
        selected_ids = encoded_normalized_text[start_index:end_index+1]
        
        # Post-Processing
        while TRANSFORMERS_TOKENIZER.cls_token_id in selected_ids:
            selected_ids.remove(TRANSFORMERS_TOKENIZER.cls_token_id)
        while TRANSFORMERS_TOKENIZER.sep_token_id in selected_ids:
            selected_ids.remove(TRANSFORMERS_TOKENIZER.sep_token_id)
        
        selected_text = TRANSFORMERS_TOKENIZER.decode(selected_ids, clean_up_tokenization_spaces=False)
        assert selected_text in normalized_text, f'{repr(selected_text)} not in {repr(normalized_text)}'
        return selected_text
    
    def _evaluate_example(self, example_input_string: str, example_selected_text: str, example_sentiment: str) -> Tuple[str, float]:
        predicted_substring = self.select_substring(example_input_string, example_sentiment)
        jaccard_score = jaccard_index_over_strings(example_selected_text, predicted_substring)
        return predicted_substring, jaccard_score

    def _random_example_for_sentiment(self, data: data.Dataset, sentiment: str) -> Tuple[str, str, int]:
        assert sentiment in SENTIMENTS
        example_for_sentiment_found = False
        for _ in range(len(data)):
            example_index = random.randrange(len(data))
            example = data.data_df.iloc[example_index]
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

    def _demonstrate_examples(self, data_set_spec: str) -> None:
        assert data_set_spec in ['training', 'validation']
        data = self.training_data_loader.dataset if data_set_spec == 'training' else self.validation_data_loader.dataset
        print('\n'*8)
        print(f'Here are some {data_set_spec} examples run through our model.')
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
                for _ in range(len(data)):
                    text, selected_text, example_index = self._random_example_for_sentiment(data, sentiment)
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

    def demonstrate_training_examples(self) -> None:
        self._demonstrate_examples('training')
        return
    
    def demonstrate_validation_examples(self) -> None:
        self._demonstrate_examples('validation')
        return

class RoBERTaPredictor(BERTPredictor):
    def __init__(self, output_directory: str, number_of_epochs: int, batch_size: int, train_portion: int, validation_portion: int, gradient_clipping_threshold: float, initial_learning_rate: float):
        transformers_model_config = RobertaConfig.from_pretrained(TRANSFORMERS_MODEL_SPEC)
        model = RobertaForTokenClassification(transformers_model_config).to(DEVICE)
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(params=model.parameters(), lr=initial_learning_rate)
        super().__init__(output_directory, number_of_epochs, batch_size, train_portion, validation_portion, gradient_clipping_threshold, model, loss_function, optimizer)

###############
# Main Driver #
###############

@debug_on_error
def train_model() -> None:
    predictor = RoBERTaPredictor(OUTPUT_DIR, NUMBER_OF_EPOCHS, BATCH_SIZE, TRAIN_PORTION, VALIDATION_PORTION, GRADIENT_CLIPPING_THRESHOLD, INITIAL_LEARNING_RATE)
    predictor.train()
    predictor.load_parameters(predictor.best_saved_model_location)
    # predictor.demonstrate_training_examples()
    predictor.demonstrate_validation_examples()
    return 

if __name__ == '__main__':
    train_model()
