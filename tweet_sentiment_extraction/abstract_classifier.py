#!/usr/bin/python3
'#!/usr/bin/python3 -OO'

"""
"""

# @todo fill this in
# @todo verify all the imports are used here and elsewhere

###########
# Imports #
###########

import random
import json
import os
import math
from statistics import mean
from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Tuple, Set, Callable, Iterable

import preprocess_data
from misc_utilities import *

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext import data

################################################
# Misc. Globals & Global State Initializations #
################################################

SEED = 1234
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = __debug__
torch.backends.cudnn.benchmark = not __debug__

NUMBER_OF_RELEVANT_RECENT_EPOCHS = 5
GOAL_NUMBER_OF_OVERSAMPLED_DATAPOINTS = 0
PORTION_OF_WORDS_TO_CROP_TO_UNK_FOR_DATA_AUGMENTATION = 0.30

SENTIMENTS = ['positive', 'neutral', 'negative']

####################
# Helper Utilities #
####################

def tensor_has_nan(tensor: torch.Tensor) -> bool:
    return (tensor != tensor).any().item()

def _safe_count_tensor_division(dividend: torch.Tensor, divisor: torch.Tensor) -> torch.Tensor:
    safe_divisor = divisor + (~(divisor.bool())).float()
    answer = dividend / safe_divisor
    assert not tensor_has_nan(answer)
    return answer

# def soft_f1_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     batch_size, output_size = tuple(y.shape)
#     assert tuple(y.shape) == (batch_size, output_size)
#     assert tuple(y_hat.shape) == (batch_size, output_size)
#     true_positive_sum = (y_hat * y.float()).sum(dim=0)
#     false_positive_sum = (y_hat * (1-y.float())).sum(dim=0)
#     false_negative_sum = ((1-y_hat) * y.float()).sum(dim=0)
#     soft_recall = true_positive_sum / (true_positive_sum + false_negative_sum + 1e-16)
#     soft_precision = true_positive_sum / (true_positive_sum + false_positive_sum + 1e-16)
#     soft_f1 = 2 * soft_precision * soft_recall / (soft_precision + soft_recall + 1e-16)
#     mean_soft_f1 = torch.mean(soft_f1)
#     loss = 1-mean_soft_f1
#     assert not tensor_has_nan(loss)
#     return loss

class BatchIterator:
    def __init__(self, non_numericalized_iterator: Iterable, x_attribute_name: str, y_attribute_name: str):
        self.non_numericalized_iterator = non_numericalized_iterator
        self.x_attribute_name: str = x_attribute_name
        self.y_attribute_name: str = y_attribute_name
        
    def __iter__(self):
        for non_numericalized_batch in self.non_numericalized_iterator:
            x = getattr(non_numericalized_batch, self.x_attribute_name)
            y = getattr(non_numericalized_batch, self.y_attribute_name)
            yield (x, y)
            
    def __len__(self):
        return len(self.non_numericalized_iterator)

#######################
# Abstract Predictor #
#######################

class Predictor(ABC):
    def __init__(self, output_directory: str, number_of_epochs: int, batch_size: int, train_portion: float, validation_portion: float, max_vocab_size: int, pre_trained_embedding_specification: str, **kwargs):
        super().__init__()
        self.best_valid_loss = float('inf')
        
        self.model_args = kwargs
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.max_vocab_size = max_vocab_size
        self.pre_trained_embedding_specification = pre_trained_embedding_specification
        
        self.train_portion = train_portion
        self.validation_portion = validation_portion
        
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        
        self.load_data()
        self.reset_jaccard_threshold()
        self.initialize_model()
        
    def load_data(self):
        self.text_field = data.Field(include_lengths=True, batch_first=True)
        label_preprocessing_pipeline = data.Pipeline(convert_token=lambda x: torch.tensor(eager_map(int, x)))
        def pad_and_batch(original_batch: List[torch.Tensor]) -> torch.Tensor:
            batch_size: int = len(original_batch)
            max_len: int = max(map(len, original_batch))
            assert batch_size == self.batch_size
            batch = torch.zeros((batch_size, max_len))
            for tensor_index, single_tensor in enumerate(original_batch):
                batch[tensor_index,:len(single_tensor)] = single_tensor
            return batch
        self.label_field = data.RawField(preprocessing=label_preprocessing_pipeline, postprocessing=pad_and_batch, is_target=True)
        self.sentiment_field = data.RawField()
        self.misc_field = data.RawField()
        self.all_data = data.dataset.TabularDataset(
            path=preprocess_data.PREPROCESSED_TRAINING_DATA_JSON_FILE,
            format='json',
            fields={
                "preprocessed_input_string": ("preprocessed_input_string", self.text_field),
                "numericalized_selected_text" : ("numericalized_selected_text", self.label_field),
                "sentiment": ("sentiment", self.sentiment_field),
                
                "textID": ("text_id", self.misc_field),
                "text": ("text", self.misc_field),
                "selected_text": ("selected_text", self.misc_field),
                "token_index_to_position_info_map": ("token_index_to_position_info_map", self.misc_field),
            })
        self.training_data, self.validation_data = self.all_data.split(split_ratio=[self.train_portion, self.validation_portion], random_state = random.seed(SEED))
        self.embedding_size = torchtext.vocab.pretrained_aliases[self.pre_trained_embedding_specification]().dim
        self.text_field.build_vocab(self.training_data, max_size = self.max_vocab_size, vectors = self.pre_trained_embedding_specification, unk_init = torch.Tensor.normal_)
        assert self.text_field.vocab.vectors.shape[0] <= self.max_vocab_size+4
        assert self.text_field.vocab.vectors.shape[1] == self.embedding_size
        self.training_iterator, self.validation_iterator = data.BucketIterator.splits(
            (self.training_data, self.validation_data),
            batch_size = self.batch_size,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            shuffle=True,
            device = DEVICE)
        self.training_iterator = BatchIterator(self.training_iterator, 'preprocessed_input_string', 'numericalized_selected_text')
        self.validation_iterator = BatchIterator(self.validation_iterator, 'preprocessed_input_string', 'numericalized_selected_text')
        self.pad_idx = self.text_field.vocab.stoi[self.text_field.pad_token]
        self.unk_idx = self.text_field.vocab.stoi[self.text_field.unk_token]
        return
    
    def determine_training_unknown_words(self) -> None:
        pretrained_embedding_vectors = torchtext.vocab.pretrained_aliases[self.pre_trained_embedding_specification]()
        pretrained_embedding_vectors_unk_default_tensor = pretrained_embedding_vectors.unk_init(torch.Tensor(pretrained_embedding_vectors.dim))
        is_unk_token = lambda token: torch.all(pretrained_embedding_vectors[token] == pretrained_embedding_vectors_unk_default_tensor)
        tokens = reduce(set.union, (set(map(str,example.text)) for example in self.training_data))
        self.training_unk_words = set(eager_filter(is_unk_token, tokens))
        return
    
    @abstractmethod
    def initialize_model(self) -> None:
        pass
        
    def train_one_epoch(self) -> Tuple[float, float]:
        epoch_loss = 0
        epoch_jaccard = 0
        number_of_training_batches = len(self.training_iterator)
        self.model.train()
        for (text_batch, text_lengths), labels in tqdm_with_message(self.training_iterator, post_yield_message_func = lambda index: f'Training Jaccard {epoch_jaccard/(index+1):.8f}', total=number_of_training_batches, bar_format='{l_bar}{bar:50}{r_bar}'):
            text_batch = self.augment_training_data_sample(text_batch)
            self.optimizer.zero_grad()
            predictions = self.model(text_batch, text_lengths)
            loss = self.loss_function(predictions, labels)
            jaccard = self.scores_of_discretized_values(predictions, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_jaccard += jaccard
        epoch_loss /= number_of_training_batches
        epoch_jaccard /= number_of_training_batches
        return epoch_loss, epoch_jaccard
    
    def evaluate(self, iterator: Iterable) -> Tuple[float, float]:
        epoch_loss = 0
        epoch_jaccard = 0
        self.model.eval()
        self.optimize_jaccard_threshold()
        iterator_size = len(iterator)
        with torch.no_grad():
            for (text_batch, text_lengths), labels in tqdm_with_message(iterator, post_yield_message_func = lambda index: f'Validation Jaccard {epoch_jaccard/(index+1):.8f}', total=iterator_size, bar_format='{l_bar}{bar:50}{r_bar}'):
                predictions = self.model(text_batch, text_lengths).squeeze(1)
                loss = self.loss_function(predictions, labels)
                jaccard = self.scores_of_discretized_values(predictions, labels)
                epoch_loss += loss.item()
                epoch_jaccard += jaccard
        self.reset_jaccard_threshold()
        epoch_loss /= iterator_size
        epoch_jaccard /= iterator_size
        return epoch_loss, epoch_jaccard
    
    def validate(self, epoch_index: int, result_is_from_final_run: bool) -> Tuple[float, float]:
        valid_loss, valid_jaccard = self.evaluate(self.validation_iterator, True)
        if not os.path.isfile('global_best_model_score.json'):
            log_current_model_as_best = True
        else:
            with open('global_best_model_score.json', 'r') as current_global_best_model_score_json_file:
                current_global_best_model_score_dict = json.load(current_global_best_model_score_json_file)
                current_global_best_model_jaccard: float = current_global_best_model_score_dict['valid_jaccard']
                log_current_model_as_best = current_global_best_model_jaccard < valid_jaccard
        self_score_dict = {
            'valid_jaccard': valid_jaccard,
            'valid_loss': valid_loss,
            'best_valid_loss': self.best_valid_loss,
            'number_of_epochs': self.number_of_epochs,
            'most_recently_completed_epoch_index': epoch_index,
            'batch_size': self.batch_size,
            'max_vocab_size': self.max_vocab_size,
            'vocab_size': self.vocab_size, 
            'pre_trained_embedding_specification': self.pre_trained_embedding_specification,
            'train_portion': self.train_portion,
            'validation_portion': self.validation_portion,
            'number_of_parameters': self.count_parameters(),
        }
        self_score_dict.update({(key, value.__name__ if hasattr(value, '__name__') else str(value)) for key, value in self.model_args.items()})
        if log_current_model_as_best:
            with open('global_best_model_score.json', 'w') as outfile:
                json.dump(self_score_dict, outfile)
        with open(self.latest_model_score_location, 'w') as outfile:
            json.dump(self_score_dict, outfile)
        if result_is_from_final_run:
            with open(self.final_model_score_location, 'w') as outfile:
                json.dump(self_score_dict, outfile)
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.save_parameters(self.best_saved_model_location)
        return valid_loss, valid_jaccard

    @property
    def vocab_size(self) -> str:
        return len(self.text_field.vocab)
    
    @property
    def best_saved_model_location(self) -> str:
        return os.path.join(self.output_directory, 'best-model.pt')
    
    @property
    def latest_model_score_location(self) -> str:
        return os.path.join(self.output_directory, 'latest_model_score.json')
    
    @property
    def final_model_score_location(self) -> str:
        return os.path.join(self.output_directory, 'final_model_score.json')
    
    def train(self) -> None:
        self.print_hyperparameters()
        most_recent_validation_jaccard_scores = [0]*NUMBER_OF_RELEVANT_RECENT_EPOCHS
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
            if reduce(bool.__or__, (valid_jaccard > previous_jaccard for previous_jaccard in most_recent_validation_jaccard_scores)):
                most_recent_validation_jaccard_scores.pop(0)
                most_recent_validation_jaccard_scores.append(valid_jaccard)
            else:
                print()
                print(f"Validation is not better than any of the {NUMBER_OF_RELEVANT_RECENT_EPOCHS} recent epochs, so training is ending early due to apparent convergence.")
                print()
                break
        self.load_parameters(self.best_saved_model_location)
        self.validate(epoch_index, True)
        return
    
    def print_hyperparameters(self) -> None:
        print()
        print(f"Model hyperparameters are:")
        print(f'        number_of_epochs: {self.number_of_epochs}')
        print(f'        batch_size: {self.batch_size}')
        print(f'        max_vocab_size: {self.max_vocab_size}')
        print(f'        vocab_size: {self.vocab_size}')
        print(f'        pre_trained_embedding_specification: {self.pre_trained_embedding_specification}')
        print(f'        output_directory: {self.output_directory}')
        for model_arg_name, model_arg_value in sorted(self.model_args.items()):
            print(f'        {model_arg_name}: {model_arg_value.__name__ if hasattr(model_arg_value, "__name__") else str(model_arg_value)}')
        print()
        print(f'The model has {self.count_parameters()} trainable parameters.')
        print(f"This processes's PID is {os.getpid()}.")
        print()
    
    def save_parameters(self, parameter_file_location: str) -> None:
        torch.save(self.model.state_dict(), parameter_file_location)
        return
    
    def load_parameters(self, parameter_file_location: str) -> None:
        self.model.load_state_dict(torch.load(parameter_file_location))
        return
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def optimize_jaccard_threshold(self) -> None:
        self.model.eval()
        with torch.no_grad():
            number_of_training_batches = len(self.training_iterator)
            training_sum_of_positives = 0.0
            training_sum_of_negatives = 0.0
            training_count_of_positives = 0.0
            training_count_of_negatives = 0.0
            for (text_batch, text_lengths), labels in tqdm_with_message(self.training_iterator, post_yield_message_func = lambda index: f'Optimizing Jaccard Threshold', total=number_of_training_batches, bar_format='{l_bar}{bar:50}{r_bar}'):
                predictions = self.model(text_batch, text_lengths)
                if __debug__:
                    batch_size = len(text_lengths)
                    max_sequence_length = text_lengths.max().item()
                assert tuple(predictions.data.shape) == (batch_size, max_sequence_length)
                assert tuple(labels.shape) == (batch_size, max_sequence_length)
                training_sum_of_positives += (predictions.data * labels).sum(dim=0).item()
                training_count_of_positives += labels.sum(dim=0).item()
                training_sum_of_negatives += (predictions.data * (1-labels)).sum(dim=0).item()
                training_count_of_negatives += (1-labels).sum(dim=0).item()
                assert training_sum_of_positives == training_sum_of_positives
                assert training_sum_of_negatives == training_sum_of_negatives
                assert training_count_of_positives == training_count_of_positives
                assert training_count_of_negatives == training_count_of_negatives
            assert 0 < training_sum_of_positives
            assert 0 < training_count_of_positives
            assert 0 < training_sum_of_negatives
            assert 0 < training_count_of_negatives
            training_mean_of_positives = training_sum_of_positives / training_count_of_positives
            training_mean_of_negatives = training_sum_of_negatives / training_count_of_negatives
            assert training_mean_of_positives == training_mean_of_positives
            assert training_mean_of_negatives == training_mean_of_negatives
            self.jaccard_threshold = (training_mean_of_positives + training_mean_of_negatives) / 2.0
            assert self.jaccard_threshold == self.jaccard_threshold
        return
    
    def reset_jaccard_threshold(self) -> None:
        if 'jaccard_threshold' in vars(self):
            self.last_jaccard_threshold = self.jaccard_threshold
        self.jaccard_threshold = 0.5
        return 
    
    def scores_of_discretized_values(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        assert y_hat.shape == y.shape
        batch_size, max_sequence_length = y.shape
        assert batch_size <= self.batch_size
        y_hat_discretized = (y_hat > self.jaccard_threshold)
        intersection_count = (y_hat_discretized & y).sum(dim=0)
        union_count = y_hat_discretized.sum(dim=0) + y.sum(dim=0) - intersection_count
        jaccard_index = _safe_count_tensor_division(intersection_count, union_count)
        assert tuple(intersection_count.shape) == (batch_size,)
        assert tuple(union_count.shape) == (batch_size,)
        assert tuple(jaccard_index.shape) == (batch_size,)
        mean_jaccard_index = jaccard_index.mean().item()
        assert isinstance(mean_jaccard_index, float)
        assert mean_jaccard_index == 0.0 or 0.0 not in (intersection_count.sum(), union_count.sum())
        return mean_jaccard_index
    
    def select_substring(self, input_string: str, sentiment: str) -> str:
        assert sentiment in SENTIMENTS
        self.model.eval()
        preprocessed_input_string, preprocessed_token_index_to_position_info_map = preprocess_data.preprocess_tweet(input_string)
        preprocessed_tokens = self.text_field.tokenize(preprocessed_input_string)
        indexed = [self.text_field.vocab.stoi[t] for t in preprocessed_tokens]
        lengths = [len(indexed)]
        input_string_tensor = torch.LongTensor(indexed).to(DEVICE)
        input_string_tensor = input_string_tensor.view(1,-1)
        length_tensor = torch.LongTensor(lengths).to(DEVICE)
        assert 'last_jaccard_threshold' in vars(self), "Model has not been trained yet and Jaccard threshold has not been optimized."
        threshold = self.last_jaccard_threshold
        predictions = self.model(input_string_tensor, length_tensor, sentiment)
        assert tuple(predictions.shape) == (1, len(indexed))
        prediction = predictions[0]
        discretized_prediction = prediction > threshold
        preprocessed_token_is_included_bools = eager_map(torch.Tensor.item, discretized_prediction)
        assert len(discretized_prediction) == len(tokenized)
        selected_tokens: List[str] = []
        for preprocessed_token_index, (preprocessed_token, preprocessed_token_is_included) in enumerate(zip(preprocessed_tokens, preprocessed_token_is_included_bools)):
            if preprocessed_token_is_included:
                position_info = preprocessed_token_index_to_position_info_map[str(preprocessed_token_index)]
                token_original_start_position = position_info['token_original_start_position']
                token_original_end_position = position_info['token_original_end_position']
                original_token = position_info['original_token']
                assert position_info['preprocessed_token'] == preprocessed_token
                selected_tokens.append(original_token)
        selected_substring = ' '.join(selected_tokens)
        return selected_substring

###############
# Main Driver #
###############

if __name__ == '__main__':
    print("This file contains the abstract class with which we wrap our torch models.")
