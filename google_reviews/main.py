'#!/usr/bin/python3 -OO' # @todo make this the default

'''
'''
# @todo update doc string

###########
# Imports #
###########

import os
import json
import argparse
import operator
import itertools
import random
import multiprocessing as mp
import pandas as pd
from abc import ABC, abstractmethod
from functools import lru_cache
from pandarallel import pandarallel
from collections import OrderedDict
from typing import Tuple, Callable, Iterable, Generator
from typing_extensions import Literal

from misc_utilities import *

from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils import data

import transformers 
from transformers import AdamW, get_linear_schedule_with_warmup

# @todo make sure all of the imports are used

###########
# Globals #
###########

pandarallel.initialize(nb_workers=mp.cpu_count(), progress_bar=False, verbose=0)

# https://bit.ly/3ez5TOO
APPS_CSV_FILE = './data/apps.csv'
REVIEWS_CSV_FILE = './data/reviews.csv'

RESULTS_DIR = './results/'
RESULT_SUMMARY_JSON_FILE_BASE_NAME = 'testing_results.json'
AGGREGATED_RESULTS_JSON_FILE = 'aggregated_results.json'

NUMBER_OF_WORKERS = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 1234

SENTIMENT_ID_TO_SENTIMENT = ['Negative', 'Neutral', 'Positive']
NUMBER_OF_SENTIMENTS = len(SENTIMENT_ID_TO_SENTIMENT)

#############################
# Sanity Checking Utilities #
#############################

from typing import Generator
from contextlib import contextmanager
@contextmanager
def _transformers_logging_suppressed() -> Generator:
    import logging
    logger_to_original_level = {}
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and name.startswith('transformers.'):
            logger_to_original_level[logger] = logger.level
            logger.setLevel(logging.ERROR)
    yield
    for logger, original_level in logger_to_original_level.items():
        logger.setLevel(original_level)
    return

def _sanity_check_sequence_length(df: pd.DataFrame, tokenizer: transformers.tokenization_utils.PreTrainedTokenizer, max_sequence_length: int) -> None:
    with _transformers_logging_suppressed():
        lengths = df.content.parallel_map(lambda input_string: len(tokenizer.encode_plus(input_string)['input_ids']))
    length_historgram = histogram(lengths)
    assert len(df) == sum(length_historgram.values())
    assert sum(number_of_strings_with_length for length, number_of_strings_with_length in length_historgram.items() if length < max_sequence_length) / len(df) > 0.99
    return 

##################################
# Application-Specific Utilities #
##################################

def set_cuda_device_id(cuda_device_id: int) -> None:
    assert DEVICE.type == 'cuda', 'No CUDA devices available.'
    assert cuda_device_id < torch.cuda.device_count()
    torch.cuda.set_device(cuda_device_id)
    return

def move_dict_value_tensors_to_device(input_dict: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in input_dict.items()}

################################
# Data Preprocessing Utilities #
################################

def score_to_sentiment_id(score: int) -> int:
    if score < 3:
        sentiment = SENTIMENT_ID_TO_SENTIMENT.index('Negative')
    elif score == 3:
        sentiment = SENTIMENT_ID_TO_SENTIMENT.index('Neutral')
    elif score > 3:
        sentiment = SENTIMENT_ID_TO_SENTIMENT.index('Positive')
    return sentiment

def preprocess_data_frame(df: pd.DataFrame) -> pd.DataFrame:
    df['sentiment_id'] = df.score.map(score_to_sentiment_id)
    columns_to_drop = df.columns.tolist()
    columns_to_drop.remove('content')
    columns_to_drop.remove('sentiment_id')
    df.drop(columns=columns_to_drop, inplace=True)
    return df

@lru_cache(maxsize=1)
def load_data_frames() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(REVIEWS_CSV_FILE)
    df = preprocess_data_frame(df)
    assert not df.isnull().any().any()
    training_df, testing_df = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
    validation_df, testing_df = train_test_split(testing_df, test_size=0.5, random_state=RANDOM_SEED)
    return training_df, validation_df, testing_df

class TransformersDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: transformers.tokenization_utils.PreTrainedTokenizer, max_sequence_length: int):
        self.df = df.reset_index()
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        
    def encode_string(self, input_string: str) -> dict:
        return self.tokenizer.encode_plus(input_string, max_length=self.max_sequence_length, truncation=True, return_token_type_ids=False, padding='max_length', return_attention_mask=True, return_tensors='pt')
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        encoding = self.encode_string(row.content)
        item = {
            'review_string': row.content,
            'sentiment_id': torch.tensor(row.sentiment_id, dtype=torch.long),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        assert len(item['input_ids']) == len(item['attention_mask']) <= self.max_sequence_length
        return item
    
    def __len__(self):
        return len(self.df)

###################################
# Transformer Metaclass Utilities #
###################################

TRANSFORMER_MODEL_SPEC_TO_MODEL_UTILS = {
    # 'albert': {
    #     'model': transformers.AlbertForSequenceClassification,
    #     'tokenizer': transformers.AlbertTokenizer,
    #     'pretrained_model_names': [
    #         'albert-base-v1',
    #         'albert-large-v1',
    #         'albert-base-v2',
    #         'albert-large-v2',
    #     ],
    # },
    # 'bart': {
    #     'model': transformers.BartForSequenceClassification,
    #     'tokenizer': transformers.BartTokenizer,
    #     'pretrained_model_names': [
    #         'facebook/bart-base',
    #     ],
    # },
    # 'bert': {
    #     'model': transformers.BertForSequenceClassification,
    #     'tokenizer': transformers.BertTokenizer,
    #     'pretrained_model_names': [
    #         'bert-base-cased',
    #         'bert-base-uncased',
    #         'bert-base-multilingual-uncased',
    #         'bert-base-multilingual-cased',
    #     ],
    # },
    # 'distilbert': {
    #     'model': transformers.DistilBertForSequenceClassification,
    #     'tokenizer': transformers.DistilBertTokenizer,
    #     'pretrained_model_names': [
    #         'distilbert-base-uncased',
    #         'distilbert-base-uncased-distilled-squad',
    #         'distilbert-base-uncased-distilled-squad',
    #         'distilbert-base-cased-distilled-squad',
    #         'distilbert-base-multilingual-cased',
    #     ],
    # },
    'longformer': {
        'model': transformers.LongformerForSequenceClassification,
        'tokenizer': transformers.LongformerTokenizer,
        'pretrained_model_names': [
            'allenai/longformer-base-4096',
        ],
    },
    # 'roberta': {
    #     'model': transformers.RobertaForSequenceClassification,
    #     'tokenizer': transformers.RobertaTokenizer,
    #     'pretrained_model_names': [
    #         'roberta-base',
    #         'distilroberta-base',
    #     ],
    # },
    'xlnet': {
        'model': transformers.XLNetForSequenceClassification,
        'tokenizer': transformers.XLNetTokenizer,
        'pretrained_model_names': [
            'xlnet-base-cased',
        ],
    },
    'xlm': {
        'model': transformers.XLMForSequenceClassification,
        'tokenizer': transformers.XLMTokenizer,
        'pretrained_model_names': [
            'xlm-mlm-en-2048',
        ],
    },
    # 'xlmroberta': {
    #     'model': transformers.XLMRobertaForSequenceClassification,
    #     'tokenizer': transformers.XLMRobertaTokenizer,
    #     'pretrained_model_names': [
    #         'xlm-roberta-base',
    #     ],
    # },
}

TransformerModelSpec = operator.getitem(Literal, tuple(TRANSFORMER_MODEL_SPEC_TO_MODEL_UTILS.keys()))

##################################
# Transformer Module Metaclasses #
##################################

class TransformerModule(nn.Module):
    
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        with _transformers_logging_suppressed():
            self.pretrained_model = self.__class__.transformer_model.from_pretrained(self.model_name, num_labels=NUMBER_OF_SENTIMENTS)
        self.softmax_layer = nn.Softmax(dim=1)
    
    @property
    def device(self) -> torch.device:
        return only_one({parameter.device for parameter in self.parameters()})
        
    def forward(self, encoding: dict) -> torch.Tensor:
        encoding = move_dict_value_tensors_to_device(encoding, self.device)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        logits = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False, output_hidden_states=False)[0]
        prediction = self.softmax_layer(logits)
        return prediction

class TransformersModuleType(type):

    def __new__(meta, class_name: str, bases: Tuple[type, ...], attributes: dict, transformer_model_spec: TransformerModelSpec):
        
        transformer_model = TRANSFORMER_MODEL_SPEC_TO_MODEL_UTILS[transformer_model_spec]['model']
        
        updated_attributes = dict(attributes)
        updated_attributes.update({
            'transformer_model': transformer_model,
        })
        
        return type(class_name, bases+(TransformerModule,), updated_attributes)

######################################
# Transformer Classifier Metaclasses #
######################################

class Classifier(ABC):
    
    @classmethod
    def hyperparameter_search(cls,
                              model_name_choices: Iterable[str],
                              number_of_epochs_choices: Iterable[int] = [15],
                              # number_of_epochs_choices: Iterable[int] = [15, 30],
                              batch_size_choices: Iterable[int] = [1],
                              # batch_size_choices: Iterable[int] = [1, 32, 64],
                              learning_rate_choices: Iterable[float] = [
                                  4e-6, 4e-5,
                                  2e-6, 2e-5,
                                  1e-6, 1e-5,
                              ],
                              max_sequence_length_choices: Iterable[int] = [160],
                              gradient_clipping_max_threshold_choices: Iterable[float] = [1.0, 10.0],
    ) -> Generator[Callable[[None], None], None, None]:
        hyparameter_list_choices = list(itertools.product(
            model_name_choices,
            number_of_epochs_choices,
            batch_size_choices,
            learning_rate_choices,
            max_sequence_length_choices,
            gradient_clipping_max_threshold_choices,
        ))
        random.shuffle(hyparameter_list_choices)
        for (model_name, number_of_epochs, batch_size, learning_rate, max_sequence_length, gradient_clipping_max_threshold) in hyparameter_list_choices:
            checkpoint_directory = cls.checkpoint_directory_for_model_hyperparameters(model_name, number_of_epochs, batch_size, learning_rate, max_sequence_length, gradient_clipping_max_threshold)
            if cls.model_already_trained(model_name, number_of_epochs, batch_size, learning_rate, max_sequence_length, gradient_clipping_max_threshold):
                print(f'Skipping {checkpoint_directory} .')
            else:
                def training_callback() -> None:
                    with timer():
                        with safe_cuda_memory():
                            print(f'Starting {checkpoint_directory} .')
                            classifier = cls(model_name, number_of_epochs, batch_size, learning_rate, max_sequence_length, gradient_clipping_max_threshold)
                            classifier.train()
                            return
                yield training_callback
    
    @classmethod
    def checkpoint_directory_for_model_hyperparameters(cls, model_name: str, number_of_epochs: int, batch_size: int, learning_rate: float, max_sequence_length: int, gradient_clipping_max_threshold: float):
        checkpoint_dir = f'{model_name.replace("-", "_")}_epochs_{number_of_epochs}_batch_{batch_size}_lr_{learning_rate}_seq_len_{max_sequence_length}_grad_clip_{gradient_clipping_max_threshold}'
        checkpoint_dir = os.path.join(RESULTS_DIR, checkpoint_dir)
        return checkpoint_dir
    
    @classmethod
    def model_already_trained(cls, model_name: str, number_of_epochs: int, batch_size: int, learning_rate: float, max_sequence_length: int, gradient_clipping_max_threshold: float) -> bool:
        checkpoint_directory = cls.checkpoint_directory_for_model_hyperparameters(model_name, number_of_epochs, batch_size, learning_rate, max_sequence_length, gradient_clipping_max_threshold)
        testing_result_file = os.path.join(checkpoint_directory, RESULT_SUMMARY_JSON_FILE_BASE_NAME)
        return os.path.exists(testing_result_file)
    
    def __init__(self, model_name: str, number_of_epochs: int, batch_size: int, learning_rate: float, max_sequence_length: int, gradient_clipping_max_threshold: float):
        self.model_name = model_name
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_sequence_length = max_sequence_length
        self.gradient_clipping_max_threshold = gradient_clipping_max_threshold
        
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
        
        self.tokenizer = self.__class__.transformer_model_tokenizer.from_pretrained(self.model_name)
        self._load_data()
        
        self.model: nn.Module = self.__class__.transformer_module(self.model_name).to(DEVICE)
        self.optimizer: torch.optim.Optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        self.loss_function: Callable = nn.CrossEntropyLoss().to(DEVICE)
        self.scheduler: torch.optim.lr_scheduler.LambdaLR = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=len(self.training_dataloader)*self.number_of_epochs)

    def _load_data(self) -> None:
        training_df, validation_df, testing_df = load_data_frames()
        _sanity_check_sequence_length(training_df, self.tokenizer, self.max_sequence_length)
        _sanity_check_sequence_length(validation_df, self.tokenizer, self.max_sequence_length)
        _sanity_check_sequence_length(testing_df, self.tokenizer, self.max_sequence_length)
        self.training_dataloader: data.DataLoader = self.dataset_to_dataloader(TransformersDataset(training_df, self.tokenizer, self.max_sequence_length))
        self.validation_dataloader: data.DataLoader = self.dataset_to_dataloader(TransformersDataset(validation_df, self.tokenizer, self.max_sequence_length))
        self.testing_dataloader: data.DataLoader = self.dataset_to_dataloader(TransformersDataset(testing_df, self.tokenizer, self.max_sequence_length))
        return
    
    def dataset_to_dataloader(self, dataset: data.Dataset) -> data.DataLoader:
        dataloader = data.DataLoader(dataset, batch_size=self.batch_size, num_workers=NUMBER_OF_WORKERS)
        return dataloader

    @property
    def checkpoint_directory(self) -> str:
        return self.__class__.checkpoint_directory_for_model_hyperparameters(self.model_name, self.number_of_epochs, self.batch_size, self.learning_rate, self.max_sequence_length, self.gradient_clipping_max_threshold)
    
    @property
    def checkpoint_file(self) -> str:
        return os.path.join(self.checkpoint_directory, 'best-performing-model.pt')
    
    def save_model_parameters(self) -> None:
        torch.save(self.model.state_dict(), self.checkpoint_file)
        return

    def load_model_parameters(self) -> None:
        self.model.load_state_dict(torch.load(self.checkpoint_file))
        return

    def note_results(self, file_base_name: str, epoch_index: int, loss_per_example: float, accuracy_per_example: float, auxillary_information: dict = {}) -> None:
        result_json_file = os.path.join(self.checkpoint_directory, file_base_name)
        result_data = dict(auxillary_information)
        result_data.update({
            'model_name': self.model_name,
            'number_of_epochs': self.number_of_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'max_sequence_length': self.max_sequence_length,
            'gradient_clipping_max_threshold': self.gradient_clipping_max_threshold,
            'epoch': epoch_index,
            'loss_per_example': loss_per_example,
            'accuracy_per_example': accuracy_per_example,
        })
        with open(result_json_file, 'w') as file_handle:
            json.dump(result_data, file_handle, indent=4)
        return

    def train_one_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        for encoding in tqdm_with_message(self.training_dataloader, post_yield_message_func = lambda index: f'Training Loss Per Batch      {total_loss/(index+1):.8f}', total=len(self.training_dataloader)):
            prediction_distributions = self.model(encoding)
            assert prediction_distributions.shape[0] <= self.batch_size
            assert prediction_distributions.shape[1] <= NUMBER_OF_SENTIMENTS
            targets = encoding['sentiment_id'].to(self.model.device)
            loss = self.loss_function(prediction_distributions, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clipping_max_threshold)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()
            total_accuracy += prediction_distributions.argmax(dim=1).eq(targets).sum().item()
        mean_loss = total_loss / len(self.training_dataloader.dataset)
        mean_accuracy = total_accuracy / len(self.training_dataloader.dataset)
        return mean_loss, mean_accuracy

    def _evaluate(self, dataloader_type: Literal['validation', 'testing']) -> Tuple[float, float]:
        total_loss = 0
        total_accuracy = 0
        self.model.eval()
        tqdm_padding = ' '*3 if dataloader_type == 'validation' else ' '*6
        dataloader = self.validation_dataloader if dataloader_type == 'validation' else self.testing_dataloader
        with torch.no_grad():
            for encoding in tqdm_with_message(dataloader, post_yield_message_func = lambda index: f'{dataloader_type.capitalize()} Loss Per Batch {tqdm_padding}{total_loss/(index+1):.8f}', total=len(dataloader)):
                prediction_distributions = self.model(encoding)
                targets = encoding['sentiment_id'].to(self.model.device)
                loss = self.loss_function(prediction_distributions, targets)
                total_loss += loss.item()
                total_accuracy += prediction_distributions.argmax(dim=1).eq(targets).sum().item()
        mean_loss = total_loss / len(dataloader.dataset)
        mean_accuracy = total_accuracy / len(dataloader.dataset)
        return mean_loss, mean_accuracy

    def validate(self) -> Tuple[float, float]:
        return self._evaluate('validation')

    def test(self) -> Tuple[float, float]:
        return self._evaluate('testing')
        
    def train(self) -> None:
        print()
        if self.__class__.model_already_trained(self.model_name, self.number_of_epochs, self.batch_size, self.learning_rate, self.max_sequence_length, self.gradient_clipping_max_threshold):
            print(f'Model already trained and tested (results stored in {self.checkpoint_directory}).')
        else:
            best_validation_loss = float('inf')
            best_validation_epoch = None
            print()
            print(f'Saving results to {self.checkpoint_directory} .')
            print(f'Using CUDA device {torch.cuda.current_device()}.' if DEVICE.type == 'cuda' else 'Using CPU.')
            print(f'Process id: {os.getpid()}')
            for epoch_index in range(self.number_of_epochs):
                print()
                print(f'Training Epoch {epoch_index}')
                training_loss, training_accuracy = self.train_one_epoch()
                validation_loss, validation_accuracy = self.validate()
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_validation_epoch = epoch_index
                    self.save_model_parameters()
                self.note_results(f'training_epoch_{epoch_index}_results.json', epoch_index, training_loss, training_accuracy)
                self.note_results(f'validation_epoch_{epoch_index}_results.json', epoch_index, validation_loss, validation_accuracy)
                print(f'Training Loss Per Example:       {training_loss:.8f}')
                print(f'Validation Loss Per Example:     {validation_loss:.8f}')
                print(f'Training Accuracy Per Example:   {training_accuracy:.8f}')
                print(f'Validation Accuracy Per Example: {validation_accuracy:.8f}')
                print(f'Current checkpoint directory: {self.checkpoint_directory}')
                print()
            self.load_model_parameters()
            testing_loss, testing_accuracy = self.test()
            self.note_results(RESULT_SUMMARY_JSON_FILE_BASE_NAME, epoch_index, testing_loss, testing_accuracy, {'best_validation_epoch': best_validation_epoch})
            print(f'Testing Loss Per Example:        {validation_loss:.8f}')
            print(f'Testing Accuracy Per Example:    {testing_accuracy:.8f}')
        return

class TransformersClassifierType(type):

    def __new__(meta, class_name: str, bases: Tuple[type, ...], attributes: dict, transformer_model_spec: TransformerModelSpec):
        
        transformer_model = TRANSFORMER_MODEL_SPEC_TO_MODEL_UTILS[transformer_model_spec]['model']
        transformer_model_tokenizer = TRANSFORMER_MODEL_SPEC_TO_MODEL_UTILS[transformer_model_spec]['tokenizer']

        class TransformerModule(metaclass=TransformersModuleType, transformer_model_spec=transformer_model_spec):
            pass
        
        updated_attributes = dict(attributes)
        updated_attributes.update({
            'transformer_model_tokenizer': transformer_model_tokenizer,
            'transformer_module': TransformerModule,
        })
        
        return type(class_name, bases+(Classifier,), updated_attributes)

###########################
# Transformer Classifiers #
###########################

for transformer_model_spec in TRANSFORMER_MODEL_SPEC_TO_MODEL_UTILS.keys():
    class_name = transformer_model_spec.capitalize()+'Classifier'
    TRANSFORMER_MODEL_SPEC_TO_MODEL_UTILS[transformer_model_spec]['classifier'] = TransformersClassifierType(class_name, (), {}, transformer_model_spec=transformer_model_spec)

###########################################
# Aggregate Hyperparameter Search Results #
###########################################

def aggregate_hyperparameter_search_results() -> None:
    trial_dirs = [os.path.join(RESULTS_DIR, trial_dir) for trial_dir in os.listdir(RESULTS_DIR)]
    result_summary_dicts = []
    for trial_dir in trial_dirs:
        result_summary_file = os.path.join(trial_dir, RESULT_SUMMARY_JSON_FILE_BASE_NAME)
        if os.path.isfile(result_summary_file):
            with open(result_summary_file, 'r') as file_handle:
                result_summary_dict = json.loads(file_handle.read())
            result_summary_dicts.append(result_summary_dict)
    with open(AGGREGATED_RESULTS_JSON_FILE, 'w') as file_handle:
        json.dump(result_summary_dicts, file_handle, indent=4)
    return

##########
# Driver #
##########

def perform_hyperparameter_search() -> None:
    # Execute Hyperparameter Search
    callback_generators = []
    for transformer_model_spec in TRANSFORMER_MODEL_SPEC_TO_MODEL_UTILS.keys():
        callback_generators.append(
            TRANSFORMER_MODEL_SPEC_TO_MODEL_UTILS[transformer_model_spec]['classifier'].hyperparameter_search(
                model_name_choices = TRANSFORMER_MODEL_SPEC_TO_MODEL_UTILS[transformer_model_spec]['pretrained_model_names'],
            ))
    random.shuffle(callback_generators)
    all_hyperparameter_search_callbacks = roundrobin(*callback_generators)
    for callback in all_hyperparameter_search_callbacks:
        callback()
    return

@debug_on_error
def main() -> None:
    parser = argparse.ArgumentParser(prog='tool', formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 9999))
    parser.add_argument('-cuda-device-id', type=int, default=0, help='Perform hyperparameter search on specified CUDA device with the given id.')
    parser.add_argument('-use-cpu', action='store_true', help='Perform hyperparameter search on the CPU.')
    parser.add_argument('-aggregate-results', action='store_true', help='Aggregate the results of our hyperparameter search.')
    args = parser.parse_args()
    if args.aggregate_results:
        aggregate_hyperparameter_search_results()
    else:
        if args.use_cpu:
            global DEVICE
            DEVICE = torch.device('cpu')
        else:
            set_cuda_device_id(args.cuda_device_id)
        perform_hyperparameter_search()
    return

if __name__ == '__main__':
    main()
