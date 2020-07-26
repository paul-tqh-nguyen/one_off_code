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

import transformers 
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils import data

# @todo make sure all of the imports are used

###########
# Globals #
###########

pandarallel.initialize(nb_workers=mp.cpu_count(), progress_bar=False, verbose=0)

# https://bit.ly/3ez5TOO
APPS_CSV_FILE = './data/apps.csv'
REVIEWS_CSV_FILE = './data/reviews.csv'

NUMBER_OF_WORKERS = mp.cpu_count()
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

def _sanity_check_classifier_class_attributes() -> None:
    assert BertClassifier.transformer_module.transformer_model == BertModel
    assert BertClassifier.transformer_model_tokenizer == BertTokenizer
    assert RobertaClassifier.transformer_module.transformer_model == RobertaModel
    assert RobertaClassifier.transformer_model_tokenizer == RobertaTokenizer
    global_classifier_names = {var_name for var_name, value in globals().items() if Classifier in parent_classes(value) and Classifier != value}
    assert len(global_classifier_names) == 2
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

TRANSFORMER_MODEL_SPEC_TO_MODEL_TOKENIZER_PAIR = {
    'bert': (BertModel, BertTokenizer),
    'roberta': (RobertaModel, RobertaTokenizer),
}

TransformerModelSpec = operator.getitem(Literal, tuple(TRANSFORMER_MODEL_SPEC_TO_MODEL_TOKENIZER_PAIR.keys()))

##################################
# Transformer Module Metaclasses #
##################################

class TransformerModule(nn.Module):
    
    def __init__(self, model_name: str, dropout_probability: float):
        super().__init__()
        self.model_name = model_name
        self.dropout_probability = dropout_probability
        self.pretrained_model = self.__class__.transformer_model.from_pretrained(self.model_name)
        self.dropout_layer = nn.Dropout(self.dropout_probability)
        self.prediction_layers = nn.Sequential(OrderedDict([
            ('fully_connected_layer', nn.Linear(self.pretrained_model.config.hidden_size, NUMBER_OF_SENTIMENTS)),
            ('softmax_layer', nn.Softmax(dim=1)),
        ]))
    
    @property
    def device(self) -> torch.device:
        return only_one({parameter.device for parameter in self.parameters()})
        
    def forward(self, encoding: dict) -> torch.Tensor:
        encoding = move_dict_value_tensors_to_device(encoding, self.device)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        _, pooled_output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout_layer(pooled_output)
        prediction = self.prediction_layers(pooled_output)
        return prediction

class TransformersModuleType(type):

    def __new__(meta, class_name: str, bases: Tuple[type, ...], attributes: dict, transformer_model_spec: TransformerModelSpec):
        
        transformer_model, transformer_model_tokenizer = TRANSFORMER_MODEL_SPEC_TO_MODEL_TOKENIZER_PAIR[transformer_model_spec]
        
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
                              number_of_epochs_choices: Iterable[int] = [10],
                              batch_size_choices: Iterable[int] = [1, 32, 64, 128],
                              learning_rate_choices: Iterable[float] = [
                                  4e-7, 4e-5, 4e-3,
                                  2e-7, 2e-5, 2e-3,
                                  1e-7, 1e-5, 1e-3,
                              ],
                              max_sequence_length_choices: Iterable[int] = [160],
                              dropout_probability_choices: Iterable[float] = [0.0, 0.25, 0.5],
                              gradient_clipping_max_threshold_choices: Iterable[float] = [1.0, 3.0, 5.0, 10.0],
    ) -> Generator[Callable[[None], None], None, None]:
        hyparameter_list_choices = list(itertools.product(
            model_name_choices,
            number_of_epochs_choices,
            batch_size_choices,
            learning_rate_choices,
            max_sequence_length_choices,
            dropout_probability_choices,
            gradient_clipping_max_threshold_choices,
        ))
        random.shuffle(hyparameter_list_choices)
        for (model_name, number_of_epochs, batch_size, learning_rate, max_sequence_length, dropout_probability, gradient_clipping_max_threshold) in hyparameter_list_choices:
            with safe_cuda_memory():
                def training_callback() -> None:
                    classifier = cls(model_name, number_of_epochs, batch_size, learning_rate, max_sequence_length, dropout_probability, gradient_clipping_max_threshold)
                    classifier.train()
                    return
                yield training_callback
    
    def __init__(self, model_name: str, number_of_epochs: int, batch_size: int, learning_rate: float, max_sequence_length: int, dropout_probability: float, gradient_clipping_max_threshold: float):
        self.model_name = model_name
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_sequence_length = max_sequence_length
        self.dropout_probability = dropout_probability
        self.gradient_clipping_max_threshold = gradient_clipping_max_threshold
        
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
        
        self.tokenizer = self.__class__.transformer_model_tokenizer.from_pretrained(self.model_name)
        self._load_data()
        
        self.model: nn.Module = self.__class__.transformer_module(self.model_name, self.dropout_probability).to(DEVICE)
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
        return f'{self.model_name.replace("-", "_")}_epochs_{self.number_of_epochs}_batch_{self.batch_size}_lr_{self.learning_rate}_seq_len_{self.max_sequence_length}_dropout_{self.dropout_probability}_grad_clip_{self.gradient_clipping_max_threshold}'
    
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
            'dropout_probability': self.dropout_probability,
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
        testing_result_file = os.path.join(self.checkpoint_directory, 'testing_results.json')
        if os.path.exists(testing_result_file):
            print(f'Model already trained and tested (results stored in {self.checkpoint_directory}).')
        else:
            best_validation_loss = float('inf')
            best_validation_epoch = None
            for epoch_index in range(self.number_of_epochs):
                print()
                print(f'Saving results to {self.checkpoint_directory} .')
                print()
                print(f'Training Epoch {epoch_index}')
                training_loss, training_accuracy = self.train_one_epoch()
                validation_loss, validation_accuracy = self.validate()
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_validation_epoch = epoch_index
                    self.save_model_parameters()
                self.note_results(f'training_epoch_{epoch_index}_results.json', epoch_index, training_loss, training_accuracy)
                self.note_results(f'valildation_epoch_{epoch_index}_results.json', epoch_index, valildation_loss, valildation_accuracy)
                print(f'Training Loss Per Example:       {training_loss:.8f}')
                print(f'Validation Loss Per Example:     {validation_loss:.8f}')
                print(f'Training Accuracy Per Example:   {training_accuracy:.8f}')
                print(f'Validation Accuracy Per Example: {validation_accuracy:.8f}')
                print()
            self.load_model_parameters()
            testing_loss, testing_accuracy = self.test()
            self.note_results('testing_results.json', epoch_index, testing_loss, testing_accuracy, {'best_validation_epoch': best_validation_epoch})
            print(f'Testing Loss Per Example:        {validation_loss:.8f}')
            print(f'Testing Accuracy Per Example:    {testing_accuracy:.8f}')
        return

class TransformersClassifierType(type):

    def __new__(meta, class_name: str, bases: Tuple[type, ...], attributes: dict, transformer_model_spec: TransformerModelSpec):
        
        transformer_model, transformer_model_tokenizer = TRANSFORMER_MODEL_SPEC_TO_MODEL_TOKENIZER_PAIR[transformer_model_spec]

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

class BertClassifier(metaclass=TransformersClassifierType, transformer_model_spec='bert'):
    pass

class RobertaClassifier(metaclass=TransformersClassifierType, transformer_model_spec='roberta'):    
    pass

##########
# Driver #
##########

def perform_hyperparameter_search() -> None:
    _sanity_check_classifier_class_attributes()
    # Bert Hyperparameter Search Callbacks
    bert_hyperparameter_search_callbacks = BertClassifier.hyperparameter_search(
        model_name_choices = [
            'bert-base-cased',
            'bert-base-uncased',
            'bert-large-cased',
            'bert-large-uncased',
            'bert-base-multilingual-uncased',
            'bert-base-multilingual-cased',
            # 'bert-large-uncased-whole-word-masking',
            # 'bert-large-cased-whole-word-masking',
            # 'bert-large-uncased-whole-word-masking-finetuned-squad',
            # 'bert-large-cased-whole-word-masking-finetuned-squad',
        ],
    )
    # Roberta Hyperparameter Search Callbacks
    roberta_hyperparameter_search_callbacks = RobertaClassifier.hyperparameter_search(
        model_name_choices = [
            'roberta-base',
            'roberta-large',
            'distilroberta-base',
            # 'roberta-large-mnli',
        ]
    )
    # Execute Hyperparameter Search
    all_hyperparameter_search_callbacks = roundrobin(
        bert_hyperparameter_search_callbacks,
        roberta_hyperparameter_search_callbacks,
    )
    for callback in all_hyperparameter_search_callbacks:
        callback()
    return

@debug_on_error
def main() -> None:
    parser = argparse.ArgumentParser(prog='tool', formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 9999))
    parser.add_argument('-cuda-device-id', type=int, default=0, help="Perform hyperparameter search on specified CUDA device with the given id.")
    args = parser.parse_args()
    set_cuda_device_id(args.cuda_device_id)
    if torch.device('cuda').type == 'cuda':
        print(f'Using CUDA device {args.cuda_device_id}.')
    else:
        print(f'Using CPU.')
    perform_hyperparameter_search()
    return

if __name__ == '__main__':
    main()
