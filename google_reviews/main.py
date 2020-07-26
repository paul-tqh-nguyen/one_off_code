'#!/usr/bin/python3 -OO' # @todo make this the default

'''
'''
# @todo update doc string

###########
# Imports #
###########

import operator
import multiprocessing as mp
import pandas as pd
from abc import ABC, abstractmethod
from functools import lru_cache
from pandarallel import pandarallel
from collections import OrderedDict
from typing import Tuple, Callable
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
APPS_CSV_FILE_LOCATION = './data/apps.csv'
REVIEWS_CSV_FILE_LOCATION = './data/reviews.csv'

NUMBER_OF_WORKERS = mp.cpu_count()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 1234

NUMBER_OF_SENTIMENTS = 3

MAX_SEQUENCE_LENGTH = 160

SAVED_MODEL_LOCATION = './best-performing-model.pt'

NUMBER_OF_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
DROPOUT_PROBABILITY = 0.3
GRADIENT_CLIPPING_MAX_THRESHOLD = 1.0

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

def move_dict_value_tensors_to_device(input_dict: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in input_dict.items()}

######################
# Data Preprocessing #
######################

def score_to_sentiment_id(score: int) -> int:
    if score < 3:
        sentiment = 0
    elif score == 3:
        sentiment = 1
    elif score > 3:
        sentiment = 2
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
    df = pd.read_csv(REVIEWS_CSV_FILE_LOCATION)
    df = preprocess_data_frame(df)
    assert not df.isnull().any().any()
    training_df, testing_df = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
    validation_df, testing_df = train_test_split(testing_df, test_size=0.5, random_state=RANDOM_SEED)
    return training_df, validation_df, testing_df

#######################
# Abstract Classifier #
#######################

class Classifier(ABC):
    
    @abstractmethod
    def __init__(self, model_name: str, number_of_epochs: int, batch_size: int, learning_rate: float, max_sequence_length: int, dropout_probability: float, gradient_clipping_max_threshold: float, *args, **kwargs):
        '''
        The initializer must populate the following attributes:
            self.model_name
            self.number_of_epochs
            self.batch_size
            self.learning_rate
            self.max_sequence_length
            self.dropout_probability
            self.gradient_clipping_max_threshold
            self.tokenizer
            self.model
            self.optimizer
            self.loss_function
            self.scheduler
        '''
        pass
    
    def dataset_to_dataloader(self, dataset: data.Dataset) -> data.DataLoader:
        dataloader = data.DataLoader(dataset, batch_size=self.batch_size, num_workers=NUMBER_OF_WORKERS)
        return dataloader

    def train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0
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
        mean_loss = total_loss / len(self.training_dataloader.dataset)
        return mean_loss

    def _evaluate(self, dataloader_type: Literal['validation', 'testing']) -> float:
        total_loss = 0
        self.model.eval()
        tqdm_padding = ' '*3 if dataloader_type == 'validation' else ''
        dataloader = self.validation_dataloader if dataloader_type == 'validation' else self.testing_dataloader
        with torch.no_grad():
            for encoding in tqdm_with_message(dataloader, post_yield_message_func = lambda index: f'{dataloader_type.capitalize()} Loss Per Batch {tqdm_padding}{total_loss/(index+1):.8f}', total=len(dataloader)):
                prediction_distributions = self.model(encoding)
                targets = encoding['sentiment_id'].to(self.model.device)
                loss = self.loss_function(prediction_distributions, targets)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clipping_max_threshold)
                total_loss += loss.item()
        mean_loss = total_loss / len(dataloader.dataset)
        return mean_loss

    def validate(self) -> float:
        return self._evaluate('validation')

    def test(self) -> float:
        return self._evaluate('testing')

    def save_model_parameters(self, saved_model_location: str) -> None:
        torch.save(self.model.state_dict(), saved_model_location)
        return

    def load_model_parameters(self, saved_model_location: str) -> None:
        self.model.load_state_dict(torch.load(saved_model_location))
        return
    
    def train(self) -> None:
        best_validation_loss = float('inf')
        for epoch_index in range(self.number_of_epochs):
            print(f'Training Epoch {epoch_index}')
            training_loss = self.train_one_epoch()
            validation_loss = self.validate()
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                self.save_model_parameters(SAVED_MODEL_LOCATION)
            print(f'Training Loss Per Example:   {training_loss:.8f}')
            print(f'Validation Loss Per Example: {validation_loss:.8f}')
            print()
        self.load_model_parameters(SAVED_MODEL_LOCATION)
        testing_loss = self.test()
        print(f'Testing Loss Per Example:    {validation_loss:.8f}')
        return

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
            'transformer_model_tokenizer': transformer_model_tokenizer,
        })
        
        return type(class_name, bases+(TransformerModule,), updated_attributes)

###################
# Bert Classifier #
###################

class BertModule(metaclass=TransformersModuleType, transformer_model_spec='bert'):
    pass

class BertDataset(data.Dataset): # @todo can we make this model agnostic?
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

class BertClassifier(Classifier): # @todo can we make this class a function of the model name alone and use it for all transformers models? Can we use metaclasses here?

    def __init__(self, model_name: str, number_of_epochs: int, batch_size: int, learning_rate: float, max_sequence_length: int, dropout_probability: float, gradient_clipping_max_threshold: float):
        self.model_name = model_name
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_sequence_length = max_sequence_length
        self.dropout_probability = dropout_probability
        self.gradient_clipping_max_threshold = gradient_clipping_max_threshold
        
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self._load_data()
        
        self.model: nn.Module = BertModule(self.model_name, self.dropout_probability).to(DEVICE)
        self.optimizer: torch.optim.Optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        self.loss_function: Callable = nn.CrossEntropyLoss().to(DEVICE)
        self.scheduler: torch.optim.lr_scheduler.LambdaLR = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=len(self.training_dataloader)*self.number_of_epochs)

    def _load_data(self) -> None:
        training_df, validation_df, testing_df = load_data_frames()
        _sanity_check_sequence_length(training_df, self.tokenizer, self.max_sequence_length)
        _sanity_check_sequence_length(validation_df, self.tokenizer, self.max_sequence_length)
        _sanity_check_sequence_length(testing_df, self.tokenizer, self.max_sequence_length)
        self.training_dataloader: data.DataLoader = self.dataset_to_dataloader(BertDataset(training_df, self.tokenizer, self.max_sequence_length))
        self.validation_dataloader: data.DataLoader = self.dataset_to_dataloader(BertDataset(validation_df, self.tokenizer, self.max_sequence_length))
        self.testing_dataloader: data.DataLoader = self.dataset_to_dataloader(BertDataset(testing_df, self.tokenizer, self.max_sequence_length))
        return
        

######################
# Roberta Classifier #
######################

class RobertaModule(metaclass=TransformersModuleType, transformer_model_spec='roberta'):
    pass

class RobertaModule(nn.Module):
    
    def __init__(self, model_name: str, dropout_probability: float):
        super().__init__()
        self.model_name = model_name
        self.dropout_probability = dropout_probability
        self.pretrained_model = RobertaModel.from_pretrained(self.model_name)
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

class RobertaDataset(data.Dataset):
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

class RobertaClassifier(Classifier):

    def __init__(self, model_name: str, number_of_epochs: int, batch_size: int, learning_rate: float, max_sequence_length: int, dropout_probability: float, gradient_clipping_max_threshold: float):
        self.model_name = model_name
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_sequence_length = max_sequence_length
        self.dropout_probability = dropout_probability
        self.gradient_clipping_max_threshold = gradient_clipping_max_threshold
        
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self._load_data()
        
        self.model: nn.Module = RobertaModule(self.model_name, self.dropout_probability).to(DEVICE)
        self.optimizer: torch.optim.Optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        self.loss_function: Callable = nn.CrossEntropyLoss().to(DEVICE)
        self.scheduler: torch.optim.lr_scheduler.LambdaLR = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=len(self.training_dataloader)*self.number_of_epochs)

    def _load_data(self) -> None:
        training_df, validation_df, testing_df = load_data_frames()
        _sanity_check_sequence_length(training_df, self.tokenizer, self.max_sequence_length)
        _sanity_check_sequence_length(validation_df, self.tokenizer, self.max_sequence_length)
        _sanity_check_sequence_length(testing_df, self.tokenizer, self.max_sequence_length)
        self.training_dataloader: data.DataLoader = self.dataset_to_dataloader(RobertaDataset(training_df, self.tokenizer, self.max_sequence_length))
        self.validation_dataloader: data.DataLoader = self.dataset_to_dataloader(RobertaDataset(validation_df, self.tokenizer, self.max_sequence_length))
        self.testing_dataloader: data.DataLoader = self.dataset_to_dataloader(RobertaDataset(testing_df, self.tokenizer, self.max_sequence_length))
        return
        
##########
# Driver #
##########

@debug_on_error
def main() -> None:
    # bert_classifier = BertClassifier('bert-base-cased', NUMBER_OF_EPOCHS, BATCH_SIZE, LEARNING_RATE, MAX_SEQUENCE_LENGTH, DROPOUT_PROBABILITY, GRADIENT_CLIPPING_MAX_THRESHOLD)
    # bert_classifier.train()
    roberta_classifier = RobertaClassifier('roberta-base', NUMBER_OF_EPOCHS, BATCH_SIZE, LEARNING_RATE, MAX_SEQUENCE_LENGTH, DROPOUT_PROBABILITY, GRADIENT_CLIPPING_MAX_THRESHOLD)
    roberta_classifier.train()
    breakpoint()
    return
        
if __name__ == '__main__':
    main()
