'#!/usr/bin/python3 -OO' # @todo make this the default

'''
'''
# @todo update doc string

###########
# Imports #
###########

import multiprocessing as mp
import pandas as pd
from functools import lru_cache
from pandarallel import pandarallel
from collections import OrderedDict
from typing import Tuple, Callable
from typing_extensions import Literal

from misc_utilities import *

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils import data

###########
# Globals #
###########

pandarallel.initialize(nb_workers=mp.cpu_count(), progress_bar=False, verbose=0)

# https://bit.ly/3ez5TOO
APPS_CSV_FILE_LOCATION = './data/apps.csv'
REVIEWS_CSV_FILE_LOCATION = './data/reviews.csv'
SAVED_MODEL_LOCATION = './best-performing-model.pt'

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
TOKENIZER = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

NUMBER_OF_WORKERS = 0 # mp.cpu_count()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 1234

NUMBER_OF_SENTIMENTS = 3
MAX_SEQUENCE_LENGTH = 160

GRADIENT_CLIPPING_MAX_THRESHOLD = 1.0
NUMBER_OF_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
DROPOUT_PROBABILITY = 0.3

###################
# Sanity Checking #
###################

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

def _string_sequence_length(input_string: str) -> int:
    return len(TOKENIZER.encode_plus(input_string)['input_ids'])

def _sanity_check_sequence_length(df: pd.DataFrame) -> None:
    with _transformers_logging_suppressed():
        lengths = df.content.parallel_map(_string_sequence_length)
    length_historgram = histogram(lengths)
    assert len(df) == sum(length_historgram.values())
    assert sum(number_of_strings_with_length for length, number_of_strings_with_length in length_historgram.items() if length < MAX_SEQUENCE_LENGTH) / len(df) > 0.99
    return 

##################################
# Application-Specific Utilities #
##################################

def encode_string(input_string: str) -> dict:
    return TOKENIZER.encode_plus(input_string, max_length=MAX_SEQUENCE_LENGTH, truncation=True, return_token_type_ids=False, padding='max_length', return_attention_mask=True, return_tensors='pt')

def move_dict_value_tensors_to_device(input_dict: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in input_dict.items()}

######################
# Data Preprocessing #
######################

class ReviewsDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index()
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        encoding = encode_string(row.content)
        return {
            'review_string': row.content,
            'sentiment_id': torch.tensor(row.sentiment_id, dtype=torch.long),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
    
    def __len__(self):
        return len(self.df)

def dataset_to_dataloader(dataset: ReviewsDataset) -> data.DataLoader:
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUMBER_OF_WORKERS)
    return dataloader
    
def score_to_sentiment_id(score: int) -> int:
    if score < 3:
        sentiment = 0
    elif score == 3:
        sentiment = 1
    elif score > 3:
        sentiment = 2
    return sentiment

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['sentiment_id'] = df.score.map(score_to_sentiment_id)
    columns_to_drop = df.columns.tolist()
    columns_to_drop.remove('content')
    columns_to_drop.remove('sentiment_id')
    df.drop(columns=columns_to_drop, inplace=True)
    return df

def load_data() -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    df = pd.read_csv(REVIEWS_CSV_FILE_LOCATION)
    df = preprocess_data(df)
    assert not df.isnull().any().any()
    _sanity_check_sequence_length(df)
    training_df, testing_df = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
    validation_df, testing_df = train_test_split(testing_df, test_size=0.5, random_state=RANDOM_SEED)
    
    training_dataset = ReviewsDataset(training_df)
    validation_dataset = ReviewsDataset(validation_df)
    testing_dataset = ReviewsDataset(testing_df)
    
    training_dataloader = dataset_to_dataloader(training_dataset)
    validation_dataloader = dataset_to_dataloader(validation_dataset)
    testing_dataloader = dataset_to_dataloader(testing_dataset)
    return training_dataloader, validation_dataloader, testing_dataloader

##########
# Models #
##########

class BERTModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.dropout_layer = nn.Dropout(DROPOUT_PROBABILITY)
        self.prediction_layers = nn.Sequential(OrderedDict([
            ("fully_connected_layer", nn.Linear(self.bert_model.config.hidden_size, NUMBER_OF_SENTIMENTS)),
            ("softmax_layer", nn.Softmax(dim=1)),
        ]))

    @property
    @lru_cache(maxsize=1)
    def device(self) -> torch.device:
        return only_one({parameter.device for parameter in self.parameters()})
        
    def forward(self, encoding: dict) -> torch.Tensor:
        encoding = move_dict_value_tensors_to_device(encoding, self.device)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        _, pooled_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout_layer(pooled_output)
        prediction = self.prediction_layers(pooled_output)
        return prediction

##############
# Classifier #
##############

class BERTClassifier:

    def __init__(self):
        self.training_dataloader: data.DataLoader
        self.validation_dataloader: data.DataLoader
        self.testing_dataloader: data.DataLoader
        self.training_dataloader, self.validation_dataloader, self.testing_dataloader = load_data()
        self.bert_module: nn.Module = BERTModule().to(DEVICE)
        self.optimizer: torch.optim.Optimizer = AdamW(self.bert_module.parameters(), lr=LEARNING_RATE, correct_bias=False)
        self.loss_function: Callable = nn.CrossEntropyLoss().to(DEVICE)
        self.scheduler: torch.optim.lr_scheduler.LambdaLR = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=len(self.training_dataloader)*NUMBER_OF_EPOCHS)
    
    def train_one_epoch(self) -> float:
        self.bert_module.train()
        total_loss = 0
        for encoding in tqdm_with_message(self.training_dataloader, post_yield_message_func = lambda index: f'Loss {total_loss/(index+1):.8f}', total=len(self.training_dataloader)):
            prediction_distributions = self.bert_module(encoding)
            targets = encoding['sentiment_id'].to(self.bert_module.device)
            loss = self.loss_function(prediction_distributions, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.bert_module.parameters(), max_norm=GRADIENT_CLIPPING_MAX_THRESHOLD)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()
        mean_loss = total_loss / len(self.training_dataloader.dataset)
        return mean_loss

    def _evaluate(self, dataloader_type: Literal['validation', 'testing']) -> float:
        total_loss = 0
        self.bert_module.eval()
        dataloader = self.validation_dataloader if dataloader_type == 'validation' else self.testing_dataloader
        with torch.no_grad():
            for encoding in tqdm_with_message(dataloader, post_yield_message_func = lambda index: f'{dataloader_type.capitalize()} Loss {total_loss/(index+1):.8f}', total=len(dataloader)):
                prediction_distributions = self.bert_module(encoding)
                targets = encoding['sentiment_id'].to(self.bert_module.device)
                loss = self.loss_function(prediction_distributions, targets)
                nn.utils.clip_grad_norm_(self.bert_module.parameters(), max_norm=GRADIENT_CLIPPING_MAX_THRESHOLD)
                total_loss += loss.item()
        mean_loss = total_loss / len(dataloader.dataset)
        return mean_loss

    def validate(self) -> float:
        return self._evaluate('validation')

    def test(self) -> float:
        return self._evaluate('testing')

    def save_model_parameters(self, saved_model_location: str) -> None:
        torch.save(self.bert_module.state_dict(), saved_model_location)
        return

    def load_model_parameters(self, saved_model_location: str) -> None:
        self.bert_module.load_state_dict(torch.load(saved_model_location))
        return
    
    def train(self) -> None:
        best_validation_loss = float('inf')
        for epoch_index in range(NUMBER_OF_EPOCHS):
            print(f'Training Epoch {epoch_index}')
            training_loss = self.train_one_epoch()
            validation_loss = self.validate()
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                self.save_model_parameters(SAVED_MODEL_LOCATION)
            print(f'Training Loss:   {training_loss:.8f}')
            print(f'Validation Loss: {validation_loss:.8f}')
            print()
        self.load_model_parameters(SAVED_MODEL_LOCATION)
        testing_loss = self.test()
        print(f'Testing Loss:    {validation_loss:.8f}')
        return
        
        
##########
# Driver #
##########

@debug_on_error
def main() -> None:
    bert_classifier = BERTClassifier()
    bert_classifier.train()
    breakpoint()
    return
        
if __name__ == '__main__':
    main()
