'#!/usr/bin/python3 -OO' # @todo make this the default

'''
'''
# @todo update doc string

###########
# Imports #
###########

import multiprocessing as mp
import pandas as pd
from pandarallel import pandarallel
from typing import Tuple

from misc_utilities import *

from transformers import BertModel, BertTokenizer
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

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
TOKENIZER = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

NUMBER_OF_WORKERS = mp.cpu_count()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 1234
NUMBER_OF_SENTIMENTS = 3

BATCH_SIZE = 64
MAX_SEQUENCE_LENGTH = 160
DROPOUT_PROBABILITY = 0.3

###################
# Sanity Checking #
###################

from contextlib import contextmanager
@contextmanager
def _transformers_logging_suppressed() -> None:
    import logging
    logger_to_original_level = {}
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and name.startswith('transformers.'):
            logger_to_original_level[logger] = logger.level
            logger.setLevel(logging.ERROR)
    yield
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and name.startswith('transformers.'):
            logger.setLevel(logger_to_original_level[logger])
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

######################
# Data Preprocessing #
######################

class ReviewsDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        encoding = TOKENIZER.encode_plus(
            row.content,
            max_length=MAX_SEQUENCE_LENGTH,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'review_string': df.content,
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
        self.out = nn.Linear(self.bert_model.config.hidden_size, NUMBER_OF_SENTIMENTS)
    
    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        _, pooled_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout_layer(pooled_output)
        prediction = self.out(pooled_output)
        return prediction

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    training_dataloader, validation_dataloader, testing_dataloader = load_data()
    bert_module = BERTModule().to(DEVICE)
    for e in testing_dataloader:
        print(f"e {repr(e)}")
    return
        
if __name__ == '__main__':
    main()
