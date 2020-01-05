#!/usr/bin/python3 -O

"""

Twitter Data Sentiment Analyzer Neural Network Model for Text Classification.

Owner : paul-tqh-nguyen

Created : 10/30/2019

File Name : models.py

File Organization:
* Imports
* Models
* Data Loading Utilities
* Main Runner

"""

###########
# Imports #
###########

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
import csv
import math
import statistics
from string import punctuation
from functools import reduce
from collections import OrderedDict
from typing import List

##########
# Models #
##########

class SentimentAnalysisNetwork(nn.Module):
    def __init__(self, vocabulary_size,
                 embedding_size=400,
                 lstm_hidden_size=300,
                 number_of_lstm_layers=2,
                 drop_out_probability=0.2):
        super().__init__()
        global NUMBER_OF_SENTIMENTS
        # @todo make this not have to be batch_first
        
        # @todo use these in printing state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # @todo use this
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.number_of_lstm_layers = number_of_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.drop_out_probability = drop_out_probability
        
        self.embedding_layers = nn.Sequential(OrderedDict([
            ("embedding_layer", nn.Embedding(vocabulary_size, embedding_size)),
            # ("activation", nn.ReLU(True)), # @todo is it correct to have this? 
        ]))
        self.encoding_layers = nn.Sequential(OrderedDict([
            ("LSTM_layers", nn.LSTM(self.embedding_size, self.lstm_hidden_size, self.number_of_lstm_layers, dropout=self.drop_out_probability, bidirectional=True, batch_first=True)),
            ("final_drop_out_layer", nn.Dropout(self.drop_out_probability)), # @todo do we need this layer?
        ]))
        # @todo add attention layers
        self.prediction_layers = nn.Sequential(OrderedDict([
            ("linear_classification_layer", nn.Linear(2*self.lstm_hidden_size, 1)),
            ("squashing_layer", nn.Sigmoid()), # @todo experiment with tanh
        ]))
        
    def forward(self, x): # @todo rename this input
        batch_size = x.size()
        embedded_inputs = self.embedding_layers(x)
        encoded_results, _ = self.encoding_layers(embedded_inputs)
        predicted_results = self.prediction_layers(out)
        predicted_results = predicted_results.view(batch_size, -1)
        predicted_results = predicted_results[:, -1]
        return predicted_results

##########################
# Data Loading Utilities #
##########################

def normalize_string(input_string: str) -> str:
    normalized_string = input_string
    normalized_string = ''.join(filter(lambda character: character not in punctuation, normalized_string))
    normalized_string = normalized_string.lower()
    return normalized_string

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
TRAINING_DATA_LOCATION = os.path.join(CURRENT_FILE_PATH, "./data/training.txt")
# @todo add global for testing data and do something with it

TRAINING_DATA_SENTIMENT_COLUMN_INDEX = 0
TRAINING_DATA_TEXT_COLUMN_INDEX = 1

def get_normalized_training_dataframe() -> pd.DataFrame:
    global TRAINING_DATA_LOCATION
    training_dataframe = pd.read_csv(TRAINING_DATA_LOCATION, sep="\t", header=None, quoting=csv.QUOTE_NONE)
    texts_series = training_dataframe[TRAINING_DATA_TEXT_COLUMN_INDEX]
    normalized_texts_series = texts_series.apply(normalize_string)
    training_dataframe.loc[:, TRAINING_DATA_TEXT_COLUMN_INDEX] = normalized_texts_series
    return training_dataframe

UNKNOWN_WORD_ID = 0
PADDING_TOKEN_ID = 1
SPECIAL_TOKENS = [UNKNOWN_WORD_ID, PADDING_TOKEN_ID]

def vocabulary_from_training_dataframe(training_dataframe: pd.DataFrame): # @todo add return types
    global UNKNOWN_WORD_ID
    global SPECIAL_TOKENS
    texts_series = training_dataframe[TRAINING_DATA_TEXT_COLUMN_INDEX]
    text_word_lists_series = texts_series.apply(str.split)
    text_word_sets_series = text_word_lists_series.apply(set)
    vocabulary = reduce(set.union, text_word_sets_series)
    number_of_special_tokens = len(SPECIAL_TOKENS)
    vocabulary_to_id_mapping = {word : index+number_of_special_tokens for index, word in enumerate(vocabulary)}
    word_tokenizer = lambda word: UNKNOWN_WORD_ID if word not in vocabulary_to_id_mapping else vocabulary_to_id_mapping[word]
    vocabulary_size = len(vocabulary) + number_of_special_tokens
    return vocabulary, word_tokenizer, vocabulary_size

def tokenized_training_dataframe_from_normalized_training_dataframe(normalized_training_dataframe: pd.DataFrame, word_tokenizer) -> pd.DataFrame:
    sentence_tokenizer = lambda sentence: list(map(word_tokenizer, sentence.split()))
    texts_series = normalized_training_dataframe[TRAINING_DATA_TEXT_COLUMN_INDEX]
    tokenized_texts_series = texts_series.apply(sentence_tokenizer)
    tokenized_training_dataframe = normalized_training_dataframe.copy()
    tokenized_training_dataframe.loc[:, TRAINING_DATA_TEXT_COLUMN_INDEX] = tokenized_texts_series
    return tokenized_training_dataframe

def pad_tokens(tokens: List[int], final_length: int) -> List[int]:
    global PADDING_TOKEN_ID
    number_of_padding_tokens_needed = final_length-len(tokens)
    padded_tokens = tokens + [PADDING_TOKEN_ID]*number_of_padding_tokens_needed
    return padded_tokens

def pad_training_tokenized_texts_series(training_tokenized_texts_series: pd.Series, final_length: int) -> pd.Series:
    padder = lambda tokens: pad_tokens(tokens, final_length)
    padded_training_tokenized_texts_series = training_tokenized_texts_series.apply(padder)
    return padded_training_tokenized_texts_series

TRAINING_DATA_PORTION = 0.75
VALIDATION_DATA_PORTION = 1 - TRAINING_DATA_PORTION

MAX_TWEET_WORD_LENGTH = None

def get_training_and_validation_dataloaders(): # @todo add return types
    # @todo abstract out chunks below
    global TRAINING_DATA_PORTION
    global VALIDATION_DATA_PORTION
    global TRAINING_DATA_SENTIMENT_COLUMN_INDEX
    global TRAINING_DATA_TEXT_COLUMN_INDEX
    global MAX_TWEET_WORD_LENGTH
    normalized_training_dataframe = get_normalized_training_dataframe()
    vocabulary, word_tokenizer, vocabulary_size = vocabulary_from_training_dataframe(normalized_training_dataframe)
    tokenized_training_dataframe = tokenized_training_dataframe_from_normalized_training_dataframe(normalized_training_dataframe, word_tokenizer)
    MAX_TWEET_WORD_LENGTH = max(tokenized_training_dataframe[TRAINING_DATA_TEXT_COLUMN_INDEX].apply(len)) # @todo abstract this global updating out
    MAX_TWEET_WORD_LENGTH = 2**math.ceil(math.log2(MAX_TWEET_WORD_LENGTH))
    assert MAX_TWEET_WORD_LENGTH <= 256
    all_training_tokenized_texts_series = tokenized_training_dataframe[TRAINING_DATA_TEXT_COLUMN_INDEX]
    all_training_padded_tokenized_texts_series = pad_training_tokenized_texts_series(all_training_tokenized_texts_series, MAX_TWEET_WORD_LENGTH)
    all_training_sentiments_series = tokenized_training_dataframe[TRAINING_DATA_SENTIMENT_COLUMN_INDEX]
    all_training_data_size = len(all_training_sentiments_series)
    training_length = round(TRAINING_DATA_PORTION * all_training_data_size)
    validation_length = round(VALIDATION_DATA_PORTION * all_training_data_size)
    training_inputs = all_training_padded_tokenized_texts_series[:training_length]
    training_outputs =  all_training_sentiments_series[:training_length]
    validation_inputs = all_training_padded_tokenized_texts_series[training_length:]
    validation_outputs = all_training_sentiments_series[training_length:]
    assert len(validation_inputs) == validation_length and len(validation_outputs) == validation_length
    training_inputs = torch.LongTensor(list(training_inputs))
    training_outputs = torch.FloatTensor(list(training_outputs))
    validation_inputs = torch.LongTensor(list(validation_inputs))
    validation_outputs = torch.FloatTensor(list(validation_outputs))
    training_dataset = TensorDataset(training_inputs, training_outputs)
    validation_dataset =  TensorDataset(validation_inputs, validation_outputs)
    training_dataloader = DataLoader(training_dataset)
    validation_dataloader = DataLoader(validation_dataset)
    return training_dataloader, validation_dataloader, vocabulary_size

###############
# Main Runner #
###############

# @todo turn these into CLI parameters

BATCH_SIZE = 64 # @todo use this
EMBEDDING_SIZE = 400
LSTM_HIDDEN_SIZE = 256
NUMBER_OF_LSTM_LAYERS = 2
DROP_OUT_PROBABILITY = 0.2
LEARNING_RATE = 0.001
MAX_GRADIENT_NORM = 5
NUMBER_OF_EPOCHS = 9999 # @todo change this

def main():
    global EMBEDDING_SIZE
    global LSTM_HIDDEN_SIZE
    global NUMBER_OF_LSTM_LAYERS
    global DROP_OUT_PROBABILITY
    global LEARNING_RATE
    global MAX_GRADIENT_NORM
    global NUMBER_OF_EPOCHS
    training_dataloader, validation_dataloader, vocabulary_size = get_training_and_validation_dataloaders()
    loss_function = nn.BCELoss()
    model = SentimentAnalysisNetwork(
        vocabulary_size, 
        embedding_size=EMBEDDING_SIZE,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        number_of_lstm_layers=NUMBER_OF_LSTM_LAYERS,
        drop_out_probability=DROP_OUT_PROBABILITY)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE) # @todo try Adam
    model.train()
    for epoch_index in range(NUMBER_OF_EPOCHS): # @todo finish writing training loop
        for training_input_index, (inputs, labels) in enumerate(training_dataloader):
            model.zero_grad()
            predicted_label = model(inputs).squeeze()
            loss = loss_function(predicted_label, labels.float()) # @todo make this support batch sizes that are not 1
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRADIENT_NORM)
            optimizer.step()
            if training_input_index % 100: # @todo parameterize via the CLI how often we calculate this
                validation_losses = []
                model.eval()
                for validation_inputs, validation_labels in validation_dataloader:
                    predicted_validation_label = model(validation_inputs).squeeze()
                    validation_loss = loss_function(predicted_validation_label, validation_labels.float()) # @todo make this support batch sizes that are not 1
                    validation_losses.append(validation_loss.item())
                model.train()
                print("Epoch: {}/{} \n".format(e+1, epochs), # @todo make this look nicer
                      "Step Index: {} \n".format(training_input_index),
                      "Loss: {:.6f} \n".format(loss.item()),
                      "Val Loss: {:.6f}".format(statistics.mean(validation_losses)))
    #################################################
    print("This module contains sentiment analysis models for Twitter text classification.")

if __name__ == '__main__':
    main()
