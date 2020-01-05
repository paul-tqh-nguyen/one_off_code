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
* Classifier Classes
* Main Runner

"""

###########
# Imports #
###########

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import sklearn.utils
import tqdm
import os
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
            # ("final_drop_out_layer", nn.Dropout(self.drop_out_probability)), # @todo do we need this layer? Also, this is causing bugs, so this is implemented incorrectly
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
        predicted_results = self.prediction_layers(encoded_results)
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

RANDOMNESS_SEED = 1

def get_training_and_validation_dataloaders(): # @todo add return types
    # @todo abstract out chunks below
    global TRAINING_DATA_PORTION
    global VALIDATION_DATA_PORTION
    global TRAINING_DATA_SENTIMENT_COLUMN_INDEX
    global TRAINING_DATA_TEXT_COLUMN_INDEX
    global MAX_TWEET_WORD_LENGTH
    normalized_training_dataframe = get_normalized_training_dataframe()
    normalized_training_dataframe = sklearn.utils.shuffle(normalized_training_dataframe, random_state=RANDOMNESS_SEED).reset_index(drop=True)
    # normalized_training_dataframe = normalized_training_dataframe.truncate(after=500) # @todo remove this
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

######################
# Classifier Classes #
######################

class SentimentAnalysisClassifier():
    def __init__(self, # @todo add types
                 batch_size=1,
                 embedding_size=400,
                 lstm_hidden_size=256,
                 number_of_lstm_layers=2,
                 drop_out_probability=0.2,
                 learning_rate=1e-3,
                 number_of_iterations_between_updates=1000,
    ) -> None:
        self.batch_size = batch_size # @todo do something with this
        self.training_dataloader, self.validation_dataloader, vocabulary_size = get_training_and_validation_dataloaders()
        self.loss_function = nn.BCELoss()
        self.model = SentimentAnalysisNetwork(
            vocabulary_size, 
            embedding_size=embedding_size,
            lstm_hidden_size=lstm_hidden_size,
            number_of_lstm_layers=number_of_lstm_layers,
            drop_out_probability=drop_out_probability)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate) # @todo try Adam
        self.number_of_iterations_between_updates = number_of_iterations_between_updates
        self.number_of_completed_epochs = 0
        self.most_recent_epoch_training_loss = 0

    def train(self, number_of_epochs_to_train: int) -> None:
        self.model.train()
        for _ in range(number_of_epochs_to_train):
            self.note_validation_statistics()
            self.print_epoch_training_preamble()
            self.most_recent_epoch_training_loss = 0
            training_process_bar = tqdm.tqdm(enumerate(self.training_dataloader), total=len(self.training_dataloader))
            for training_input_index, (inputs, labels) in training_process_bar:
                self.train_one_iteration(inputs, labels)
                mean_training_loss = round(self.most_recent_epoch_training_loss/(1+training_input_index), 8)
                training_process_bar.set_description("Mean training loss: {:0.8f}".format(mean_training_loss))
                if 0 == (training_input_index % self.number_of_iterations_between_updates) and training_input_index != 0:
                    self.note_validation_statistics()
            self.print_epoch_training_postamble()
            self.number_of_completed_epochs += 1
        return None
    
    def train_one_iteration(self, inputs, labels) -> None: # @todo add input types
        self.model.zero_grad()
        predicted_label = self.model(inputs).squeeze()
        label = labels.float().squeeze()
        loss = self.loss_function(predicted_label, label) # @todo make this support batch sizes that are not 1
        self.most_recent_epoch_training_loss += float(loss)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRADIENT_NORM)
        self.optimizer.step()
        return None
    
    def print_epoch_training_preamble(self) -> None:
        print("Training for epoch {number_of_completed_epochs}".format(number_of_completed_epochs=self.number_of_completed_epochs))
        return None
    
    def print_epoch_training_postamble(self) -> None:
        print("Training for epoch {number_of_completed_epochs}.".format(number_of_completed_epochs=self.number_of_completed_epochs))
        print("Total loss for epoch {number_of_completed_epochs}: {loss}".format(number_of_completed_epochs=self.number_of_completed_epochs, loss=self.most_recent_epoch_training_loss))
        print("Mean loss for epoch {number_of_completed_epochs}: {loss}".format(
            number_of_completed_epochs=self.number_of_completed_epochs,
            loss=self.most_recent_epoch_training_loss/len(self.training_dataloader)))
        return None
    
    def note_validation_statistics(self) -> None:
        number_of_validation_datapoints = len(self.validation_dataloader.dataset)
        total_validation_loss = 0
        total_number_of_incorrect_results = number_of_validation_datapoints
        self.model.eval()
        for validation_inputs, validation_labels in self.validation_dataloader:
            validation_labels = validation_labels.float()
            validation_label = validation_labels.squeeze()
            predicted_validation_labels = self.model(validation_inputs)
            predicted_validation_label = predicted_validation_labels.squeeze()
            validation_loss = self.loss_function(predicted_validation_label, validation_label) # @todo make this support batch sizes that are not 1
            total_validation_loss += float(validation_loss)
            number_of_correct_results = torch.sum(validation_labels.eq(torch.round(predicted_validation_labels))).squeeze()
            total_number_of_incorrect_results -= number_of_correct_results
        mean_validation_loss = total_validation_loss / number_of_validation_datapoints
        self.model.train()
        # @todo make this look nicer
        print()
        print("    Mean Validation Loss: {:.6f}".format(mean_validation_loss))
        print("    Incorrect Results: {} / {}".format(total_number_of_incorrect_results, number_of_validation_datapoints))
        print()
        return None

###############
# Main Runner #
###############

# @todo turn these into CLI parameters

BATCH_SIZE = 64
EMBEDDING_SIZE = 400
LSTM_HIDDEN_SIZE = 256
NUMBER_OF_LSTM_LAYERS = 2
DROP_OUT_PROBABILITY = 0.2
LEARNING_RATE = 1e-3
MAX_GRADIENT_NORM = 5
NUMBER_OF_EPOCHS = 9999 # @todo change this
NUMBER_OF_ITERATIONS_BETWEEN_UPDATES = 5000000000000

def main() -> None:
    global BATCH_SIZE 
    global EMBEDDING_SIZE
    global LSTM_HIDDEN_SIZE
    global NUMBER_OF_LSTM_LAYERS
    global DROP_OUT_PROBABILITY
    global LEARNING_RATE
    global MAX_GRADIENT_NORM
    global NUMBER_OF_EPOCHS
    global NUMBER_OF_ITERATIONS_BETWEEN_UPDATES
    print("Initializing classifier...")
    classifier = SentimentAnalysisClassifier(
        batch_size=BATCH_SIZE,
        embedding_size=EMBEDDING_SIZE,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        number_of_lstm_layers=NUMBER_OF_LSTM_LAYERS,
        drop_out_probability=DROP_OUT_PROBABILITY,
        learning_rate=LEARNING_RATE,
        number_of_iterations_between_updates=NUMBER_OF_ITERATIONS_BETWEEN_UPDATES,
    )
    print("Starting training...")
    classifier.train(NUMBER_OF_EPOCHS)
    #################################################
    print("This module contains sentiment analysis models for Twitter text classification.")

if __name__ == '__main__':
    main()
