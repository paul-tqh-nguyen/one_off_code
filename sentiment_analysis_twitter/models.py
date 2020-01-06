#!/usr/bin/python3 -O

"""

Twitter Data Sentiment Analyzer Neural Network Model for Text Classification.

Owner : paul-tqh-nguyen

Created : 10/30/2019

File Name : models.py

File Organization:
* Imports
* Misc Utilities
* Models
* Data Loading Utilities
* Classifier Classes
* Main Runner

"""

###########
# Imports #
###########

# @todo verify that these are all used
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import sklearn.utils
from tqdm import tqdm
import warnings
import os
import csv
import math
import statistics
import time
from contextlib import contextmanager
from string import punctuation
from functools import reduce
from collections import OrderedDict
from typing import List

##################
# Misc Utilities #
##################

@contextmanager
def timer(section_name=None, exitCallback=None):
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    if exitCallback != None:
        exitCallback(elapsed_time)
    elif section_name:
        pass
        #print('Execution of "{section_name}" took {elapsed_time} seconds.'.format(section_name=section_name, elapsed_time=elapsed_time))
    else:
        print('Execution took {elapsed_time} seconds.'.format(elapsed_time=elapsed_time))


##########
# Models #
##########

class SentimentAnalysisNetwork(nn.Module):
    def __init__(self, vocabulary_size,
                 embedding_size=400,
                 lstm_hidden_size=300,
                 number_of_lstm_layers=2,
                 drop_out_probability=0.2) -> None:
        super().__init__()
        # @todo make this not have to be batch_first
        
        self.number_of_lstm_layers=number_of_lstm_layers
        self.lstm_hidden_size=lstm_hidden_size
        #Embedding and LSTM layers
        self.embedding=nn.Embedding(vocabulary_size, embedding_size)
        self.lstm=nn.LSTM(embedding_size, lstm_hidden_size, number_of_lstm_layers, dropout=drop_out_probability, batch_first=True)
        #dropout layer
        self.dropout=nn.Dropout(0.3) # @todo do we need this?
        #Linear and sigmoid layer
        self.fc1=nn.Linear(lstm_hidden_size, 64)
        self.fc2=nn.Linear(64, 16)
        self.fc3=nn.Linear(16,1)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x): # @todo rename this input ; add types
        batch_size=x.size()
        #Embadding and LSTM output
        embedd=self.embedding(x)
        lstm_out, _ = self.lstm(embedd)
        #stack up the lstm output
        lstm_out=lstm_out.contiguous().view(-1, self.lstm_hidden_size)
        #dropout and fully connected layers
        out=self.dropout(lstm_out)
        out=self.fc1(out)
        out=self.dropout(out)
        out=self.fc2(out)
        out=self.dropout(out)
        out=self.fc3(out)
        sig_out=self.sigmoid(out)
        sig_out=sig_out.view(batch_size, -1)
        sig_out=sig_out[:, -1]
        return sig_out

##########################
# Data Loading Utilities #
##########################

def normalize_string(input_string: str) -> str:
    normalized_string = input_string
    normalized_string = ''.join(filter(lambda character: character not in punctuation, normalized_string))
    normalized_string = ' '.join(filter(lambda word: word is not '', normalized_string.split()))
    normalized_string = normalized_string.lower()
    return normalized_string

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
TEXT_DATA_LOCATION = os.path.join(CURRENT_FILE_PATH, "./data/reviews.txt")
LABEL_DATA_LOCATION = os.path.join(CURRENT_FILE_PATH, "./data/labels.txt")

def sentiment_to_id(sentiment_label: str) -> int:
    id = 0 if sentiment_label == "negative" else 1 if sentiment_label == "positive" else None
    assert id is not None, "Input data not handled correctly."
    return id

TRAINING_DATA_SENTIMENT_COLUMN_NAME = "label"
TRAINING_DATA_TEXT_COLUMN_NAME = "text"

def get_normalized_training_dataframe() -> pd.DataFrame:
    global TRAINING_DATA_LOCATION
    texts_dataframe = pd.read_csv(TEXT_DATA_LOCATION, names=[TRAINING_DATA_TEXT_COLUMN_NAME], quoting=csv.QUOTE_NONE)
    texts_series = texts_dataframe[TRAINING_DATA_TEXT_COLUMN_NAME]
    tqdm.pandas(desc="Normalizing texts")
    normalized_texts_series = texts_series.progress_apply(normalize_string)
    tqdm.pandas(desc="Loading labels")
    labels_dataframe = pd.read_csv(LABEL_DATA_LOCATION, names=[TRAINING_DATA_SENTIMENT_COLUMN_NAME], quoting=csv.QUOTE_NONE)
    labels_series = labels_dataframe[TRAINING_DATA_SENTIMENT_COLUMN_NAME]
    label_ids_series = labels_series.progress_apply(sentiment_to_id)
    training_dataframe = pd.DataFrame({
        TRAINING_DATA_TEXT_COLUMN_NAME: normalized_texts_series,
        TRAINING_DATA_SENTIMENT_COLUMN_NAME: label_ids_series,
    })
    return training_dataframe

UNKNOWN_WORD_ID = 0
PADDING_TOKEN_ID = 1
SPECIAL_TOKENS = [UNKNOWN_WORD_ID, PADDING_TOKEN_ID]

def vocabulary_from_training_dataframe(training_dataframe: pd.DataFrame): # @todo add return types
    global UNKNOWN_WORD_ID
    global SPECIAL_TOKENS
    texts_series = training_dataframe[TRAINING_DATA_TEXT_COLUMN_NAME]
    all_text = " ".join(texts_series)
    all_words = all_text.split()
    all_words_progress_bar = tqdm(all_words)
    all_words_progress_bar.set_description("Determining unique vocabulary")
    vocabulary = set(all_words_progress_bar)
    number_of_special_tokens = len(SPECIAL_TOKENS)
    vocabulary_to_id_mapping = {word : index+number_of_special_tokens for index, word in enumerate(vocabulary)}
    word_tokenizer = lambda word: UNKNOWN_WORD_ID if word not in vocabulary_to_id_mapping else vocabulary_to_id_mapping[word]
    vocabulary_size = len(vocabulary) + number_of_special_tokens
    return vocabulary, word_tokenizer, vocabulary_size

def tokenized_training_dataframe_from_normalized_training_dataframe(normalized_training_dataframe: pd.DataFrame, word_tokenizer) -> pd.DataFrame:
    sentence_tokenizer = lambda sentence: list(map(word_tokenizer, sentence.split()))
    texts_series = normalized_training_dataframe[TRAINING_DATA_TEXT_COLUMN_NAME]
    tokenized_texts_series = texts_series.apply(sentence_tokenizer)
    tokenized_training_dataframe = normalized_training_dataframe.copy()
    tokenized_training_dataframe.loc[:, TRAINING_DATA_TEXT_COLUMN_NAME] = tokenized_texts_series
    return tokenized_training_dataframe

def pad_tokens(tokens: List[int], final_length: int) -> List[int]:
    global PADDING_TOKEN_ID
    number_of_padding_tokens_needed = final_length-len(tokens)
    if number_of_padding_tokens_needed < 0:
        padded_tokens = tokens[:final_length]
    else:
        padded_tokens = tokens + [PADDING_TOKEN_ID]*number_of_padding_tokens_needed
    return padded_tokens

def pad_training_tokenized_texts_series(training_tokenized_texts_series: pd.Series, final_length: int) -> pd.Series:
    padder = lambda tokens: pad_tokens(tokens, final_length)
    padded_training_tokenized_texts_series = training_tokenized_texts_series.apply(padder)
    return padded_training_tokenized_texts_series

TRAINING_DATA_PORTION = 0.90
VALIDATION_DATA_PORTION = 1 - TRAINING_DATA_PORTION

MAX_TEXT_WORD_LENGTH = None # @todo use this

RANDOMNESS_SEED = 1

def get_training_and_validation_dataloaders(batch_size: int): # @todo add return types
    # @todo abstract out chunks below
    global TRAINING_DATA_PORTION
    global VALIDATION_DATA_PORTION
    global TRAINING_DATA_SENTIMENT_COLUMN_NAME
    global TRAINING_DATA_TEXT_COLUMN_NAME
    global MAX_TEXT_WORD_LENGTH
    normalized_training_dataframe = get_normalized_training_dataframe()
    normalized_training_dataframe = sklearn.utils.shuffle(normalized_training_dataframe, random_state=RANDOMNESS_SEED).reset_index(drop=True)
    # normalized_training_dataframe = normalized_training_dataframe.truncate(after=200) # @todo remove this
    vocabulary, word_tokenizer, vocabulary_size = vocabulary_from_training_dataframe(normalized_training_dataframe)
    tokenized_training_dataframe = tokenized_training_dataframe_from_normalized_training_dataframe(normalized_training_dataframe, word_tokenizer)
    text_lengths_series = tokenized_training_dataframe[TRAINING_DATA_TEXT_COLUMN_NAME].apply(len)
    unpadded_max_text_word_length = max(text_lengths_series)
    if unpadded_max_text_word_length > 1024:
        warnings.warn("Got an unexpectedly large input text size of {max_text_word_length}".format(max_text_word_length=unpadded_max_text_word_length))
    unpadded_median_text_word_length = statistics.median(text_lengths_series)
    MAX_TEXT_WORD_LENGTH = 2**math.ceil(math.log2(unpadded_median_text_word_length))
    print("Max Sequence Length: {max_sequence_length}".format(max_sequence_length=MAX_TEXT_WORD_LENGTH))
    all_training_tokenized_texts_series = tokenized_training_dataframe[TRAINING_DATA_TEXT_COLUMN_NAME]
    all_training_padded_tokenized_texts_series = pad_training_tokenized_texts_series(all_training_tokenized_texts_series, MAX_TEXT_WORD_LENGTH)
    all_training_sentiments_series = tokenized_training_dataframe[TRAINING_DATA_SENTIMENT_COLUMN_NAME]
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
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
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
        self.training_dataloader, self.validation_dataloader, vocabulary_size = get_training_and_validation_dataloaders(batch_size)
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
            self.print_epoch_training_preamble()
            self.note_validation_statistics()
            self.most_recent_epoch_training_loss = 0
            training_process_bar = tqdm(enumerate(self.training_dataloader), total=len(self.training_dataloader))
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
        #print("\n\n")
        with timer(section_name="zero out gradients"):
            self.model.zero_grad()
        with timer(section_name="get predicted label"):
            predicted_label = self.model(inputs).squeeze()
        with timer(section_name="get loss"):
            label = labels.float().squeeze()
            loss = self.loss_function(predicted_label, label) # @todo make this support batch sizes that are not 1
        with timer(section_name="self.most_recent_epoch_training_loss += float(loss)"):
            self.most_recent_epoch_training_loss += float(loss)
        with timer(section_name="loss.backward()"):
            loss.backward()
        with timer(section_name="clip gradient"):
            nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRADIENT_NORM)
        with timer(section_name="optimizer step"):
            self.optimizer.step()
        return None
    
    def print_epoch_training_preamble(self) -> None:
        print()
        print("Training for epoch {number_of_completed_epochs}".format(number_of_completed_epochs=self.number_of_completed_epochs))
        return None
    
    def print_epoch_training_postamble(self) -> None:
        print("Training for epoch {number_of_completed_epochs}.".format(number_of_completed_epochs=self.number_of_completed_epochs))
        print("Total loss for epoch {number_of_completed_epochs}: {loss}".format(number_of_completed_epochs=self.number_of_completed_epochs, loss=self.most_recent_epoch_training_loss))
        print("Mean loss for epoch {number_of_completed_epochs}: {loss}".format(
            number_of_completed_epochs=self.number_of_completed_epochs,
            loss=self.most_recent_epoch_training_loss/len(self.training_dataloader)))
        print()
        return None
    
    def note_validation_statistics(self) -> None:
        number_of_validation_datapoints = len(self.validation_dataloader.dataset)
        total_validation_loss = 0
        total_number_of_incorrect_results = number_of_validation_datapoints
        self.model.eval()
        validation_data_progress_bar = tqdm(self.validation_dataloader, total=len(self.validation_dataloader))
        validation_data_progress_bar.set_description("Calculating validation statistics...")
        for validation_inputs, validation_labels in validation_data_progress_bar:
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
