#!/usr/bin/python3 -O

"""

Simple Sentiment Analyzer.

Owner : paul-tqh-nguyen

Created : 10/30/2019

File Name : sentiment_analysis.py

File Organization:
* Imports
* Misc. Utilities
* String <-> Tensor Utilities
* Model Definitions
* Dataset Definitions
* Classifier Definitions
* Main Runner

"""

###########
# Imports #
###########

from word2vec_utilities import WORD2VEC_MODEL, WORD2VEC_VECTOR_LENGTH
from string_processing_utilities import normalized_words_from_text_string
from typing import Iterable, List
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from contextlib import contextmanager
import time
import csv
import random

UNSEEN_WORD_TO_TENSOR_MAP = {}

###################
# Misc. Utilities #
###################
 
def dt(var_name_string):
    return 'print("{{variable}} : {{value}}".format(variable="{var_name_string}", value=locals()["{var_name_string}"]))'.format(var_name_string=var_name_string)

@contextmanager
def timeout(time, functionToExecuteOnTimeout=None):
    """NB: This cannot be nested."""
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(time)
    try:
        yield
    except TimeoutError:
        if functionToExecuteOnTimeout is not None:
            functionToExecuteOnTimeout()
    finally:
        signal.signal(signal.SIGALRM, signal.SIG_IGN)

@contextmanager
def timer(print_function_callback):
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    print_function_callback(elapsed_time)

###############################
# String <-> Tensor Utilities #
###############################

def random_vector_for_unseen_word():
    """Similar to Parikh et al. 2016"""
    random_vector = torch.randn(WORD2VEC_VECTOR_LENGTH)
    normalized_vector = F.normalize(random_vector, dim=0)
    return normalized_vector

def tensor_from_normalized_word(word: str):
    tensor = None
    if word in WORD2VEC_MODEL:
        tensor = torch.from_numpy(WORD2VEC_MODEL[word])
    elif word in UNSEEN_WORD_TO_TENSOR_MAP:
        tensor = UNSEEN_WORD_TO_TENSOR_MAP[word]
    else:
        tensor = random_vector_for_unseen_word()
        UNSEEN_WORD_TO_TENSOR_MAP[word] = tensor
    return tensor

def tensors_from_text_string(text_string: str):
    normalized_words = normalized_words_from_text_string(text_string)
    tensors = map(tensor_from_normalized_word, normalized_words)
    return tensors

def text_string_matrix_from_text_string(text_string: str):
    word_tensors = tuple(tensors_from_text_string(text_string))
    text_string_matrix = torch.stack(word_tensors)
    return text_string_matrix

#######################
# Dataset Definitions #
#######################

RAW_VALUE_TO_SENTIMENT_PAIRS = [
    ("0", "Negative"),
    ("1", "Positive"),
]

RAW_VALUE_TO_SENTIMENT_MAP = dict(RAW_VALUE_TO_SENTIMENT_PAIRS)

SENTIMENTS = list(map(lambda x:x[1], RAW_VALUE_TO_SENTIMENT_PAIRS))

NUMBER_OF_SENTIMENTS = len(SENTIMENTS)

SENTIMENT_INDEX_TO_SENTIMENT_MAP = {index:sentiment for index, sentiment in enumerate(SENTIMENTS)}
SENTIMENT_TO_SENTIMENT_INDEX_MAP = {sentiment:index for index, sentiment in enumerate(SENTIMENTS)}

def sentiment_to_one_hot_vector(sentiment):
    assert sentiment in SENTIMENTS
    sentiment_index = SENTIMENT_TO_SENTIMENT_INDEX_MAP[sentiment]
    one_hot_vector = torch.zeros(NUMBER_OF_SENTIMENTS)
    one_hot_vector[sentiment_index] = 1
    return one_hot_vector

def truncate_sentiment_result(sentiment_result):
    truncated_sentiment_result = torch.floor(sentiment_result/sentiment_result.max())
    assert torch.sum(truncated_sentiment_result) == 1
    return truncated_sentiment_result

TORCH_ARANGE_NUMBER_OF_SENTIMENTS = torch.arange(NUMBER_OF_SENTIMENTS, dtype=torch.float32)

def sentiment_result_to_string(sentiment_result_0):
    sentiment_result = truncate_sentiment_result(sentiment_result_0)
    sentiment_result_string = None
    assert tuple(sentiment_result.shape) == (NUMBER_OF_SENTIMENTS,)
    assert torch.sum(sentiment_result) == 1
    sentiment_index = int(sentiment_result.dot(TORCH_ARANGE_NUMBER_OF_SENTIMENTS))
    sentiment_string = SENTIMENT_INDEX_TO_SENTIMENT_MAP[sentiment_index]
    return sentiment_string

TRAINING_DATA_LOCATION = "./data/train.csv"
TEST_DATA_LOCATION = "./data/test.csv"

VALIDATION_DATA_PORTION = 0.10

TRAINING_DATA_ID_TO_DATA_MAP = {}
TRAINING_DATA_ID_TO_DATA_MAP = {}
TEST_DATA_ID_TO_TEXT_MAP = {} # @todo do something with this

PORTION_OF_TRAINING_DATA_TO_USE = 0.001

with open(TRAINING_DATA_LOCATION, encoding='ISO-8859-1') as training_data_csv_file:
    training_data_csv_reader = csv.DictReader(training_data_csv_file, delimiter=',')
    row_dicts = list(training_data_csv_reader)
    number_of_row_dicts = len(row_dicts)
    for row_dict_index, row_dict in enumerate(row_dicts):
        if row_dict_index/number_of_row_dicts >= PORTION_OF_TRAINING_DATA_TO_USE:
            break
        id = row_dict.pop('ItemID')
        TRAINING_DATA_ID_TO_DATA_MAP[id]=row_dict

with open(TEST_DATA_LOCATION, encoding='ISO-8859-1') as test_data_csv_file:
    test_data_csv_reader = csv.DictReader(test_data_csv_file, delimiter=',')
    for row_dict in test_data_csv_reader:
        id = row_dict['ItemID']
        text = row_dict['SentimentText']
        TEST_DATA_ID_TO_TEXT_MAP[id]=text

class SentimentLabelledDataset(data.Dataset):
    def __init__(self, texts, one_hot_sentiment_vectors):
        self.x_data = texts
        self.y_data = one_hot_sentiment_vectors
        assert len(self.x_data) == len(self.y_data)
        
    def __len__(self):
        assert len(self.x_data) == len(self.y_data)
        return len(self.x_data)
    
    def __getitem__(self, index):
        x_datum = self.x_data[index]
        y_datum = self.y_data[index]
        return x_datum, y_datum

def determine_training_and_validation_datasets():
    data_dictionaries = list(TRAINING_DATA_ID_TO_DATA_MAP.values())
    random.shuffle(data_dictionaries)
    number_of_validation_data_points = round(VALIDATION_DATA_PORTION*len(data_dictionaries))
    training_inputs = []
    training_labels = []
    validation_inputs = []
    validation_labels = []
    for data_dictionary_index, data_dictionary in enumerate(data_dictionaries):
        sentiment_text = data_dictionary['SentimentText']
        raw_sentiment = data_dictionary['Sentiment']
        sentiment = RAW_VALUE_TO_SENTIMENT_MAP[raw_sentiment]
        one_hot_vector = sentiment_to_one_hot_vector(sentiment)
        if data_dictionary_index < number_of_validation_data_points:
            validation_inputs.append(sentiment_text)
            validation_labels.append(one_hot_vector)
        else:
            training_inputs.append(sentiment_text)
            training_labels.append(one_hot_vector)
    training_dataset = SentimentLabelledDataset(training_inputs, training_labels)
    validation_dataset = SentimentLabelledDataset(validation_inputs, validation_labels)
    return training_dataset, validation_dataset

#####################
# Model Definitions #
#####################

class SelfAttentionLayers(nn.Module):
    def __init__(self, input_size=400, number_of_attention_heads=2, hidden_size=None):
        super().__init__()
        if __debug__: # only used for assertion checking
            self.number_of_attention_heads = number_of_attention_heads
        if hidden_size == None:
            hidden_size = input_size // 2
        self.attention_layers = nn.Sequential(OrderedDict([
            ("reduction_layer", nn.Linear(input_size, hidden_size)),
            ("activation", nn.ReLU(True)),
            ("attention_layer", nn.Linear(hidden_size, number_of_attention_heads)),
        ]))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def to(self, device):
        self.device = device
        self.attention_layers.to(device)
        
    def forward(self, input_matrix):
        max_number_of_words, batch_size, input_size = input_matrix.shape
        attention_weights_pre_softmax = self.attention_layers(input_matrix) 
        assert tuple(attention_weights_pre_softmax.shape) == (max_number_of_words, batch_size, self.number_of_attention_heads), "attention_weights_pre_softmax has unexpected dimensions."
        attention_weights = F.softmax(attention_weights_pre_softmax, dim=0)
        assert tuple(attention_weights.shape) == (max_number_of_words, batch_size, self.number_of_attention_heads), "attention_weights has unexpected dimensions."
        attention_weights_batch_first = attention_weights.transpose(0,1)
        assert tuple(attention_weights_batch_first.shape) == (batch_size, max_number_of_words, self.number_of_attention_heads), "attention_weights_batch_first has unexpected dimensions."
        attention_weights_batch_first_transpose = attention_weights_batch_first.transpose(1,2)
        assert tuple(attention_weights_batch_first_transpose.shape) == (batch_size, self.number_of_attention_heads, max_number_of_words), \
            "attention_weights_batch_first_transpose has unexpected dimensions."
        attention_weights_times_transpose = attention_weights_batch_first.matmul(attention_weights_batch_first_transpose)
        assert tuple(attention_weights_times_transpose.shape) == (batch_size, max_number_of_words, max_number_of_words), "attention_weights_times_transpose has unexpected dimensions."
        identity_matrix = torch.eye(max_number_of_words).repeat(batch_size,1,1).to(self.device)
        assert tuple(identity_matrix.shape) == (batch_size, max_number_of_words, max_number_of_words), "identity_matrix has unexpected dimensions."
        attenion_regularization_penalty_unnormalized = attention_weights_times_transpose - identity_matrix
        assert tuple(attenion_regularization_penalty_unnormalized.shape) == (batch_size, max_number_of_words, max_number_of_words), \
            "attenion_regularization_penalty_unnormalized has unexpected dimensions."
        attenion_regularization_penalty_per_batch = torch.sqrt((attenion_regularization_penalty_unnormalized**2).sum(dim=1).sum(dim=1))
        attenion_regularization_penalty = attenion_regularization_penalty_per_batch.sum(dim=0)
        attention_weights_duplicated = attention_weights.view(-1,1).repeat(1,input_size).view(max_number_of_words, batch_size, self.number_of_attention_heads*input_size)
        assert tuple(attention_weights_duplicated.shape) == (max_number_of_words, batch_size, self.number_of_attention_heads*input_size), "attention_weights_duplicated has unexpected dimensions."
        input_matrix_duplicated = input_matrix.repeat(1,1,self.number_of_attention_heads) 
        assert tuple(input_matrix_duplicated.shape) == (max_number_of_words, batch_size, self.number_of_attention_heads*input_size), "input_matrix_duplicated has unexpected dimensions."
        weight_adjusted_input_matrix = torch.mul(attention_weights_duplicated, input_matrix_duplicated)
        assert tuple(weight_adjusted_input_matrix.shape) == (max_number_of_words, batch_size, self.number_of_attention_heads*input_size), "weight_adjusted_input_matrix has unexpected dimensions."
        attended_matrix = torch.sum(weight_adjusted_input_matrix, dim=0)
        assert tuple(attended_matrix.shape) == (batch_size, self.number_of_attention_heads*input_size), "attended_matrix has unexpected dimensions."
        return attended_matrix, attenion_regularization_penalty

class SentimentAnalysisNetwork(nn.Module):
    def __init__(self, embedding_hidden_size=200, lstm_dropout_prob=0.2, number_of_attention_heads=2, attention_hidden_size=24):
        super().__init__()
        if __debug__: # only used for assertion checking
            self.embedding_hidden_size = embedding_hidden_size
            self.number_of_attention_heads = number_of_attention_heads
        self.embedding_layers = nn.Sequential(OrderedDict([
            ("reduction_layer", nn.Linear(WORD2VEC_VECTOR_LENGTH, embedding_hidden_size)),
            ("activation", nn.ReLU(True)),
        ]))
        self.encoding_layers = nn.LSTM(embedding_hidden_size, embedding_hidden_size, num_layers=2, dropout=lstm_dropout_prob, bidirectional=True)
        encoding_hidden_size = 2*embedding_hidden_size 
        self.attention_layers = SelfAttentionLayers(input_size=encoding_hidden_size, number_of_attention_heads=number_of_attention_heads, hidden_size=attention_hidden_size)
        attention_size = encoding_hidden_size*number_of_attention_heads
        self.prediction_layers = nn.Sequential(OrderedDict([
            ("linear_classification_layer", nn.Linear(attention_size, NUMBER_OF_SENTIMENTS)),
            ("softmax", nn.Softmax(dim=1)),
        ]))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_layers.to(self.device)
        self.encoding_layers.to(self.device)
        self.attention_layers.to(self.device)
        self.prediction_layers.to(self.device)
        
    def forward(self, text_strings: Iterable[str]):
        batch_size = len(text_strings)
        text_string_matrices_unpadded = [text_string_matrix_from_text_string(text_string) for text_string in text_strings]
        text_string_batch_matrix = torch.nn.utils.rnn.pad_sequence(text_string_matrices_unpadded)
        text_string_batch_matrix = text_string_batch_matrix.to(self.device)
        max_number_of_words = max(map(len, text_string_matrices_unpadded))
        assert tuple(text_string_batch_matrix.shape) == (max_number_of_words, batch_size, WORD2VEC_VECTOR_LENGTH), "text_string_batch_matrix has unexpected dimensions."
        embeddeding_batch_matrix = self.embedding_layers(text_string_batch_matrix)
        assert tuple(embeddeding_batch_matrix.shape) == (max_number_of_words, batch_size, self.embedding_hidden_size)
        encoding_batch_matrix, _ = self.encoding_layers(embeddeding_batch_matrix)
        assert tuple(encoding_batch_matrix.shape) == (max_number_of_words, batch_size, 2*self.embedding_hidden_size)
        attention_matrix, attenion_regularization_penalty = self.attention_layers(encoding_batch_matrix)
        assert tuple(attention_matrix.shape) == (batch_size, self.number_of_attention_heads*2*self.embedding_hidden_size)
        prediction_scores = self.prediction_layers(attention_matrix)
        assert tuple(prediction_scores.shape) == (batch_size, NUMBER_OF_SENTIMENTS)
        return prediction_scores, attenion_regularization_penalty

##########################
# Classifier Definitions #
##########################

class SentimentAnalysisClassifier():
    def __init__(self, batch_size=1, learning_rate=1e-2, attenion_regularization_penalty_multiplicative_factor=0.1,
                 embedding_hidden_size=200, lstm_dropout_prob=0.2, number_of_attention_heads=2, attention_hidden_size=241
    ):
        self.attenion_regularization_penalty_multiplicative_factor = attenion_regularization_penalty_multiplicative_factor
        self.number_of_completed_epochs = 0
        self.most_recent_epoch_loss = 0
        self.loss_function = nn.BCELoss()
        self.model = SentimentAnalysisNetwork(
            embedding_hidden_size=embedding_hidden_size,
            lstm_dropout_prob=lstm_dropout_prob,
            number_of_attention_heads=number_of_attention_heads,
            attention_hidden_size=attention_hidden_size)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        training_set, validation_set = determine_training_and_validation_datasets()
        self.training_generator = data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
        self.validation_generator = data.DataLoader(validation_set, batch_size=1, shuffle=False)
        
    def print_dataset_stats(self):
        print("Training Size: {training_size}".format(training_size=len(self.training_generator.dataset)))
        print("Validation Size: {validation_size}".format(validation_size=len(self.validation_generator.dataset)))
        
    def train(self, number_of_epochs_to_train, number_of_iterations_per_status_update=None):
        self.model.train()
        for new_epoch_index in range(number_of_epochs_to_train):
            epoch_loss = 0
            total_number_of_iterations = len(self.training_generator.dataset)
            for iteration_index, (x_batch, y_batch) in enumerate(self.training_generator):
                if number_of_iterations_per_status_update is not None:
                    if (iteration_index != 0) and (iteration_index % number_of_iterations_per_status_update) == 0:
                        current_global_epoch=self.number_of_completed_epochs+new_epoch_index
                        print("Completed Iteration {iteration_index} / {total_number_of_iterations} of epoch {current_global_epoch}".format(
                            iteration_index=iteration_index,
                            total_number_of_iterations=total_number_of_iterations,
                            current_global_epoch=current_global_epoch))
                y_batch_predicted, attenion_regularization_penalty = self.model(x_batch)
                batch_loss = self.loss_function(y_batch_predicted, y_batch) + attenion_regularization_penalty * self.attenion_regularization_penalty_multiplicative_factor
                self.optimizer.zero_grad()
                batch_loss.backward()
                epoch_loss += float(batch_loss)
                self.optimizer.step()
            self.most_recent_epoch_loss = epoch_loss
            self.number_of_completed_epochs += 1
            
    def evaluate(self, strings: List[str]):
        self.model.eval()
        return self.model(strings)
    
    def print_current_state(self, verbose=False):
        print()
        if verbose:
            print("===================================================================")
        correct_result_number = 0
        for x_batch, y_batch in self.validation_generator:
            assert isinstance(x_batch, tuple)
            assert len(x_batch) == 1
            assert tuple(y_batch.shape) == (1, NUMBER_OF_SENTIMENTS)
            y_datum = y_batch[0]
            expected_result = sentiment_result_to_string(y_datum)
            y_batch_predicted, _ = self.evaluate(x_batch)
            assert y_batch_predicted.shape == (1,NUMBER_OF_SENTIMENTS)
            actual_result = sentiment_result_to_string(y_batch_predicted[0])
            if verbose:
                input_string = x_batch[0]
                print("Input: {x}".format(x=input_string))
                print("Expected Output: {x}".format(x=expected_result))
                print("Actual Output: {x}".format(x=actual_result))
                print("Raw Expected Output: {x}".format(x=y_datum))
                print("Raw Actual Output: {x}".format(x=y_batch_predicted))
                print("\n")
            if actual_result == expected_result:
                correct_result_number += 1
        total_result_number = len(self.validation_generator.dataset)
        print("Truncated Correctness Portion: {correct_result_number} / {total_result_number}".format(correct_result_number=correct_result_number, total_result_number=total_result_number))
        print("Loss per datapoint for epoch {epoch_index} is {loss}".format(epoch_index=self.number_of_completed_epochs,loss=self.most_recent_epoch_loss/total_result_number))
        print("Total loss for epoch {epoch_index} is {loss}".format(epoch_index=self.number_of_completed_epochs,loss=self.most_recent_epoch_loss))
        if verbose:
            print("===================================================================")
            
###############
# Main Runner #
###############

def main(batch_size=1,
         learning_rate=1e-2,
         attenion_regularization_penalty_multiplicative_factor=0.1,
         embedding_hidden_size=200,
         lstm_dropout_prob=0.2,
         number_of_attention_heads=2,
         attention_hidden_size=24
):
    classifier = SentimentAnalysisClassifier(
        batch_size=batch_size,
        learning_rate=learning_rate,
        attenion_regularization_penalty_multiplicative_factor=attenion_regularization_penalty_multiplicative_factor,
        embedding_hidden_size=embedding_hidden_size,
        lstm_dropout_prob=lstm_dropout_prob,
        number_of_attention_heads=number_of_attention_heads,
        attention_hidden_size=attention_hidden_size)
    number_of_epochs = 9000
    number_of_epochs_between_updates = 1
    number_of_iterations_between_updates = 500
    number_of_updates = number_of_epochs//number_of_epochs_between_updates
    print_verbosely = False
    print()
    print("Starting Training.")
    print()
    classifier.print_dataset_stats()
    classifier.print_current_state(print_verbosely)
    for update_index in range(number_of_updates):
        with timer(lambda number_of_seconds: print("Time for epochs {start_epoch_index} to {end_epoch_index}: {time_for_epochs} seconds".format(
                start_epoch_index=update_index*number_of_epochs_between_updates,
                end_epoch_index=(update_index+1)*number_of_epochs_between_updates-1,
                time_for_epochs=number_of_seconds,
        ))):
            classifier.train(number_of_epochs_between_updates, number_of_iterations_between_updates)
            classifier.print_current_state(print_verbosely)
    classifier.print_current_state(print_verbosely)
    print("Training Complete.")

if __name__ == '__main__':
    main()
