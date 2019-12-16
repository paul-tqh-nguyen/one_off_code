#!/usr/bin/python3 -O

"""

Twitter Data Sentiment Analyzer via Neural Network Based Text Classification.

Owner : paul-tqh-nguyen

Created : 10/30/2019

File Name : text_classifier.py

File Organization:
* Imports
* String <-> Tensor Utilities
* Model Definitions
* Dataset Definitions
* Classifier Definitions
* Main Runner

"""

###########
# Imports #
###########

from contextlib import redirect_stdout
from typing import Iterable, List
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import csv
import random
import time
import pickle
import socket
import pandas as pd
if socket.gethostname() != 'phact': # @todo remove this exception
    from matplotlib import pyplot as plt
from string_processing_utilities import timeout, timer, normalized_words_from_text_string, word_string_resembles_meaningful_special_character_sequence_placeholder, PUNCTUATION_SET
from word2vec_utilities import WORD2VEC_MODEL, WORD2VEC_VECTOR_LENGTH
from misc_utilities import *

###############################
# String <-> Tensor Utilities #
###############################

def random_word_vector():
    random_vector = torch.randn(WORD2VEC_VECTOR_LENGTH)
    normalized_vector = F.normalize(random_vector, dim=0)
    return normalized_vector

WORD_VECTOR_FOR_UNKNOWN_WORD = random_word_vector()

UNSEEN_WORD_TO_TENSOR_MAP = {}

def tensor_from_normalized_word(word: str):
    global UNSEEN_WORD_TO_TENSOR_MAP
    global WORD_VECTOR_FOR_UNKNOWN_WORD
    tensor = None
    if word in WORD2VEC_MODEL:
        tensor = torch.from_numpy(WORD2VEC_MODEL[word])
    elif word in UNSEEN_WORD_TO_TENSOR_MAP:
        tensor = UNSEEN_WORD_TO_TENSOR_MAP[word]
    elif word_string_resembles_meaningful_special_character_sequence_placeholder(word) or word in PUNCTUATION_SET:
        tensor = random_word_vector()
        UNSEEN_WORD_TO_TENSOR_MAP[word] = tensor
    else:
        tensor = WORD_VECTOR_FOR_UNKNOWN_WORD
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
    global SENTIMENTS
    global SENTIMENT_TO_SENTIMENT_INDEX_MAP
    global NUMBER_OF_SENTIMENTS
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
    global TORCH_ARANGE_NUMBER_OF_SENTIMENTS
    global SENTIMENT_INDEX_TO_SENTIMENT_MAP
    sentiment_result = truncate_sentiment_result(sentiment_result_0)
    sentiment_result_string = None
    assert tuple(sentiment_result.shape) == (NUMBER_OF_SENTIMENTS,)
    assert torch.sum(sentiment_result) == 1
    sentiment_index = int(sentiment_result.dot(TORCH_ARANGE_NUMBER_OF_SENTIMENTS))
    sentiment_string = SENTIMENT_INDEX_TO_SENTIMENT_MAP[sentiment_index]
    return sentiment_string

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))

RAW_TRAINING_DATA_LOCATION = os.path.join(CURRENT_FILE_PATH, "data/train.csv")
RAW_TEST_DATA_LOCATION = os.path.join(CURRENT_FILE_PATH, "data/test.csv")

TRAINING_DATA_ID_TO_DATA_MAP = {}
TEST_DATA_ID_TO_TEXT_MAP = OrderedDict()

VALIDATION_DATA_PORTION = 0.1 #0.001 # @todo fix this
PORTION_OF_TRAINING_DATA_TO_USE = 0.001 #1.0 # @todo fix this
PORTION_OF_TESTING_DATA_TO_USE = 0.001 #1.0 # @todo fix this

'''
if socket.gethostname() == 'phact': # @todo get rid of this
    VALIDATION_DATA_PORTION = 0.001
    PORTION_OF_TRAINING_DATA_TO_USE = 1.0
    PORTION_OF_TESTING_DATA_TO_USE = 1.0
    #'''

with open(RAW_TRAINING_DATA_LOCATION, encoding='ISO-8859-1') as training_data_csv_file:
    training_data_csv_reader = csv.DictReader(training_data_csv_file, delimiter=',')
    row_dicts = list(training_data_csv_reader)
    number_of_row_dicts = len(row_dicts)
    for row_dict_index, row_dict in enumerate(row_dicts):
        if row_dict_index/number_of_row_dicts >= PORTION_OF_TRAINING_DATA_TO_USE:
            break
        id = row_dict.pop('ItemID')
        TRAINING_DATA_ID_TO_DATA_MAP[id]=row_dict

with open(RAW_TEST_DATA_LOCATION, encoding='ISO-8859-1') as test_data_csv_file:
    test_data_csv_reader = csv.DictReader(test_data_csv_file, delimiter=',')
    row_dicts = list(test_data_csv_reader)
    number_of_row_dicts = len(row_dicts)
    for row_dict_index, row_dict in enumerate(row_dicts):
        if row_dict_index/number_of_row_dicts >= PORTION_OF_TESTING_DATA_TO_USE:
            break
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
    global TRAINING_DATA_ID_TO_DATA_MAP
    global VALIDATION_DATA_PORTION
    global RAW_VALUE_TO_SENTIMENT_MAP
    data_dictionaries = list(TRAINING_DATA_ID_TO_DATA_MAP.values())
    # random.shuffle(data_dictionaries)
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

def get_new_checkpoint_directory():
    return "/tmp/sentiment_classifier_{timestamp}".format(timestamp=time.strftime("%Y%m%d-%H%M%S"))

STATE_DICT_TO_BE_SAVED_FILE_LOCAL_NAME = "state_dict.pth"
UNSEEN_WORD_TO_TENSOR_MAP_PICKLED_FILE_LOCAL_NAME = "unseen_word_to_tensor_map.pickle"
VALIDATION_RESULTS_LOCAL_NAME = "validation_results.txt"
PROGRESS_CSV_LOCAL_NAME = "loss_per_epoch.csv"
PROGRESS_PNG_LOCAL_NAME = "loss_per_epoch.png"

class SentimentAnalysisClassifier():
    def __init__(self, batch_size=1, learning_rate=1e-2, attenion_regularization_penalty_multiplicative_factor=0.1,
                 embedding_hidden_size=200, lstm_dropout_prob=0.2, number_of_attention_heads=2, attention_hidden_size=24,
                 checkpoint_directory=get_new_checkpoint_directory(), loading_directory=None, print_verbosely=False, 
    ):
        global PROGRESS_CSV_LOCAL_NAME
        self.print_verbosely = print_verbosely
        self.attenion_regularization_penalty_multiplicative_factor = attenion_regularization_penalty_multiplicative_factor
        self.number_of_completed_epochs = 0
        self.most_recent_epoch_loss = 0
        self.most_recent_epoch_loss_via_correctness = 0
        self.most_recent_epoch_loss_via_attention_regularization = 0
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
        self.checkpoint_directory = checkpoint_directory
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
        csv_location = os.path.join(self.checkpoint_directory, PROGRESS_CSV_LOCAL_NAME)
        with open(csv_location, 'w') as csvfile:
            headers = ['epoch_index', 'total_loss', 'attention_regularization_loss', 'correctness_loss']
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)
            writer.writeheader()
        if loading_directory is not None:
            self.load(loading_directory)
        
    def print_static_information(self):
        logging_print("Checkpoint Directory: {checkpoint_directory}".format(checkpoint_directory=self.checkpoint_directory))
        logging_print("Training Size: {training_size}".format(training_size=len(self.training_generator.dataset)))
        logging_print("Validation Size: {validation_size}".format(validation_size=len(self.validation_generator.dataset)))
        
    def _update_loss_per_epoch_logs(self, current_global_epoch):
        global PROGRESS_CSV_LOCAL_NAME
        global PROGRESS_PNG_LOCAL_NAME 
        loss_per_epoch_csv_location = os.path.join(self.checkpoint_directory, PROGRESS_CSV_LOCAL_NAME)
        current_csv_dataframe = pd.read_csv(loss_per_epoch_csv_location)
        updated_csv_dataframe = current_csv_dataframe.append({
            'epoch_index': current_global_epoch,
            'total_loss': self.most_recent_epoch_loss,
            'attention_regularization_loss': self.most_recent_epoch_loss_via_attention_regularization,
            'correctness_loss': self.most_recent_epoch_loss_via_correctness,
        }, ignore_index=True)
        updated_csv_dataframe.loc[:, 'epoch_index'] = updated_csv_dataframe['epoch_index'].apply(int)
        updated_csv_dataframe.to_csv(loss_per_epoch_csv_location, index=False)
        if socket.gethostname() != 'phact': # @todo remove this exception
            print("updated_csv_dataframe.total_loss {}".format(updated_csv_dataframe.total_loss))
            print("updated_csv_dataframe.attention_regularization_loss {}".format(updated_csv_dataframe.attention_regularization_loss))
            print("updated_csv_dataframe.correctness_loss {}".format(updated_csv_dataframe.correctness_loss))
            plt.plot(updated_csv_dataframe.epoch_index, updated_csv_dataframe.total_loss, label="Total Loss")
            plt.plot(updated_csv_dataframe.epoch_index, updated_csv_dataframe.attention_regularization_loss, label="Attention Regularization Loss")
            plt.plot(updated_csv_dataframe.epoch_index, updated_csv_dataframe.correctness_loss, label="Correctness Loss")
            plt.title('Loss Per Epoch')
            plt.ylabel('Loss')
            plt.xlabel('Epoch Index')
            plt.legend(loc='lower left')
            loss_per_epoch_png_location = os.path.join(self.checkpoint_directory, PROGRESS_PNG_LOCAL_NAME)
            plt.savefig(loss_per_epoch_png_location)
        
    def train(self, number_of_epochs_to_train, number_of_iterations_between_checkpoints=None):
        self.model.train()
        for new_epoch_index in range(number_of_epochs_to_train):
            total_epoch_loss = 0
            epoch_loss_via_correctness = 0
            epoch_loss_via_attention_regularization = 0
            total_number_of_iterations = len(self.training_generator.dataset)
            current_global_epoch=self.number_of_completed_epochs+new_epoch_index
            for iteration_index, (x_batch, y_batch) in enumerate(self.training_generator):
                if number_of_iterations_between_checkpoints is not None:
                    if (iteration_index != 0) and (iteration_index % number_of_iterations_between_checkpoints) == 0:
                        logging_print("Completed Iteration {iteration_index} / {total_number_of_iterations} of epoch {current_global_epoch}".format(
                            iteration_index=iteration_index,
                            total_number_of_iterations=total_number_of_iterations,
                            current_global_epoch=current_global_epoch))
                        sub_directory_to_checkpoint_in = os.path.join(self.checkpoint_directory, "checkpoint_{timestamp}_for_epoch_{current_global_epoch}_iteration_{iteration_index}".format(
                            timestamp=time.strftime("%Y%m%d-%H%M%S"), current_global_epoch=current_global_epoch, iteration_index=iteration_index))
                        self.save(sub_directory_to_checkpoint_in)
                y_batch_predicted, attenion_regularization_penalty = self.model(x_batch)
                loss_via_correctness = self.loss_function(y_batch_predicted, y_batch)
                loss_via_attention_regularization = attenion_regularization_penalty * self.attenion_regularization_penalty_multiplicative_factor
                batch_loss = loss_via_correctness + loss_via_attention_regularization
                self.optimizer.zero_grad()
                batch_loss.backward()
                total_epoch_loss += float(batch_loss)
                epoch_loss_via_correctness += float(loss_via_correctness)
                epoch_loss_via_attention_regularization += float(loss_via_attention_regularization)
                self.optimizer.step()
            self.most_recent_epoch_loss = total_epoch_loss
            self.most_recent_epoch_loss_via_correctness = epoch_loss_via_correctness
            self.most_recent_epoch_loss_via_attention_regularization = epoch_loss_via_attention_regularization
            sub_directory_to_checkpoint_in = os.path.join(self.checkpoint_directory, "checkpoint_{timestamp}_for_epoch_{current_global_epoch}".format(
                timestamp=time.strftime("%Y%m%d-%H%M%S"), current_global_epoch=current_global_epoch))
            self._update_loss_per_epoch_logs(current_global_epoch)
            self.print_current_state()
            self.save(sub_directory_to_checkpoint_in)
            self.number_of_completed_epochs += 1
            
    def evaluate(self, strings: List[str]):
        self.model.eval()
        return self.model(strings)
    
    def print_current_state(self):
        logging_print()
        logging_print("===================================================================")
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
            if self.print_verbosely:
                input_string = x_batch[0]
                logging_print("Input: {x}".format(x=input_string))
                logging_print("Expected Output: {x}".format(x=expected_result))
                logging_print("Actual Output: {x}".format(x=actual_result))
                logging_print("Raw Expected Output: {x}".format(x=y_datum))
                logging_print("Raw Actual Output: {x}".format(x=y_batch_predicted))
                logging_print("\n")
            if actual_result == expected_result:
                correct_result_number += 1
        total_result_number = len(self.validation_generator.dataset)
        logging_print("Truncated Validation Set Correctness Portion: {correct_result_number} / {total_result_number}".format(
            correct_result_number=correct_result_number, total_result_number=total_result_number))
        logging_print("Total training loss for model prior to training for epoch {epoch_index} is {loss}".format(epoch_index=self.number_of_completed_epochs,loss=self.most_recent_epoch_loss))
        logging_print("Training loss via correctness for model prior to training for epoch {epoch_index} is {loss}".format(
            epoch_index=self.number_of_completed_epochs,loss=self.most_recent_epoch_loss_via_correctness))
        logging_print("Training loss via attention regularization for model prior to training for epoch {epoch_index} is {loss}".format(
            epoch_index=self.number_of_completed_epochs,loss=self.most_recent_epoch_loss_via_attention_regularization))
        assert implies(self.most_recent_epoch_loss==0, epoch_index==0)
        if self.most_recent_epoch_loss == 0:
            logging_print("    NB: Loss has not yet been calculated for model prior to training for epoch {epoch_index}.".format(epoch_index=self.number_of_completed_epochs))
        logging_print("===================================================================")
    
    def save(self, sub_directory_name):
        global UNSEEN_WORD_TO_TENSOR_MAP
        global STATE_DICT_TO_BE_SAVED_FILE_LOCAL_NAME
        global UNSEEN_WORD_TO_TENSOR_MAP_PICKLED_FILE_LOCAL_NAME
        global VALIDATION_RESULTS_LOCAL_NAME
        directory_to_save_in = os.path.join(self.checkpoint_directory, sub_directory_name)
        if not os.path.exists(directory_to_save_in):
            os.makedirs(directory_to_save_in)
        state_dict_file_location = os.path.join(directory_to_save_in, STATE_DICT_TO_BE_SAVED_FILE_LOCAL_NAME)
        torch.save(self.model.state_dict(), state_dict_file_location)
        unseen_word_to_tensor_map_pickled_file_to_be_saved_name = os.path.join(directory_to_save_in, UNSEEN_WORD_TO_TENSOR_MAP_PICKLED_FILE_LOCAL_NAME)
        with open(unseen_word_to_tensor_map_pickled_file_to_be_saved_name, 'wb') as handle:
            pickle.dump(UNSEEN_WORD_TO_TENSOR_MAP, handle, protocol=pickle.HIGHEST_PROTOCOL)
        validation_results_file = os.path.join(directory_to_save_in, VALIDATION_RESULTS_LOCAL_NAME)
        with open(validation_results_file, 'w') as f:
            with redirect_stdout(f):
                self.print_current_state()
        logging_print()
        logging_print("Saved checkpoint to {directory_to_save_in}".format(directory_to_save_in=directory_to_save_in))
        logging_print()
    
    def load(self, saved_directory_name):
        global UNSEEN_WORD_TO_TENSOR_MAP
        global STATE_DICT_TO_BE_SAVED_FILE_LOCAL_NAME
        global UNSEEN_WORD_TO_TENSOR_MAP_PICKLED_FILE_LOCAL_NAME
        state_dict_file_location = os.path.join(saved_directory_name, STATE_DICT_TO_BE_SAVED_FILE_LOCAL_NAME)
        self.model.load_state_dict(torch.load(state_dict_file_location))
        unseen_word_to_tensor_map_pickled_file_name = os.path.join(saved_directory_name, UNSEEN_WORD_TO_TENSOR_MAP_PICKLED_FILE_LOCAL_NAME)
        with open(unseen_word_to_tensor_map_pickled_file_name, 'rb') as handle:
            UNSEEN_WORD_TO_TENSOR_MAP = pickle.load(handle)
        logging_print()
        logging_print("Loaded checkpoint from {saved_directory_name}".format(saved_directory_name=saved_directory_name))
        logging_print()

###############
# Main Runner #
###############

def train_classifier(batch_size=1,
                     learning_rate=1e-2,
                     attenion_regularization_penalty_multiplicative_factor=0.1,
                     embedding_hidden_size=200,
                     lstm_dropout_prob=0.2,
                     number_of_attention_heads=2,
                     attention_hidden_size=24,
                     checkpoint_directory=get_new_checkpoint_directory(),
                     number_of_epochs=8,
                     number_of_iterations_between_checkpoints=1000,
                     print_verbosely=False,
                     loading_directory=None,
):
    classifier = SentimentAnalysisClassifier(
        batch_size=batch_size,
        learning_rate=learning_rate,
        attenion_regularization_penalty_multiplicative_factor=attenion_regularization_penalty_multiplicative_factor,
        embedding_hidden_size=embedding_hidden_size,
        lstm_dropout_prob=lstm_dropout_prob,
        number_of_attention_heads=number_of_attention_heads,
        attention_hidden_size=attention_hidden_size,
        checkpoint_directory=checkpoint_directory,
        loading_directory=loading_directory,
        print_verbosely=print_verbosely,
    )
    number_of_epochs_between_updates = 1
    number_of_updates = number_of_epochs//number_of_epochs_between_updates
    logging_print()
    logging_print("Starting Training on {machine_name}.".format(machine_name=socket.gethostname()))
    logging_print()
    classifier.print_static_information()
    classifier.print_current_state()
    for update_index in range(number_of_updates):
        with timer(exitCallback=lambda number_of_seconds: logging_print("\nTime for epochs {start_epoch_index} to {end_epoch_index} (inclusive): {time_for_epochs} seconds".format(
                start_epoch_index=update_index*number_of_epochs_between_updates,
                end_epoch_index=(update_index+1)*number_of_epochs_between_updates-1,
                time_for_epochs=number_of_seconds,
        ))):
            classifier.train(number_of_epochs_between_updates, number_of_iterations_between_checkpoints)
    logging_print("Training Complete.")

def test_classifier(loading_directory):
    global TEST_DATA_ID_TO_TEXT_MAP
    classifier = SentimentAnalysisClassifier(loading_directory=loading_directory)
    logging_print()
    logging_print("Starting Testing on {machine_name}.".format(machine_name=socket.gethostname()))
    logging_print()
    sentiment_texts = TEST_DATA_ID_TO_TEXT_MAP.values()
    print("Number of Sentiment Texts for Testing: {sentiment_texts_len}".format(sentiment_texts_len=len(sentiment_texts)))
    results, _ = classifier.evaluate(sentiment_texts)
    for (id, sentiment_text), result in zip(TEST_DATA_ID_TO_TEXT_MAP.items(), results):
        logging_print("ID: {id}".format(id=id))
        logging_print("Sentiment Text: {sentiment_text}".format(sentiment_text=sentiment_text))
        result_as_string = sentiment_result_to_string(result)
        logging_print("Result: {result_as_string}".format(result_as_string=result_as_string))
        logging_print("Raw Result: {result}".format(result=result))
        logging_print()
    logging_print("Testing Complete.")

def main():
    print("This module contains utilities for sentiment analysis for Twitter data via neural network based text classification.")
    
if __name__ == '__main__':
    main()
