#!/usr/bin/python3

"""

Simple Sentiment Analyzer.

Owner : paul-tqh-nguyen

Created : 10/30/2019

File Name : sentiment_analysis.py

File Organization:
* Imports
* Misc. Utilities
* String Utilities
* String <-> Tensor Utilities
* Model Definitions
* Dataset Definitions
* Classifier Definitions
* Main Runner

"""

###########
# Imports #
###########

from typing import List
from gensim.models import KeyedVectors
import string
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

WORD2VEC_BIN_LOCATION = '/home/pnguyen/code/datasets/GoogleNews-vectors-negative300.bin'
WORD2VEC_MODEL = KeyedVectors.load_word2vec_format('/home/pnguyen/code/datasets/GoogleNews-vectors-negative300.bin', binary=True)
WORD2VEC_VECTOR_LENGTH = 300
PUNCTUATION_SET = set(string.punctuation)
UNSEEN_WORD_TO_TENSOR_MAP = {}

###################
# Misc. Utilities #
###################
 
def dt(var_name_string):
    return 'print("{{variable}} : {{value}}".format(variable="{var_name_string}", value=locals()["{var_name_string}"]))'.format(var_name_string=var_name_string)

####################
# String Utilities #
####################

def remove_punctuation(input_string):
    return ''.join(char for char in input_string if char not in PUNCTUATION_SET)

def normalize_word_string(word_string):
    normalized_word_string = word_string.lower()
    normalized_word_string = remove_punctuation(normalized_word_string)
    return normalized_word_string

def normalized_words_from_sentence_string(sentence_string):
    """
    sentence_string is assumed to contain only one sentence. 
    For example, 'Red is a color. Green is a color.' violates this assumption while 'We play spots, e.g. football.' does not.
    """
    normalized_sentence_string = sentence_string
    words = normalized_sentence_string.split(' ')
    normalized_sentence_string_words = map(normalize_word_string, words)
    return normalized_sentence_string_words

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

def tensors_from_sentence_string(sentence_string: str):
    normalized_words = normalized_words_from_sentence_string(sentence_string)
    tensors = map(tensor_from_normalized_word, normalized_words)
    return tensors

def sentence_matrix_from_sentence_string(sentence_string: str):
    word_tensors = tuple(tensors_from_sentence_string(sentence_string))
    sentence_matrix = torch.stack(word_tensors)
    return sentence_matrix

#####################
# Model Definitions #
#####################

NUMBER_OF_SENTIMENTS = 2

class SelfAttentionLayers(nn.Module):
    def __init__(self, input_size=400, number_of_attention_heads=2, hidden_size=None):
        super().__init__()
        self.number_of_attention_heads = number_of_attention_heads # only used for assertion checking
        if hidden_size == None:
            hidden_size = input_size // 2
        self.attention_layers = nn.Sequential(OrderedDict([
            ("reduction_layer", nn.Linear(input_size, hidden_size)),
            ("activation", nn.ReLU(True)),
            ("attention_layer", nn.Linear(hidden_size, number_of_attention_heads)),
        ]))
        
    def to(self, device):
        self.attention_layers.to(device)
        
    def forward(self, input_matrix):
        max_number_of_words, batch_size, input_size = input_matrix.shape
        attention_weights_pre_softmax = self.attention_layers(input_matrix) 
        assert tuple(attention_weights_pre_softmax.shape) == (max_number_of_words, batch_size, self.number_of_attention_heads), "attention_weights_pre_softmax has unexpected dimensions."
        attention_weights = F.softmax(attention_weights_pre_softmax, dim=0)
        assert tuple(attention_weights.shape) == (max_number_of_words, batch_size, self.number_of_attention_heads), "attention_weights has unexpected dimensions."
        attention_weights_duplicated = attention_weights.view(-1,1).repeat(1,input_size).view(max_number_of_words, batch_size, self.number_of_attention_heads*input_size)
        assert tuple(attention_weights_duplicated.shape) == (max_number_of_words, batch_size, self.number_of_attention_heads*input_size), "attention_weights_duplicated has unexpected dimensions."
        input_matrix_duplicated = input_matrix.repeat(1,1,self.number_of_attention_heads) 
        assert tuple(input_matrix_duplicated.shape) == (max_number_of_words, batch_size, self.number_of_attention_heads*input_size), "input_matrix_duplicated has unexpected dimensions."
        weight_adjusted_input_matrix = torch.mul(attention_weights_duplicated, input_matrix_duplicated)
        assert tuple(weight_adjusted_input_matrix.shape) == (max_number_of_words, batch_size, self.number_of_attention_heads*input_size), "weight_adjusted_input_matrix has unexpected dimensions."
        attended_matrix = torch.sum(weight_adjusted_input_matrix, dim=0)
        assert tuple(attended_matrix.shape) == (batch_size, self.number_of_attention_heads*input_size), "attended_matrix has unexpected dimensions."
        return attended_matrix

class SentimentAnalysisNetwork(nn.Module):
    def __init__(self, embedding_hidden_size=200, lstm_dropout_prob=0.2, number_of_attention_heads=2, attention_hidden_size=24):
        super().__init__()
        self.embedding_hidden_size = embedding_hidden_size # only used for assertion checking
        self.number_of_attention_heads = number_of_attention_heads # only used for assertion checking
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
        
    def forward(self, sentence_strings: List[str]):
        batch_size = len(sentence_strings)
        sentence_matrices_unpadded = [sentence_matrix_from_sentence_string(sentence_string) for sentence_string in sentence_strings]
        sentence_batch_matrix = torch.nn.utils.rnn.pad_sequence(sentence_matrices_unpadded)
        sentence_batch_matrix = sentence_batch_matrix.to(self.device)
        max_number_of_words = max(map(len, sentence_matrices_unpadded))
        assert tuple(sentence_batch_matrix.shape) == (max_number_of_words, batch_size, WORD2VEC_VECTOR_LENGTH), "sentence_batch_matrix has unexpected dimensions."
        embeddeding_batch_matrix = self.embedding_layers(sentence_batch_matrix)
        assert tuple(embeddeding_batch_matrix.shape) == (max_number_of_words, batch_size, self.embedding_hidden_size)
        encoding_batch_matrix, _ = self.encoding_layers(embeddeding_batch_matrix)
        assert tuple(encoding_batch_matrix.shape) == (max_number_of_words, batch_size, 2*self.embedding_hidden_size)
        attention_matrix = self.attention_layers(encoding_batch_matrix)
        assert tuple(attention_matrix.shape) == (batch_size, self.number_of_attention_heads*2*self.embedding_hidden_size)
        prediction_scores = self.prediction_layers(attention_matrix)
        return prediction_scores

#######################
# Dataset Definitions #
#######################

NEGATIVE_STRINGS = [
    "Welds didn't hold... So disappointed", 
    "The item was bent inside the undamaged box.", 
    "Bad craftsmanship at welding position.", 
    "Knock Off version of the USA Made Vortex, THIS IS NOT THE SAME QUALITY, chinese crap.", 
]

POSITIVE_STRINGS = [
    "Makes the best chicken wings.", 
    "Works great at about half the price of others.", 
    "Very pleased.", 
    "Excellent product. Works as advertised.", 
]

POSITIVE_RESULT = torch.tensor([1,0]).float()
NEGATIVE_RESULT = torch.tensor([0,1]).float() 

class SentimentDataset(data.Dataset):
  def __init__(self, strings, expected_classification_tensors):
        self.x_data = strings
        self.y_data = expected_classification_tensors
    
  def __len__(self):
        return len(self.x_data)
    
  def __getitem__(self, index):
        x_datum = self.x_data[index]
        y_datum = self.y_data[index]
        return x_datum, y_datum

##########################
# Classifier Definitions #
##########################

def sentiment_result_to_string(sentiment_result):
    sentiment_result_string = None
    positive_result = POSITIVE_RESULT.to(sentiment_result.device)
    negative_result = NEGATIVE_RESULT.to(sentiment_result.device)
    if torch.all(sentiment_result == positive_result):
        sentiment_result_string = "Positive"
    elif torch.all(sentiment_result == negative_result):
        sentiment_result_string = "Negative"
    else:
        raise Exception('The following is not a supported sentiment: {sentiment_result}'.format(sentiment_result=sentiment_result))
    return sentiment_result_string

class SentimentAnalysisClassifier():
    def __init__(self, batch_size=1, learning_rate=1e-2):
        self.number_of_completed_epochs = 0
        self.most_recent_epoch_loss = 0
        self.loss_function = nn.BCELoss()
        self.model = SentimentAnalysisNetwork()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)        
        self.x_data = POSITIVE_STRINGS+NEGATIVE_STRINGS
        self.y_data = torch.stack([POSITIVE_RESULT]*len(POSITIVE_STRINGS)+[NEGATIVE_RESULT]*len(NEGATIVE_STRINGS)).to(self.model.device)
        training_set = SentimentDataset(self.x_data, self.y_data)
        self.training_generator = data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
        
    def train(self, number_of_epochs_to_train):
        for new_epoch_index in range(number_of_epochs_to_train):
            epoch_loss = 0
            for x_batch, y_batch in self.training_generator:
                y_batch_predicted = self.model(x_batch)
                batch_loss = self.loss_function(y_batch_predicted, y_batch)
                # @todo account for the penalty in the cost
                self.optimizer.zero_grad()
                batch_loss.backward()
                epoch_loss += float(batch_loss)
                self.optimizer.step()
            self.most_recent_epoch_loss = epoch_loss
            self.number_of_completed_epochs += 1
                
    def print_current_state(self, verbose=False):
        if verbose:
            print("\n")
            print("===================================================================")
        correct_result_number = 0
        for x_datum, y_datum in zip(self.x_data, self.y_data):
            expected_result = sentiment_result_to_string(y_datum)
            y_batch_predicted = self.model(x_datum)
            actual_result = sentiment_result_to_string(torch.round(y_batch_predicted[0]))
            if verbose:
                print("Input: {x}".format(x=x_datum))
                print("Expected Output: {x}".format(x=expected_result))
                print("Actual Output: {x}".format(x=actual_result))
                print("\n")
            if actual_result == expected_result:
                correct_result_number += 1
        total_result_number = len(self.x_data)
        if verbose:
            print("Truncated Correctness Portion: {correct_result_number} / {total_result_number}".format(correct_result_number=correct_result_number, total_result_number=total_result_number))
            print("Loss per datapoint for {epoch_index} is {loss}".format(epoch_index=self.number_of_completed_epochs,loss=self.most_recent_epoch_loss/total_result_number))
        print("Total loss for epoch {epoch_index} is {loss}".format(epoch_index=self.number_of_completed_epochs,loss=self.most_recent_epoch_loss))
        if verbose:
            print("===================================================================")
            
###############
# Main Runner #
###############

def main():
    classifier = SentimentAnalysisClassifier()
    number_of_epochs = 9000
    number_of_epochs_between_updates = 50
    number_of_updates = number_of_epochs//number_of_epochs_between_updates
    for update_index in range(number_of_updates):
        classifier.print_current_state()
        classifier.train(number_of_epochs_between_updates)
    classifier.print_current_state()
            
if __name__ == '__main__':
    main()

