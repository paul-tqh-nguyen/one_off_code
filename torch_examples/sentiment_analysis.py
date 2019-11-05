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
* Main Runner

"""

###########
# Imports #
###########

from gensim.models import KeyedVectors
import string
from collections import OrderedDict
import torch
import torch.nn as nn

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
    normalized_vector = nn.functional.normalize(random_vector, dim=0)
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

def tensors_from_sentence_string(sentence_string):
    normalized_words = normalized_words_from_sentence_string(sentence_string)
    tensors = map(tensor_from_normalized_word, normalized_words)
    return tensors

#####################
# Model Definitions #
#####################

NUMBER_OF_SENTIMENTS = 2

class SentimentAnalysisNetwork(nn.Module):
    def __init__(self, hidden_size=200, lstm_dropout_prob=0.2):
        super().__init__()
        self.embedding_layer = nn.Linear(WORD2VEC_VECTOR_LENGTH, hidden_size)
        self.encoding_layer = nn.LSTM(hidden_size, hidden_size, num_layers=2, dropout=lstm_dropout_prob, bidirectional=True)
        self.attention_layer = nn.Linear(2*hidden_size, 2*hidden_size)
        self.prediction_layers = nn.Sequential(OrderedDict([
            ("linear_classification_layer", nn.Linear(2*hidden_size, NUMBER_OF_SENTIMENTS)),
            ("softmax", nn.Softmax()),
        ]))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_layer.to(self.device)
        self.encoding_layer.to(self.device)
        self.attention_layer.to(self.device)
        self.prediction_layers.to(self.device)
    
    def forward(self, sentence_string):
        word_tensors = tuple(tensors_from_sentence_string(sentence_string))
        number_of_words = len(word_tensors)
        sentence_matrix = torch.stack(word_tensors)
        batch_size = 1
        sentence_batch_matrix = sentence_matrix.view(number_of_words, batch_size, -1)
        sentence_batch_matrix = sentence_batch_matrix.to(self.device)
        embedded_tensors = self.embedding_layer(sentence_batch_matrix)
        encoded_tensors, _ = self.encoding_layer(embedded_tensors)
        attention_tensor = torch.sum(self.attention_layer(encoded_tensors), 0)
        prediction_scores = self.prediction_layers(attention_tensor)
        return prediction_scores

###############
# Main Runner #
###############

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

NUMBER_OF_EPOCHS = 3000
LEARNING_RATE = 1e-3

DEBUG_MODE = False

def main():
    if DEBUG_MODE:
        torch.manual_seed(1)
    loss_function = nn.BCELoss()
    model = SentimentAnalysisNetwork()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    x_data = POSITIVE_STRINGS+NEGATIVE_STRINGS
    y_data = torch.stack([POSITIVE_RESULT]*len(POSITIVE_STRINGS)+[NEGATIVE_RESULT]*len(NEGATIVE_STRINGS)).to(model.device)
    for epoch_index in range(NUMBER_OF_EPOCHS):
        epoch_loss = 0
        for x_datum, y_datum in zip(x_data, y_data):
            y_batch_predicted = model(x_datum)
            y_batch = y_datum.view(1,-1).to(model.device)
            loss = loss_function(y_batch_predicted, y_batch)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (0==epoch_index%50):
            print("\n\n")
            print("===================================================================")
            correct_result_number = 0
            for x_datum, y_datum in zip(x_data, y_data):
                print("Input: {x}".format(x=x_datum))
                expected_result = sentiment_result_to_string(y_datum)
                print("Expected Output: {x}".format(x=expected_result))
                y_batch_predicted = model(x_datum)
                actual_result = sentiment_result_to_string(torch.round(y_batch_predicted[0]))
                print("Actual Output: {x}".format(x=actual_result))
                print("\n")
                if actual_result == expected_result:
                    correct_result_number += 1
            total_result_number = len(x_data)
            print("Truncated Correctness Portion: {correct_result_number} / {total_result_number}".format(correct_result_number=correct_result_number, total_result_number=len(x_data)))
            print("Total loss for epoch {epoch_index} is {loss}".format(epoch_index=epoch_index,loss=epoch_loss))
            print("Loss per datapoint for {epoch_index} is {loss}".format(epoch_index=epoch_index,loss=epoch_loss/total_result_number))
            print("===================================================================")

if __name__ == '__main__':
    main()
