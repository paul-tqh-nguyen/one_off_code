#!/usr/bin/python3
'#!/usr/bin/python3 -OO'

"""
"""

# @todo update doc string

###########
# Imports #
###########

from functools import reduce
from typing import List, Callable, Iterable
from collections import OrderedDict

from abstract_classifier import Predictor, DEVICE, SENTIMENTS
from misc_utilities import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

###########
# Globals #
###########

OUTPUT_DIR = './default_output'
TRAIN_PORTION = 0.75
VALIDATION_PORTION = 1-TRAIN_PORTION
NUMBER_OF_EPOCHS = 100

BATCH_SIZE = 128
MAX_VOCAB_SIZE = 15_000
PRE_TRAINED_EMBEDDING_SPECIFICATION = 'fasttext.en.300d'

SENTIMENT_EMBEDDING_SIZE = 256
ENCODING_HIDDEN_SIZE = 256
NUMBER_OF_ENCODING_LAYERS = 2
DROPOUT_PROBABILITY = 0.5

##########
# Models #
##########

class RNNNetwork(nn.Module):
    def __init__(self, vocab_size: int, sentiment_size: int, embedding_size: int, encoding_hidden_size: int, number_of_encoding_layers: int, dropout_probability: float, pad_idx: int, unk_idx: int, initial_embedding_vectors: torch.Tensor):
        super().__init__()
        if __debug__:
            self.sentiment_size = sentiment_size
            self.embedding_size = embedding_size
            self.encoding_hidden_size = encoding_hidden_size
            self.number_of_encoding_layers = number_of_encoding_layers
        self.sentiment_vectors = {sentiment: nn.Parameter(torch.nn.functional.normalize(torch.randn([sentiment_size]), dim=0)) for sentiment in SENTIMENTS}
        self.embedding_layers = nn.Sequential(OrderedDict([
            ("embedding_layer", nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx, max_norm=1.0)),
            ("dropout_layer", nn.Dropout(dropout_probability)),
        ]))
        self.embedding_layers.embedding_layer.weight.data.copy_(initial_embedding_vectors)
        self.embedding_layers.embedding_layer.weight.data[unk_idx] = torch.zeros(embedding_size)
        self.embedding_layers.embedding_layer.weight.data[pad_idx] = torch.zeros(embedding_size)
        self.encoding_layers = nn.LSTM(embedding_size,
                                       encoding_hidden_size,
                                       num_layers=number_of_encoding_layers,
                                       bidirectional=True,
                                       dropout=dropout_probability)
        self.prediction_layers = nn.Sequential(OrderedDict([
            ("dropout_layer", nn.Dropout(dropout_probability)),
            ("fully_connected_layer", nn.Linear(sentiment_size+encoding_hidden_size*2, 1)),
            ("sigmoid_layer", nn.Sigmoid()),
        ]))
        self.to(DEVICE)

    def forward(self, text_batch: torch.Tensor, text_lengths: torch.Tensor, sentiment: str):
        sentiment_vector = self.sentiment_vectors[sentiment]
        if __debug__:
            max_sequence_length = max(text_lengths)
            batch_size = text_batch.shape[0]
        assert tuple(text_batch.shape) == (batch_size, max_sequence_length)
        assert tuple(text_lengths.shape) == (batch_size,)

        embedded_batch = self.embedding_layers(text_batch)
        assert tuple(embedded_batch.shape) == (batch_size, max_sequence_length, self.embedding_size)

        embedded_batch_packed = nn.utils.rnn.pack_padded_sequence(embedded_batch, text_lengths, batch_first=True)
        if __debug__:
            encoded_batch_packed, (encoding_hidden_state, encoding_cell_state) = self.encoding_layers(embedded_batch_packed)
            encoded_batch, encoded_batch_lengths = nn.utils.rnn.pad_packed_sequence(encoded_batch_packed, batch_first=True)
        else:
            encoded_batch_packed, _ = self.encoding_layers(embedded_batch_packed)
            encoded_batch, _ = nn.utils.rnn.pad_packed_sequence(encoded_batch_packed, batch_first=True)
        assert tuple(encoded_batch.shape) == (batch_size, max_sequence_length, self.encoding_hidden_size*2)
        assert tuple(encoding_hidden_state.shape) == (self.number_of_encoding_layers*2, batch_size, self.encoding_hidden_size)
        assert tuple(encoding_cell_state.shape) == (self.number_of_encoding_layers*2, batch_size, self.encoding_hidden_size)
        assert tuple(encoded_batch_lengths.shape) == (batch_size,)
        assert (encoded_batch_lengths.to(text_lengths.device) == text_lengths).all()

        duplicated_sentiment_vectors = sentiment_vector.repeat(batch_size, max_sequence_length).view(batch_size, max_sequence_length, -1)
        assert tuple(duplicated_sentiment_vectors.shape) == (batch_size, max_sequence_length, self.sentiment_size)
        concatenated_batch = torch.torch.cat((encoded_batch, duplicated_sentiment_vectors), dim=2)
        assert tuple(concatenated_batch.shape) == (batch_size, max_sequence_length, self.sentiment_size+self.encoding_hidden_size*2)
        
        prediction = self.prediction_layers(concatenated_batch)
        assert tuple(prediction.shape) == (batch_size, max_sequence_length)
        
        return prediction

##############
# Predictors #
##############

class RNNPredictor(Predictor):
    def initialize_model(self) -> None:
        self.sentiment_embedding_size = self.model_args['sentiment_embedding_size']
        self.encoding_hidden_size = self.model_args['encoding_hidden_size']
        self.number_of_encoding_layers = self.model_args['number_of_encoding_layers']
        self.dropout_probability = self.model_args['dropout_probability']
        vocab_size = len(self.text_field.vocab)
        self.model = RNNNetwork(vocab_size, self.sentiment_embedding_size, self.embedding_size, self.encoding_hidden_size, self.number_of_encoding_layers, self.dropout_probability, self.pad_idx, self.unk_idx, self.text_field.vocab.vectors)
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_function = nn.BCELoss().to(DEVICE)
        return

###############
# Main Driver #
###############

@debug_on_error
def main() -> None:
    predictor = RNNPredictor(OUTPUT_DIR, NUMBER_OF_EPOCHS, BATCH_SIZE, TRAIN_PORTION, VALIDATION_PORTION, MAX_VOCAB_SIZE, PRE_TRAINED_EMBEDDING_SPECIFICATION,
                             sentiment_embedding_size=SENTIMENT_EMBEDDING_SIZE, 
                             encoding_hidden_size=ENCODING_HIDDEN_SIZE,
                             number_of_encoding_layers=NUMBER_OF_ENCODING_LAYERS,
                             dropout_probability=DROPOUT_PROBABILITY)
    predictor.train()
    return 

if __name__ == '__main__':
    main()
