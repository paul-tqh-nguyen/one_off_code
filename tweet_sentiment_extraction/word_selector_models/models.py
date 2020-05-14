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

import sys ; sys.path.append("..")
from misc_utilities import *
from .abstract_predictor import Predictor, DEVICE, SENTIMENTS, soft_jaccard_loss

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

BATCH_SIZE = 32
MAX_VOCAB_SIZE = 25_000
PRE_TRAINED_EMBEDDING_SPECIFICATION = 'fasttext.en.300d'
LOSS_FUNCTION_SPEC = 'BCELoss' # 'soft_jaccard_loss'

SENTIMENT_EMBEDDING_SIZE = 512
ENCODING_HIDDEN_SIZE = 512
NUMBER_OF_ENCODING_LAYERS = 2
DROPOUT_PROBABILITY = 0.5

##########
# Models #
##########

class LSTMSentimentConcatenationNetwork(nn.Module):
    def __init__(self, vocab_size: int, sentiment_size: int, embedding_size: int, encoding_hidden_size: int, number_of_encoding_layers: int, dropout_probability: float, pad_idx: int, unk_idx: int, initial_embedding_vectors: torch.Tensor):
        super().__init__()
        if __debug__:
            self.sentiment_size = sentiment_size
            self.embedding_size = embedding_size
            self.encoding_hidden_size = encoding_hidden_size
            self.number_of_encoding_layers = number_of_encoding_layers
        self.sentiment_to_sentiment_vector_map = {sentiment: nn.Parameter(torch.nn.functional.normalize(torch.randn([sentiment_size]), dim=0)).to(DEVICE) for sentiment in SENTIMENTS}
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
    
    def forward(self, text_batch: torch.Tensor, text_lengths: torch.Tensor, sentiments: List[str]):
        if __debug__:
            max_sequence_length = max(text_lengths)
            batch_size = text_batch.shape[0]
        assert tuple(text_batch.shape) == (batch_size, max_sequence_length)
        assert tuple(text_lengths.shape) == (batch_size,)
        
        sentiment_vectors = torch.stack([self.sentiment_to_sentiment_vector_map[sentiment] for sentiment in sentiments])
        assert tuple(sentiment_vectors.shape) == (batch_size, self.sentiment_size)

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

        duplicated_sentiment_vectors = sentiment_vectors.repeat(1, max_sequence_length).view(batch_size, max_sequence_length, -1)
        assert tuple(duplicated_sentiment_vectors.shape) == (batch_size, max_sequence_length, self.sentiment_size)
        concatenated_batch = torch.torch.cat((encoded_batch, duplicated_sentiment_vectors), dim=2)
        assert tuple(concatenated_batch.shape) == (batch_size, max_sequence_length, self.sentiment_size+self.encoding_hidden_size*2)
        
        prediction = self.prediction_layers(concatenated_batch)
        assert tuple(prediction.shape) == (batch_size, max_sequence_length, 1)
        prediction = prediction.view(batch_size, max_sequence_length)
        assert tuple(prediction.shape) == (batch_size, max_sequence_length)
        
        return prediction

##############
# Predictors #
##############

class LSTMSentimentConcatenationPredictor(Predictor):
    def initialize_model(self) -> None:
        self.loss_function_spec = self.model_args['loss_function_spec']
        self.sentiment_embedding_size = self.model_args['sentiment_embedding_size']
        self.encoding_hidden_size = self.model_args['encoding_hidden_size']
        self.number_of_encoding_layers = self.model_args['number_of_encoding_layers']
        self.dropout_probability = self.model_args['dropout_probability']
        vocab_size = len(self.text_field.vocab)
        self.model = LSTMSentimentConcatenationNetwork(vocab_size, self.sentiment_embedding_size, self.embedding_size, self.encoding_hidden_size, self.number_of_encoding_layers, self.dropout_probability, self.pad_idx, self.unk_idx, self.text_field.vocab.vectors)
        self.optimizer = optim.Adam(self.model.parameters())
        assert self.loss_function_spec in ['BCELoss', 'soft_jaccard_loss'] 
        self.loss_function = soft_jaccard_loss if self.loss_function_spec == 'soft_jaccard_loss' else nn.BCELoss().to(DEVICE)
        return

###############
# Main Driver #
###############

@debug_on_error
def train_model() -> None:
    predictor = LSTMSentimentConcatenationPredictor(OUTPUT_DIR, NUMBER_OF_EPOCHS, BATCH_SIZE, TRAIN_PORTION, VALIDATION_PORTION, MAX_VOCAB_SIZE, PRE_TRAINED_EMBEDDING_SPECIFICATION,
                                                    loss_function_spec=LOSS_FUNCTION_SPEC,
                                                    sentiment_embedding_size=SENTIMENT_EMBEDDING_SIZE, 
                                                    encoding_hidden_size=ENCODING_HIDDEN_SIZE,
                                                    number_of_encoding_layers=NUMBER_OF_ENCODING_LAYERS,
                                                    dropout_probability=DROPOUT_PROBABILITY)
    predictor.train()
    predictor.load_parameters(predictor.best_saved_model_location)
    # predictor.demonstrate_training_examples()
    predictor.demonstrate_validation_examples()
    return 

if __name__ == '__main__':
    train_model()
