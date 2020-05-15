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

##########
# Models #
##########

# Concatenation Networks

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

# Attention Networks

class LSTMScaledDotProductAttentionNetwork(nn.Module):
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
        self.attention_query_generating_layers = nn.Sequential(OrderedDict([
            ("dropout_layer", nn.Dropout(dropout_probability)),
            ("fully_connected_layer", nn.Linear(sentiment_size+encoding_hidden_size*2, encoding_hidden_size*2)),
            ("sigmoid_layer", nn.ReLU()),
        ]))
        self.prediction_layers = nn.Sequential(OrderedDict([
            ("dropout_layer", nn.Dropout(dropout_probability)),
            ("fully_connected_layer", nn.Linear(encoding_hidden_size*2, 1)),
            ("sigmoid_layer", nn.Sigmoid()),
        ]))
        self.to(DEVICE)
    
    def attend(self, encoded_batch: torch.Tensor, sentiment_vectors: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        batch_size = sentiment_vectors.shape[0]
        max_sequence_length = max(text_lengths)
        encoding_hidden_size_times_two = encoded_batch.shape[2]
        assert tuple(encoded_batch.shape) == (batch_size, max_sequence_length, self.encoding_hidden_size*2)
        assert tuple(sentiment_vectors.shape) == (batch_size, self.sentiment_size)

        attended_batch = Variable(torch.zeros(batch_size, max_sequence_length, encoding_hidden_size_times_two).to(encoded_batch.device))

        for batch_index in range(batch_size):
            sequence_length = text_lengths[batch_index]
            encoded_matrix = encoded_batch[batch_index, :sequence_length, :]
            assert tuple(encoded_matrix.shape) == (sequence_length, self.encoding_hidden_size*2)

            sentiment_vector = sentiment_vectors[batch_index]
            assert tuple(sentiment_vector.shape) == (self.sentiment_size,)
            
            for word_index in range(sequence_length):
                encoded_word_concatenated_with_sentiment_vector = torch.cat([encoded_matrix[word_index],sentiment_vector])
                assert tuple(encoded_word_concatenated_with_sentiment_vector.shape) == (self.sentiment_size+self.encoding_hidden_size*2,)
                query_vector = self.attention_query_generating_layers(encoded_word_concatenated_with_sentiment_vector)
                assert tuple(query_vector.shape) == (self.encoding_hidden_size*2,)

                duplicated_query_vector_transposed = query_vector.unsqueeze(1).expand(encoding_hidden_size_times_two, sequence_length)
                assert tuple(duplicated_query_vector_transposed.shape) == (self.encoding_hidden_size*2, sequence_length)
                
                dot_products = encoded_matrix.mm(duplicated_query_vector_transposed)
                assert tuple(dot_products.shape) == (sequence_length, sequence_length)
                assert tuple(dot_products.unique(dim=1).shape) == (sequence_length, 1)
                dot_products = dot_products[:,0]

                scaled_dot_products = dot_products / torch.sqrt(torch.tensor(encoding_hidden_size_times_two, dtype=float))
                assert tuple(scaled_dot_products.shape) == (sequence_length,)
                
                attention_weights = F.softmax(scaled_dot_products, dim=0)
                assert tuple(attention_weights.shape) == (sequence_length,)
                attention_weights = attention_weights.unsqueeze(1).expand(sequence_length, encoding_hidden_size_times_two)
                        
                attended_word = torch.sum(encoded_matrix * attention_weights, dim=0)
                assert tuple(attended_word.shape) == (self.encoding_hidden_size*2,)

                attended_batch[batch_index, word_index, :encoding_hidden_size_times_two] = attended_word

        if __debug__:
            for batch_index, text_length_tensor in enumerate(text_lengths):
                text_length = text_length_tensor.item()
                assert attended_batch[batch_index, text_length:, :].sum().item() == 0

        assert tuple(attended_batch.shape) == (batch_size, max_sequence_length, self.encoding_hidden_size*2)
        return attended_batch
    
    def forward(self, text_batch: torch.Tensor, text_lengths: torch.Tensor, sentiments: List[str]):
        if __debug__:
            batch_size = text_batch.shape[0]
            max_sequence_length = max(text_lengths)
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
        
        attended_batch = self.attend(encoded_batch, sentiment_vectors, text_lengths)
        #attended_batch = encoded_batch
        assert tuple(attended_batch.shape) == (batch_size, max_sequence_length, self.encoding_hidden_size*2)
        
        prediction = self.prediction_layers(attended_batch)
        assert tuple(prediction.shape) == (batch_size, max_sequence_length, 1)
        prediction = prediction.squeeze(2)
        assert tuple(prediction.shape) == (batch_size, max_sequence_length)
        
        return prediction

##############
# Predictors #
##############

class LSTMSentimentConcatenationPredictor(Predictor):
    def initialize_model(self) -> None:
        self.loss_function_spec = self.model_args['loss_function_spec']
        self.sentiment_size = self.model_args['sentiment_size']
        self.encoding_hidden_size = self.model_args['encoding_hidden_size']
        self.number_of_encoding_layers = self.model_args['number_of_encoding_layers']
        self.dropout_probability = self.model_args['dropout_probability']
        vocab_size = len(self.text_field.vocab)
        self.model = LSTMSentimentConcatenationNetwork(vocab_size, self.sentiment_size, self.embedding_size, self.encoding_hidden_size, self.number_of_encoding_layers, self.dropout_probability, self.pad_idx, self.unk_idx, self.text_field.vocab.vectors)
        self.optimizer = optim.Adam(self.model.parameters())
        assert self.loss_function_spec in ['BCELoss', 'soft_jaccard_loss'] 
        self.loss_function = soft_jaccard_loss if self.loss_function_spec == 'soft_jaccard_loss' else nn.BCELoss().to(DEVICE)
        return

class LSTMScaledDotProductAttentionPredictor(Predictor):
    def initialize_model(self) -> None:
        self.loss_function_spec = self.model_args['loss_function_spec']
        self.sentiment_size = self.model_args['sentiment_size']
        self.encoding_hidden_size = self.model_args['encoding_hidden_size']
        self.number_of_encoding_layers = self.model_args['number_of_encoding_layers']
        self.dropout_probability = self.model_args['dropout_probability']
        vocab_size = len(self.text_field.vocab)
        self.model = LSTMScaledDotProductAttentionNetwork(vocab_size, self.sentiment_size, self.embedding_size, self.encoding_hidden_size, self.number_of_encoding_layers, self.dropout_probability, self.pad_idx, self.unk_idx, self.text_field.vocab.vectors)
        self.optimizer = optim.Adam(self.model.parameters())
        assert self.loss_function_spec in ['BCELoss', 'soft_jaccard_loss'] 
        self.loss_function = soft_jaccard_loss if self.loss_function_spec == 'soft_jaccard_loss' else nn.BCELoss().to(DEVICE)
        return

###############
# Main Driver #
###############

def get_default_LSTMSentimentConcatenationPredictor() -> LSTMSentimentConcatenationPredictor:
    batch_size = 32
    max_vocab_size = 25_000
    pre_trained_embedding_specification = 'fasttext.en.300d'
    loss_function_spec = 'soft_jaccard_loss'

    sentiment_size = 512
    encoding_hidden_size = 512
    number_of_encoding_layers = 2
    dropout_probability = 0.5

    return LSTMSentimentConcatenationPredictor(OUTPUT_DIR, NUMBER_OF_EPOCHS, batch_size, TRAIN_PORTION, VALIDATION_PORTION, max_vocab_size, pre_trained_embedding_specification,
                                               loss_function_spec=loss_function_spec,
                                               sentiment_size=sentiment_size, 
                                               encoding_hidden_size=encoding_hidden_size,
                                               number_of_encoding_layers=number_of_encoding_layers,
                                               dropout_probability=dropout_probability)

def get_default_LSTMScaledDotProductAttentionPredictor() -> LSTMScaledDotProductAttentionPredictor:
    batch_size = 1
    max_vocab_size = 10_000
    pre_trained_embedding_specification = 'fasttext.en.300d'
    loss_function_spec = 'soft_jaccard_loss'
    
    sentiment_size = 64
    encoding_hidden_size = 64
    number_of_encoding_layers = 1
    dropout_probability = 0.5
    
    return LSTMScaledDotProductAttentionPredictor(OUTPUT_DIR, NUMBER_OF_EPOCHS, batch_size, TRAIN_PORTION, VALIDATION_PORTION, max_vocab_size, pre_trained_embedding_specification,
                                                  loss_function_spec=loss_function_spec,
                                                  sentiment_size=sentiment_size, 
                                                  encoding_hidden_size=encoding_hidden_size,
                                                  number_of_encoding_layers=number_of_encoding_layers,
                                                  dropout_probability=dropout_probability)

@debug_on_error
def train_model() -> None:
    predictor = get_default_LSTMScaledDotProductAttentionPredictor()
    predictor.train()
    predictor.load_parameters(predictor.best_saved_model_location)
    # predictor.demonstrate_training_examples()
    predictor.demonstrate_validation_examples()
    return 

if __name__ == '__main__':
    train_model()
