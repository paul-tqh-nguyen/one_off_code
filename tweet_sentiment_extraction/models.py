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

from abstract_classifier import Classifier, DEVICE, soft_f1_loss

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

##########
# Models #
##########

class RNNNetwork(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, encoding_hidden_size: int, number_of_encoding_layers: int, attention_intermediate_size: int, number_of_attention_heads: int, output_size: int, dropout_probability: float, pad_idx: int, unk_idx: int, initial_embedding_vectors: torch.Tensor):
        super().__init__()
        if __debug__:
            self.embedding_size = embedding_size
            self.encoding_hidden_size = encoding_hidden_size
            self.number_of_encoding_layers = number_of_encoding_layers
            self.number_of_attention_heads = number_of_attention_heads
            self.output_size = output_size
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
        self.attention_layers = AttentionLayers(encoding_hidden_size, attention_intermediate_size, number_of_attention_heads, dropout_probability)
        self.prediction_layers = nn.Sequential(OrderedDict([
            ("fully_connected_layer", nn.Linear(encoding_hidden_size*2*number_of_attention_heads, output_size)),
            ("dropout_layer", nn.Dropout(dropout_probability)),
            ("sigmoid_layer", nn.Sigmoid()),
        ]))
        self.to(DEVICE)

    def forward(self, text_batch: torch.Tensor, text_lengths: torch.Tensor):
        if __debug__:
            max_sentence_length = max(text_lengths)
            batch_size = text_batch.shape[0]
        assert tuple(text_batch.shape) == (batch_size, max_sentence_length)
        assert tuple(text_lengths.shape) == (batch_size,)

        embedded_batch = self.embedding_layers(text_batch)
        assert tuple(embedded_batch.shape) == (batch_size, max_sentence_length, self.embedding_size)

        embedded_batch_packed = nn.utils.rnn.pack_padded_sequence(embedded_batch, text_lengths, batch_first=True)
        if __debug__:
            encoded_batch_packed, (encoding_hidden_state, encoding_cell_state) = self.encoding_layers(embedded_batch_packed)
            encoded_batch, encoded_batch_lengths = nn.utils.rnn.pad_packed_sequence(encoded_batch_packed, batch_first=True)
        else:
            encoded_batch_packed, _ = self.encoding_layers(embedded_batch_packed)
            encoded_batch, _ = nn.utils.rnn.pad_packed_sequence(encoded_batch_packed, batch_first=True)
        assert tuple(encoded_batch.shape) == (batch_size, max_sentence_length, self.encoding_hidden_size*2)
        assert tuple(encoding_hidden_state.shape) == (self.number_of_encoding_layers*2, batch_size, self.encoding_hidden_size)
        assert tuple(encoding_cell_state.shape) == (self.number_of_encoding_layers*2, batch_size, self.encoding_hidden_size)
        assert tuple(encoded_batch_lengths.shape) == (batch_size,)
        assert (encoded_batch_lengths.to(text_lengths.device) == text_lengths).all()

        attended_batch = self.attention_layers(encoded_batch, text_lengths)
        assert tuple(attended_batch.shape) == (batch_size, self.encoding_hidden_size*2*self.number_of_attention_heads)
        
        prediction = self.prediction_layers(attended_batch)
        assert tuple(prediction.shape) == (batch_size, self.output_size)
        
        return prediction

class ConvNetwork(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, convolution_hidden_size: int, kernel_sizes: List[int], pooling_method: Callable[[torch.Tensor], torch.Tensor], output_size: int, dropout_probability: float, pad_idx: int, unk_idx: int, initial_embedding_vectors: torch.Tensor):
        super().__init__()
        if __debug__:
            self.embedding_size = embedding_size
            self.convolution_hidden_size = convolution_hidden_size
            self.kernel_sizes = kernel_sizes
            self.output_size = output_size
        self.embedding_layers = nn.Sequential(OrderedDict([
            ("embedding_layer", nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx, max_norm=1.0)),
            ("dropout_layer", nn.Dropout(dropout_probability)),
        ]))
        self.embedding_layers.embedding_layer.weight.data.copy_(initial_embedding_vectors)
        self.embedding_layers.embedding_layer.weight.data[unk_idx] = torch.zeros(embedding_size)
        self.embedding_layers.embedding_layer.weight.data[pad_idx] = torch.zeros(embedding_size)
        self.convolutional_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_size, out_channels=convolution_hidden_size, kernel_size=kernel_size)
            for kernel_size in kernel_sizes
        ])
        self.pooling_method = pooling_method
        self.prediction_layers = nn.Sequential(OrderedDict([
            ("fully_connected_layers", nn.Linear(len(kernel_sizes) * convolution_hidden_size, output_size)),
            ("dropout_layer", nn.Dropout(dropout_probability)),
            ("sigmoid_layer", nn.Sigmoid()),
        ]))
        self.dropout_layers = nn.Dropout(dropout_probability)
        self.to(DEVICE)
        
    def forward(self, text_batch: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        batch_size = text_lengths.shape[0]
        max_sentence_length = text_batch.shape[1]
        assert tuple(text_batch.shape) == (batch_size, max_sentence_length)
        assert tuple(text_lengths.shape) == (batch_size,)
        
        embedded_batch = self.embedding_layers(text_batch)
        assert tuple(embedded_batch.shape) == (batch_size, max_sentence_length, self.embedding_size)
        embedded_batch = embedded_batch.permute(0, 2, 1)
        assert tuple(embedded_batch.shape) == (batch_size, self.embedding_size, max_sentence_length)
        
        convolved_batches = [F.relu(conv(embedded_batch)) for conv in self.convolutional_layers]
        assert reduce(bool.__and__, [tuple(convolved_batch.shape) == (batch_size, self.convolution_hidden_size, max_sentence_length-kernel_size+1)
                                     for convolved_batch, kernel_size in zip(convolved_batches, self.kernel_sizes)])
        
        pooled_batches = [self.pooling_method(convolved_batch, kernel_size=convolved_batch.shape[2]).squeeze(2) for convolved_batch in convolved_batches]
        assert reduce(bool.__and__, [tuple(pooled_batch.shape) == (batch_size, self.convolution_hidden_size) for pooled_batch in pooled_batches])
        
        concatenated_pooled_batches = torch.cat(pooled_batches, dim = 1)
        concatenated_pooled_batches = self.dropout_layers(concatenated_pooled_batches)
        assert tuple(concatenated_pooled_batches.shape) == (batch_size, self.convolution_hidden_size * len(self.kernel_sizes))

        prediction = self.prediction_layers(concatenated_pooled_batches)
        assert tuple(prediction.shape) == (batch_size, self.output_size)
        
        return prediction


###############
# Classifiers #
###############

class RNNClassifier(Classifier):
    def initialize_model(self) -> None:
        self.encoding_hidden_size = self.model_args['encoding_hidden_size']
        self.number_of_encoding_layers = self.model_args['number_of_encoding_layers']
        self.attention_intermediate_size = self.model_args['attention_intermediate_size']
        self.number_of_attention_heads = self.model_args['number_of_attention_heads']
        self.dropout_probability = self.model_args['dropout_probability']
        vocab_size = len(self.text_field.vocab)
        self.model = RNNNetwork(vocab_size, self.embedding_size, self.encoding_hidden_size, self.number_of_encoding_layers, self.attention_intermediate_size, self.number_of_attention_heads, self.output_size, self.dropout_probability, self.pad_idx, self.unk_idx, self.text_field.vocab.vectors)
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_function = nn.BCELoss().to(DEVICE)
        return

###############
# Main Driver #
###############

def main() -> None:
    return 

if __name__ == '__main__':
    main()
