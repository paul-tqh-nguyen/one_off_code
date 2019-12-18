#!/usr/bin/python3 -O

###########
# Imports #
###########

from typing import Iterable, List
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

#####################
# Model Definitions #
#####################

class SelfAttentionLayers(nn.Module):
    def __init__(self, hidden_size, input_size=400, number_of_attention_heads=2):
        super().__init__()
        self.number_of_attention_heads = number_of_attention_heads
        self.reduction_layer = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU(True)
        self.attention_layer = nn.Linear(hidden_size, number_of_attention_heads)
        '''
        self.attention_layers = nn.Sequential(OrderedDict([
            ("reduction_layer", nn.Linear(input_size, hidden_size)),
            ("activation", nn.ReLU(True)),
            ("attention_layer", nn.Linear(hidden_size, number_of_attention_heads)),
        ]))
        #'''
                
    def forward(self, input_matrix):
        max_number_of_words, batch_size, input_size = input_matrix.shape
        
        reduced = self.reduction_layer(input_matrix)
        activated = self.activation(reduced)
        attention_weights_pre_softmax = self.attention_layer(activated)
        #attention_weights_pre_softmax = self.attention_layers(input_matrix)
        attention_weights = F.softmax(attention_weights_pre_softmax, dim=0)
        attention_weights_batch_first = attention_weights.transpose(0,1)
        attention_weights_batch_first_transpose = attention_weights_batch_first.transpose(1,2)
        attention_weights_times_transpose = attention_weights_batch_first.matmul(attention_weights_batch_first_transpose)
        identity_matrix = torch.eye(max_number_of_words).repeat(batch_size,1,1)
        attenion_regularization_penalty_unnormalized = attention_weights_times_transpose - identity_matrix
        attenion_regularization_penalty_per_batch = torch.sqrt((attenion_regularization_penalty_unnormalized**2).sum(dim=1).sum(dim=1))
        attenion_regularization_penalty = attenion_regularization_penalty_per_batch.sum(dim=0)
        attention_weights_duplicated = attention_weights.view(-1,1).repeat(1,input_size).view(max_number_of_words, batch_size, self.number_of_attention_heads*input_size)
        input_matrix_duplicated = input_matrix.repeat(1,1,self.number_of_attention_heads) 
        weight_adjusted_input_matrix = torch.mul(attention_weights_duplicated, input_matrix_duplicated)
        attended_matrix = torch.sum(weight_adjusted_input_matrix, dim=0)
        print("\n\n\n")
        print("input_matrix {}".format(input_matrix))
        print("attention_weights_pre_softmax {}".format(attention_weights_pre_softmax))
        print("attention_weights {}".format(attention_weights))
        
        assert tuple(attention_weights_pre_softmax.shape) == (max_number_of_words, batch_size, self.number_of_attention_heads), "attention_weights_pre_softmax has unexpected dimensions."
        assert tuple(attention_weights.shape) == (max_number_of_words, batch_size, self.number_of_attention_heads), "attention_weights has unexpected dimensions."
        assert tuple(attention_weights_batch_first.shape) == (batch_size, max_number_of_words, self.number_of_attention_heads), "attention_weights_batch_first has unexpected dimensions."
        assert tuple(attention_weights_batch_first_transpose.shape) == (batch_size, self.number_of_attention_heads, max_number_of_words), \
            "attention_weights_batch_first_transpose has unexpected dimensions."
        assert tuple(attention_weights_times_transpose.shape) == (batch_size, max_number_of_words, max_number_of_words), "attention_weights_times_transpose has unexpected dimensions."
        assert tuple(identity_matrix.shape) == (batch_size, max_number_of_words, max_number_of_words), "identity_matrix has unexpected dimensions."
        assert tuple(attenion_regularization_penalty_unnormalized.shape) == (batch_size, max_number_of_words, max_number_of_words), \
            "attenion_regularization_penalty_unnormalized has unexpected dimensions."
        assert tuple(attention_weights_duplicated.shape) == (max_number_of_words, batch_size, self.number_of_attention_heads*input_size), "attention_weights_duplicated has unexpected dimensions."
        assert tuple(input_matrix_duplicated.shape) == (max_number_of_words, batch_size, self.number_of_attention_heads*input_size), "input_matrix_duplicated has unexpected dimensions."
        assert tuple(weight_adjusted_input_matrix.shape) == (max_number_of_words, batch_size, self.number_of_attention_heads*input_size), "weight_adjusted_input_matrix has unexpected dimensions."
        assert tuple(attended_matrix.shape) == (batch_size, self.number_of_attention_heads*input_size), "attended_matrix has unexpected dimensions."
        return attended_matrix, attenion_regularization_penalty

##########################
# Classifier Definitions #
##########################

class SentimentAnalysisClassifier():
    def __init__(self, batch_size=1, learning_rate=1e-2, attenion_regularization_penalty_multiplicative_factor=0.1,
                 number_of_attention_heads=2, attention_hidden_size=24,
    ):
        self.attenion_regularization_penalty_multiplicative_factor = attenion_regularization_penalty_multiplicative_factor
        self.loss_function = nn.BCELoss()
        self.model = SelfAttentionLayers(input_size=6, number_of_attention_heads=number_of_attention_heads, hidden_size=attention_hidden_size)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.x_data = [torch.tensor([[[0.0111,  0.0222,  0.0333, 0.0444, 0.0555,  0.0666]]],), torch.tensor([[[0.0634,  0.0424,  0.0726, 0.1276, 0.3193,  0.1732]],
                                                                                                             [[0.1098,  0.0531,  0.1112, 0.0798, 0.1852,  0.0991]]])]
        self.y_data = [torch.tensor([0.0649,  0.0347,  0.0702, 0.0811, 0.1790,  0.1028]), torch.tensor([0.1098,  0.0531,  0.1112, 0.0798, 0.1852,  0.0991])]
        
    def train(self, number_of_epochs_to_train: int) -> None:
        self.model.train()
        for new_epoch_index in range(number_of_epochs_to_train):
            for iteration_index, (x_batch, y_batch) in enumerate(zip(self.x_data, self.y_data)):
                y_batch_predicted, attenion_regularization_penalty = self.model(x_batch)
                #'''
                loss_via_correctness = self.loss_function(y_batch_predicted, y_batch)
                print("loss_via_correctness {}".format(loss_via_correctness))
                loss_via_attention_regularization = attenion_regularization_penalty * self.attenion_regularization_penalty_multiplicative_factor
                batch_loss = loss_via_correctness + loss_via_attention_regularization
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                #'''

###############
# Main Runner #
###############

def main():
    classifier = SentimentAnalysisClassifier(**{
        'batch_size': 1,
        'learning_rate': 0.01, 
        'attenion_regularization_penalty_multiplicative_factor': 0.1, 
        'attention_hidden_size': 16, 
        'number_of_attention_heads': 1,
    }) ; classifier.train(2)
    
    
if __name__ == '__main__':
    main()
