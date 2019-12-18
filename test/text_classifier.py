#!/usr/bin/python3 -O

###########
# Imports #
###########

import torch
import torch.nn as nn
import torch.nn.functional as F

#####################
# Model Definitions #
#####################

class ToyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.output_size = output_size
        self.reduction_layer = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU(True)
        self.final_layer = nn.Linear(hidden_size, output_size)
        '''
        if True: # @todo remove this
            def hook_fn(m, i, o):
                print("m {}".format(m))
                print("i {}".format(i))
                print("o {}".format(o))
                print(m)
                print("------------Input Grad------------")
                for grad in i:
                    try:
                        print(grad)
                        print(grad.shape)
                    except AttributeError: 
                        print ("None found for Gradient")
                print("------------Output Grad------------")
                for grad in o:  
                    try:
                        print(grad.shape)
                    except AttributeError: 
                        print ("None found for Gradient")
                print("\n")
            def get_hook_fn_that_prints_header(header: str):
                def new_hook_fn(m, i, o):
                    print(header)
                    return hook_fn(m, i, o)
                return new_hook_fn
            self.reduction_layer.register_backward_hook(get_hook_fn_that_prints_header('reduction_layer'))
            self.activation.register_backward_hook(get_hook_fn_that_prints_header('activation'))
            self.final_layer.register_backward_hook(get_hook_fn_that_prints_header('final_layer'))
            #'''
            
    def forward(self, input_matrix):
        reduced = self.reduction_layer(input_matrix)
        activated = self.activation(reduced)
        output = self.final_layer(activated)
        max_sequence_length, batch_size, embedding_size = input_matrix.shape
        attention_weights = F.softmax(output, dim=0)
        attention_weights_duplicated = attention_weights.view(-1,1).repeat(1,embedding_size).view(max_sequence_length, batch_size, self.output_size*embedding_size)
        input_matrix_duplicated = input_matrix.repeat(1,1,self.output_size) 
        weight_adjusted_input_matrix = torch.mul(attention_weights_duplicated, input_matrix_duplicated)
        attended_matrix = torch.sum(weight_adjusted_input_matrix, dim=0)
        print("\n\n\n\n\n\n")
        print("input_matrix {}".format(input_matrix))
        print("reduced {}".format(reduced))
        print("self.reduction_layer.weight {}".format(self.reduction_layer.weight))
        print("activated {}".format(activated))
        print("output {}".format(output))
        
        print("=============================================================================================================================================================================")
        print("max_sequence_length {}".format(max_sequence_length))
        print("batch_size {}".format(batch_size))
        print("embedding_size {}".format(embedding_size))
        attention_weights_batch_first = output.transpose(0,1)
        #attention_weights_batch_first = attention_weights.transpose(0,1)
        attention_weights_batch_first_transpose = attention_weights_batch_first.transpose(1,2)
        attention_weights_times_transpose = attention_weights_batch_first.matmul(attention_weights_batch_first_transpose)
        identity_matrix = torch.eye(max_sequence_length).repeat(batch_size,1,1)
        attenion_regularization_penalty_unnormalized = attention_weights_times_transpose - identity_matrix
        attenion_regularization_penalty_per_batch = torch.sqrt((attenion_regularization_penalty_unnormalized**2).sum(dim=1).sum(dim=1))
        attenion_regularization_penalty = attenion_regularization_penalty_per_batch.sum(dim=0)
        print("attention_weights {}".format(attention_weights))
        print("attention_weights_batch_first_transpose {}".format(attention_weights_batch_first_transpose))
        print("attention_weights_times_transpose {}".format(attention_weights_times_transpose))
        print("identity_matrix {}".format(identity_matrix))
        print("attenion_regularization_penalty_unnormalized {}".format(attenion_regularization_penalty_unnormalized))
        print("attenion_regularization_penalty_per_batch {}".format(attenion_regularization_penalty_per_batch))
        print("attenion_regularization_penalty {}".format(attenion_regularization_penalty))
        #print(" {}".format())
        return attended_matrix, attenion_regularization_penalty

###############
# Main Runner #
###############

def main():
    x_data = [torch.tensor([[[0.0111,  0.0222,  0.0333, 0.0444, 0.0555,  0.0666]]],), torch.tensor([[[0.0634,  0.0424,  0.0726, 0.1276, 0.3193,  0.1732]],
                                                                                                    [[0.1098,  0.0531,  0.1112, 0.0798, 0.1852,  0.0991]]])]
    y_data = [torch.tensor([0.0649,  0.0347,  0.0702, 0.0811, 0.1790,  0.1028]), torch.tensor([0.1098,  0.0531,  0.1112, 0.0798, 0.1852,  0.0991])]
    input_size = 6
    attention_hidden_size = 16
    output_size = 1
    loss_function = nn.BCELoss()
    model = ToyModel(input_size=input_size, output_size=output_size, hidden_size=attention_hidden_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    number_of_epochs_to_train = 30000
    for _ in range(number_of_epochs_to_train):
        for x_batch, y_batch in zip(x_data, y_data):
            y_batch_predicted, attenion_regularization_penalty = model(x_batch)
            loss_via_correctness = loss_function(y_batch_predicted, y_batch)
            loss_via_attention_regularization = attenion_regularization_penalty
            batch_loss = loss_via_correctness + loss_via_attention_regularization
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
if __name__ == '__main__':
    main()
