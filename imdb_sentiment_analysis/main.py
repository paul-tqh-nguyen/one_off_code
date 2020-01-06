#!/usr/bin/python3 -O

"""

This file contains the main driver for a neural network based sentiment analyzer for IMDB data. 

Owner : paul-tqh-nguyen

Created : 01/03/2020

File Name : sentiment_analysis.py

File Organization:
* Imports
* Main Runner

"""

###########
# Imports #
###########

import numpy as np
from string import punctuation
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
from contextlib import contextmanager

##################
# Misc Utilities #
##################

@contextmanager
def timer(section_name=None, exitCallback=None):
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    if exitCallback != None:
        exitCallback(elapsed_time)
    elif section_name:
        pass
        print('Execution of "{section_name}" took {elapsed_time} seconds.'.format(section_name=section_name, elapsed_time=elapsed_time))
    else:
        print('Execution took {elapsed_time} seconds.'.format(elapsed_time=elapsed_time))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class SentimentalLSTM(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):    
        super().__init__()
        print("vocab_size {}".format(vocab_size))
        print("output_size {}".format(output_size))
        print("embedding_dim {}".format(embedding_dim))
        print("hidden_dim {}".format(hidden_dim))
        print("n_layers {}".format(n_layers))
        self.output_size=output_size
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim
        #Embedding and LSTM layers
        print("vocab_size {}".format(vocab_size))
        print("embedding_dim {}".format(embedding_dim))
        self.embedding=nn.Embedding(vocab_size, embedding_dim)
        self.lstm=nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        #dropout layer
        self.dropout=nn.Dropout(0.3) # @todo do we need this?
        #Linear and sigmoid layer
        self.fc1=nn.Linear(hidden_dim, 64)
        self.fc2=nn.Linear(64, 16)
        self.fc3=nn.Linear(16,output_size)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self, x):
        batch_size=x.size()
        #Embadding and LSTM output
        embedd=self.embedding(x)
        lstm_out, _ = self.lstm(embedd)
        #stack up the lstm output
        lstm_out=lstm_out.contiguous().view(-1, self.hidden_dim)
        #dropout and fully connected layers
        out=self.dropout(lstm_out)
        out=self.fc1(out)
        out=self.dropout(out)
        out=self.fc2(out)
        out=self.dropout(out)
        out=self.fc3(out)
        sig_out=self.sigmoid(out)
        sig_out=sig_out.view(batch_size, -1)
        sig_out=sig_out[:, -1]
        return sig_out

def main():
    with open('data/reviews.txt', 'r') as f:
        reviews = f.readlines()
    with open('data/labels.txt', 'r') as f:
        labels = f.readlines()
    all_reviews = []
    for review in reviews:
        review = review.lower()
        review = "".join([character for character in review if character not in punctuation])
        all_reviews.append(review)
    all_text = " ".join(all_reviews)
    all_words = all_text.split()
    count_words = Counter(all_words)
    total_words = len(all_words)
    sorted_words = count_words.most_common(total_words)
    vocab_to_int = {w:i+1 for i,(w,c) in enumerate(sorted_words)}
    encoded_reviews = list()
    for review in all_reviews:
        encoded_review = list()
        for word in review.split():
            if word not in vocab_to_int.keys():
                encoded_review.append(0)
            else:
                encoded_review.append(vocab_to_int[word])
        encoded_reviews.append(encoded_review)
    #print("list(encoded_reviews)[:5] {}".format(list(encoded_reviews)[:5]))
    sequence_length=250
    features=np.zeros((len(encoded_reviews), sequence_length), dtype=int)
    for i, review in enumerate(encoded_reviews):
        review_len=len(review)
        if (review_len<=sequence_length):
            zeros=list(np.zeros(sequence_length-review_len))
            new=zeros+review
        else:
            new=review[:sequence_length]
        features[i,:]=np.array(new)
    #print("features[:3,:] {}".format(features[:3,:]))
    labels = [1 if label.strip()=='positive' else 0 for label in labels]
    #split_dataset into 80% training , 10% test and 10% Validation Dataset
    train_x = features[:int(0.8*len(features))]
    train_y = labels[:int(0.8*len(features))]
    valid_x = features[int(0.8*len(features)):int(0.9*len(features))]
    valid_y = labels[int(0.8*len(features)):int(0.9*len(features))]
    test_x = features[int(0.9*len(features)):]
    test_y = labels[int(0.9*len(features)):]
    
    train_data = TensorDataset(torch.LongTensor(train_x), torch.FloatTensor(train_y))
    valid_data = TensorDataset(torch.LongTensor(valid_x), torch.FloatTensor(valid_y))
    test_data = TensorDataset(torch.LongTensor(test_x), torch.FloatTensor(test_y))
    #dataloader
    batch_size=50
    train_loader = DataLoader(train_data, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size) 
   
    vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2

    net = SentimentalLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    
    lr=0.001
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()
    
    # training params
    
    epochs = 3 # 3-4 is approx where I noticed the validation loss stop decreasing
    
    print_every = 100
    clip=5 # gradient clipping
    
    # move model to GPU, if available
    if(train_on_gpu):
        net.cuda()
    
    net.train()
    print("count_parameters(net) {}".format(count_parameters(net)))
    
    # train for some number of epochs
    for e in range(epochs):

        print("len(train_loader) {}".format(len(train_loader)))
        # batch loop
        for training_input_index, (inputs, labels) in enumerate(train_loader):
    
            if(train_on_gpu):
                inputs=inputs.cuda()
                labels=labels.cuda()
            print('\n\n')
            print("training_input_index {}".format(training_input_index))
            # zero accumulated gradients
            with timer(section_name="zero out gradients"):
                net.zero_grad()

            print("training_input_index {}".format(training_input_index))
            
            # get the output from the model
            with timer(section_name="get predicted label"):
                output = net(inputs)
    
            # calculate the loss and perform backprop
            with timer(section_name="get loss"):
                loss = criterion(output.squeeze(), labels.float())
            print("loss {}".format(float(loss)))
            with timer(section_name="loss.backward()"):
                loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            with timer(section_name="clip gradient"):
                nn.utils.clip_grad_norm_(net.parameters(), clip)
            with timer(section_name="optimizer step"):
                optimizer.step()
            
            # loss stats
            if training_input_index % print_every == 0:
                # Get validation loss
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:
    
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    if train_on_gpu:
                        inputs, labels = inputs.cuda(), labels.cuda()  
                    output = net(inputs)
                    val_loss = criterion(output.squeeze(), labels.float())
    
                    val_losses.append(val_loss.item())
    
                net.train()
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step Index: {}...".format(training_input_index),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))
    test_losses = [] # track loss
    num_correct = 0

    net.eval()
    # iterate over test data
    for inputs, labels in test_loader:

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()


        output = net(inputs)

        # calculate loss
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze())  # rounds to the nearest integer

        # compare predictions to true label
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)


    # -- stats! -- ##
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))

if __name__ == '__main__':
    main()
