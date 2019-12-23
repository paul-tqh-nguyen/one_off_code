#!/usr/bin/python3 -O

"""
https://medium.com/@bhadreshpsavani/tutorial-on-sentimental-analysis-using-pytorch-b1431306a2d7
"""

import numpy as np
from string import punctuation
from collections import Counter 
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

class SentimentalLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):    
        """
        Initialize the model by setting up the layers
        """
        super().__init__()
        self.output_size=output_size
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim
        #Embedding and LSTM layers
        print("vocab_size {}".format(vocab_size))
        print("embedding_dim {}".format(embedding_dim))
        self.embedding=nn.Embedding(vocab_size, embedding_dim)
        self.lstm=nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        #dropout layer
        self.dropout=nn.Dropout(0.3)
        #Linear and sigmoid layer
        self.fc1=nn.Linear(hidden_dim, 64)
        self.fc2=nn.Linear(64, 16)
        self.fc3=nn.Linear(16,output_size)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size=x.size()
        #Embadding and LSTM output
        embedd=self.embedding(x)
        lstm_out, hidden=self.lstm(embedd, hidden)
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
        return sig_out, hidden
    
    def init_hidden(self, batch_size):
        """Initialize Hidden STATE"""
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        train_on_gpu = torch.cuda.is_available()
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

def main():
    with open('data/reviews.txt', 'r') as f:
        reviews = f.readlines()
    with open('data/labels.txt', 'r') as f:
        labels = f.readlines()
    for review, label in list(zip(reviews, labels))[:5]:
        print("review {}".format(review))
        print("label {}".format(label))
        print()
    print("punctuation {}".format(punctuation))
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
    print("total_words {}".format(total_words))
    print("list(sorted_words)[:5] {}".format(list(sorted_words)[:5]))
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
    print("list(encoded_reviews)[:5] {}".format(list(encoded_reviews)[:5]))
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
    print("features[:3,:] {}".format(features[:3,:]))
    labels = [1 if label.strip()=='positive' else 0 for label in labels]
    #split_dataset into 80% training , 10% test and 10% Validation Dataset
    train_x = features[:int(0.8*len(features))]
    train_y = labels[:int(0.8*len(features))]
    valid_x = features[int(0.8*len(features)):int(0.9*len(features))]
    valid_y = labels[int(0.8*len(features)):int(0.9*len(features))]
    test_x = features[int(0.9*len(features)):]
    test_y = labels[int(0.9*len(features)):]
    print(len(train_y), len(valid_y), len(test_y))
    print("train_x[:2] {}".format(train_x[:2]))
    train_data = TensorDataset(torch.LongTensor(train_x), torch.FloatTensor(train_y))
    valid_data = TensorDataset(torch.LongTensor(valid_x), torch.FloatTensor(valid_y))
    test_data = TensorDataset(torch.LongTensor(test_x), torch.FloatTensor(test_y))
    #dataloader
    batch_size=50
    train_loader = DataLoader(train_data, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    '''
    train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader=DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_loader=DataLoader(test_data, batch_size=batch_size, shuffle=True)
    #'''
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
    
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)
    
        # batch loop
        for training_input_index, (inputs, labels) in enumerate(train_loader):
    
            if(train_on_gpu):
                inputs=inputs.cuda()
                labels=labels.cuda()
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])
    
            # zero accumulated gradients
            net.zero_grad()

            print("training_input_index {}".format(training_input_index))
            
            # get the output from the model
            output, h = net(inputs, h)
    
            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            print("loss {}".format(float(loss)))
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()
            
            # loss stats
            if training_input_index % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:
    
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])
                    if train_on_gpu:
                        inputs, labels = inputs.cuda(), labels.cuda()  
                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())
    
                    val_losses.append(val_loss.item())
    
                net.train()
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step Index: {}...".format(training_input_index),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))
    test_losses = [] # track loss
    num_correct = 0

    # init hidden state
    h = net.init_hidden(batch_size)

    net.eval()
    # iterate over test data
    for inputs, labels in test_loader:

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()


        output, h = net(inputs, h)

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
