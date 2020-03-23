#!/usr/bin/python3
"#!/usr/bin/python3 -OO"

"""
https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb
"""

###########
# Imports #
###########

import pdb
import traceback
import sys
import random
import time
import spacy
from contextlib import contextmanager
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torchtext import datasets

################################################
# Misc. Globals & Global State Initializations #
################################################

SEED = 1234
MAX_VOCAB_SIZE = 25_000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NLP = spacy.load('en')

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

NUMBER_OF_EPOCHS = 5
BATCH_SIZE = 64

DROPOUT_PROBABILITY = 0.5
NUMBER_OF_ENCODING_LAYERS = 2
EMBEDDING_SIZE = 100
ENCODING_HIDDEN_SIZE = 256
OUTPUT_SIZE = 2

####################
# Model Definition #
####################

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, encoding_hidden_size, output_size, number_of_encoding_layers, dropout_probability):
        super().__init__()
        if __debug__:
            self.embedding_size = embedding_size
            self.encoding_hidden_size = encoding_hidden_size
            self.number_of_encoding_layers = number_of_encoding_layers
        self.embedding_layers = nn.Sequential(OrderedDict([
            ("embedding_layer", nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_IDX)),
            ("dropout_layer", nn.Dropout(dropout_probability)),
        ]))
        self.rnn = nn.LSTM(embedding_size, 
                           encoding_hidden_size, 
                           num_layers=number_of_encoding_layers, 
                           bidirectional=True, 
                           dropout=dropout_probability)
        self.dropout = nn.Dropout(dropout_probability)
        self.prediction_layers = nn.Sequential(OrderedDict([
            ("fully_connected_layer", nn.Linear(encoding_hidden_size * 2, output_size)),
            ("dropout_layer", nn.Dropout(dropout_probability)),
            ("softmax_layer", nn.Softmax(dim=1)),
        ]))
        
    def forward(self, text_batch, text_lengths):
        if __debug__:
            max_sentence_length = max(text_lengths)
            batch_size = text_batch.shape[1]
        assert batch_size <= BATCH_SIZE
        assert tuple(text_batch.shape) == (max_sentence_length, batch_size)
        assert tuple(text_lengths.shape) == (batch_size,)
                
        embedded_batch = self.embedding_layers(text_batch)
        assert tuple(embedded_batch.shape) == (max_sentence_length, batch_size, self.embedding_size)
        
        embedded_batch_packed = nn.utils.rnn.pack_padded_sequence(embedded_batch, text_lengths)
        if __debug__:
            encoded_batch_packed, (encoding_hidden_state, encoding_cell_state) = self.rnn(embedded_batch_packed)
        else:
            encoded_batch_packed, _ = self.rnn(embedded_batch_packed)
        encoded_batch, encoded_batch_lengths = nn.utils.rnn.pad_packed_sequence(encoded_batch_packed)
        assert tuple(encoded_batch.shape) == (max_sentence_length, batch_size, self.encoding_hidden_size*2)
        assert tuple(encoding_hidden_state.shape) == (self.number_of_encoding_layers*2, batch_size, self.encoding_hidden_size)
        assert tuple(encoding_cell_state.shape) == (self.number_of_encoding_layers*2, batch_size, self.encoding_hidden_size)
        
        hidden = self.dropout(torch.cat((encoding_hidden_state[-2,:,:], encoding_hidden_state[-1,:,:]), dim = 1))
        #hidden = [batch size, hid dim * num directions]
        
        prediction = self.prediction_layers(hidden)
        assert tuple(prediction.shape) == (batch_size, OUTPUT_SIZE)
        
        return prediction

#############
# Load Data #
#############

TEXT = data.Field(tokenize = 'spacy', include_lengths = True)
LABEL = data.LabelField(dtype = torch.long)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state = random.seed(SEED))

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE, vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_)
LABEL.build_vocab(train_data)

assert TEXT.vocab.vectors.shape[0] <= MAX_VOCAB_SIZE+2
assert TEXT.vocab.vectors.shape[1] == EMBEDDING_SIZE

VOCAB_SIZE = len(TEXT.vocab)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    device = DEVICE)

#print(f'An arbitrary training batch: {" ".join(map(lambda dim: dim[0], map(lambda int_list: list(map(lambda index :TEXT.vocab.itos[index], int_list)), next(iter(train_iterator)).text[0])))}')

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

###########
# Helpers #
###########

@contextmanager
def timer(section_name=None, exitCallback=None):
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    if exitCallback != None:
        exitCallback(elapsed_time)
    elif section_name:
        print('Execution of "{section_name}" took {elapsed_time} seconds.'.format(section_name=section_name, elapsed_time=elapsed_time))
    else:
        print('Execution took {elapsed_time} seconds.'.format(elapsed_time=elapsed_time))

def debug_on_error(func):
    def func_wrapped(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as err:
            print(f'Exception Class: {type(err)}')
            print(f'Exception Args: {err.args}')
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
    return func_wrapped

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def discrete_accuracy(y_hat, y):
    y_hat_indices_of_max = y_hat.argmax(dim=1)
    number_of_correct_answers = (y_hat_indices_of_max == y).float().sum(dim=0)
    mean_accuracy = number_of_correct_answers / y.shape[0]
    return mean_accuracy

def predict_sentiment(model, sentence):
    model.eval()
    tokenized = [token.text for token in NLP.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    lengths = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(DEVICE)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(lengths)
    prediction_as_index = model(tensor, length_tensor).item()
    prediction = LABEL.vocab.itos(prediction_as_index)
    return prediction

###############
# Main Driver #
###############

def train(model, iterator, optimizer, loss_function):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths)
        loss = loss_function(predictions, batch.label)
        acc = discrete_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, loss_function):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = loss_function(predictions, batch.label)
            acc = discrete_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

@debug_on_error
def main():
    model = RNN(VOCAB_SIZE, 
                EMBEDDING_SIZE,
                ENCODING_HIDDEN_SIZE, 
                OUTPUT_SIZE, 
                NUMBER_OF_ENCODING_LAYERS, 
                DROPOUT_PROBABILITY)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    pretrained_embeddings = TEXT.vocab.vectors

    print(f'pretrained_embeddings.shape {pretrained_embeddings.shape}')
    embedding_layer_weight_data = model.embedding_layers.embedding_layer.weight.data
    embedding_layer_weight_data.copy_(pretrained_embeddings)
    embedding_layer_weight_data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)
    embedding_layer_weight_data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
    print(f'embedding_layer_weight_data.shape {embedding_layer_weight_data.shape}')
    
    optimizer = optim.Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()
    model = model.to(DEVICE)
    loss_function = loss_function.to(DEVICE)
    best_valid_loss = float('inf')
    
    for epoch_index in range(NUMBER_OF_EPOCHS):
        with timer(section_name=f"Epoch {epoch_index}"):
            train_loss, train_acc = train(model, train_iterator, optimizer, loss_function)
            valid_loss, valid_acc = evaluate(model, valid_iterator, loss_function)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut2-model.pt')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    model.load_state_dict(torch.load('tut2-model.pt'))
    test_loss, test_acc = evaluate(model, test_iterator, loss_function)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    
    print(f'predict_sentiment(model, "This film is terrible") {predict_sentiment(model, "This film is terrible")}')
    print(f'predict_sentiment(model, "This film is great") {predict_sentiment(model, "This film is great")}')

if __name__ == '__main__':
    main()

