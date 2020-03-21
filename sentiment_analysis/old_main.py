#!/usr/bin/python3 -O

"""

Following this tutorial https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb

"""

###########
# Imports #
###########

import time
import torch
from torchtext import data
from torchtext import datasets
import random
import torch.nn as nn
import torch.optim as optim

###########
# Globals #
###########

SEED = 1234
MAX_VOCAB_SIZE = 25_000
BATCH_SIZE = 64
EMBEDDING_SIZE = 100
RNN_HIDDEN_SIZE = 256
OUTPUT_SIZE = 1
N_EPOCHS = 5

###################
# Initializations #
###################

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

###############
# Main Driver #
###############

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, rnn_hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, rnn_hidden_size)
        self.fc = nn.Linear(rnn_hidden_size, output_size)

    def forward(self, text):
        # text = [sent len, batch size]
        embedded = self.embedding(text)
        # embedded = [sent len, batch size, emb dim]
        output, hidden = self.rnn(embedded)
        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        return self.fc(hidden.squeeze(0))

TEXT = data.Field(tokenize = 'spacy')
LABEL = data.LabelField(dtype = torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state = random.seed(SEED))

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VOCAB_SIZE = len(TEXT.vocab)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device)

model = RNN(VOCAB_SIZE, EMBEDDING_SIZE, RNN_HIDDEN_SIZE, OUTPUT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def main() -> None:
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of valid examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')
    print(f'First training example: {" ".join(train_data[0].text)}')
    print(f'First valid example: {" ".join(valid_data[0].text)}')
    print(f'First testing example: {" ".join(test_data[0].text)}')
    print(f'Most common words: {TEXT.vocab.freqs.most_common(20)}')
    print(f'First vocab words: {TEXT.vocab.itos[:10]}')
    print(f'Labels: {LABEL.vocab.stoi}')
    print(f'Device: {device}')
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    model.load_state_dict(torch.load('tut1-model.pt'))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

if __name__ == '__main__':
    main()
