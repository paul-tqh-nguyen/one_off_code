#!/usr/bin/python3
"#!/usr/bin/python3 -OO"

"""
"""

# @todo fill in the top-level doc string

###########
# Imports #
###########

import random

import torch
from torchtext import data

################################################
# Misc. Globals & Global State Initializations #
################################################

SEED = 1234
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

MAX_VOCAB_SIZE = 25_000

#############
# Load Data #
#############

TEXT = data.Field(tokenize = 'spacy', include_lengths = True, batch_first = True)
LABEL = data.LabelField(dtype = torch.long)

train_data, test_data = datasets.TREC.splits(TEXT, LABEL, fine_grained=False)
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

###############
# Main Driver #
###############

if __name__ == '__main__':
    print() # @todo fill this in
