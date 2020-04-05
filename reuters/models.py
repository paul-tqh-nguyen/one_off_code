#!/usr/bin/python3
"#!/usr/bin/python3 -OO"

"""
"""

# @todo fill in the top-level doc string
# @todo add type declarations 
###########
# Imports #
###########

import random

import preprocess_data
from misc_utilites import eager_map
from misc_utilites import debug_on_error # @todo remove this

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
BATCH_SIZE = 32

EMBEDDING_SIZE = 100

#############
# Load Data #
#############

TEXT = data.Field(tokenize = 'spacy', include_lengths = True, batch_first = True)
LABEL = data.LabelField(dtype = torch.long)

with open(preprocess_data.TOPICS_DATA_OUTPUT_CSV_FILE, 'r') as topics_csv_file:
    column_names = eager_map(str.strip, topics_csv_file.readline().split(','))
    # @todo account for text_title column in prediction as well
    column_name_to_field_map = [(column_name, TEXT if column_name=='text' else
                                 None if column_name in preprocess_data.COLUMNS_RELEVANT_TO_TOPICS_DATA else
                                 LABEL) for column_name in column_names]

    TOPICS = list(set(column_names)-preprocess_data.COLUMNS_RELEVANT_TO_TOPICS_DATA)

TRAIN_PORTION, VALIDATION_PORTION, TESTING_PORTION = (0.50, 0.20, 0.3)

all_data = data.dataset.TabularDataset(
    path=preprocess_data.TOPICS_DATA_OUTPUT_CSV_FILE,
    format='csv',
    skip_header=True,
    fields=column_name_to_field_map)

training_data, validation_data, testing_data = all_data.split(split_ratio=[TRAIN_PORTION, VALIDATION_PORTION, TESTING_PORTION], random_state = random.seed(SEED))

TEXT.build_vocab(training_data, max_size = MAX_VOCAB_SIZE, vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_)
LABEL.build_vocab(training_data)

assert TEXT.vocab.vectors.shape[0] <= MAX_VOCAB_SIZE+2
assert TEXT.vocab.vectors.shape[1] == EMBEDDING_SIZE

VOCAB_SIZE = len(TEXT.vocab)

training_iterator, validation_iterator, testing_iterator = data.BucketIterator.splits(
    (training_data, validation_data, testing_data),
    batch_size = BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch = True,
    repeat=False,
    device = DEVICE)

class NumericalizedBatchIterator:
    def __init__(self, non_numericalized_iterator, x_attribute_name, y_attribute_names):
        self.non_numericalized_iterator = non_numericalized_iterator
        self.x_attribute_name: str = x_attribute_name
        self.y_attribute_names: List[str] = y_attribute_names

    def __iter__(self):
        for non_numericalized_batch in self.non_numericalized_iterator:
            x = getattr(non_numericalized_batch, self.x_attribute_name)
            y = torch.cat([getattr(non_numericalized_batch, y_attribute_names).unsqueeze(1) for feat in self.y_attribute_names], dim=1).int()
            yield (x, y)

    def __len__(self):
        return len(self.non_numericalized_iterator)

training_iterator = NumericalizedBatchIterator(training_iterator, 'text', TOPICS)
validation_iterator = NumericalizedBatchIterator(validation_iterator, 'text', TOPICS)
testing_iterator = NumericalizedBatchIterator(testing_iterator, 'text', TOPICS)

###############
# Main Driver #
###############

@debug_on_error
def main(): # @todo get rid of this
    for batch, labels in training_iterator:
        print(f'training_iterator batch {batch}')
        print(f'training_iterator labels {labels}')
    for batch, labels in validation_iterator:
        print(f'validation_iterator batch {batch}')
        print(f'validation_iterator labels {labels}')
    for batch, labels in testing_iterator:
        print(f'testing_iterator batch {batch}')
        print(f'testing_iterator labels {labels}')

if __name__ == '__main__':
    print() # @todo fill this in
    main()
