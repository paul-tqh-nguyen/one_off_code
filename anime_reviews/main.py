'#!/usr/bin/python3 -OO' # @todo use this

'''
'''

# @todo update doc string

###########
# Imports #
###########

from misc_utilities import *
from global_values import *
from trainer import train_model
from hyperparameter_search import hyperparameter_search

# @todo make sure these imports are used

#################
# Default Model #
#################

NUMBER_OF_EPOCHS = 15
BATCH_SIZE = 256
GRADIENT_CLIP_VAL = 1.0

LEARNING_RATE = 1e-3
EMBEDDING_SIZE = 100
REGULARIZATION_FACTOR = 1
DROPOUT_PROBABILITY = 0.5

def train_default_model() -> None:
    train_model(
        learning_rate=LEARNING_RATE,
        number_of_epochs=NUMBER_OF_EPOCHS,
        batch_size=BATCH_SIZE,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        embedding_size=EMBEDDING_SIZE,
        regularization_factor=REGULARIZATION_FACTOR,
        dropout_probability=DROPOUT_PROBABILITY,
        gpus=GPU_IDS,
    )
    return

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    train_default_model()
    # hyperparameter_search()
    return

if __name__ == '__main__':
    main()

