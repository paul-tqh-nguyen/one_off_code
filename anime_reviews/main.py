#!/usr/bin/python3 -OO

'''

This module is the main interface towards our collaborative filtering model. 

Sections:
* Imports
* Default Model
* Driver

'''

###########
# Imports #
###########

import argparse

from misc_utilities import *
from global_values import GPU_IDS

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
    from models import LinearColaborativeFilteringModel
    LinearColaborativeFilteringModel.train_model(
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
    parser = argparse.ArgumentParser(prog='tool', formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 9999))
    parser.add_argument('-train-default-model', action='store_true', help='Train the default model.')
    parser.add_argument('-hyperparameter-search', action='store_true', help='Perform several trials of hyperparameter search.')
    parser.add_argument('-analyze-hyperparameter-search-results', action='store_true', help=f'Analyze completed hyperparameter search trials so far.')
    args = parser.parse_args()
    number_of_args_specified = sum(map(int,map(bool,vars(args).values())))
    if number_of_args_specified == 0:
        parser.print_help()
    elif number_of_args_specified > 1:
        print('Please specify exactly one action.')
    elif args.train_default_model:
        train_default_model()
    elif args.hyperparameter_search:
        from hyperparameter_search import hyperparameter_search
        from models import LinearColaborativeFilteringModel
        hyperparameter_search(LinearColaborativeFilteringModel)
    elif args.analyze_hyperparameter_search_results:
        from hyperparameter_search import  analyze_hyperparameter_search_results
        analyze_hyperparameter_search_results()
    else:
        raise ValueError('Unexpected args received.')
    return

if __name__ == '__main__':
    main()

