#!/usr/bin/python3 -OO

'''

This module is the main interface towards our collaborative filtering models. 

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

def train_default_model_linear() -> None:
    from models import LinearColaborativeFilteringModel
    number_of_epochs = 15
    batch_size = 256
    gradient_clip_val = 1.0
    learning_rate = 1e-3
    embedding_size = 100
    regularization_factor = 1
    dropout_probability = 0.5
    LinearColaborativeFilteringModel.train_model(
        gpus=GPU_IDS,
        learning_rate=learning_rate,
        number_of_epochs=number_of_epochs,
        batch_size=batch_size,
        gradient_clip_val=gradient_clip_val,
        embedding_size=embedding_size,
        regularization_factor=regularization_factor,
        dropout_probability=dropout_probability,
    )
    return

def train_default_model_deep_concat() -> None:
    from models import DeepConcatenationColaborativeFilteringModel
    # @todo change these defaults
    number_of_epochs = 13
    batch_size = 2048
    gradient_clip_val = 1.0
    learning_rate = 0.04051541383517857
    embedding_size = 230
    dense_layer_count = 2
    regularization_factor = 2.1908736439954413
    dropout_probability = 0.3954686174994483
    DeepConcatenationColaborativeFilteringModel.train_model(
        gpus=GPU_IDS,
        learning_rate=learning_rate,
        number_of_epochs=number_of_epochs,
        batch_size=batch_size,
        gradient_clip_val=gradient_clip_val,
        embedding_size=embedding_size,
        dense_layer_count=dense_layer_count, 
        regularization_factor=regularization_factor,
        dropout_probability=dropout_probability,
    )
    return

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    parser = argparse.ArgumentParser(prog='tool', formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 9999))
    parser.add_argument('-train-default-model-linear', action='store_true', help='Train the default linear model.')
    parser.add_argument('-hyperparameter-search-linear', action='store_true', help='Perform several trials of hyperparameter search for the linear model.')
    parser.add_argument('-analyze-hyperparameter-search-results-linear', action='store_true', help=f'Analyze completed hyperparameter search trials so far for the linear model.')
    parser.add_argument('-train-default-model-deep-concat', action='store_true', help='Train the default deep concatenation model.')
    parser.add_argument('-hyperparameter-search-deep-concat', action='store_true', help='Perform several trials of hyperparameter search for the deep concatenation model.')
    parser.add_argument('-analyze-hyperparameter-search-results-deep-concat', action='store_true', help=f'Analyze completed hyperparameter search trials so far for the deep concatenation model.')
    args = parser.parse_args()
    number_of_args_specified = sum(map(int,map(bool,vars(args).values())))
    if number_of_args_specified == 0:
        parser.print_help()
    elif number_of_args_specified > 1:
        print('Please specify exactly one action.')
    elif args.train_default_model_linear:
        train_default_model_linear()
    elif args.hyperparameter_search_linear:
        from hyperparameter_search import hyperparameter_search
        from models import LinearColaborativeFilteringModel
        hyperparameter_search(LinearColaborativeFilteringModel)
    elif args.analyze_hyperparameter_search_results_linear:
        from hyperparameter_search import  analyze_hyperparameter_search_results
        from models import LinearColaborativeFilteringModel
        analyze_hyperparameter_search_results(LinearColaborativeFilteringModel)
    elif args.train_default_model_deep_concat:
        train_default_model_deep_concat()
    elif args.hyperparameter_search_deep_concat:
        from hyperparameter_search import hyperparameter_search
        from models import DeepConcatenationColaborativeFilteringModel
        hyperparameter_search(DeepConcatenationColaborativeFilteringModel)
    elif args.analyze_hyperparameter_search_results_deep_concat:
        from hyperparameter_search import  analyze_hyperparameter_search_results
        from models import DeepConcatenationColaborativeFilteringModel
        analyze_hyperparameter_search_results(DeepConcatenationColaborativeFilteringModel)
    else:
        raise ValueError('Unexpected args received.')
    return

if __name__ == '__main__':
    main()

