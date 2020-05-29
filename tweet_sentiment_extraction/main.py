#!/usr/bin/python3
'#!/usr/bin/python3 -OO'

'''
'''

# @todo update doc string

###########
# Imports #
###########

import argparse
import random
import os
import itertools
from typing import Generator

from model_utilities import *
from misc_utilities import *

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    
    parser = argparse.ArgumentParser(prog='tool', formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 9999))

    # Misc. Commands
    parser.add_argument('-cuda-device-id', help=f'Set the CUDA device ID.')
    parser.add_argument('-preprocess-data', action='store_true', help=f'Preprocess the raw data. Results stored in {PREPROCESSED_TRAINING_DATA_JSON_FILE}.')

    # RoBERTa Commands
    from hyperparameter_search import ROBERTA_RESULTS_DIRECTORY
    from hyperparameter_search import ROBERTA_BEST_MODEL_SCORE_JSON_FILE_LOCATION
    parser.add_argument('-train-roberta-model', action='store_true', help=f'Trains & evaluates our RoBERTa model on the raw dataset. Results are saved to {ROBERTA_RESULTS_DIRECTORY}.')
    parser.add_argument('-hyperparameter-search-roberta', action='store_true',
                        help=f'Exhaustively performs hyperparameter random search using only the RoBERTa model. Details of the best performance are tracked in {ROBERTA_BEST_MODEL_SCORE_JSON_FILE_LOCATION}')

    # Word Selector Commands
    from hyperparameter_search import WORD_SELECTOR_RESULTS_DIRECTORY
    from hyperparameter_search import WORD_SELECTOR_BEST_MODEL_SCORE_JSON_FILE_LOCATION
    parser.add_argument('-train-word-selector-model', action='store_true', help=f'Trains & evaluates our word selector model on our preprocessed dataset. Results are saved to {WORD_SELECTOR_RESULTS_DIRECTORY}.')
    parser.add_argument('-hyperparameter-search-word-selector-models', action='store_true',
                        help=f'Exhaustively performs hyperparameter random search. Details of the best performance are tracked in {WORD_SELECTOR_BEST_MODEL_SCORE_JSON_FILE_LOCATION}.')
    parser.add_argument('-hyperparameter-search-lstm-sentiment-concatenation-predictor', action='store_true', help=f'Exhaustively performs hyperparameter random search using only the LSTMSentimentConcatenationPredictor model.')
    parser.add_argument('-hyperparameter-search-lstm-scaled-dot-product-attention-predictor', action='store_true',
                        help=f'Exhaustively performs hyperparameter random search using only the LSTMScaledDotProductAttentionPredictor model.')
    parser.add_argument('-hyperparameter-search-naive-dense-predictor', action='store_true', help=f'Exhaustively performs hyperparameter random search using only the NaiveDensePredictor model.')
    
    args = parser.parse_args()
    number_of_args_specified = sum(map(lambda arg_value: 
                                       int(arg_value) if isinstance(arg_value, bool) else 
                                       0 if arg_value is None else 1 if (isinstance(arg_value, str) and arg_value.isnumeric()) else 
                                       arg_value, vars(args).values()))
    if number_of_args_specified == 0:
        parser.print_help()
    if isinstance(args.cuda_device_id, str):
        assert args.cuda_device_id.isnumeric()
        import word_selector_models.abstract_predictor
        word_selector_models.abstract_predictor.set_global_device_id(int(args.cuda_device_id))
        import roberta_models.models
        roberta_models.models.set_global_device_id(int(args.cuda_device_id))
    if args.preprocess_data:
        import preprocess_data
        preprocess_data.preprocess_data()
    
    # Hyperparameter Search
    
    if args.hyperparameter_search_word_selector_models:
        from hyperparameter_search import (
            LSTMSentimentConcatenationPredictor_generator,
            LSTMScaledDotProductAttentionPredictor_generator,
            # NaiveDensePredictor_generator,
            RoBERTaPredictor_generator,
        )
        predictors = roundrobin(
            LSTMSentimentConcatenationPredictor_generator(),
            LSTMScaledDotProductAttentionPredictor_generator(),
            # NaiveDensePredictor_generator(),
            RoBERTaPredictor_generator(),
        )
        from hyperparameter_search import hyperparameter_search
        hyperparameter_search(predictors)

    # RoBERTa
    if args.train_roberta_model:
        import roberta_models.models
        roberta_models.models.train_model()
    if args.hyperparameter_search_roberta:
        from hyperparameter_search import RoBERTaPredictor_generator, hyperparameter_search
        predictors = RoBERTaPredictor_generator()
        hyperparameter_search(predictors)

    # Word Selector
    if args.train_word_selector_model:
        from word_selector_models.models import train_model
        train_model()
    if args.hyperparameter_search_lstm_sentiment_concatenation_predictor:
        from hyperparameter_search import LSTMSentimentConcatenationPredictor_generator, hyperparameter_search
        predictors = LSTMSentimentConcatenationPredictor_generator()
        hyperparameter_search(predictors)
    if args.hyperparameter_search_lstm_scaled_dot_product_attention_predictor:
        from hyperparameter_search import LSTMScaledDotProductAttentionPredictor_generator, hyperparameter_search
        predictors = LSTMScaledDotProductAttentionPredictor_generator()
        hyperparameter_search(predictors)
    if args.hyperparameter_search_naive_dense_predictor:
        from hyperparameter_search import NaiveDensePredictor_generator, hyperparameter_search
        predictors = NaiveDensePredictor_generator()
        hyperparameter_search(predictors)
        
    return

if __name__ == '__main__':
    main()
 
