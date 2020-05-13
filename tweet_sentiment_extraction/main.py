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

from misc_utilities import *
from preprocess_data import PREPROCESSED_TRAINING_DATA_JSON_FILE
from word_selector_models.abstract_predictor import GLOBAL_BEST_MODEL_SCORE_JSON_FILE_LOCATION as WORD_SELECTOR_BEST_MODEL_SCORE_JSON_FILE_LOCATION_BASENAME

###########
# Globals #
###########

WORD_SELECTOR_RESULTS_DIRECTORY = './word_selector_models/default_output/'
WORD_SELECTOR_BEST_MODEL_SCORE_JSON_FILE_LOCATION = os.path.join('./word_selector_models/',WORD_SELECTOR_BEST_MODEL_SCORE_JSON_FILE_LOCATION_BASENAME)

#########################
# Hyperparameter Search #
#########################

def hyperparameter_search_word_selector_models() -> None:
    from word_selector_models.models import LSTMSentimentConcatenationPredictor
    from word_selector_models.abstract_predictor import FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME
    
    number_of_epochs = 99999
    train_portion, validation_portion = (0.75, 0.25)
    
    batch_size_choices = [1, 32, 256, 512, 1024]
    max_vocab_size_choices = [10_000, 25_000]
    pre_trained_embedding_specification_choices = ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']

    sentiment_embedding_size_choices = [256, 512, 1024]
    encoding_hidden_size_choices = [256, 512, 1024]
    number_of_encoding_layers_choices = [2,4]
    dropout_probability_choices = [0.5]

    loss_function_spec_choices = ['BCELoss', 'soft_jaccard_loss']
    
    hyparameter_list_choices = list(itertools.product(batch_size_choices,
                                                      max_vocab_size_choices,
                                                      pre_trained_embedding_specification_choices,
                                                      loss_function_spec_choices,
                                                      sentiment_embedding_size_choices,
                                                      encoding_hidden_size_choices,
                                                      number_of_encoding_layers_choices,
                                                      dropout_probability_choices))
    random.seed()
    random.shuffle(hyparameter_list_choices)
    for (batch_size, max_vocab_size, pre_trained_embedding_specification, loss_function_spec, sentiment_embedding_size, encoding_hidden_size, number_of_encoding_layers, dropout_probability) in hyparameter_list_choices:
        output_directory = f'./results/epochs_{number_of_epochs}_batch_size_{batch_size}_train_frac_{train_portion}_validation_frac_{validation_portion}_max_vocab_{max_vocab_size}_embed_spec_{pre_trained_embedding_specification}_loss_func_{loss_function_spec}_sentiment_embed_size_{sentiment_embedding_size}_encoding_size_{encoding_hidden_size}_num_encoding_layers_{number_of_encoding_layers}_dropout_{dropout_probability}/'
        final_output_results_file = os.path.join(output_directory, FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME)
        if os.path.isfile(final_output_results_file):
            print(f'Skipping result generation for {final_output_results_file}.')
        else:
            with safe_cuda_memory():
                predictor = LSTMSentimentConcatenationPredictor(output_directory, number_of_epochs, batch_size, train_portion, validation_portion, max_vocab_size, pre_trained_embedding_specification,
                                                                loss_function_spec=loss_function_spec,
                                                                sentiment_embedding_size=sentiment_embedding_size, 
                                                                encoding_hidden_size=encoding_hidden_size,
                                                                number_of_encoding_layers=number_of_encoding_layers,
                                                                dropout_probability=dropout_probability)
                predictor.train()
    return

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 9999))
    parser.add_argument('-preprocess-data', action='store_true', help=f'Preprocess the raw data. Results stored in {PREPROCESSED_TRAINING_DATA_JSON_FILE}.')
    parser.add_argument('-train-word-selector-model', action='store_true', help=f'Trains & evaluates our word selector model on our preprocessed dataset. Results are saved to {WORD_SELECTOR_RESULTS_DIRECTORY}.')
    parser.add_argument('-hyperparameter-search-word-selector-models', action='store_true', help=f'Exhaustively performs hyperparameter random search. Details of the best performance are tracked in {WORD_SELECTOR_BEST_MODEL_SCORE_JSON_FILE_LOCATION}.')
    args = parser.parse_args()
    number_of_args_specified = sum(map(int,vars(args).values()))
    if number_of_args_specified == 0:
        parser.print_help()
    elif number_of_args_specified > 1:
        print('Please specify exactly one action.')
    elif args.preprocess_data:
        import preprocess_data
        preprocess_data.preprocess_data()
    elif args.train_word_selector_model:
        from word_selector_models.models import train_model
        train_model()
    elif args.hyperparameter_search_word_selector_models:
        hyperparameter_search_word_selector_models()
    else:
        raise Exception('Unexpected args received.')
    return

if __name__ == '__main__':
    main()
