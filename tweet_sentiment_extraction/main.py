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

def LSTMSentimentConcatenationPredictor_generator() -> Generator:
    from word_selector_models.models import LSTMSentimentConcatenationPredictor
    from word_selector_models.abstract_predictor import FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME
    
    number_of_epochs = 99999
    train_portion, validation_portion = (0.75, 0.25)
    
    batch_size_choices = [1, 32, 256, 512, 1024]
    max_vocab_size_choices = [10_000, 25_000]
    pre_trained_embedding_specification_choices = ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']

    sentiment_size_choices = [256, 512, 1024]
    encoding_hidden_size_choices = [256, 512, 1024]
    number_of_encoding_layers_choices = [2,4]
    dropout_probability_choices = [0.5]

    loss_function_spec_choices = ['BCELoss', 'soft_jaccard_loss']
    
    hyparameter_list_choices = list(itertools.product(batch_size_choices,
                                                      max_vocab_size_choices,
                                                      pre_trained_embedding_specification_choices,
                                                      loss_function_spec_choices,
                                                      sentiment_size_choices,
                                                      encoding_hidden_size_choices,
                                                      number_of_encoding_layers_choices,
                                                      dropout_probability_choices))
    random.seed()
    random.shuffle(hyparameter_list_choices)
    for (batch_size, max_vocab_size, pre_trained_embedding_specification, loss_function_spec, sentiment_size, encoding_hidden_size, number_of_encoding_layers, dropout_probability) in hyparameter_list_choices:
        output_directory = f'./results/LSTMSentimentConcatenationPredictor_batch_{batch_size}_train_frac_{train_portion}_valid_frac_{validation_portion}_vocab_{max_vocab_size}_embed_spec_{pre_trained_embedding_specification}_{loss_function_spec}_sentiment_size_{sentiment_size}_encod_size_{encoding_hidden_size}_encod_layer_{number_of_encoding_layers}_dropout_{dropout_probability}'
        final_output_results_file = os.path.join(output_directory, FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME)
        if os.path.isfile(final_output_results_file):
            print(f'Skipping result generation for {final_output_results_file}.')
        else:
            predictor = LSTMSentimentConcatenationPredictor(output_directory, number_of_epochs, batch_size, train_portion, validation_portion, max_vocab_size, pre_trained_embedding_specification,
                                                            loss_function_spec=loss_function_spec,
                                                            sentiment_size=sentiment_size, 
                                                            encoding_hidden_size=encoding_hidden_size,
                                                            number_of_encoding_layers=number_of_encoding_layers,
                                                            dropout_probability=dropout_probability)
            yield predictor

def LSTMScaledDotProductAttentionPredictor_generator() -> Generator:
    from word_selector_models.models import LSTMScaledDotProductAttentionPredictor
    from word_selector_models.abstract_predictor import FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME
    
    number_of_epochs = 99999
    train_portion, validation_portion = (0.75, 0.25)
    
    batch_size_choices = [1, 32, 256]
    max_vocab_size_choices = [10_000, 25_000]
    pre_trained_embedding_specification_choices = ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']

    sentiment_size_choices = [32, 64, 128, 256]
    encoding_hidden_size_choices = [32, 64, 128, 256]
    number_of_encoding_layers_choices = [1, 2, 4]
    dropout_probability_choices = [0.5]

    loss_function_spec_choices = ['BCELoss', 'soft_jaccard_loss']
    
    hyparameter_list_choices = list(itertools.product(batch_size_choices,
                                                      max_vocab_size_choices,
                                                      pre_trained_embedding_specification_choices,
                                                      loss_function_spec_choices,
                                                      sentiment_size_choices,
                                                      encoding_hidden_size_choices,
                                                      number_of_encoding_layers_choices,
                                                      dropout_probability_choices))
    random.seed()
    random.shuffle(hyparameter_list_choices)
    for (batch_size, max_vocab_size, pre_trained_embedding_specification, loss_function_spec, sentiment_size, encoding_hidden_size, number_of_encoding_layers, dropout_probability) in hyparameter_list_choices:
        output_directory = f'./results/LSTMScaledDotProductAttentionPredictor_batch_{batch_size}_train_frac_{train_portion}_valid_frac_{validation_portion}_vocab_{max_vocab_size}_embed_spec_{pre_trained_embedding_specification}_{loss_function_spec}_sentiment_size_{sentiment_size}_encod_size_{encoding_hidden_size}_encod_layer_{number_of_encoding_layers}_dropout_{dropout_probability}'
        final_output_results_file = os.path.join(output_directory, FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME)
        if os.path.isfile(final_output_results_file):
            print(f'Skipping result generation for {final_output_results_file}.')
        else:
            predictor = LSTMScaledDotProductAttentionPredictor(output_directory, number_of_epochs, batch_size, train_portion, validation_portion, max_vocab_size, pre_trained_embedding_specification,
                                                               loss_function_spec=loss_function_spec,
                                                               sentiment_size=sentiment_size, 
                                                               encoding_hidden_size=encoding_hidden_size,
                                                               number_of_encoding_layers=number_of_encoding_layers,
                                                               dropout_probability=dropout_probability)
            yield predictor


def LSTMScaledDotProductAttentionPredictor_generator() -> Generator:
    from word_selector_models.models import LSTMScaledDotProductAttentionPredictor
    from word_selector_models.abstract_predictor import FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME
    
    number_of_epochs = 99999
    train_portion, validation_portion = (0.75, 0.25)
    
    batch_size_choices = [1, 32, 256]
    max_vocab_size_choices = [10_000, 25_000]
    pre_trained_embedding_specification_choices = ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']

    sentiment_size_choices = [32, 64, 128, 256]
    encoding_hidden_size_choices = [32, 64, 128, 256]
    number_of_encoding_layers_choices = [1, 2, 4]
    dropout_probability_choices = [0.5]

    loss_function_spec_choices = ['BCELoss', 'soft_jaccard_loss']
    
    hyparameter_list_choices = list(itertools.product(batch_size_choices,
                                                      max_vocab_size_choices,
                                                      pre_trained_embedding_specification_choices,
                                                      loss_function_spec_choices,
                                                      sentiment_size_choices,
                                                      encoding_hidden_size_choices,
                                                      number_of_encoding_layers_choices,
                                                      dropout_probability_choices))
    random.seed()
    random.shuffle(hyparameter_list_choices)
    for (batch_size, max_vocab_size, pre_trained_embedding_specification, loss_function_spec, sentiment_size, encoding_hidden_size, number_of_encoding_layers, dropout_probability) in hyparameter_list_choices:
        output_directory = f'./results/LSTMScaledDotProductAttentionPredictor_batch_{batch_size}_train_frac_{train_portion}_valid_frac_{validation_portion}_vocab_{max_vocab_size}_embed_spec_{pre_trained_embedding_specification}_{loss_function_spec}_sentiment_size_{sentiment_size}_encod_size_{encoding_hidden_size}_encod_layer_{number_of_encoding_layers}_dropout_{dropout_probability}'
        final_output_results_file = os.path.join(output_directory, FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME)
        if os.path.isfile(final_output_results_file):
            print(f'Skipping result generation for {final_output_results_file}.')
        else:
            predictor = LSTMScaledDotProductAttentionPredictor(output_directory, number_of_epochs, batch_size, train_portion, validation_portion, max_vocab_size, pre_trained_embedding_specification,
                                                               loss_function_spec=loss_function_spec,
                                                               sentiment_size=sentiment_size, 
                                                               encoding_hidden_size=encoding_hidden_size,
                                                               number_of_encoding_layers=number_of_encoding_layers,
                                                               dropout_probability=dropout_probability)
            yield predictor

def NaiveDensePredictor_generator() -> Generator:
    from word_selector_models.models import NaiveDensePredictor
    from word_selector_models.abstract_predictor import FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME
    
    number_of_epochs = 99999
    train_portion, validation_portion = (0.75, 0.25)
    
    batch_size_choices = [1]
    max_vocab_size_choices = [10_000, 25_000]
    pre_trained_embedding_specification_choices = ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']

    sentiment_size_choices = [32, 64, 128, 256]
    dense_sequence_lengths_choices = [
        [110, 1],
        [110, 64, 1],
        [110, 64, 32, 1],
        [110, 64, 32, 16, 1],
        [110, 64, 32, 16, 8, 1],
        [110, 64, 32, 16, 8, 4, 1],
        [110, 32, 1],
        [110, 32, 8, 1],
    ]
    dropout_probability_choices = [0.5]

    loss_function_spec_choices = ['BCELoss', 'soft_jaccard_loss']
    
    hyparameter_list_choices = list(itertools.product(batch_size_choices,
                                                      max_vocab_size_choices,
                                                      pre_trained_embedding_specification_choices,
                                                      loss_function_spec_choices,
                                                      sentiment_size_choices,
                                                      dense_sequence_lengths_choices,
                                                      dropout_probability_choices))
    random.seed()
    random.shuffle(hyparameter_list_choices)
    for (batch_size, max_vocab_size, pre_trained_embedding_specification, loss_function_spec, sentiment_size, dense_sequence_lengths, dropout_probability) in hyparameter_list_choices:
        output_directory = f'./results/NaiveDensePredictor_batch_{batch_size}_train_frac_{train_portion}_valid_frac_{validation_portion}_vocab_{max_vocab_size}_embed_spec_{pre_trained_embedding_specification}_{loss_function_spec}_sentiment_size_{sentiment_size}_dense_lens_{str(dense_sequence_lengths).replace(" ","")}_dropout_{dropout_probability}'
        final_output_results_file = os.path.join(output_directory, FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME)
        if os.path.isfile(final_output_results_file):
            print(f'Skipping result generation for {final_output_results_file}.')
        else:
            predictor = NaiveDensePredictor(output_directory, number_of_epochs, batch_size, train_portion, validation_portion, max_vocab_size, pre_trained_embedding_specification,
                                            loss_function_spec=loss_function_spec,
                                            sentiment_size=sentiment_size, 
                                            dense_sequence_lengths=dense_sequence_lengths,
                                            dropout_probability=dropout_probability)
            yield predictor

def hyperparameter_search(predictors: Iterable) -> None:
    for predictor in predictors:
        with safe_cuda_memory():
            predictor.train()
    return

##########
# Driver #
##########

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 9999))
    parser.add_argument('-preprocess-data', action='store_true', help=f'Preprocess the raw data. Results stored in {PREPROCESSED_TRAINING_DATA_JSON_FILE}.')
    parser.add_argument('-train-word-selector-model', action='store_true', help=f'Trains & evaluates our word selector model on our preprocessed dataset. Results are saved to {WORD_SELECTOR_RESULTS_DIRECTORY}.')
    parser.add_argument('-hyperparameter-search-word-selector-models', action='store_true',
                        help=f'Exhaustively performs hyperparameter random search. Details of the best performance are tracked in {WORD_SELECTOR_BEST_MODEL_SCORE_JSON_FILE_LOCATION}.')
    parser.add_argument('-hyperparameter-search-lstm-sentiment-concatenation-predictor', action='store_true', help=f'Exhaustively performs hyperparameter random search using only the LSTMSentimentConcatenationPredictor model.')
    parser.add_argument('-hyperparameter-search-lstm-scaled-dot-product-attention-predictor', action='store_true',
                        help=f'Exhaustively performs hyperparameter random search using only the LSTMScaledDotProductAttentionPredictor model.')
    parser.add_argument('-hyperparameter-search-naive-dense-predictor', action='store_true', help=f'Exhaustively performs hyperparameter random search using only the NaiveDensePredictor model.')
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
        predictors = roundrobin(
            LSTMSentimentConcatenationPredictor_generator(),
            LSTMScaledDotProductAttentionPredictor_generator(),
            NaiveDensePredictor_generator(),
        )
        hyperparameter_search(predictors)
    elif args.hyperparameter_search_lstm_sentiment_concatenation_predictor:
        predictors = LSTMSentimentConcatenationPredictor_generator()
        hyperparameter_search(predictors)
    elif args.hyperparameter_search_lstm_scaled_dot_product_attention_predictor:
        predictors = LSTMScaledDotProductAttentionPredictor_generator()
        hyperparameter_search(predictors)
    elif args.hyperparameter_search_naive_dense_predictor:
        predictors = NaiveDensePredictor_generator()
        hyperparameter_search(predictors)
    else:
        raise Exception('Unexpected args received.')
    return

if __name__ == '__main__':
    main()
