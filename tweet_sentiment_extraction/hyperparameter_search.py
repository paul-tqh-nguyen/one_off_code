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

###########
# Globals #
###########

# WORD_SELECTOR_RESULTS_DIRECTORY = os.path.join('./word_selector_models/', OUTPUT_DIR)
# WORD_SELECTOR_BEST_MODEL_SCORE_JSON_FILE_LOCATION = os.path.join('./word_selector_models/', GLOBAL_BEST_MODEL_SCORE_JSON_FILE_LOCATION)

ROBERTA_RESULTS_DIRECTORY = os.path.join('./roberta_models/', OUTPUT_DIR)
ROBERTA_BEST_MODEL_SCORE_JSON_FILE_LOCATION = os.path.join('./roberta_models/', GLOBAL_BEST_MODEL_SCORE_JSON_FILE_LOCATION)

ELECTRA_RESULTS_DIRECTORY = os.path.join('./electra_models/', OUTPUT_DIR)
ELECTRA_BEST_MODEL_SCORE_JSON_FILE_LOCATION = os.path.join('./electra_models/', GLOBAL_BEST_MODEL_SCORE_JSON_FILE_LOCATION)

#########################
# Hyperparameter Search #
#########################

# def LSTMSentimentConcatenationPredictor_generator() -> Generator:
#     from word_selector_models.models import LSTMSentimentConcatenationPredictor
    
#     number_of_epochs = 99999
#     train_portion, validation_portion = (0.8, 0.2)
    
#     batch_size_choices = [1, 32, 256, 512, 1024]
#     max_vocab_size_choices = [10_000, 25_000]
#     pre_trained_embedding_specification_choices = ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']

#     sentiment_size_choices = [256, 512, 1024]
#     encoding_hidden_size_choices = [256, 512, 1024]
#     number_of_encoding_layers_choices = [2,4]
#     dropout_probability_choices = [0.5]

#     loss_function_spec_choices = ['BCELoss', 'soft_jaccard_loss']
    
#     hyparameter_list_choices = list(itertools.product(batch_size_choices,
#                                                       max_vocab_size_choices,
#                                                       pre_trained_embedding_specification_choices,
#                                                       loss_function_spec_choices,
#                                                       sentiment_size_choices,
#                                                       encoding_hidden_size_choices,
#                                                       number_of_encoding_layers_choices,
#                                                       dropout_probability_choices))
#     random.seed()
#     random.shuffle(hyparameter_list_choices)
#     for (batch_size, max_vocab_size, pre_trained_embedding_specification, loss_function_spec, sentiment_size, encoding_hidden_size, number_of_encoding_layers, dropout_probability) in hyparameter_list_choices:
#         output_directory = f'./results/LSTMSentimentConcatenationPredictor_batch_{batch_size}_train_frac_{train_portion}_valid_frac_{validation_portion}_vocab_{max_vocab_size}_embed_spec_{pre_trained_embedding_specification}_{loss_function_spec}_sentiment_size_{sentiment_size}_encod_size_{encoding_hidden_size}_encod_layer_{number_of_encoding_layers}_dropout_{dropout_probability}'
#         final_output_results_file = os.path.join(output_directory, FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME)
#         if os.path.isfile(final_output_results_file):
#             print(f'Skipping result generation for {final_output_results_file}.')
#         else:
#             predictor = LSTMSentimentConcatenationPredictor(output_directory, number_of_epochs, batch_size, train_portion, validation_portion, max_vocab_size, pre_trained_embedding_specification,
#                                                             loss_function_spec=loss_function_spec,
#                                                             sentiment_size=sentiment_size, 
#                                                             encoding_hidden_size=encoding_hidden_size,
#                                                             number_of_encoding_layers=number_of_encoding_layers,
#                                                             dropout_probability=dropout_probability)
#             yield predictor

# def LSTMScaledDotProductAttentionPredictor_generator() -> Generator:
#     from word_selector_models.models import LSTMScaledDotProductAttentionPredictor
    
#     number_of_epochs = 99999
#     train_portion, validation_portion = (0.8, 0.2)
    
#     batch_size_choices = [1, 32, 256]
#     max_vocab_size_choices = [10_000, 25_000]
#     pre_trained_embedding_specification_choices = ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']

#     sentiment_size_choices = [32, 64, 128, 256]
#     encoding_hidden_size_choices = [32, 64, 128, 256]
#     number_of_encoding_layers_choices = [1, 2, 4]
#     dropout_probability_choices = [0.5]

#     loss_function_spec_choices = ['BCELoss', 'soft_jaccard_loss']
    
#     hyparameter_list_choices = list(itertools.product(batch_size_choices,
#                                                       max_vocab_size_choices,
#                                                       pre_trained_embedding_specification_choices,
#                                                       loss_function_spec_choices,
#                                                       sentiment_size_choices,
#                                                       encoding_hidden_size_choices,
#                                                       number_of_encoding_layers_choices,
#                                                       dropout_probability_choices))
#     random.seed()
#     random.shuffle(hyparameter_list_choices)
#     for (batch_size, max_vocab_size, pre_trained_embedding_specification, loss_function_spec, sentiment_size, encoding_hidden_size, number_of_encoding_layers, dropout_probability) in hyparameter_list_choices:
#         output_directory = f'./results/LSTMScaledDotProductAttentionPredictor_batch_{batch_size}_train_frac_{train_portion}_valid_frac_{validation_portion}_vocab_{max_vocab_size}_embed_spec_{pre_trained_embedding_specification}_{loss_function_spec}_sentiment_size_{sentiment_size}_encod_size_{encoding_hidden_size}_encod_layer_{number_of_encoding_layers}_dropout_{dropout_probability}'
#         final_output_results_file = os.path.join(output_directory, FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME)
#         if os.path.isfile(final_output_results_file):
#             print(f'Skipping result generation for {final_output_results_file}.')
#         else:
#             predictor = LSTMScaledDotProductAttentionPredictor(output_directory, number_of_epochs, batch_size, train_portion, validation_portion, max_vocab_size, pre_trained_embedding_specification,
#                                                                loss_function_spec=loss_function_spec,
#                                                                sentiment_size=sentiment_size, 
#                                                                encoding_hidden_size=encoding_hidden_size,
#                                                                number_of_encoding_layers=number_of_encoding_layers,
#                                                                dropout_probability=dropout_probability)
#             yield predictor


# def LSTMScaledDotProductAttentionPredictor_generator() -> Generator:
#     from word_selector_models.models import LSTMScaledDotProductAttentionPredictor
    
#     number_of_epochs = 99999
#     train_portion, validation_portion = (0.8, 0.2)
    
#     batch_size_choices = [1, 32, 256]
#     max_vocab_size_choices = [10_000, 25_000]
#     pre_trained_embedding_specification_choices = ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']

#     sentiment_size_choices = [32, 64, 128, 256]
#     encoding_hidden_size_choices = [32, 64, 128, 256]
#     number_of_encoding_layers_choices = [1, 2, 4]
#     dropout_probability_choices = [0.5]

#     loss_function_spec_choices = ['BCELoss', 'soft_jaccard_loss']
    
#     hyparameter_list_choices = list(itertools.product(batch_size_choices,
#                                                       max_vocab_size_choices,
#                                                       pre_trained_embedding_specification_choices,
#                                                       loss_function_spec_choices,
#                                                       sentiment_size_choices,
#                                                       encoding_hidden_size_choices,
#                                                       number_of_encoding_layers_choices,
#                                                       dropout_probability_choices))
#     random.seed()
#     random.shuffle(hyparameter_list_choices)
#     for (batch_size, max_vocab_size, pre_trained_embedding_specification, loss_function_spec, sentiment_size, encoding_hidden_size, number_of_encoding_layers, dropout_probability) in hyparameter_list_choices:
#         output_directory = f'./results/LSTMScaledDotProductAttentionPredictor_batch_{batch_size}_train_frac_{train_portion}_valid_frac_{validation_portion}_vocab_{max_vocab_size}_embed_spec_{pre_trained_embedding_specification}_{loss_function_spec}_sentiment_size_{sentiment_size}_encod_size_{encoding_hidden_size}_encod_layer_{number_of_encoding_layers}_dropout_{dropout_probability}'
#         final_output_results_file = os.path.join(output_directory, FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME)
#         if os.path.isfile(final_output_results_file):
#             print(f'Skipping result generation for {final_output_results_file}.')
#         else:
#             predictor = LSTMScaledDotProductAttentionPredictor(output_directory, number_of_epochs, batch_size, train_portion, validation_portion, max_vocab_size, pre_trained_embedding_specification,
#                                                                loss_function_spec=loss_function_spec,
#                                                                sentiment_size=sentiment_size, 
#                                                                encoding_hidden_size=encoding_hidden_size,
#                                                                number_of_encoding_layers=number_of_encoding_layers,
#                                                                dropout_probability=dropout_probability)
#             yield predictor

# def NaiveDensePredictor_generator() -> Generator:
#     from word_selector_models.models import NaiveDensePredictor
    
#     number_of_epochs = 99999
#     train_portion, validation_portion = (0.8, 0.2)
    
#     batch_size_choices = [1]
#     max_vocab_size_choices = [10_000, 25_000]
#     pre_trained_embedding_specification_choices = ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']

#     sentiment_size_choices = [32, 64, 128, 256]
#     dense_sequence_lengths_choices = [
#         [110, 1],
#         [110, 64, 1],
#         [110, 64, 32, 1],
#         [110, 64, 32, 16, 1],
#         [110, 64, 32, 16, 8, 1],
#         [110, 64, 32, 16, 8, 4, 1],
#         [110, 32, 1],
#         [110, 32, 8, 1],
#     ]
#     dropout_probability_choices = [0.5]

#     loss_function_spec_choices = ['BCELoss', 'soft_jaccard_loss']
    
#     hyparameter_list_choices = list(itertools.product(batch_size_choices,
#                                                       max_vocab_size_choices,
#                                                       pre_trained_embedding_specification_choices,
#                                                       loss_function_spec_choices,
#                                                       sentiment_size_choices,
#                                                       dense_sequence_lengths_choices,
#                                                       dropout_probability_choices))
#     random.seed()
#     random.shuffle(hyparameter_list_choices)
#     for (batch_size, max_vocab_size, pre_trained_embedding_specification, loss_function_spec, sentiment_size, dense_sequence_lengths, dropout_probability) in hyparameter_list_choices:
#         output_directory = f'./results/NaiveDensePredictor_batch_{batch_size}_train_frac_{train_portion}_valid_frac_{validation_portion}_vocab_{max_vocab_size}_embed_spec_{pre_trained_embedding_specification}_{loss_function_spec}_sentiment_size_{sentiment_size}_dense_lens_{str(dense_sequence_lengths).replace(" ","")}_dropout_{dropout_probability}'
#         final_output_results_file = os.path.join(output_directory, FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME)
#         if os.path.isfile(final_output_results_file):
#             print(f'Skipping result generation for {final_output_results_file}.')
#         else:
#             predictor = NaiveDensePredictor(output_directory, number_of_epochs, batch_size, train_portion, validation_portion, max_vocab_size, pre_trained_embedding_specification,
#                                             loss_function_spec=loss_function_spec,
#                                             sentiment_size=sentiment_size, 
#                                             dense_sequence_lengths=dense_sequence_lengths,
#                                             dropout_probability=dropout_probability)
#             yield predictor


# def NaiveDensePredictor_generator() -> Generator:
#     from word_selector_models.models import NaiveDensePredictor
    
#     number_of_epochs = 99999
#     train_portion, validation_portion = (0.8, 0.2)
    
#     batch_size_choices = [1]
#     max_vocab_size_choices = [10_000, 25_000]
#     pre_trained_embedding_specification_choices = ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']

#     sentiment_size_choices = [32, 64, 128, 256]
#     dense_sequence_lengths_choices = [
#         [110, 1],
#         [110, 64, 1],
#         [110, 64, 32, 1],
#         [110, 64, 32, 16, 1],
#         [110, 64, 32, 16, 8, 1],
#         [110, 64, 32, 16, 8, 4, 1],
#         [110, 32, 1],
#         [110, 32, 8, 1],
#     ]
#     dropout_probability_choices = [0.5]

#     loss_function_spec_choices = ['BCELoss', 'soft_jaccard_loss']
    
#     hyparameter_list_choices = list(itertools.product(batch_size_choices,
#                                                       max_vocab_size_choices,
#                                                       pre_trained_embedding_specification_choices,
#                                                       loss_function_spec_choices,
#                                                       sentiment_size_choices,
#                                                       dense_sequence_lengths_choices,
#                                                       dropout_probability_choices))
#     random.seed()
#     random.shuffle(hyparameter_list_choices)
#     for (batch_size, max_vocab_size, pre_trained_embedding_specification, loss_function_spec, sentiment_size, dense_sequence_lengths, dropout_probability) in hyparameter_list_choices:
#         output_directory = f'./results/NaiveDensePredictor_batch_{batch_size}_train_frac_{train_portion}_valid_frac_{validation_portion}_vocab_{max_vocab_size}_embed_spec_{pre_trained_embedding_specification}_{loss_function_spec}_sentiment_size_{sentiment_size}_dense_lens_{str(dense_sequence_lengths).replace(" ","")}_dropout_{dropout_probability}'
#         final_output_results_file = os.path.join(output_directory, FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME)
#         if os.path.isfile(final_output_results_file):
#             print(f'Skipping result generation for {final_output_results_file}.')
#         else:
#             predictor = NaiveDensePredictor(output_directory, number_of_epochs, batch_size, train_portion, validation_portion, max_vocab_size, pre_trained_embedding_specification,
#                                             loss_function_spec=loss_function_spec,
#                                             sentiment_size=sentiment_size, 
#                                             dense_sequence_lengths=dense_sequence_lengths,
#                                             dropout_probability=dropout_probability)
#             yield predictor

            
# def NaiveDensePredictor_generator() -> Generator:
#     from word_selector_models.models import NaiveDensePredictor
    
#     number_of_epochs = 99999
#     train_portion, validation_portion = (0.8, 0.2)
    
#     batch_size_choices = [1]
#     max_vocab_size_choices = [10_000, 25_000]
#     pre_trained_embedding_specification_choices = ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']

#     sentiment_size_choices = [32, 64, 128, 256]
#     dense_sequence_lengths_choices = [
#         [110, 1],
#         [110, 64, 1],
#         [110, 64, 32, 1],
#         [110, 64, 32, 16, 1],
#         [110, 64, 32, 16, 8, 1],
#         [110, 64, 32, 16, 8, 4, 1],
#         [110, 32, 1],
#         [110, 32, 8, 1],
#     ]
#     dropout_probability_choices = [0.5]

#     loss_function_spec_choices = ['BCELoss', 'soft_jaccard_loss']
    
#     hyparameter_list_choices = list(itertools.product(batch_size_choices,
#                                                       max_vocab_size_choices,
#                                                       pre_trained_embedding_specification_choices,
#                                                       loss_function_spec_choices,
#                                                       sentiment_size_choices,
#                                                       dense_sequence_lengths_choices,
#                                                       dropout_probability_choices))
#     random.seed()
#     random.shuffle(hyparameter_list_choices)
#     for (batch_size, max_vocab_size, pre_trained_embedding_specification, loss_function_spec, sentiment_size, dense_sequence_lengths, dropout_probability) in hyparameter_list_choices:
#         output_directory = f'./results/NaiveDensePredictor_batch_{batch_size}_train_frac_{train_portion}_valid_frac_{validation_portion}_vocab_{max_vocab_size}_embed_spec_{pre_trained_embedding_specification}_{loss_function_spec}_sentiment_size_{sentiment_size}_dense_lens_{str(dense_sequence_lengths).replace(" ","")}_dropout_{dropout_probability}'
#         final_output_results_file = os.path.join(output_directory, FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME)
#         if os.path.isfile(final_output_results_file):
#             print(f'Skipping result generation for {final_output_results_file}.')
#         else:
#             predictor = NaiveDensePredictor(output_directory, number_of_epochs, batch_size, train_portion, validation_portion, max_vocab_size, pre_trained_embedding_specification,
#                                             loss_function_spec=loss_function_spec,
#                                             sentiment_size=sentiment_size, 
#                                             dense_sequence_lengths=dense_sequence_lengths,
#                                             dropout_probability=dropout_probability)
#             yield predictor

def RoBERTaPredictor_generator() -> Generator:
    from roberta_models.models import RoBERTaPredictor
    
    number_of_epochs = 99999
    number_of_folds = 5
    
    batch_size_choices = [32] # [1, 32, 64]
    gradient_clipping_threshold_choices = [10, 20, 30, 50]
    initial_learning_rate_choices = [1e-5, 3e-5, 5e-5, 1e-6, 1e-7]
    
    hyparameter_list_choices = list(itertools.product(batch_size_choices,
                                                      gradient_clipping_threshold_choices,
                                                      initial_learning_rate_choices))
    random.seed()
    random.shuffle(hyparameter_list_choices)
    for (batch_size, gradient_clipping_threshold, initial_learning_rate) in hyparameter_list_choices:
        output_directory = f'./results/RoBERTaPredictor_batch_{batch_size}_folds_{number_of_folds}_gradient_clip_{gradient_clipping_threshold}_learning_rate_{initial_learning_rate}'
        final_output_results_file = os.path.join(output_directory, 'best_model_for_fold_0', FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME)
        if os.path.isfile(final_output_results_file):
            print(f'Skipping result generation for {final_output_results_file}.')
        else:
            predictor = RoBERTaPredictor(output_directory, number_of_epochs, batch_size, number_of_folds, gradient_clipping_threshold, initial_learning_rate)
            yield predictor

def ElectraPredictor_generator() -> Generator:
    from electra_models.models import ElectraPredictor
    
    number_of_epochs = 99999
    number_of_folds = 5

    pretrained_model_spec_choices = {
        'google/electra-small-generator',
        'google/electra-base-generator',
        'google/electra-large-generator',
        'google/electra-small-discriminator',
        'google/electra-base-discriminator',
        'google/electra-large-discriminator',
    }
    
    batch_size_choices = [1, 32, 64]
    gradient_clipping_threshold_choices = [10, 20, 30, 50]
    initial_learning_rate_choices = [5e-5, 3e-5, 1e-5, 1e-6, 1e-7]
    
    hyparameter_list_choices = list(itertools.product(pretrained_model_spec_choices,
                                                      batch_size_choices,
                                                      gradient_clipping_threshold_choices,
                                                      initial_learning_rate_choices))
    random.seed()
    random.shuffle(hyparameter_list_choices)
    for (pretrained_model_spec, batch_size, gradient_clipping_threshold, initial_learning_rate) in hyparameter_list_choices:
        output_directory = f'./results/ElectraPredictor_batch_{batch_size}_folds_{number_of_folds}_gradient_clip_{gradient_clipping_threshold}_learning_rate_{initial_learning_rate}'
        final_output_results_file = os.path.join(output_directory, 'best_model_for_fold_0', FINAL_MODEL_SCORE_JSON_FILE_BASE_NAME)
        if os.path.isfile(final_output_results_file):
            print(f'Skipping result generation for {final_output_results_file}.')
        else:
            import electra_models.models
            electra_models.models.TRANSFORMERS_MODEL_SPEC = pretrained_model_spec # @todo this is leaky ; training depends on global state set here that can change prior to training.
            predictor = ElectraPredictor(output_directory, number_of_epochs, batch_size, number_of_folds, gradient_clipping_threshold, initial_learning_rate)
            yield predictor

def hyperparameter_search(predictors: Iterable) -> None:
    for predictor in predictors:
        with safe_cuda_memory():
            predictor.train()
    return

##########
# Driver #
##########

if __name__ == '__main__':
    print("This module contains utilities for performing hyperparameter search over our models.")
 
