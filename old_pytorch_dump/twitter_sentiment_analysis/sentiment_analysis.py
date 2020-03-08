#!/usr/bin/python3 -O

"""

This file contains the main driver for a neural network based sentiment analyzer for Twitter data. 

Owner : paul-tqh-nguyen

Created : 10/30/2019

File Name : sentiment_analysis.py

File Organization:
* Imports
* Data Preprocessing
* Hyperparameter Grid Search Utilities
* Main Runner

"""

###########
# Imports #
###########

import os
import sys
import argparse
import csv
import tqdm
from typing import List
from misc_utilities import logging_print

######################
# Data Preprocessing #
######################

def preprocess_data_file(raw_data_csv_location: str, normalized_data_csv_location: str) -> None:
    from string_processing_utilities import normalized_words_from_text_string
    with open(raw_data_csv_location, encoding='ISO-8859-1') as raw_data_csv_file:
        raw_data_csv_reader = csv.DictReader(raw_data_csv_file, delimiter=',')
        updated_row_dicts = []
        for row_dict in tqdm.tqdm(raw_data_csv_reader):
            updated_row_dict = row_dict
            sentiment_text = row_dict['SentimentText']
            normalized_sentiment_text = ' '.join(normalized_words_from_text_string(sentiment_text))
            # logging_print("Raw Text: {sentiment_text}".format(sentiment_text=sentiment_text))
            # logging_print("Normalized Text: {normalized_sentiment_text}".format(normalized_sentiment_text=normalized_sentiment_text))
            updated_row_dict['SentimentText'] = normalized_sentiment_text
            updated_row_dicts.append(updated_row_dict)
        with open(normalized_data_csv_location, mode='w') as normalized_data_csv_file:
            assert len(updated_row_dicts) > 0, "Data at {raw_data_csv_location} is empty.".format(raw_data_csv_location=raw_data_csv_location)
            arbitrary_updated_row_dict = updated_row_dicts[0]
            fieldnames = arbitrary_updated_row_dict.keys()
            writer = csv.DictWriter(normalized_data_csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for updated_row_dict in tqdm.tqdm(updated_row_dicts):
                writer.writerow(updated_row_dict)
    return None

def preprocess_data() -> None:
    import text_classifier
    raw_training_data_csv_location = text_classifier.RAW_TRAINING_DATA_LOCATION
    normalized_training_data_csv_location = text_classifier.NORMALIZED_TRAINING_DATA_LOCATION
    logging_print("Storing processed training data from {raw_training_data_csv_location} to {normalized_training_data_csv_location}".format(
        raw_training_data_csv_location=raw_training_data_csv_location,
        normalized_training_data_csv_location=normalized_training_data_csv_location))
    preprocess_data_file(raw_training_data_csv_location, normalized_training_data_csv_location)
    raw_testing_data_csv_location = text_classifier.RAW_TEST_DATA_LOCATION
    normalized_testing_data_csv_location = text_classifier.NORMALIZED_TEST_DATA_LOCATION
    logging_print("Storing processed test data from {raw_testing_data_csv_location} to {normalized_testing_data_csv_location}".format(
        raw_testing_data_csv_location=raw_testing_data_csv_location,
        normalized_testing_data_csv_location=normalized_testing_data_csv_location))
    preprocess_data_file(raw_testing_data_csv_location, normalized_testing_data_csv_location)
    import importlib
    importlib.reload(text_classifier)
    return None

###############
# Main Runner #
###############

def string_is_float(input_string: str) -> bool:
    string_is_float = None
    try:
        string_is_float = True
    except ValueError:
        string_is_float = False
    assert string_is_float is not None
    return string_is_float

def possibly_print_complaint_strings_and_exit(complaint_strings: List[str]) -> None:
    if len(complaint_strings) != 0:
        logging_print("We encountered the following issues:")
        for complaint_string in complaint_strings:
            logging_print("    {complaint_string}".format(complaint_string=complaint_string))
        sys.exit(1)
    return None

def validate_cli_args_for_hyperparameter_grid_search(arg_to_value_map: dict) -> None:
    complaint_strings = []
    output_directory = arg_to_value_map['perform_hyperparameter_grid_search_in_directory']
    if output_directory is None:
        complaint_strings.append("No output directory specified.")
    else:
        assert isinstance(output_directory, str)
    possibly_print_complaint_strings_and_exit(complaint_strings)
    return None

def validate_cli_args_for_training(arg_to_value_map: dict) -> None:
    complaint_strings = []
    loading_directory = arg_to_value_map['loading_directory']
    checkpoint_directory = arg_to_value_map['checkpoint_directory']
    number_of_epochs = arg_to_value_map['number_of_epochs']
    number_of_attention_heads = arg_to_value_map['number_of_attention_heads']
    attention_hidden_size = arg_to_value_map['attention_hidden_size']
    learning_rate = arg_to_value_map['learning_rate']
    embedding_hidden_size = arg_to_value_map['embedding_hidden_size']
    attenion_regularization_penalty_multiplicative_factor = arg_to_value_map['attenion_regularization_penalty_multiplicative_factor']
    lstm_dropout_prob = arg_to_value_map['lstm_dropout_prob']
    batch_size = arg_to_value_map['batch_size']
    number_of_iterations_between_checkpoints = arg_to_value_map['number_of_iterations_between_checkpoints']
    if loading_directory is not None:
        assert isinstance(loading_directory, str)
        if not os.path.exists(loading_directory):
            complaint_strings.append("Loading directory does not exist.")
    if checkpoint_directory is not None:
        assert isinstance(checkpoint_directory, str)
        pass
    if number_of_epochs is not None:
        if not number_of_epochs.isdigit():
            complaint_strings.append("Number of epochs must be an integer.")
    if number_of_attention_heads is not None:
        if not number_of_attention_heads.isdigit():
            complaint_strings.append("Number of attention heads must be an integer.")
    if attention_hidden_size is not None:
        if not attention_hidden_size.isdigit():
            complaint_strings.append("Attention hidden size must be an integer.")
    if learning_rate is not None:
        if not string_is_float(learning_rate):
            complaint_strings.append("Learning rate must be a number.")
    if embedding_hidden_size is not None:
        if not embedding_hidden_size.isdigit():
            complaint_strings.append("Embedding hidden size must be an integer.")
    if attenion_regularization_penalty_multiplicative_factor is not None:
        if not string_is_float(attenion_regularization_penalty_multiplicative_factor):
            complaint_strings.append("Attenion regularization penalty multiplicative factor must be a number.")
    if lstm_dropout_prob is not None:
        if not string_is_float(lstm_dropout_prob):
            complaint_strings.append("LSTM dropout probability must be a number.")
    if batch_size is not None:
        if not batch_size.isdigit():
            complaint_strings.append("Batch size must be an integer.")
    if number_of_iterations_between_checkpoints is not None:
        if not number_of_iterations_between_checkpoints.isdigit():
            complaint_strings.append("Number of iterations between checkpoints must be an integer.")
    possibly_print_complaint_strings_and_exit(complaint_strings)
    return None

def validate_cli_args_for_testing(arg_to_value_map: dict) -> None:
    complaint_strings = []
    loading_directory = arg_to_value_map['loading_directory']
    if loading_directory is None:
        complaint_strings.append("No loading directory specified.")
    else:
        assert isinstance(loading_directory, str)
        if not os.path.exists(loading_directory):
            complaint_strings.append("Loading directory does not exist.")
    possibly_print_complaint_strings_and_exit(complaint_strings)
    return None

def main():
    '''
    Example Use:
        ./sentiment_analysis.py -train-sentiment-analyzer -number-of-epochs 3 -batch-size  1 -learning-rate 1e-2 -attenion-regularization-penalty-multiplicative-factor 0.1 -embedding-hidden-size 200 -lstm-dropout-prob 0.2 -number-of-attention-heads 2 -attention-hidden-size 16 -number-of-iterations-between-checkpoints 20000 -checkpoint-directory /tmp/checkpoint_dir
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-run-tests', action='store_true', help="To run all of the tests.")
    parser.add_argument('-preprocess-data', action='store_true', help="To normalize our training and test data so it doesn't have to happen during training or testing.")
    parser.add_argument('-train-sentiment-analyzer', action='store_true', help="To run the training process for the sentiment analyzer.")
    parser.add_argument('-test-sentiment-analyzer', action='store_true', help="To evaluate the sentiment analyzer's performance against the test set.")
    parser.add_argument('-number-of-epochs', help="Number of epochs to train the sentiment analyzer.")
    parser.add_argument('-batch-size', help="Sentiment analyzer batch size.")
    parser.add_argument('-learning-rate', help="Sentiment analyzer learning_rate.")
    parser.add_argument('-attenion-regularization-penalty-multiplicative-factor', help="Sentiment analyzer attenion regularization penalty multiplicative factor.")
    parser.add_argument('-embedding-hidden-size', help="Sentiment analyzer embedding hidden size.")
    parser.add_argument('-lstm-dropout-prob', help="Sentiment analyzer LSTM dropout probability.")
    parser.add_argument('-number-of-attention-heads', help="Sentiment analyzer number of attention heads.")
    parser.add_argument('-attention-hidden-size', help="Sentiment analyzer attention hidden size.")
    parser.add_argument('-number-of-iterations-between-checkpoints', help="Sentiment analyzer number of iterations between checkpoints.")
    parser.add_argument('-checkpoint-directory', help="Sentiment analyzer checkpoint directory for saving intermediate results.")
    parser.add_argument('-loading-directory', help="Sentiment analyzer directory for loading a model.")
    parser.add_argument('-perform-hyperparameter-grid-search-in-directory', help="Perform grid search and save results in specified directory via distributed search on hard-coded set of machines.")
    args = parser.parse_args()
    arg_to_value_map = vars(args)
    no_args_specified = not any(arg_to_value_map.values())
    if no_args_specified:
        parser.print_help()
    data_preprocessing_requested = arg_to_value_map['preprocess_data']
    if data_preprocessing_requested:
        preprocess_data()
    hyperparameter_grid_search_specified = bool(arg_to_value_map['perform_hyperparameter_grid_search_in_directory'])
    if hyperparameter_grid_search_specified:
        validate_cli_args_for_hyperparameter_grid_search(arg_to_value_map)
        result_directory = arg_to_value_map['perform_hyperparameter_grid_search_in_directory']
        import distributed_hyperparameter_grid_search
        distributed_hyperparameter_grid_search.perform_distributed_hyperparameter_grid_search(result_directory)
    test_run_requested = arg_to_value_map['run_tests']
    if test_run_requested:
        import string_processing_tests
        string_processing_tests.run_all_tests()
    training_requested = arg_to_value_map['train_sentiment_analyzer']
    if training_requested:
        validate_cli_args_for_training(arg_to_value_map)
        keyword_args = dict()
        if arg_to_value_map['loading_directory'] is not None:
            keyword_args['loading_directory'] = arg_to_value_map['loading_directory']
        if arg_to_value_map['checkpoint_directory'] is not None:
            keyword_args['checkpoint_directory'] = arg_to_value_map['checkpoint_directory']
        if arg_to_value_map['number_of_epochs'] is not None:
            keyword_args['number_of_epochs'] = int(arg_to_value_map['number_of_epochs'])
        if arg_to_value_map['number_of_attention_heads'] is not None:
            keyword_args['number_of_attention_heads'] = int(arg_to_value_map['number_of_attention_heads'])
        if arg_to_value_map['attention_hidden_size'] is not None:
            keyword_args['attention_hidden_size'] = int(arg_to_value_map['attention_hidden_size'])
        if arg_to_value_map['learning_rate'] is not None:
            keyword_args['learning_rate'] = float(arg_to_value_map['learning_rate'])
        if arg_to_value_map['embedding_hidden_size'] is not None:
            keyword_args['embedding_hidden_size'] = int(arg_to_value_map['embedding_hidden_size'])
        if arg_to_value_map['attenion_regularization_penalty_multiplicative_factor'] is not None:
            keyword_args['attenion_regularization_penalty_multiplicative_factor'] = float(arg_to_value_map['attenion_regularization_penalty_multiplicative_factor'])
        if arg_to_value_map['lstm_dropout_prob'] is not None:
            keyword_args['lstm_dropout_prob'] = float(arg_to_value_map['lstm_dropout_prob'])
        if arg_to_value_map['batch_size'] is not None:
            keyword_args['batch_size'] = int(arg_to_value_map['batch_size'])
        if arg_to_value_map['number_of_iterations_between_checkpoints'] is not None:
            keyword_args['number_of_iterations_between_checkpoints'] = int(arg_to_value_map['number_of_iterations_between_checkpoints'])
        import text_classifier
        text_classifier.train_classifier(**keyword_args)
    testing_requested = arg_to_value_map['test_sentiment_analyzer']
    if testing_requested:
        validate_cli_args_for_testing(arg_to_value_map)
        import text_classifier
        loading_directory = arg_to_value_map['loading_directory']
        text_classifier.test_classifier(loading_directory=loading_directory)

if __name__ == '__main__':
    main()
