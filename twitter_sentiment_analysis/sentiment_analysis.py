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
import random
import csv
from pssh.clients import ParallelSSHClient
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
        for row_dict in raw_data_csv_reader:
            updated_row_dict = row_dict
            sentiment_text = row_dict['SentimentText']
            normalized_sentiment_text = ' '.join(normalized_words_from_text_string(sentiment_text))
            updated_row_dict['SentimentText'] = normalized_sentiment_text
            updated_row_dicts.append(updated_row_dict)
            with open(normalized_data_csv_location, mode='w') as normalized_data_csv_file:
                assert len(updated_row_dicts) > 0, "Data is empty at {raw_data_csv_location}.".format(raw_data_csv_location=raw_data_csv_location)
                arbitrary_updated_row_dict = updated_row_dicts[0]
                fieldnames = arbitrary_updated_row_dict.keys()
                writer = csv.DictWriter(normalized_data_csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for updated_row_dict in updated_row_dicts:
                    writer.writerow(updated_row_dict)
    return None

def preprocess_data() -> None:
    import text_classifier
    raw_training_data_csv_location = text_classifier.RAW_TRAINING_DATA_LOCATION
    normalized_training_data_csv_location = text_classifier.NORMALIZED_TRAINING_DATA_LOCATION
    logging_print("Storing processed training data from {raw_training_data_csv_location} to {normalized_training_data_csv_location}.".format(
        raw_training_data_csv_location=raw_training_data_csv_location,
        normalized_training_data_csv_location=normalized_training_data_csv_location))
    preprocess_data_file(raw_training_data_csv_location, normalized_training_data_csv_location)
    raw_testing_data_csv_location = text_classifier.RAW_TEST_DATA_LOCATION
    normalized_testing_data_csv_location = text_classifier.NORMALIZED_TEST_DATA_LOCATION
    logging_print("Storing processed test data from {raw_testing_data_csv_location} to {normalized_testing_data_csv_location}.".format(
        raw_testing_data_csv_location=raw_testing_data_csv_location,
        normalized_testing_data_csv_location=normalized_testing_data_csv_location))
    preprocess_data_file(raw_testing_data_csv_location, normalized_testing_data_csv_location)
    import importlib
    importlib.reload(text_classifier)
    return None

########################################
# Hyperparameter Grid Search Utilities #
########################################

def determine_training_commands_for_distributed_grid_search(result_directory: str) -> List[str]:
    training_commands = []
    number_of_iterations_between_checkpoints = 20000
    number_of_epochs = 300 #8 # @todo fix this
    options_for_batch_size = [1]
    options_for_learning_rate = [1e-1, 1e-2, 1e-3]
    options_for_attenion_regularization_penalty_multiplicative_factor = [1e-4, 1e-2, 1]
    options_for_embedding_hidden_size = [200]
    options_for_lstm_dropout_prob = [0.2]
    options_for_number_of_attention_heads = [1, 2]
    options_for_attention_hidden_size = [32, 16]
    for batch_size in options_for_batch_size:
        for learning_rate in options_for_learning_rate:
            for attenion_regularization_penalty_multiplicative_factor in options_for_attenion_regularization_penalty_multiplicative_factor:
                for embedding_hidden_size in options_for_embedding_hidden_size:
                    for lstm_dropout_prob in options_for_lstm_dropout_prob:
                        for number_of_attention_heads in options_for_number_of_attention_heads:
                            for attention_hidden_size in options_for_attention_hidden_size:
                                checkpoint_directory = os.path.expanduser("{result_directory}/batch_size_{batch_size}__learning_rate_{learning_rate}__attention_hidden_size_{attention_hidden_size}__attenion_regularization_penalty_multiplicative_factor_{attenion_regularization_penalty_multiplicative_factor}__embedding_hidden_size_{embedding_hidden_size}__lstm_dropout_prob_{lstm_dropout_prob}__number_of_attention_heads_{number_of_attention_heads}/".format(
                                    result_directory=result_directory,
                                    batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    attention_hidden_size=attention_hidden_size,
                                    attenion_regularization_penalty_multiplicative_factor=attenion_regularization_penalty_multiplicative_factor,
                                    embedding_hidden_size=embedding_hidden_size,
                                    lstm_dropout_prob=lstm_dropout_prob,
                                    number_of_attention_heads=number_of_attention_heads,
                                ))
                                training_command = os.path.expanduser("~/code/one_off_code/twitter_sentiment_analysis/sentiment_analysis.py -train-sentiment-analyzer -number-of-epochs {number_of_epochs} -batch-size {batch_size} -learning-rate {learning_rate} -attenion-regularization-penalty-multiplicative-factor {attenion_regularization_penalty_multiplicative_factor} -embedding-hidden-size {embedding_hidden_size} -lstm-dropout-prob {lstm_dropout_prob} -number-of-attention-heads {number_of_attention_heads} -attention-hidden-size {attention_hidden_size} -number-of-iterations-between-checkpoints {number_of_iterations_between_checkpoints} -checkpoint-directory {checkpoint_directory}".format(
                                    number_of_epochs=number_of_epochs,
                                    batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    attenion_regularization_penalty_multiplicative_factor=attenion_regularization_penalty_multiplicative_factor,
                                    embedding_hidden_size=embedding_hidden_size,
                                    lstm_dropout_prob=lstm_dropout_prob,
                                    number_of_attention_heads=number_of_attention_heads,
                                    attention_hidden_size=attention_hidden_size,
                                    number_of_iterations_between_checkpoints=number_of_iterations_between_checkpoints,
                                    checkpoint_directory=checkpoint_directory,
                                ))
                                training_commands.append(training_command)
    return training_commands

HOST_NAMES_FOR_DISTRIBUTED_GRID_SEARCH = [
    'anscombe',
    'aquinas',
    'arendt',
    # 'aristotle',
    'aurelius',
    'bentham',
    'berkeley',
    'carnap',
    'chisholm',
    'davidson',
    'descartes',
    'emerson',
    'foot',
    'frege',
    'gettier',
    'hegel',
    'heidegger',
    'hobbes',
    'hume',
    'james',
    'kant',
    'kuhn',
    'leibniz',
    'locke',
    'mill',
    'montesquieu',
    'nietzsche',
    'parmenides',
    'pierce',
    'plato',
    'popper',
    'putnam',
    'quine',
    'rawls',
    'rousseau',
    'russell',
    'seneca',
    'socrates',
    'sperber',
    'spinoza',
    'thoreau',
    'wilkson',
    'wittgenstein',
    'zeno',
]

def perform_distributed_hyperparameter_grid_search(result_directory: str) -> None:
    # @todo verify that this works to some extent
    training_commands = determine_training_commands_for_distributed_grid_search(result_directory)
    random.shuffle(training_commands)
    if True: # @todo remove this
        '''
        num_jobs = 2
        training_commands = training_commands[:num_jobs]
        global HOST_NAMES_FOR_DISTRIBUTED_GRID_SEARCH
        HOST_NAMES_FOR_DISTRIBUTED_GRID_SEARCH = HOST_NAMES_FOR_DISTRIBUTED_GRID_SEARCH[:num_jobs]
        #'''
    number_of_training_commands = len(training_commands)
    logging_print()
    logging_print("{number_of_training_commands} jobs to run.".format(number_of_training_commands=number_of_training_commands))
    logging_print()
    hosts = HOST_NAMES_FOR_DISTRIBUTED_GRID_SEARCH
    host_to_training_commands_map = {host:[] for host in hosts}
    while len(training_commands)!=0:
        for host in hosts:
            if len(training_commands)!=0:
                training_command = training_commands.pop()
                host_to_training_commands_map[host].append(training_command)
    host_args = []
    for host in hosts:
        training_commands = host_to_training_commands_map[host]
        logging_print("Job commands for {host}: ".format(host=host))
        for training_command in training_commands:
            logging_print("    {training_command}".format(training_command=training_command))
        logging_print()
        whole_command_for_host = ' ; '.join(training_commands) if len(training_commands)>0 else ":"
        host_arg = (whole_command_for_host,)
        host_args.append(host_arg)
    client = ParallelSSHClient(hosts, user="pnguyen", password="fridaywinner")
    output = client.run_command('%s', host_args=host_args)
    for host, host_output in output.items():
        logging_print()
        logging_print("host {host}".format(host=host))
        logging_print("STDOUT")
        for stdout_line in host_output.stdout:
            logging_print(stdout_line)
        logging_print()
        logging_print("STDERR")
        for stderr_line in host_output.stderr:
            logging_print(stderr_line)
        logging_print()
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
        ./sentiment_analysis.py -train-sentiment-analyzer -number-of-epochs 3 -batch-size  1 -learning-rate 1e-2 -attenion-regularization-penalty-multiplicative-factor 0.1 -embedding-hidden-size 200 -lstm-dropout-prob 0.2 -number-of-attention-heads 2 -attention-hidden-size 24 -number-of-iterations-between-checkpoints 1 -checkpoint-directory /tmp/checkpoint_dir
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
        perform_distributed_hyperparameter_grid_search(result_directory) # @todo abstract this out to its own module
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
