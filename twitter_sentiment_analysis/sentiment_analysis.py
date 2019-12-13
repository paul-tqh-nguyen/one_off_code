#!/usr/bin/python3 -O

"""

This file contains the main driver for a neural network based sentiment analyzer for Twitter data. 

Owner : paul-tqh-nguyen

Created : 10/30/2019

File Name : sentiment_analysis.py

File Organization:
* Imports
* Main Runner

"""

###########
# Imports #
###########

import argparse

###############
# Main Runner #
###############

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-run-tests', action='store_true', help="To run all of the tests.")
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
    parser.add_argument('-checkpoint-directory', help="Sentiment analyzer checkpoint directory for saving intermediate results.")
    parser.add_argument('-loading-directory', help="Sentiment analyzer directory for loading a model.")
    parser.add_argument('-number-of-iterations-between-checkpoints', help="Sentiment analyzer number of iterations between checkpoints.")
    args = parser.parse_args()
    arg_to_value_map = vars(args)
    test_run_requested = arg_to_value_map['run_tests']
    no_args_specified = not any(arg_to_value_map.values())
    if no_args_specified:
        parser.print_help()
    if test_run_requested:
        import string_processing_tests
        string_processing_tests.run_all_tests()
    training_requested = arg_to_value_map['train_sentiment_analyzer']
    if training_requested:
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
            keyword_args['attenion_regularization_penalty_multiplicative_factor'] = int(arg_to_value_map['attenion_regularization_penalty_multiplicative_factor'])
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
        import text_classifier
        keyword_args = dict()
        loading_directory = arg_to_value_map['loading_directory']
        text_classifier.test_classifier(loading_directory=loading_directory)

if __name__ == '__main__':
    main()
