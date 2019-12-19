#!/usr/bin/python3 -O

"""

This file contains utilities for performing a distributed hyperparameter grid search for Twitter sentiment analysis.

Owner : paul-tqh-nguyen

Created : 12/19/2019

File Name : distributed_hyperparameter_grid_search.py

File Organization:
* Imports
* Hyperparameter Grid Search Utilities

"""

###########
# Imports #
###########

import os
import random
from typing import List
from pssh.clients import ParallelSSHClient
from misc_utilities import logging_print

########################################
# Hyperparameter Grid Search Utilities #
########################################

def determine_training_commands_for_distributed_grid_search(result_directory: str) -> List[str]:
    training_commands = []
    number_of_iterations_between_checkpoints = 20000
    number_of_epochs = 8
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
