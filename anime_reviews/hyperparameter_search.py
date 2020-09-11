'#!/usr/bin/python3 -OO' # @todo use this

'''
'''

# @todo update doc string

###########
# Imports #
###########

import json
import numpy as np
import pandas as pd
from contextlib import contextmanager
from typing import Generator, Optional

import optuna
import joblib

from misc_utilities import *
from global_values import *
from data_modules import AnimeRatingDataModule
from models import LinearColaborativeFilteringModel
from trainer import train_model, checkpoint_directory_from_hyperparameters

# @todo make sure these imports are used

###########
# Globals #
###########

DB_URL = 'sqlite:///collaborative_filtering.db'
STUDY_NAME = 'collaborative-filtering'

ANALYSIS_OUTPUT_DIR = './result_analysis'

NUMBER_OF_HYPERPARAMETER_SEARCH_TRIALS = 200
NUMBER_OF_BEST_HYPERPARAMETER_RESULTS_TO_DISPLAY = 10

#########################
# Hyperparameter Search #
#########################

@contextmanager
def _training_logging_suppressed() -> Generator:
    logger = logging.root.manager.loggerDict['lightning']
    lightning_original_disability = logger.disabled
    logger.disabled = True
    logger_stream_handler_original_stream = LOGGER_STREAM_HANDLER.stream
    with open(os.devnull, 'w') as dev_null:
        LOGGER_STREAM_HANDLER.setStream(dev_null)
        yield
    logger.disabled = lightning_original_disability
    LOGGER_STREAM_HANDLER.setStream(logger_stream_handler_original_stream)
    return

class HyperParameterSearchObjective:
    def __init__(self, gpu_id_queue: Optional[object]):
        # gpu_id_queue is an mp.managers.AutoProxy[Queue] and an mp.managers.BaseProxy ; can't declare statically since the classs are generated dyanmically
        self.gpu_id_queue = gpu_id_queue

    def __call__(self, trial: optuna.Trial) -> float:
        gpu_id = self.gpu_id_queue.get() if self.gpu_id_queue else DEFAULT_GPU
        learning_rate = trial.suggest_uniform('learning_rate', 1e-5, 1e-1)
        number_of_epochs = int(trial.suggest_int('number_of_epochs', 3, 15))
        batch_size = int(trial.suggest_categorical('batch_size', [2**power for power in range(6,12)]))
        gradient_clip_val = trial.suggest_uniform('gradient_clip_val', 1.0, 1.0)
        embedding_size = int(trial.suggest_int('embedding_size', 100, 500))
        regularization_factor = trial.suggest_uniform('regularization_factor', 1, 100)
        dropout_probability = trial.suggest_uniform('dropout_probability', 0.0, 1.0)
        
        checkpoint_dir = checkpoint_directory_from_hyperparameters(learning_rate, number_of_epochs, batch_size, gradient_clip_val, embedding_size, regularization_factor, dropout_probability)
        print(f'Starting training for {checkpoint_dir} on GPU {gpu_id}.')
        
        try:
            with suppressed_output():
                with warnings_suppressed():
                    best_validation_loss = train_model(
                        learning_rate=learning_rate,
                        number_of_epochs=number_of_epochs,
                        batch_size=batch_size,
                        gradient_clip_val=gradient_clip_val,
                        embedding_size=embedding_size,
                        regularization_factor=regularization_factor,
                        dropout_probability=dropout_probability,
                        gpus=[gpu_id],
                    )
        except Exception as e:
            if self.gpu_id_queue is not None:
                self.gpu_id_queue.put(gpu_id)
            raise e
        if self.gpu_id_queue is not None:
            self.gpu_id_queue.put(gpu_id)
        return best_validation_loss

def hyperparameter_search() -> None:
    study = optuna.create_study(study_name=STUDY_NAME, sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.SuccessiveHalvingPruner(), storage=DB_URL, direction='minimize', load_if_exists=True)
    optimize_kawrgs = dict(
        n_trials=NUMBER_OF_HYPERPARAMETER_SEARCH_TRIALS,
        gc_after_trial=True,
        catch=(Exception,),
    )
    with _training_logging_suppressed():
        if not HYPERPARAMETER_SEARCH_IS_DISTRIBUTED:
            optimize_kawrgs['func'] = HyperParameterSearchObjective(None)
            study.optimize(**optimize_kawrgs)
        else:
            with mp.Manager() as manager:
                gpu_id_queue = manager.Queue()
                more_itertools.consume((gpu_id_queue.put(gpu_id) for gpu_id in GPU_IDS))
                optimize_kawrgs['func'] = HyperParameterSearchObjective(gpu_id_queue)
                optimize_kawrgs['n_jobs'] = len(GPU_IDS)
                with joblib.parallel_backend('multiprocessing', n_jobs=len(GPU_IDS)):
                    study.optimize(**optimize_kawrgs)
    best_params = study.best_params
    LOGGER.info('Best Validation Parameters:\n'+'\n'.join((f'    {param}: {repr(param_value)}' for param, param_value in best_params.items())))
    trials_df = study.trials_dataframe()
    best_trials_df = trials_df.nsmallest(NUMBER_OF_BEST_HYPERPARAMETER_RESULTS_TO_DISPLAY, 'value')
    parameter_name_prefix = 'params_'
    for rank, row in enumerate(best_trials_df.itertuples()):
        hyperparameters = {attr_name[len(parameter_name_prefix):]: getattr(row, attr_name) for attr_name in dir(row) if attr_name.startswith(parameter_name_prefix)}
        checkpoint_dir = checkpoint_directory_from_hyperparameters(**hyperparameters)
        result_summary_json_file_location = os.path.join(checkpoint_dir, 'result_summary.json')
        with open(result_summary_json_file_location, 'r') as file_handle:
            result_summary_dict = json.loads(file_handle.read())
        LOGGER.info('')
        LOGGER.info(f'Rank: {rank}')
        LOGGER.info(f'Trial: {row.number}')
        LOGGER.info('')
        LOGGER.info(f'Best Validation Loss: {result_summary_dict["best_validation_loss"]}')
        LOGGER.info(f'Best Validation Model Path: {result_summary_dict["best_validation_model_path"]}')
        LOGGER.info('')
        LOGGER.info(f'Testing Loss: {result_summary_dict["testing_loss"]}')
        LOGGER.info(f'Testing Regularization Loss: {result_summary_dict["testing_regularization_loss"]}')
        LOGGER.info(f'Testing MSE Loss: {result_summary_dict["testing_mse_loss"]}')
        LOGGER.info('')
        LOGGER.info(f'Learning Rate: {result_summary_dict["learning_rate"]}')
        LOGGER.info(f'Number of Epochs: {result_summary_dict["number_of_epochs"]}')
        LOGGER.info(f'Batch Size: {result_summary_dict["batch_size"]}')
        LOGGER.info(f'number of Animes: {result_summary_dict["number_of_animes"]}')
        LOGGER.info(f'Number of Users: {result_summary_dict["number_of_users"]}')
        LOGGER.info(f'Embedding Size: {result_summary_dict["embedding_size"]}')
        LOGGER.info(f'Regularization Factor: {result_summary_dict["regularization_factor"]}')
        LOGGER.info(f'Dropout Probability: {result_summary_dict["dropout_probability"]}')
        LOGGER.info('')
    return

###################
# Result Analysis #
###################

def analyze_hyperparameter_search_results() -> None:
    study = optuna.create_study(study_name=STUDY_NAME, sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.SuccessiveHalvingPruner(), storage=DB_URL, direction='minimize', load_if_exists=True)
    trials_df = study.trials_dataframe()
    best_trials_df = trials_df.nsmallest(NUMBER_OF_BEST_HYPERPARAMETER_RESULTS_TO_DISPLAY, 'value')
    
    parameter_name_prefix = 'params_'
    for rank, row in enumerate(best_trials_df.itertuples()):
        hyperparameters = {attr_name[len(parameter_name_prefix):]: getattr(row, attr_name) for attr_name in dir(row) if attr_name.startswith(parameter_name_prefix)}
        checkpoint_dir = checkpoint_directory_from_hyperparameters(**hyperparameters)
        result_summary_json_file_location = os.path.join(checkpoint_dir, 'result_summary.json')
        with open(result_summary_json_file_location, 'r') as file_handle:
            result_summary_dict = json.loads(file_handle.read())
        
        model = LinearColaborativeFilteringModel.load_from_checkpoint(result_summary_dict['best_validation_model_path'])
        model.eval()

        data_module = AnimeRatingDataModule(1, NUM_WORKERS)
        data_module.prepare_data()
        data_module.setup()

        anime_df = pd.DataFrame(index=pd.Index(data_module.anime_id_index_to_anime_id, name='anime_id'))
        anime_df['total_mse_loss'] = 0
        anime_df['example_count'] = 0
        
        user_df = pd.DataFrame(index=pd.Index(data_module.user_id_index_to_user_id, name='user_id'))
        user_df['total_mse_loss'] = 0
        user_df['example_count'] = 0

        for batch_dict in data_module.test_dataloader():
            user_id_indices: torch.Tensor = batch_dict['user_id_index']
            anime_id_indices: torch.Tensor = batch_dict['anime_id_index']
            user_ids: np.ndarray = data_module.user_id_index_to_user_id[user_id_indices]
            anime_ids: np.ndarray = data_module.anime_id_index_to_anime_id[anime_id_indices]
            target_scores, _ = model(batch_dict)
            target_scores = target_scores.detach()
            
            batch_df = pd.DataFrame({'user_id': user_ids, 'anime_id': anime_ids, 'target_score': target_scores})
            batch_anime_df = batch_df.groupby('anime_id').agg({'target_score': ['sum', 'count']})
            batch_user_df = batch_df.groupby('user_id').agg({'target_score': ['sum', 'count']})

            anime_df['total_mse_loss'][batch_anime_df.index] += batch_anime_df['target_score']['sum']
            anime_df['example_count'][batch_anime_df.index] += batch_anime_df['target_score']['count']
            
            user_df['total_mse_loss'][batch_user_df.index] += batch_user_df['target_score']['sum']
            user_df['example_count'][batch_user_df.index] += batch_user_df['target_score']['count']

        anime_df['mean_mse_loss'] = anime_df['total_mse_loss'] / anime_df['example_count']
        user_df['mean_mse_loss'] = user_df['total_mse_loss'] / user_df['example_count']
        
        result_summary_dict['anime_data'] = anime_df.to_dict(orient='index')
        result_summary_dict['user_data'] = user_df.to_dict(orient='index')

        with open(os.path.join(ANALYSIS_OUTPUT_DIR, f'rank_{rank}_summary.json'), 'w') as file_handle:
            json.dump(result_summary_dict, file_handle, indent=4)
    
    return
