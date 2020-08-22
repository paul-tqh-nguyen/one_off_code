#!/usr/bin/python3 -OO

'''
'''
# @todo update doc string

###########
# Imports #
###########

import os
import json
import logging
import more_itertools
import numpy as np
import pandas as pd
import multiprocessing as mp
from pandarallel import pandarallel
from collections import OrderedDict
from contextlib import contextmanager
from typing import Tuple, List, Callable, Generator, Optional
from typing_extensions import Literal

from misc_utilities import *

import torch
from torch import nn
from torch.utils import data
import transformers
import pytorch_lightning as pl
import optuna
import nvgpu
import joblib

# @todo make sure these are used

###########
# Globals #
###########

DB_URL = 'sqlite:///collaborative_filtering.db'
STUDY_NAME = 'collaborative-filtering'

HYPERPARAMETER_SEARCH_IS_DISTRIBUTED = True
NUMBER_OF_HYPERPARAMETER_SEARCH_TRIALS = 200
NUMBER_OF_BEST_HYPERPARAMETER_RESULTS_TO_DISPLAY = 5

CPU_COUNT = mp.cpu_count()
if not HYPERPARAMETER_SEARCH_IS_DISTRIBUTED:
    pandarallel.initialize(nb_workers=CPU_COUNT, progress_bar=False, verbose=0)

GPU_IDS = eager_map(int, nvgpu.available_gpus())
DEFAULT_GPU = GPU_IDS[0]

NUM_WORKERS = 0 if HYPERPARAMETER_SEARCH_IS_DISTRIBUTED else 2

if not os.path.isdir('./checkpoints'):
    os.makedirs('./checkpoints')

# https://www.kaggle.com/CooperUnion/anime-recommendations-database
ANIME_CSV_FILE_LOCATION = './data/anime.csv'
RATING_CSV_FILE_LOCATION = './data/rating.csv'

PROCESSED_DATA_CSV_FILE_LOCATION = './data/processed_data.csv'

RATING_HISTORGRAM_PNG_FILE_LOCATION = './data/rating_histogram.png'

TRAINING_LABEL, VALIDATION_LABEL, TESTING_LABEL = 0, 1, 2

MSE_LOSS = nn.MSELoss(reduction='none')

TRAINING_PORTION = 0.65
VALIDATION_PORTION = 0.15
TESTING_PORTION = 0.20

MINIMUM_NUMBER_OF_RATINGS_PER_ANIME = 100
MINIMUM_NUMBER_OF_RATINGS_PER_USER = 100

###########################
# Default Hyperparameters #
###########################

NUMBER_OF_EPOCHS = 15
BATCH_SIZE = 256
GRADIENT_CLIP_VAL = 1.0

LEARNING_RATE = 1e-3
EMBEDDING_SIZE = 100
REGULARIZATION_FACTOR = 1
DROPOUT_PROBABILITY = 0.5

###########
# Logging #
###########

LOGGER_NAME = 'anime_collaborative_filtering_logger'
LOGGER = logging.getLogger(LOGGER_NAME)
LOGGER_OUTPUT_FILE = './tuning_logs.txt'

def _initialize_logger() -> None:
    LOGGER.setLevel(logging.INFO)
    logging_formatter = logging.Formatter('{asctime} - pid: {process} - threadid: {thread} - func: {funcName} - {levelname}: {message}', style='{')
    logging_file_handler = logging.FileHandler(LOGGER_OUTPUT_FILE)
    logging_file_handler.setFormatter(logging_formatter)
    LOGGER.addHandler(logging_file_handler)
    LOGGER.addHandler(logging.StreamHandler())
    return

_initialize_logger()

################
# Data Modules #
################

def preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    LOGGER.info('')
    LOGGER.info(f'Preprocessing data.')
    LOGGER.info('')
    rating_df = pd.read_csv(RATING_CSV_FILE_LOCATION)
    assert all(rating_df[rating_df.isnull()].count() == 0)
    rating_df.drop(rating_df[rating_df.rating == -1].index, inplace=True)
    original_anime_count = len(rating_df.anime_id.unique())
    original_user_count = len(rating_df.user_id.unique())
    LOGGER.info(f'Original Number of Animes: {original_anime_count:,}')
    LOGGER.info(f'Original Number of Users: {original_user_count:,}')
    
    LOGGER.info('')
    LOGGER.info(f'Anime Rating Count Threshold: {MINIMUM_NUMBER_OF_RATINGS_PER_ANIME:,}')
    anime_to_rating_count = rating_df[['anime_id', 'rating']].groupby('anime_id').count().rating
    anime_to_rating_count.drop(anime_to_rating_count[anime_to_rating_count < MINIMUM_NUMBER_OF_RATINGS_PER_ANIME].index, inplace=True)
    assert not anime_to_rating_count.isnull().any().any()
    LOGGER.info(f'Number Animes Kept: {len(anime_to_rating_count):,}')
    LOGGER.info(f'Number Animes Dropped: {original_anime_count - len(anime_to_rating_count):,}')
    LOGGER.info(f'Min Number Of Ratings Per Anime: {int(anime_to_rating_count.min()):,}')
    LOGGER.info(f'Max Number Of Ratings Per Anime: {int(anime_to_rating_count.max()):,}')
    LOGGER.info(f'Mean Number Of Ratings Per Anime: {anime_to_rating_count.mean():,}')
    rating_df.drop(rating_df[~rating_df.anime_id.isin(anime_to_rating_count.index)].index, inplace=True)
    
    LOGGER.info('')
    LOGGER.info(f'User Rating Count Threshold: {MINIMUM_NUMBER_OF_RATINGS_PER_USER:,}')
    user_to_rating_count = rating_df[['user_id', 'rating']].groupby('user_id').count().rating
    user_to_rating_count.drop(user_to_rating_count[user_to_rating_count < MINIMUM_NUMBER_OF_RATINGS_PER_USER].index, inplace=True)
    assert not user_to_rating_count.isnull().any().any()
    LOGGER.info(f'Number Users Kept: {len(user_to_rating_count):,}')
    LOGGER.info(f'Number Users Dropped: {original_user_count - len(user_to_rating_count):,}')
    LOGGER.info(f'Min Number Of Ratings Per User: {int(user_to_rating_count.min()):,}')
    LOGGER.info(f'Max Number Of Ratings Per User: {int(user_to_rating_count.max()):,}')
    LOGGER.info(f'Mean Number Of Ratings Per User: {user_to_rating_count.mean():,}')
    rating_df.drop(rating_df[~rating_df.user_id.isin(user_to_rating_count.index)].index, inplace=True)

    def _split_group(group: pd.DataFrame) -> pd.DataFrame:
        rating_count = len(group)
        train_index_start = 0
        validation_index_start = round(rating_count * TRAINING_PORTION)
        testing_index_start = round(rating_count * (TRAINING_PORTION+VALIDATION_PORTION))
        assert abs((validation_index_start - train_index_start) - (TRAINING_PORTION*rating_count)) <= 1
        assert abs((testing_index_start - validation_index_start) - (VALIDATION_PORTION*rating_count)) <= 1
        assert abs((rating_count - testing_index_start) - (TESTING_PORTION*rating_count)) <= 1
        labels = np.empty(rating_count,  dtype=int)
        labels[train_index_start:validation_index_start] = TRAINING_LABEL
        labels[validation_index_start:testing_index_start] = VALIDATION_LABEL
        labels[testing_index_start:] = TESTING_LABEL
        np.random.shuffle(labels)
        group['split_label'] = labels
        return group
    
    rating_df = rating_df.groupby('user_id').parallel_apply(_split_group)
    rating_df.to_csv(PROCESSED_DATA_CSV_FILE_LOCATION, index=False)
    LOGGER.info(f'Preprocessed data saved to {PROCESSED_DATA_CSV_FILE_LOCATION} .')

    with temp_plt_figure(figsize=(20.0,10.0)) as figure:
        anime_to_rating_count.sort_values(inplace=True, ascending=False)
        anime_rating_count_plot = figure.add_subplot(221)
        anime_rating_count_plot.set_xlabel('Anime ID')
        anime_rating_count_plot.set_ylabel('Rating Count')
        anime_rating_count_plot.set_title('Number of Ratings Per Anime')
        anime_rating_count_plot.set_xlim(0, len(anime_to_rating_count))
        anime_rating_count_plot.set_ylim(0, anime_to_rating_count.max())
        anime_rating_count_plot.plot(*zip(*enumerate(anime_to_rating_count)))
        anime_rating_count_plot.grid(True)

        anime_rating_count_histogram = histogram(anime_to_rating_count)
        anime_rating_count_histogram_plot = figure.add_subplot(222)
        anime_rating_count_histogram_plot.set_xlabel('Rating Count')
        anime_rating_count_histogram_plot.set_ylabel('Number of Animes')
        anime_rating_count_histogram_plot.set_title('Anime Rating Count Histogram')
        anime_rating_count_histogram_plot.set_xlim(0, len(anime_rating_count_histogram))
        anime_rating_count_histogram_plot.set_ylim(0, max(anime_rating_count_histogram.values()))
        anime_rating_count_histogram_plot.bar(*zip(*anime_rating_count_histogram.items()))
        anime_rating_count_histogram_plot.grid(True)
        
        user_to_rating_count.sort_values(inplace=True, ascending=False)
        user_rating_count_plot = figure.add_subplot(223)
        user_rating_count_plot.set_xlabel('User ID')
        user_rating_count_plot.set_ylabel('Rating Count')
        user_rating_count_plot.set_title('Number of Ratings Per User')
        user_rating_count_plot.set_xlim(0, len(user_to_rating_count))
        user_rating_count_plot.set_ylim(0, user_to_rating_count.max())
        user_rating_count_plot.plot(*zip(*enumerate(user_to_rating_count)))
        user_rating_count_plot.grid(True)

        user_rating_count_histogram = histogram(user_to_rating_count)
        user_rating_count_histogram_plot = figure.add_subplot(224)
        user_rating_count_histogram_plot.set_xlabel('Rating Count') 
        user_rating_count_histogram_plot.set_ylabel('Number Of Users')
        user_rating_count_histogram_plot.set_title('User Rating Count Histogram')
        user_rating_count_histogram_plot.set_xlim(0, len(user_rating_count_histogram))
        user_rating_count_histogram_plot.set_ylim(0, max(user_rating_count_histogram.values()))
        user_rating_count_histogram_plot.bar(*zip(*user_rating_count_histogram.items()))
        user_rating_count_histogram_plot.grid(True)

        figure.savefig(RATING_HISTORGRAM_PNG_FILE_LOCATION)
    LOGGER.info(f'Data distribution visualization saved to {RATING_HISTORGRAM_PNG_FILE_LOCATION} .')

    LOGGER.info('')
    LOGGER.info(f'Data preprocessing complete.')
    LOGGER.info('')
    return rating_df

class AnimeRatingDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, user_id_to_user_id_index: dict, anime_id_to_anime_id_index: dict):
        self.df = df
        self.user_id_to_user_id_index = user_id_to_user_id_index
        self.anime_id_to_anime_id_index = anime_id_to_anime_id_index        
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        item = {
            'user_id_index': self.user_id_to_user_id_index[row.user_id],
            'anime_id_index': self.anime_id_to_anime_id_index[row.anime_id],
            'rating': torch.tensor(row.rating, dtype=torch.float32),
        }
        return item
    
    def __len__(self):
        return len(self.df)

class AnimeRatingDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int, num_workers: int):
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        if os.path.isfile(PROCESSED_DATA_CSV_FILE_LOCATION):
            self.rating_df = pd.read_csv(PROCESSED_DATA_CSV_FILE_LOCATION)
        else:
            self.rating_df = preprocess_data()
        return
    
    def setup(self) -> None:
        data_splits = {split_label: df for split_label, df in self.rating_df.groupby('split_label')}
        more_itertools.consume((df.drop(columns=['split_label'], inplace=True) for df in data_splits.values()))
        more_itertools.consume((df.reset_index(drop=True, inplace=True) for df in data_splits.values()))
        
        training_df = data_splits[TRAINING_LABEL]
        validation_df = data_splits[VALIDATION_LABEL]
        testing_df = data_splits[TESTING_LABEL]
        assert set(training_df.user_id.unique()) == set(validation_df.user_id.unique()) == set(testing_df.user_id.unique()) == set(self.rating_df.user_id.unique())
        
        self.user_id_index_to_user_id: np.ndarray = self.rating_df.user_id.unique() # @todo use this in printing function for predictions
        self.anime_id_index_to_anime_id: np.ndarray = self.rating_df.anime_id.unique() # @todo use this in printing function for predictions
        self.user_id_index_to_user_id.sort()
        self.anime_id_index_to_anime_id.sort()
        
        self.user_id_to_user_id_index: dict = dict(map(reversed, enumerate(self.user_id_index_to_user_id)))
        self.anime_id_to_anime_id_index: dict = dict(map(reversed, enumerate(self.anime_id_index_to_anime_id)))
        
        training_dataset = AnimeRatingDataset(training_df, self.user_id_to_user_id_index, self.anime_id_to_anime_id_index)
        validation_dataset = AnimeRatingDataset(validation_df, self.user_id_to_user_id_index, self.anime_id_to_anime_id_index)
        testing_dataset = AnimeRatingDataset(testing_df, self.user_id_to_user_id_index, self.anime_id_to_anime_id_index)

        # https://github.com/PyTorchLightning/pytorch-lightning/issues/408 forces us to use shuffle in training and drop_last pervasively
        self.training_dataloader = data.DataLoader(training_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)
        self.validation_dataloader = data.DataLoader(validation_dataset, batch_size=len(validation_dataset)//4, num_workers=self.num_workers, shuffle=False, drop_last=True)
        self.testing_dataloader = data.DataLoader(testing_dataset, batch_size=len(testing_dataset)//4, num_workers=self.num_workers, shuffle=False, drop_last=True)
        assert iff(self.num_workers == 0, HYPERPARAMETER_SEARCH_IS_DISTRIBUTED)
        
        return

    @property
    def number_of_users(self) -> int:
        return len(self.user_id_index_to_user_id)
    
    @property
    def number_of_animes(self) -> int:
        return len(self.anime_id_index_to_anime_id)
    
    def train_dataloader(self) -> data.DataLoader:
        return self.training_dataloader

    def val_dataloader(self) -> data.DataLoader:
        return self.validation_dataloader

    def test_dataloader(self) -> data.DataLoader:
        return self.testing_dataloader

##########
# Models #
##########

class LinearColaborativeFilteringModel(pl.LightningModule):
    
    def __init__(
            self,
            learning_rate: float,
            number_of_epochs: int,
            number_of_training_batches: int,
            number_of_animes: int, 
            number_of_users: int, 
            embedding_size: int, 
            regularization_factor: float,
            dropout_probability: float, 
    ):
        super().__init__()
        self.save_hyperparameters(
            'learning_rate',
            'number_of_epochs',
            'number_of_training_batches',
            'number_of_animes',
            'number_of_users',
            'embedding_size',
            'regularization_factor',
            'dropout_probability',
        )
        
        self.anime_embedding_layers = nn.Sequential(OrderedDict([
            ('anime_embedding_layer', nn.Embedding(self.hparams.number_of_animes, self.hparams.embedding_size)),
            ('dropout_layer', nn.Dropout(self.hparams.dropout_probability)),
        ]))
        self.user_embedding_layers = nn.Sequential(OrderedDict([
            ('user_embedding_layer', nn.Embedding(self.hparams.number_of_users, self.hparams.embedding_size)),
            ('dropout_layer', nn.Dropout(self.hparams.dropout_probability)),
        ]))

    def forward(self, batch_dict: dict):
        user_id_indices: torch.Tensor = batch_dict['user_id_index']
        batch_size = user_id_indices.shape[0]
        assert tuple(user_id_indices.shape) == (batch_size,)
        anime_id_indices: torch.Tensor = batch_dict['anime_id_index']
        assert tuple(anime_id_indices.shape) == (batch_size,)

        user_embedding = self.user_embedding_layers(user_id_indices)
        assert tuple(user_embedding.shape) == (batch_size, self.hparams.embedding_size)
        anime_embedding = self.anime_embedding_layers(anime_id_indices)
        assert tuple(anime_embedding.shape) == (batch_size, self.hparams.embedding_size)
        
        scores = user_embedding.mul(anime_embedding).sum(dim=1)
        assert tuple(scores.shape) == (batch_size,)

        get_norm = lambda batch_matrix: batch_matrix.mul(batch_matrix).sum(dim=1)
        regularization_loss = get_norm(user_embedding[:,:-1]) + get_norm(anime_embedding[:,:-1])
        assert tuple(regularization_loss.shape) == (batch_size,)
        assert regularization_loss.gt(0).all()
        return scores, regularization_loss

    def backward(self, _trainer: pl.Trainer, loss: torch.Tensor, _optimizer: torch.optim.Optimizer, _optimizer_idx: int) -> None:
        del _trainer, _optimizer, _optimizer_idx
        loss.mean().backward()
        return
    
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        optimizer: torch.optim.Optimizer = transformers.AdamW(self.parameters(), lr=self.hparams.learning_rate, correct_bias=False)
        scheduler: torch.optim.lr_scheduler.LambdaLR = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=self.hparams.number_of_training_batches*self.hparams.number_of_epochs)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def _get_batch_loss(self, batch_dict: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_scores = batch_dict['rating']
        batch_size = only_one(target_scores.shape)
        predicted_scores, regularization_loss = self(batch_dict)
        assert tuple(predicted_scores.shape) == (batch_size,)
        assert tuple(regularization_loss.shape) == (batch_size,)
        mse_loss = MSE_LOSS(predicted_scores, target_scores)
        assert tuple(mse_loss.shape) == (batch_size,)
        assert mse_loss.gt(0).all()
        loss = regularization_loss + mse_loss
        assert tuple(loss.shape) == (batch_size,)
        assert tuple(loss.shape) == tuple(mse_loss.shape) == tuple(regularization_loss.shape) == (batch_size,)
        return loss, mse_loss, regularization_loss
    
    def training_step(self, batch_dict: dict, _: int) -> pl.TrainResult:
        loss, _, _ = self._get_batch_loss(batch_dict)
        result = pl.TrainResult(minimize=loss)
        return result

    def training_step_end(self, training_step_result: pl.TrainResult) -> pl.TrainResult:
        assert len(training_step_result.minimize.shape) == 1
        mean_loss = training_step_result.minimize.mean()
        result = pl.TrainResult(minimize=mean_loss)
        result.log('training_loss', mean_loss, prog_bar=True)
        return result
    
    def _eval_step(self, batch_dict: dict) -> pl.EvalResult:
        loss, mse_loss, regularization_loss = self._get_batch_loss(batch_dict)
        assert tuple(loss.shape) == tuple(mse_loss.shape) == tuple(regularization_loss.shape)
        assert len(loss.shape) == 1 # batch_size
        result = pl.EvalResult()
        result.log('loss', loss)
        result.log('regularization_loss', regularization_loss)
        result.log('mse_loss', mse_loss)
        return result
    
    def _eval_epoch_end(self, step_result: pl.EvalResult, eval_type: Literal['validation', 'testing']) -> pl.EvalResult:
        loss = step_result.loss.mean()
        mse_loss = step_result.mse_loss.mean()
        regularization_loss = step_result.regularization_loss.mean()
        result = pl.EvalResult(checkpoint_on=loss)
        result.log(f'{eval_type}_loss', loss)
        result.log(f'{eval_type}_regularization_loss', regularization_loss)
        result.log(f'{eval_type}_mse_loss', mse_loss)
        return result
        
    def validation_step(self, batch_dict: dict, _: int) -> pl.EvalResult:
        return self._eval_step(batch_dict)

    def validation_epoch_end(self, validation_step_results: pl.EvalResult) -> pl.EvalResult:
        return self._eval_epoch_end(validation_step_results, 'validation')

    def test_step(self, batch_dict: dict, _: int) -> pl.EvalResult:
        return self._eval_step(batch_dict)

    def test_epoch_end(self, test_step_results: pl.EvalResult) -> pl.EvalResult:
        return self._eval_epoch_end(test_step_results, 'testing')

############
# Training #
############

class PrintingCallback(pl.Callback):

    def __init__(self, checkpoint_callback: pl.callbacks.ModelCheckpoint):
        super().__init__()
        self.checkpoint_callback = checkpoint_callback
    
    def on_init_start(self, trainer: pl.Trainer) -> None:
        LOGGER.info('')
        LOGGER.info('Initializing trainer.')
        LOGGER.info('')
        return
    
    def on_train_start(self, trainer: pl.Trainer, model: pl.LightningDataModule) -> None:
        LOGGER.info('')
        LOGGER.info('Model: ')
        LOGGER.info(model)
        LOGGER.info('')
        LOGGER.info(f'Training GPUs: {trainer.gpus}')
        LOGGER.info(f'Number of Epochs: {model.hparams.number_of_epochs:,}')
        LOGGER.info(f'Learning Rate: {model.hparams.learning_rate:,}')
        LOGGER.info(f'Number of Animes: {model.hparams.number_of_animes:,}')
        LOGGER.info(f'Number of Users: {model.hparams.number_of_users:,}')
        LOGGER.info(f'Embedding Size: {model.hparams.embedding_size:,}')
        LOGGER.info(f'Regularization Factor: {model.hparams.regularization_factor:,}')
        LOGGER.info(f'Dropout Probability: {model.hparams.dropout_probability:,}')
        LOGGER.info('')
        LOGGER.info('Data:')
        LOGGER.info('')
        LOGGER.info(f'Training Batch Size: {trainer.train_dataloader.batch_size:,}')
        LOGGER.info(f'Validation Batch Size: {only_one(trainer.val_dataloaders).batch_size:,}')
        LOGGER.info('')
        LOGGER.info(f'Training Batch Count: {len(trainer.train_dataloader):,}')
        LOGGER.info(f'Validation Batch Count: {len(only_one(trainer.val_dataloaders)):,}')
        LOGGER.info('')
        LOGGER.info(f'Training Example Count: {len(trainer.train_dataloader)*trainer.train_dataloader.batch_size:,}')
        LOGGER.info(f'Validation Example Count: {len(only_one(trainer.val_dataloaders))*only_one(trainer.val_dataloaders).batch_size:,}')
        LOGGER.info('')
        LOGGER.info('Starting training.')
        LOGGER.info('')
        return
    
    def on_train_end(self, trainer: pl.Trainer, model: pl.LightningDataModule) -> None:
        LOGGER.info('')
        LOGGER.info('Training complete.')
        LOGGER.info('')
        return

    def on_test_start(self, trainer: pl.Trainer, model: pl.LightningDataModule) -> None:
        LOGGER.info('')
        LOGGER.info('Starting testing.')
        LOGGER.info('')
        LOGGER.info(f'Testing Batch Size: {only_one(trainer.test_dataloaders).batch_size:,}')
        LOGGER.info(f'Testing Example Count: {len(only_one(trainer.test_dataloaders))*only_one(trainer.test_dataloaders).batch_size:,}')
        LOGGER.info(f'Testing Batch Count: {len(only_one(trainer.test_dataloaders)):,}')
        LOGGER.info('')
        return
    
    def on_test_end(self, trainer: pl.Trainer, model: pl.LightningDataModule) -> None:
        LOGGER.info('')
        LOGGER.info('Testing complete.')
        LOGGER.info('')
        LOGGER.info(f'Best validation model checkpoint saved to {self.checkpoint_callback.best_model_path} .')
        LOGGER.info('')
        return

def checkpoint_directory_from_hyperparameters(learning_rate: float, number_of_epochs: int, batch_size: int, gradient_clip_val: float, embedding_size: int, regularization_factor: float, dropout_probability: float) -> str:
    checkpoint_dir = f'./checkpoints/' \
        f'linear_cf_lr_{learning_rate:.5g}_' \
        f'epochs_{number_of_epochs}_' \
        f'batch_{batch_size}_' \
        f'gradient_clip_{gradient_clip_val:.3g}_' \
        f'embed_{embedding_size}_' \
        f'regularization_{regularization_factor:.5g}_' \
        f'dropout_{dropout_probability:.5g}'
    return checkpoint_dir

def train_model(learning_rate: float, number_of_epochs: int, batch_size: int, gradient_clip_val: float, embedding_size: int, regularization_factor: float, dropout_probability: float, gpus: List[int]) -> float:

    checkpoint_dir = checkpoint_directory_from_hyperparameters(learning_rate, number_of_epochs, batch_size, gradient_clip_val, embedding_size, regularization_factor, dropout_probability)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'checkpoint_{epoch:03d}_{val_checkpoint_on}'),
            save_top_k=1,
            verbose=True,
            save_last=True,
            monitor='val_checkpoint_on',
            mode='min',
        )

    trainer = pl.Trainer(
        callbacks=[PrintingCallback(checkpoint_callback)],
        max_epochs=number_of_epochs,
        min_epochs=number_of_epochs,
        gradient_clip_val=gradient_clip_val,
        terminate_on_nan=True,
        gpus=gpus,
        distributed_backend='dp',
        deterministic=False,
        # precision=16, # not supported for data parallel (e.g. multiple GPUs) https://github.com/NVIDIA/apex/issues/227
        logger=pl.loggers.TensorBoardLogger(checkpoint_dir, name='linear_cf_model'),
        checkpoint_callback=checkpoint_callback,
    )

    data_module = AnimeRatingDataModule(batch_size, NUM_WORKERS)
    data_module.prepare_data()
    data_module.setup()

    model = LinearColaborativeFilteringModel(
        learning_rate = learning_rate,
        number_of_epochs = number_of_epochs,
        number_of_training_batches = len(data_module.training_dataloader),
        number_of_animes = data_module.number_of_animes,
        number_of_users = data_module.number_of_users,
        embedding_size = embedding_size,
        regularization_factor = regularization_factor,
        dropout_probability = dropout_probability,
    )

    trainer.fit(model, data_module)
    test_results = only_one(trainer.test(model, datamodule=data_module, verbose=False, ckpt_path=checkpoint_callback.best_model_path))
    best_validation_loss = checkpoint_callback.best_model_score.item()
    
    output_json_file_location = os.path.join(checkpoint_dir, 'result_summary.json')
    with open(output_json_file_location, 'w') as file_handle:
        json.dump({
            'testing_loss': test_results['testing_loss'],
            'testing_regularization_loss': test_results['testing_regularization_loss'],
            'testing_mse_loss': test_results['testing_mse_loss'],
            
            'best_validation_loss': best_validation_loss,
            'best_validation_model_path': checkpoint_callback.best_model_path,
            
            'learning_rate': learning_rate,
            'number_of_epochs': number_of_epochs,
            'batch_size': batch_size,
            'number_of_animes': data_module.number_of_animes,
            'number_of_users': data_module.number_of_users,
            'embedding_size': embedding_size,
            'regularization_factor': regularization_factor,
            'dropout_probability': dropout_probability,

            'training_set_batch_size': data_module.training_dataloader.batch_size,
            'training_set_batch_count': len(data_module.training_dataloader),
            'validation_set_batch_size': data_module.validation_dataloader.batch_size,
            'validation_set_batch_count': len(data_module.validation_dataloader),
            'testing_set_batch_size': data_module.test_dataloader().batch_size,
            'testing_set_batch_count': len(data_module.test_dataloader()),
        }, file_handle, indent=4)

    return best_validation_loss

#########################
# Hyperparameter Search #
#########################

@contextmanager
def _training_logging_suppressed() -> Generator:
    logger_to_original_disability = {}
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and name in ('lightning', LOGGER_NAME):
            logger_to_original_disability[logger] = logger.disabled
            logger.disabled = True
    yield
    for logger, original_disability in logger_to_original_disability.items():
        logger.disabled = original_disability
    return

class HyperParameterSearchObjective:
    def __init__(self, gpu_id_queue: Optional[object]):
        # gpu_id_queue is an mp.managers.AutoProxy[Queue] and an mp.managers.BaseProxy ; can't declare statically since the classs are generated dyanmically
        self.gpu_id_queue = gpu_id_queue

    def __call__(self, trial: optuna.Trial) -> float:
        gpu_id = self.gpu_id_queue.get() if self.gpu_id_queue else DEFAULT_GPU
        learning_rate = trial.suggest_uniform('learning_rate', 1e-5, 1e-1)
        number_of_epochs = int(trial.suggest_int('number_of_epochs', 3, 15))
        batch_size = int(trial.suggest_categorical('batch_size', [2**power for power in range(5)]))
        gradient_clip_val = trial.suggest_uniform('gradient_clip_val', 1.0, 1.0)
        embedding_size = int(trial.suggest_int('embedding_size', 100, 500))
        regularization_factor = trial.suggest_uniform('regularization_factor', 1, 100)
        dropout_probability = trial.suggest_uniform('dropout_probability', 0.0, 1.0)
        # @todo remove these debugging lines
        # learning_rate = trial.suggest_uniform('learning_rate', 1e-3, 1e-3)
        # number_of_epochs = int(trial.suggest_int('number_of_epochs', 3, 3))
        # batch_size = int(trial.suggest_categorical('batch_size', [1024]))
        # gradient_clip_val = trial.suggest_uniform('gradient_clip_val', 1.0, 1.0)
        # embedding_size = int(trial.suggest_int('embedding_size', 100, 100))
        # regularization_factor = trial.suggest_uniform('regularization_factor', 1, 100)
        # dropout_probability = trial.suggest_uniform('dropout_probability', 0.0, 1.0)
        
        checkpoint_dir = checkpoint_directory_from_hyperparameters(learning_rate, number_of_epochs, batch_size, gradient_clip_val, embedding_size, regularization_factor, dropout_probability)
        LOGGER.info(f'Starting raining for {checkpoint_dir} on GPU {gpu_id}.')
        
        try:
            with _training_logging_suppressed():
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
            self.gpu_id_queue.put(gpu_id)
            raise e
        self.gpu_id_queue.put(gpu_id)
        return best_validation_loss

def hyperparameter_search() -> None:
    study = optuna.create_study(study_name=STUDY_NAME, sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.SuccessiveHalvingPruner(), storage=DB_URL, direction='minimize', load_if_exists=True)
    optimize_kawrgs = dict(
        n_trials=NUMBER_OF_HYPERPARAMETER_SEARCH_TRIALS,
        gc_after_trial=True,
        catch=(Exception,),
    )
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

##########
# Driver #
##########

def train_default_mode() -> None:
    train_model(
        learning_rate=LEARNING_RATE,
        number_of_epochs=NUMBER_OF_EPOCHS,
        batch_size=BATCH_SIZE,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        embedding_size=EMBEDDING_SIZE,
        regularization_factor=REGULARIZATION_FACTOR,
        dropout_probability=DROPOUT_PROBABILITY,
        gpus=GPU_IDS,
    )
    return

@debug_on_error
def main() -> None:
    # train_default_mode()
    hyperparameter_search()
    return

if __name__ == '__main__':
    main()

