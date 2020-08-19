#!/usr/bin/python3 -OO

'''
'''
# @todo update doc string

###########
# Imports #
###########

import os
import more_itertools
import numpy as np
import pandas as pd
import multiprocessing as mp
from pandarallel import pandarallel
from collections import OrderedDict
from typing import Tuple, Callable
from typing_extensions import Literal

from misc_utilities import *

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
import transformers
import pytorch_lightning as pl

# @todo make sure these are used

###########
# Globals #
###########

CPU_COUNT = mp.cpu_count()
pandarallel.initialize(nb_workers=CPU_COUNT, progress_bar=False, verbose=0)

NUM_WORKERS = 4

# https://www.kaggle.com/CooperUnion/anime-recommendations-database
ANIME_CSV_FILE_LOCATION = './data/anime.csv'
RATING_CSV_FILE_LOCATION = './data/rating.csv'

PROCESSED_DATA_CSV_FILE_LOCATION = './data/processed_data.csv'

RATING_HISTORGRAM_PNG_FILE_LOCATION = './data/rating_histogram.png'

TRAINING_LABEL, VALIDATION_LABEL, TESTING_LABEL = 0, 1, 2

MSE_LOSS = nn.MSELoss(reduction='none')

EVALUATION_BATCH_SIZE = 2**12

###################
# Hyperparameters #
###################

TRAINING_PORTION = 0.65
VALIDATION_PORTION = 0.15
TESTING_PORTION = 0.20

MINIMUM_NUMBER_OF_RATINGS_PER_ANIME = 100
MINIMUM_NUMBER_OF_RATINGS_PER_USER = 100

NUMBER_OF_EPOCHS = 15 
BATCH_SIZE = 2**10
GRADIENT_CLIP_VAL = 1.0

LEARNING_RATE = 1e-3
EMBEDDING_SIZE = 100
REGULARIZATION_FACTOR = 1
DROPOUT_PROBABILITY = 0.5
            
################
# Data Modules #
################

def preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print()
    print(f'Preprocessing data.')
    print()
    rating_df = pd.read_csv(RATING_CSV_FILE_LOCATION)
    assert all(rating_df[rating_df.isnull()].count() == 0)
    rating_df.drop(rating_df[rating_df.rating == -1].index, inplace=True)
    original_anime_count = len(rating_df.anime_id.unique())
    original_user_count = len(rating_df.user_id.unique())
    print(f'Original Number of Animes: {original_anime_count:,}')
    print(f'Original Number of Users: {original_user_count:,}')
    
    print()
    print(f'Anime Rating Count Threshold: {MINIMUM_NUMBER_OF_RATINGS_PER_ANIME:,}')
    anime_to_rating_count = rating_df[['anime_id', 'rating']].groupby('anime_id').count().rating
    anime_to_rating_count.drop(anime_to_rating_count[anime_to_rating_count < MINIMUM_NUMBER_OF_RATINGS_PER_ANIME].index, inplace=True)
    assert not anime_to_rating_count.isnull().any().any()
    print(f'Number Animes Kept: {len(anime_to_rating_count):,}')
    print(f'Number Animes Dropped: {original_anime_count - len(anime_to_rating_count):,}')
    print(f'Min Number Of Ratings Per Anime: {int(anime_to_rating_count.min()):,}')
    print(f'Max Number Of Ratings Per Anime: {int(anime_to_rating_count.max()):,}')
    print(f'Mean Number Of Ratings Per Anime: {anime_to_rating_count.mean():,}')
    rating_df.drop(rating_df[~rating_df.anime_id.isin(anime_to_rating_count.index)].index, inplace=True)
    
    print()
    print(f'User Rating Count Threshold: {MINIMUM_NUMBER_OF_RATINGS_PER_USER:,}')
    user_to_rating_count = rating_df[['user_id', 'rating']].groupby('user_id').count().rating
    user_to_rating_count.drop(user_to_rating_count[user_to_rating_count < MINIMUM_NUMBER_OF_RATINGS_PER_USER].index, inplace=True)
    assert not user_to_rating_count.isnull().any().any()
    print(f'Number Users Kept: {len(user_to_rating_count):,}')
    print(f'Number Users Dropped: {original_user_count - len(user_to_rating_count):,}')
    print(f'Min Number Of Ratings Per User: {int(user_to_rating_count.min()):,}')
    print(f'Max Number Of Ratings Per User: {int(user_to_rating_count.max()):,}')
    print(f'Mean Number Of Ratings Per User: {user_to_rating_count.mean():,}')
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
    print(f'Preprocessed data saved to {PROCESSED_DATA_CSV_FILE_LOCATION} .')

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
    print(f'Data distribution visualization saved to {RATING_HISTORGRAM_PNG_FILE_LOCATION} .')

    print()
    print(f'Data preprocessing complete.')
    print()
    return rating_df

class AnimeRatingDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, user_id_to_user_id_index: dict, anime_id_to_anime_id_index: dict):
        self.df = df.copy().iloc[:3000] # @todo remove this
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
            # Common Hyperparameters
            learning_rate: float,
            number_of_epochs: int,
            # Information Passing
            number_of_training_batches: int,
            # Mode-specific Hyperparameters
            number_of_animes: int, 
            number_of_users: int, 
            embedding_size: int, 
            regularization_factor: float,
            dropout_probability: float, 
    ):
        super().__init__()
        
        self.learning_rate = learning_rate
        # @todo can we abstract these two away?
        self.number_of_epochs = number_of_epochs
        self.number_of_training_batches = number_of_training_batches
        self.number_of_animes = number_of_animes
        self.number_of_users = number_of_users
        self.embedding_size = embedding_size
        self.regularization_factor = regularization_factor
        self.dropout_probability = dropout_probability
        
        self.anime_embedding_layers = nn.Sequential(OrderedDict([
            ("anime_embedding_layer", nn.Embedding(self.number_of_animes, self.embedding_size)),
            ("dropout_layer", nn.Dropout(self.dropout_probability)),
        ]))
        self.user_embedding_layers = nn.Sequential(OrderedDict([
            ("user_embedding_layer", nn.Embedding(self.number_of_users, self.embedding_size)),
            ("dropout_layer", nn.Dropout(self.dropout_probability)),
        ]))

    def forward(self, batch_dict: dict):
        user_id_indices: torch.Tensor = batch_dict['user_id_index']
        batch_size = user_id_indices.shape[0]
        assert tuple(user_id_indices.shape) == (batch_size,)
        anime_id_indices: torch.Tensor = batch_dict['anime_id_index']
        assert tuple(anime_id_indices.shape) == (batch_size,)

        user_embedding = self.user_embedding_layers(user_id_indices)
        assert tuple(user_embedding.shape) == (batch_size, self.embedding_size)
        anime_embedding = self.anime_embedding_layers(anime_id_indices)
        assert tuple(anime_embedding.shape) == (batch_size, self.embedding_size)
        
        scores = user_embedding.mul(anime_embedding).sum(dim=1)
        assert tuple(scores.shape) == (batch_size,)

        regularization_loss = torch.sum(self.regularization_factor * (F.normalize(user_embedding[:,:-1], p=2, dim=1) + F.normalize(anime_embedding[:,:-1], p=2, dim=1)), dim=1)
        assert tuple(regularization_loss.shape) == (batch_size,)
        return scores, regularization_loss

    def backward(self, _trainer: pl.Trainer, loss: torch.Tensor, _optimizer: torch.optim.Optimizer, _optimizer_idx: int) -> None:
        del _trainer, _optimizer, _optimizer_idx
        loss.mean().backward()
        return
    
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        optimizer: torch.optim.Optimizer = transformers.AdamW(self.parameters(), lr=self.learning_rate, correct_bias=False)
        scheduler: torch.optim.lr_scheduler.LambdaLR = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=self.number_of_training_batches*self.number_of_epochs)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def _get_batch_loss(self, batch_dict: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_scores = batch_dict['rating']
        batch_size = only_one(target_scores.shape)
        predicted_scores, regularization_loss = self(batch_dict)
        assert tuple(predicted_scores.shape) == (batch_size,)
        assert tuple(regularization_loss.shape) == (batch_size,)
        mse_loss = MSE_LOSS(predicted_scores, target_scores)
        assert tuple(mse_loss.shape) == (batch_size,)
        loss = regularization_loss + mse_loss
        assert tuple(loss.shape) == (batch_size,)
        assert tuple(loss.shape) == tuple(mse_loss.shape) == tuple(regularization_loss.shape) == (batch_size, )
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

    def test_epoch_end(self) -> pl.EvalResult:
        return self._eval_epoch_end(validation_step_results, 'testing')

##########
# Driver #
##########

class PrintingCallback(pl.Callback):

    def on_init_start(self, trainer: pl.Trainer) -> None:
        print()
        print('Initializing trainer.')
        return
    
    def on_fit_start(trainer: pl.Trainer, pl_module: pl.LightningDataModule) -> None:
        print()
        print('Starting training.')
        return
    
    def on_fit_end(trainer: pl.Trainer, pl_module: pl.LightningDataModule) -> None:
        print()
        print('Training complete.')
        return

@debug_on_error
@raise_on_warn
def main() -> None:

    trainer = pl.Trainer(
        callbacks=[PrintingCallback()],
        max_epochs=NUMBER_OF_EPOCHS,
        min_epochs=NUMBER_OF_EPOCHS,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        terminate_on_nan=True,
        gpus=[0,1,2,3],
        distributed_backend='dp',
        logger=pl.loggers.TensorBoardLogger("lightning_logs", name="linear_cf_model"),
    )
        
    data_module = AnimeRatingDataModule(BATCH_SIZE, NUM_WORKERS)
    data_module.prepare_data()
    data_module.setup()
    
    model = LinearColaborativeFilteringModel(
        learning_rate = LEARNING_RATE,
        number_of_epochs = NUMBER_OF_EPOCHS,
        number_of_training_batches = len(data_module.training_dataloader),
        number_of_animes = data_module.number_of_animes,
        number_of_users = data_module.number_of_users,
        embedding_size = EMBEDDING_SIZE,
        regularization_factor = REGULARIZATION_FACTOR,
        dropout_probability = DROPOUT_PROBABILITY,
    )
    
    trainer.fit(model, data_module)
    return

if __name__ == '__main__':
    main()
        
