#!/usr/bin/python3 -OO

'''

This is a module for data processing and data loading utilities.

Sections: 
* Imports
* Globals
* Data Modules

'''

###########
# Imports #
###########

import os
import more_itertools
import numpy as np
import pandas as pd

import torch
from torch.utils import data
import pytorch_lightning as pl

from misc_utilities import *
from global_values import *

###########
# Globals #
###########

# https://www.kaggle.com/CooperUnion/anime-recommendations-database
# ANIME_CSV_FILE_LOCATION = './data/compressed_data/anime.csv'
RATING_CSV_FILE_LOCATION = './data/rating.csv'

COMPRESSED_RATING_CSV_FILE_PIECES = [
    './data/compressed_data/rating_1.csv',
    './data/compressed_data/rating_2.csv',
    './data/compressed_data/rating_3.csv',
    './data/compressed_data/rating_4.csv',
    './data/compressed_data/rating_5.csv',
    './data/compressed_data/rating_6.csv',
    './data/compressed_data/rating_7.csv',
    './data/compressed_data/rating_8.csv',
    './data/compressed_data/rating_9.csv',
    './data/compressed_data/rating_10.csv',
    './data/compressed_data/rating_11.csv',
    './data/compressed_data/rating_12.csv',
    './data/compressed_data/rating_13.csv',
    './data/compressed_data/rating_14.csv',
    './data/compressed_data/rating_15.csv',
    './data/compressed_data/rating_16.csv',
    './data/compressed_data/rating_17.csv',
    './data/compressed_data/rating_18.csv',
    './data/compressed_data/rating_19.csv',
    './data/compressed_data/rating_20.csv',
    './data/compressed_data/rating_21.csv',
    './data/compressed_data/rating_22.csv',
    './data/compressed_data/rating_23.csv',
]

PROCESSED_DATA_CSV_FILE_LOCATION = './data/processed_data.csv'

RATING_HISTORGRAM_PNG_FILE_LOCATION = './data/rating_histogram.png'

TRAINING_LABEL, VALIDATION_LABEL, TESTING_LABEL = 0, 1, 2

TRAINING_PORTION = 0.65
VALIDATION_PORTION = 0.15
TESTING_PORTION = 0.20

MINIMUM_NUMBER_OF_RATINGS_PER_ANIME = 100
MINIMUM_NUMBER_OF_RATINGS_PER_USER = 100

################
# Data Modules #
################

def _preprocess_data() -> pd.DataFrame:
    LOGGER.info('')
    LOGGER.info(f'Preprocessing data.')
    LOGGER.info('')
    if not os.path.isfile(RATING_CSV_FILE_LOCATION):
        full_data = ''
        for piece in COMPRESSED_RATING_CSV_FILE_PIECES:
            with open(piece, 'r') as f:
                full_data += f.read()
        with open(RATING_CSV_FILE_LOCATION, 'w') as f:
            f.write(full_data)
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

def preprocess_data() -> pd.DataFrame:
    if os.path.isfile(PROCESSED_DATA_CSV_FILE_LOCATION):
        rating_df = pd.read_csv(PROCESSED_DATA_CSV_FILE_LOCATION)
    else:
        rating_df = _preprocess_data()
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
        
        self.user_id_index_to_user_id: np.ndarray = self.rating_df.user_id.unique()
        self.anime_id_index_to_anime_id: np.ndarray = self.rating_df.anime_id.unique()
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

if __name__ == '__main__':
    print('This is a module for data processing and data loading utilities.')

