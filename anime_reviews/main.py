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
from typing import Tuple, Callable

from misc_utilities import *

import torch
from torch import nn
import pytorch_lightning as pl

# @todo make sure these are used

###########
# Globals #
###########

CPU_COUNT = mp.cpu_count()
pandarallel.initialize(nb_workers=CPU_COUNT, progress_bar=False, verbose=0)

# https://www.kaggle.com/CooperUnion/anime-recommendations-database
ANIME_CSV_FILE_LOCATION = './data/anime.csv'
RATING_CSV_FILE_LOCATION = './data/rating.csv'

PROCESSED_DATA_CSV_FILE_LOCATION = './data/processed_data.csv'

RATING_HISTORGRAM_PNG_FILE_LOCATION = './data/rating_histogram.png'

TRAINING_LABEL, VALIDATION_LABEL, TESTING_LABEL = 0, 1, 2

###################
# Hyperparameters #
###################

TRAINING_PORTION = 0.65
VALIDATION_PORTION = 0.15
TESTING_PORTION = 0.20

MINIMUM_NUMBER_OF_RATINGS_PER_ANIME = 100
MINIMUM_NUMBER_OF_RATINGS_PER_USER = 100

######################
# Data Preprocessing #
######################

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

    print()
    print(f'Data preprocessing complete.')
    print()
    return rating_df

##########
# Models #
##########



##########
# Driver #
##########

@debug_on_error
def main() -> None:
    if os.path.isfile(PROCESSED_DATA_CSV_FILE_LOCATION):
        rating_df = pd.read_csv(PROCESSED_DATA_CSV_FILE_LOCATION)
    else:
        rating_df = preprocess_data()
    data_splits = {split_label: df for split_label, df in rating_df.groupby('split_label')}
    more_itertools.consume((df.drop(columns=['split_label'], inplace=True) for df in data_splits.values()))
    training_df = data_splits[TRAINING_LABEL]
    validation_df = data_splits[VALIDATION_LABEL]
    testing_df = data_splits[TESTING_LABEL]
    assert set(training_df.user_id.unique()) == set(validation_df.user_id.unique()) == set(testing_df.user_id.unique())
    
    return

if __name__ == '__main__':
    main()
    
