#!/usr/bin/python3 -OO

'''
'''
# @todo update doc string

###########
# Imports #
###########

import pandas as pd
import multiprocessing as mp
from pandarallel import pandarallel
from typing import Callable

from misc_utilities import *

# @todo make sure these are used

###########
# Globals #
###########

CPU_COUNT = mp.cpu_count()
pandarallel.initialize(nb_workers=CPU_COUNT, progress_bar=False, verbose=0)

# https://www.kaggle.com/CooperUnion/anime-recommendations-database
ANIME_CSV_FILE_LOCATION = './data/anime.csv'
RATING_CSV_FILE_LOCATION = './data/rating.csv'

MINIMUM_NUMBER_OF_RATINGS_PER_ANIME = 10
MINIMUM_NUMBER_OF_RATINGS_PER_USER = 10

#############################
# Domain-Specific Utilities #
#############################

def series_filter(series: pd.Series, func: Callable, return_copy: bool = True) -> pd.Series:
    keep_mask = series.map(func).values
    filtered_series = series.iloc[keep_mask]
    if return_copy:
        filtered_series = filtered_series.copy()
    return filtered_series

##########
# Driver #
##########

@debug_on_error
def process_data() -> None:
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

    rating_df.reset_index(drop=True, inplace=True)
    breakpoint()
    return

if __name__ == '__main__':
    process_data()
    
