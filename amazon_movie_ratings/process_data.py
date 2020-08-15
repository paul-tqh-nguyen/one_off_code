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

from misc_utilities import *

# @todo make sure these are used

###########
# Globals #
###########

CPU_COUNT = mp.cpu_count()
pandarallel.initialize(nb_workers=CPU_COUNT, progress_bar=False, verbose=0)

# https://www.kaggle.com/eswarchandt/amazon-movie-ratings
DATA_CSV_FILE_LOCATION = './data/Amazon.csv'

MINIMUM_NUMBER_OF_RATINGS_PER_MOVIE = 10
MINIMUM_NUMBER_OF_RATINGS_PER_USER = 10

##########
# Driver #
##########

@debug_on_error
def process_data() -> None:
    df = pd.read_csv(DATA_CSV_FILE_LOCATION, index_col='user_id')
    
    print(f'Movie Rating Count Threshold: {MINIMUM_NUMBER_OF_RATINGS_PER_MOVIE}')
    movies_to_rating_count = df.count()
    movies_to_keep_mask = movies_to_rating_count > MINIMUM_NUMBER_OF_RATINGS_PER_MOVIE
    movies_to_drop = movies_to_rating_count[~movies_to_keep_mask].index
    movies_to_rating_count = movies_to_rating_count[movies_to_keep_mask]
    print(f'Number Movies Dropped: {len(movies_to_drop)}')
    print(f'Min Number Of Ratings Per Movie: {movies_to_rating_count.min()}')
    print(f'Max Number Of Ratings Per Movie: {movies_to_rating_count.max()}')
    print(f'Mean Number Of Ratings Per Movie: {movies_to_rating_count.mean()}')

    print(f'User Rating Count Threshold: {MINIMUM_NUMBER_OF_RATINGS_PER_USER}')
    users_to_rating_count = df.count(axis=1)
    users_to_keep_mask = users_to_rating_count > MINIMUM_NUMBER_OF_RATINGS_PER_USER
    users_to_drop = users_to_rating_count[~users_to_keep_mask].index
    users_to_rating_count = users_to_rating_count[users_to_keep_mask]
    print(f'Number Users Dropped: {len(users_to_drop)}')
    print(f'Min Number Of Ratings Per User: {users_to_rating_count.min()}')
    print(f'Max Number Of Ratings Per User: {users_to_rating_count.max()}')
    print(f'Mean Number Of Ratings Per User: {users_to_rating_count.mean()}')
    return

if __name__ == '__main__':
    process_data()
    
