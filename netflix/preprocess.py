#!/usr/bin/python3

import pandas as pd
from typing import List, Tuple

RAW_DATA_CSV = './netflix_titles.csv'
OUTPUT_CSV = './processed.csv'

RELEVANT_COLUMNS = ['title', 'cast', 'director', 'country', 'type', 'listed_in']
COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING = ['cast', 'director']
assert set(COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING).issubset(RELEVANT_COLUMNS)

def expand_dataframe_list_values_for_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    return df.apply(lambda x: pd.Series(x[column_name].split(', ')), axis=1) \
                  .stack() \
                  .reset_index(level=1, drop=True) \
                  .to_frame(column_name) \
                  .join(df.drop(columns=[column_name]))

def preprocess_data() -> None:
    all_df = pd.read_csv(RAW_DATA_CSV, usecols=RELEVANT_COLUMNS)
    movies_df = all_df[all_df['type']=='Movie'].drop(columns=['type'])
    movies_df = movies_df.dropna()
    for column in COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING:
        movies_df = expand_dataframe_list_values_for_column(movies_df, column)
    movies_df.to_csv(OUTPUT_CSV)
    return

def main() -> None:
    preprocess_data()
    return

if __name__ == '__main__':
    main()
