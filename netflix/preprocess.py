#!/usr/bin/python3

import math
import pandas as pd
import networkx as nx
from typing import List

from misc_utilities import timer, histogram, debug_on_error

RAW_DATA_CSV = './netflix_titles.csv'

RELEVANT_COLUMNS = ['title', 'cast', 'director', 'country', 'type', 'listed_in']
COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING = ['cast', 'director']
assert set(COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING).issubset(RELEVANT_COLUMNS)

DIRECTOR_ACTOR_EDGE_LIST_CSV = './director_actor_edge_list.csv'
PROJECTED_ACTORS_CSV = './projected_actors.csv'
PROJECTED_DIRECTORS_CSV = './projected_directors.csv'

K_CORE_PROJECTED_ACTORS_CSV_TEMPLATE = './projected_actors_k_core_%d.csv'
K_CORE_PROJECTED_DIRECTORS_CSV_TEMPLATE = './projected_directors_k_core_%d.csv'
K_CORE_CHOICE_FOR_K = [10, 20, 30, 45, 60, 75, 100]

def expand_dataframe_list_values_for_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    return df.apply(lambda x: pd.Series(x[column_name].split(', ')), axis=1) \
                  .stack() \
                  .reset_index(level=1, drop=True) \
                  .to_frame(column_name) \
                  .join(df.drop(columns=[column_name]))

def load_raw_data() -> nx.Graph:
    all_df = pd.read_csv(RAW_DATA_CSV, usecols=RELEVANT_COLUMNS)
    movies_df = all_df[all_df['type']=='Movie'].drop(columns=['type']).dropna()
    #all_df = all_df[all_df.country.str.find(r'United States')>=0]
    for column in COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING:
        movies_df = expand_dataframe_list_values_for_column(movies_df, column)
    movies_df = movies_df.rename(columns={'cast': 'actor'})
    movies_df = movies_df[movies_df.director != movies_df.actor]
    print(f'Original Number of Directors: {len(movies_df.director.unique())}')
    print(f'Original Number of Actors: {len(movies_df.actor.unique())}')
    movies_df.to_csv(DIRECTOR_ACTOR_EDGE_LIST_CSV, index=False)
    movies_graph = nx.from_pandas_edgelist(movies_df, 'actor', 'director')
    return movies_graph    

def preprocess_data() -> None:
    with timer(section_name="Initial raw data loading"):
    with timer(section_name="Actor graph projection"):
        full_projected_actors_graph = nx.projected_graph(movies_graph, movies_df['actor'])
        projected_actors_edgelist = nx.convert_matrix.to_pandas_edgelist(full_projected_actors_graph)
        projected_actors_edgelist = projected_actors_edgelist.rename(columns={"source": "source_actor", "target": "target_actor"})
        projected_actors_edgelist.to_csv(PROJECTED_ACTORS_CSV, index=False)
        print(f"Number of Actors: {len(full_projected_actors_graph.nodes)}")
        assert set(projected_actors_edgelist.stack().unique()).issubset(set(full_projected_actors_graph.nodes))
        #assert set(projected_actors_edgelist.stack().unique()).issubset(set(movies_df.actor.unique()))
    with timer(section_name="Director graph projection"):
        full_projected_directors_graph = nx.projected_graph(movies_graph, movies_df['director'])
        projected_directors_edgelist = nx.convert_matrix.to_pandas_edgelist(full_projected_directors_graph)
        projected_directors_edgelist = projected_directors_edgelist.rename(columns={"source": "source_director", "target": "target_director"})
        projected_directors_edgelist.to_csv(PROJECTED_DIRECTORS_CSV, index=False)
        print(f"Number of Directors: {len(full_projected_directors_graph.nodes)}")
        assert set(projected_directors_edgelist.stack().unique()).issubset(set(full_projected_directors_graph.nodes))
        #assert set(projected_directors_edgelist.stack().unique()).issubset(set(movies_df.director.unique()))
    with timer(section_name="Actor graph K-core computation"):
        for k in K_CORE_CHOICE_FOR_K:
            k_core_actors_graph = nx.k_core(full_projected_actors_graph, k)
            k_core_actors_edgelist = nx.convert_matrix.to_pandas_edgelist(k_core_actors_graph)
            k_core_actors_edgelist = k_core_actors_edgelist.rename(columns={"source": "source_actor", "target": "target_actor"})
            k_core_actors_edgelist.to_csv(K_CORE_PROJECTED_ACTORS_CSV_TEMPLATE%k, index=False)
    with timer(section_name="Director graph K-core computation"):
        for k in K_CORE_CHOICE_FOR_K:
            k_core_directors_graph = nx.k_core(full_projected_directors_graph, k)
            k_core_directors_edgelist = nx.convert_matrix.to_pandas_edgelist(k_core_directors_graph)
            k_core_directors_edgelist = k_core_directors_edgelist.rename(columns={"source": "source_director", "target": "target_director"})
            k_core_directors_edgelist.to_csv(K_CORE_PROJECTED_DIRECTORS_CSV_TEMPLATE%k, index=False)
    return

@debug_on_error
def main() -> None:
    preprocess_data()
    return

if __name__ == '__main__':
    main()

