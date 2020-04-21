#!/usr/bin/python3

import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

from misc_utilities import timer

matplotlib.use("Agg")

RAW_DATA_CSV = './netflix_titles.csv'
EDGE_LIST_CSV = './edge_list.csv'
PROJECTED_ACTORS_CSV = './projected_actors.csv'
PROJECTED_DIRECTORS_CSV = './projected_directors.csv'
PROJECTED_ACTORS_PNG = './projected_actors.png'
PROJECTED_DIRECTORS_PNG = './projected_directors.png'

RELEVANT_COLUMNS = ['title', 'cast', 'director', 'country', 'type', 'listed_in']
COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING = ['cast', 'director']
assert set(COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING).issubset(RELEVANT_COLUMNS)

def draw_graph_to_file(graph: nx.Graph, output_location: str) -> None:
    figure = plt.figure()
    plot = figure.add_subplot(111)
    nx.draw(graph, ax=plot)
    figure.savefig(output_location)
    return

def expand_dataframe_list_values_for_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    return df.apply(lambda x: pd.Series(x[column_name].split(', ')), axis=1) \
                  .stack() \
                  .reset_index(level=1, drop=True) \
                  .to_frame(column_name) \
                  .join(df.drop(columns=[column_name]))

def preprocess_data() -> None:
    with timer(section_name="Initial raw data loading"):
        all_df = pd.read_csv(RAW_DATA_CSV, usecols=RELEVANT_COLUMNS)
        movies_df = all_df[all_df['type']=='Movie'].drop(columns=['type'])
        movies_df = movies_df.dropna()
        for column in COLUMNS_WITH_LIST_VALUES_WORTH_EXPANDING:
            movies_df = expand_dataframe_list_values_for_column(movies_df, column)
        movies_df.to_csv(EDGE_LIST_CSV, index=False)
        movies_graph = nx.from_pandas_edgelist(movies_df, 'cast', 'director', True)
    with timer(section_name="Actor graph projection"):
        projected_actors_graph = nx.projected_graph(movies_graph, movies_df['cast'])
        projected_actors_edgelist = nx.convert_matrix.to_pandas_edgelist(projected_actors_graph)
        projected_actors_edgelist.rename(columns={"source": "source_actor", "target": "target_actor"})
        projected_actors_edgelist.to_csv(PROJECTED_ACTORS_CSV, index=False)
    with timer(section_name="Actor graph projection visualization"):
        draw_graph_to_file(projected_actors_graph, PROJECTED_ACTORS_PNG)
    with timer(section_name="Director graph projection"):
        projected_directors_graph = nx.projected_graph(movies_graph, movies_df['director'])
        projected_directors_edgelist = nx.convert_matrix.to_pandas_edgelist(projected_directors_graph)
        projected_directors_edgelist.rename(columns={"source": "source_director", "target": "target_director"})
        projected_directors_edgelist.to_csv(PROJECTED_DIRECTORS_CSV, index=False)
    with timer(section_name="Director graph projection visualization"):
        draw_graph_to_file(projected_directors_graph, PROJECTED_DIRECTORS_PNG)
    return

def main() -> None:
    preprocess_data()
    return

if __name__ == '__main__':
    main()
